"""Hand-calculated cases for ranking, missing judgments, and review integrity."""
import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.metrics import Evaluation, Gap, score, compare_systems
from evaluation.ranking import bootstrap_interval, score_ranking, weighted_kappa
from evaluation.review import export_reviews, merge_reviews

HERE = Path(__file__).parent


def judgment(real=5, open=5, doable=5):
    return {"grader": "adjudicator", "real": real, "open": open, "doable": doable,
            "reason": "Independent reviewed evidence.",
            "doability_rubric": {"data": 2, "compute": 1, "blocks": 2, "scope": 1}}


def gap(id, canonical, **extra):
    return Gap.model_validate_json(json.dumps({"id": id, "statement": id, "candidates": [],
                                               "canonical_id": canonical, **extra}))


@pytest.fixture
def ranked():
    return [gap("bad", "bad", judgment=judgment(real=2)),
            gap("good", "good", gold_id="gold-a", judgment=judgment()),
            gap("duplicate", "good", gold_id="gold-a", judgment=judgment()),
            gap("medium", "medium", gold_id="gold-b", judgment=judgment(doable=3))]


def test_rank_metrics_use_frozen_order_and_penalize_duplicates(ranked):
    result = score_ranking(ranked, {"good": {}, "duplicate": {}, "medium": {}}, None, [1, 3, 5])
    metrics = result["metrics"]
    assert result["order"] == ["bad", "good", "duplicate", "medium"]
    assert [item["future_gain"] for item in result["items"]] == [0, 1, 0, 1]
    assert metrics["future_precision@1"]["value"] == 0
    assert metrics["future_precision@3"]["value"] == pytest.approx(1 / 3)
    assert metrics["future_precision@5"]["value"] == 0.4
    assert metrics["future_recall_within_list@3"]["value"] == 0.5
    assert metrics["future_ap_within_list@3"]["value"] == 0.25
    assert metrics["future_mrr"]["value"] == 0.5
    assert metrics["duplicate_rate"]["value"] == 0.25
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert metrics["future_ndcg_within_list@3"]["value"] == pytest.approx(expected)
    assert metrics["actionable_precision@3"]["value"] == pytest.approx(1 / 3)
    assert metrics["quality_ndcg_within_list@3"]["value"] == pytest.approx((7 / math.log2(3)) / (7 + 3 / math.log2(3)))
    assert metrics["list_fill@5"]["value"] == 0.8


def test_reordering_changes_ranking_scores_without_changing_hit_count(ranked):
    old = score_ranking(ranked, {"good": {}}, None, [1])["metrics"]
    new = score_ranking([ranked[1], ranked[0], *ranked[2:]], {"good": {}}, None, [1])["metrics"]
    assert old["future_precision@1"]["value"] == 0
    assert new["future_precision@1"]["value"] == 1
    assert old["future_mrr"]["value"] < new["future_mrr"]["value"]


def test_fixed_catalogue_recall_does_not_reward_duplicate_mapping(ranked):
    from evaluation.metrics import GoldGap
    gold = [GoldGap(id="gold-a", statement="A", relevance=3), GoldGap(id="gold-b", statement="B", relevance=2),
            GoldGap(id="gold-unseen", statement="C", relevance=1)]
    metrics = score_ranking(ranked, {}, gold, [3, 5], True)["metrics"]
    assert metrics["gold_recall@3"]["value"] == pytest.approx(1 / 3)
    assert metrics["gold_recall@5"]["value"] == pytest.approx(2 / 3)
    expected = (7 / math.log2(3) + 3 / math.log2(5)) / (7 + 3 / math.log2(3) + 1 / math.log2(4))
    assert metrics["gold_ndcg@5"]["value"] == pytest.approx(expected)
    assert score_ranking(ranked, {}, gold, [3])["metrics"]["gold_recall@3"]["value"] is None


def test_unknown_judgments_do_not_become_negatives(ranked):
    ranked[0] = gap("unknown", "unknown")
    metrics = score_ranking(ranked, {}, None, [1, 5])["metrics"]
    assert metrics["actionable_precision@5"]["value"] is None
    assert metrics["quality_ndcg_within_list@5"]["value"] is None
    assert metrics["gap_judgment_coverage"]["value"] == 0.75
    assert metrics["fully_supported_gap_rate"]["value"] is None
    assert metrics["gold_recall@5"]["value"] is None


def test_empty_list_is_zero_yield_and_precision_with_undefined_recall():
    metrics = score_ranking([], {}, None, [5])["metrics"]
    assert metrics["gap_yield"]["value"] == 0
    assert metrics["future_precision@5"]["value"] == 0
    assert metrics["actionable_precision@5"]["value"] == 0
    assert metrics["future_mrr"]["value"] == 0
    assert metrics["future_recall_within_list@5"]["value"] is None
    assert metrics["future_ndcg_within_list@5"]["value"] is None
    assert metrics["duplicate_rate"]["value"] is None


def test_evidence_closure_doability_and_calibration():
    evidence = {"passage_id": "p-1", "paper_id": "p", "passage_text": "The quoted limitation.",
                "quote": "quoted limitation", "supported": True, "grader": "human"}
    one = gap("one", "one", evidence=[evidence], judgment=judgment(), future_probability=1.0,
              predicted_status="open", reference_status="closed", predicted_doability=8)
    two = gap("two", "two", evidence=[{**evidence, "supported": False}], future_probability=0.0,
              predicted_status="unchecked", reference_status="open")
    result = score_ranking([one, two], {"one": {}}, None, [1])
    m = result["metrics"]
    assert m["gap_evidence_precision"]["value"] == 0.5
    assert m["fully_supported_gap_rate"]["value"] == 0.5
    assert m["independent_support_papers"]["value"] == 0.5
    assert m["false_open_rate"]["value"] == 1
    assert m["closure_accuracy"]["value"] == 0
    assert m["doability_mae"]["value"] == 2
    assert m["future_brier"]["value"] == 0
    assert m["future_ece"]["value"] == 0
    assert result["calibration_bins"][-1]["count"] == 1
    assert result["closure_confusion"]["open"]["unchecked"] == 1
    one.future_probability = 0.0
    assert score_ranking([one], {"one": {}}, None, [1])["metrics"]["future_brier"]["value"] == 1


@pytest.mark.parametrize("extra", [
    {"future_probability": float("nan")}, {"future_probability": 1.1}, {"predicted_doability": 9},
    {"judgment": judgment(real=True)}, {"judgment": judgment(open=0)},
    {"judgment": {**judgment(), "doability_rubric": {"data": 3, "compute": 1, "blocks": 2, "scope": 1}}},
    {"evidence": [{"passage_id": "p", "paper_id": "p", "passage_text": "actual", "quote": "invented",
                   "supported": True, "grader": "h"}]},
    {"ratings": [{"rater": "r1", "real": 5, "open": 5, "doable": 5, "reason": "ok"}] * 2},
])
def test_rejects_invalid_annotations(extra):
    with pytest.raises(ValidationError):
        gap("invalid", "invalid", **extra)


def test_weighted_kappa_and_deterministic_topic_bootstrap():
    assert weighted_kappa(list(zip(range(1, 6), [1, 2, 3, 5, 4]))) == pytest.approx(0.75)
    assert weighted_kappa([(1, 1), (5, 5)]) == 1
    assert weighted_kappa([(5, 5)] * 4) is None
    assert weighted_kappa([]) is None
    assert bootstrap_interval([1]) is None
    assert bootstrap_interval([1, 1])["lower"] == 1
    assert bootstrap_interval([-1, 0, 1]) == bootstrap_interval([-1, 0, 1])


def raw_fixture():
    return (HERE / "fixture.json").read_bytes()


def fill_forms(forms):
    for form in forms:
        for item in form["ratings"]:
            item.update(real=5, open=4, doable=3, reason="Rated independently from supplied source evidence.")
    return forms


def test_blinded_export_and_import_deduplicate_repeats():
    data = json.loads(raw_fixture())
    repeat = copy.deepcopy(data["runs"][0])
    repeat["id"] = "full-repeat"
    data["runs"].append(repeat)
    raw = json.dumps(data).encode()
    public, key, forms = export_reviews(raw, 3)
    assert len(public["items"]) == 3
    assert key["population_size"] == 3
    text = json.dumps(public)
    assert all(forbidden not in text for forbidden in ("system", "candidates", "judgment", "rank", "future", "both-colors"))
    assert export_reviews(raw, 3) == export_reviews(raw, 3)
    merged = merge_reviews(raw, key, fill_forms(forms))
    result = score(Evaluation.model_validate_json(json.dumps(merged)))
    assert result["human_agreement"]["double_rated_unique_gaps"] == 3
    assert result["human_agreement"]["pairs"][0]["paired_gaps"] == 3
    assert result["human_agreement"]["rating_summary"]["axes"]["real"]["gaps"] == 3
    assert result["human_agreement"]["rating_summary_by_system"]["full"]["axes"]["real"]["gaps"] == 2
    assert result["human_agreement"]["pairs"][0]["axes"]["real"]["linear_weighted_kappa"] is None
    # Raw rating forms do not silently become adjudicated quality labels.
    assert result["systems"]["full"]["runs"][0]["metrics"]["actionable_precision@5"]["value"] is None


@pytest.mark.parametrize("mutation, message", [
    (lambda k, f: k.update(manifest_sha256="wrong"), "exact manifest"),
    (lambda k, f: f[0].update(manifest_sha256="wrong"), "form does not match"),
    (lambda k, f: f[0]["ratings"].pop(), "Every sampled gap"),
    (lambda k, f: f[0]["ratings"].append(copy.deepcopy(f[0]["ratings"][0])), "Duplicate or unknown"),
    (lambda k, f: f[1].update(rater=f[0]["rater"]), "Duplicate rater"),
])
def test_review_import_rejects_corrupt_or_incomplete_forms(mutation, message):
    raw = raw_fixture()
    _, key, forms = export_reviews(raw, 3)
    fill_forms(forms)
    mutation(key, forms)
    with pytest.raises(ValueError, match=message):
        merge_reviews(raw, key, forms)


def test_protocol_rejects_unknown_gold_and_out_of_corpus_evidence():
    data = json.loads(raw_fixture())
    data["runs"][0]["topics"][0]["gaps"][0]["gold_id"] = "unknown"
    with pytest.raises(ValidationError, match="Unknown gold"):
        Evaluation.model_validate_json(json.dumps(data))
    del data["runs"][0]["topics"][0]["gaps"][0]["gold_id"]
    data["ranking_ks"] = [5, 5]
    with pytest.raises(ValidationError, match="unique"):
        Evaluation.model_validate_json(json.dumps(data))
    data["ranking_ks"] = [5]
    data["runs"][0]["topics"][0]["gaps"][0]["evidence"] = [{
        "passage_id": "x", "paper_id": "outside", "passage_text": "text", "quote": "text", "supported": True, "grader": "h"}]
    with pytest.raises(ValidationError, match="outside the corpus"):
        Evaluation.model_validate_json(json.dumps(data))


def test_comparison_reports_adjusted_tests_and_topic_units():
    result = score(Evaluation.model_validate_json(raw_fixture()), ("full", "vector"))
    compared = result["paired_comparison"]["metrics"]
    assert compared["future_precision@5"]["direction"] == "higher"
    assert compared["future_brier"]["direction"] == "lower"
    assert compared["answer_accuracy"]["topic_bootstrap_95_ci"]["lower"] == 0
    assert compared["answer_accuracy"]["holm_p_value"] >= compared["answer_accuracy"]["p_value"]
    assert compared["future_brier"]["p_value"] is None


def test_exact_sign_test_and_holm_adjustment_against_known_values():
    def run(values):
        return {"metrics": {name: {} for name in values},
                "topics": {f"t{i}": {"metrics": {name: {"value": series[i]} for name, series in values.items()}}
                           for i in range(6)}}
    left = run({"a": [1] * 6, "b": [1] * 6, "c": [1, 1, 1, -1, -1, -1]})
    right = run({"a": [0] * 6, "b": [0] * 6, "c": [0] * 6})
    compared = compare_systems([left], [right], [f"t{i}" for i in range(6)])
    assert compared["a"]["p_value"] == 2 / 64
    assert compared["a"]["holm_p_value"] == 3 * 2 / 64
    assert compared["b"]["holm_p_value"] == 3 * 2 / 64
    assert compared["c"]["p_value"] == 1
    assert compared["a"]["topic_bootstrap_95_ci"]["lower"] == 1


def test_zero_ideal_catalogue_and_missing_canonical_are_explicit(ranked):
    from evaluation.metrics import GoldGap
    catalogue = [GoldGap(id="negative", statement="Already done", relevance=0)]
    m = score_ranking(ranked, {}, catalogue, [5], True)["metrics"]
    assert m["gold_ndcg@5"]["value"] is None
    assert m["gold_recall@5"]["value"] is None
    ranked[0].canonical_id = None
    result = score_ranking(ranked, {}, None, [5])
    assert result["metrics"]["duplicate_rate"]["value"] is None
    assert "duplicate_rate" in result["incomplete_metrics"]


def test_incomplete_labels_withhold_primary_run_metric_even_with_empty_topic():
    report = score(Evaluation.model_validate_json(raw_fixture()))
    metric = report["systems"]["full"]["runs"][0]["metrics"]["actionable_precision@5"]
    assert metric["value"] is None
    assert metric["incomplete_labels"] is True
    assert metric["available_subset"]["value"] == 0


def test_annotated_example_is_valid_and_reports_expected_synthetic_scores():
    evaluation = Evaluation.model_validate_json((HERE / "examples" / "ranking-fixture.json").read_bytes())
    report = score(evaluation)
    summary = report["systems"]["full"]["repeat_summary"]
    assert summary["future_precision@5"]["mean"] == 0.1
    assert summary["actionable_precision@5"]["mean"] == 0.1
    assert summary["gold_recall@5"]["mean"] == 1
    assert summary["future_brier"]["mean"] == pytest.approx(0.05125)
    assert report["human_agreement"]["double_rated_unique_gaps"] == 3


def test_review_cli_and_metrics_cli_round_trip(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(raw_fixture())
    directory = tmp_path / "review"
    command = [sys.executable, str(HERE / "review.py")]
    subprocess.run([*command, "export", str(manifest), "--out", str(directory), "--sample-size", "3"], check=True, timeout=15)
    paths = [directory / f"rater-{i}.json" for i in (1, 2)]
    forms = fill_forms([json.loads(path.read_text()) for path in paths])
    for path, form in zip(paths, forms):
        path.write_text(json.dumps(form), encoding="utf-8")
    merged = tmp_path / "merged.json"
    subprocess.run([*command, "merge", str(manifest), "--key", str(directory / "private-key.json"),
                    "--ratings", *map(str, paths), "--out", str(merged)], check=True, timeout=15)
    output = tmp_path / "report.json"
    subprocess.run([sys.executable, str(HERE / "metrics.py"), str(merged), "--out", str(output)], check=True, timeout=15)
    assert json.loads(output.read_text())["human_agreement"]["double_rated_unique_gaps"] == 3
