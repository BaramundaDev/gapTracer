"""Synthetic controls for scientific importance, leakage and independent labels."""
import copy
import json
from datetime import date

import pytest
from pydantic import ValidationError

from evaluation.reactivation import Label, annotation_template, freeze, nested_papers, replay, score, verify, sha
from evaluation.knowledge_benchmark import score_queries
from gaptrace.ingest import PaperImport


def analysis():
    return json.dumps({"kind": "reactivation-analysis", "opportunities": [
        {"id": "trivial", "status": "reactivation_hypothesis", "importance_estimate": 1},
        {"id": "ambitious", "status": "reactivation_hypothesis", "importance_estimate": 4},
        {"id": "duplicate", "status": "reactivation_hypothesis", "importance_estimate": 4}]}).encode()


def label(pid, importance=4, **overrides):
    value = {"prediction_id": pid, "gold_id": "important" if importance == 4 else "minor", "canonical_id": pid,
        "reference": "reactivation", "old_supported": True, "new_supported": True, "bridge_specific": True, "still_open": True,
        "importance": importance, "consequence": importance, "information_gain": importance, "reach": importance,
        "reason": "Synthetic test-only reference judgment.", "label_authors": ["test-label-author"],
        "reviews": [{"reviewer": reviewer, "reviewer_type": "human", "decision": "accept", "reason": "Synthetic unit-test declaration, not human evidence."}
                    for reviewer in ("test-reviewer-a", "test-reviewer-b")]}
    return {**value, **overrides}


def annotations(raw):
    return {"analysis_sha256": sha(raw), "provenance": "SYNTHETIC UNIT TEST ONLY", "gold_mapping_complete": True,
        "gold": [{"id": "important", "statement": "Ambitious and scientifically important", "importance": 4},
                 {"id": "minor", "statement": "Easy but low consequence", "importance": 1},
                 {"id": "missed", "statement": "Important unreturned opportunity", "importance": 4}],
        "labels": [label("trivial", 1), label("ambitious"), label("duplicate", canonical_id="ambitious")]}


def test_importance_is_independent_of_easy_resources_and_duplicates_do_not_gain():
    raw = analysis()
    report = score(raw, annotations(raw))
    assert report["metrics"]["opportunity_precision@5"] == 1 / 5
    assert report["metrics"]["gold_recall@5"] == 2 / 3
    assert report["metrics"]["duplicate_rate"] == 1 / 3
    assert 0 < report["metrics"]["gold_importance_ndcg@5"] < 1
    # The primary label is importance, not small-team feasibility.
    assert "feasibility" not in Label.model_fields


def test_missing_labels_are_undefined_not_zero_and_wrong_snapshot_is_rejected():
    raw = analysis()
    labels = annotations(raw)
    labels["labels"] = labels["labels"][:1]
    assert score(raw, labels)["metrics"]["opportunity_precision@5"] is None
    assert score(raw, labels)["metrics"]["gold_importance_ndcg@5"] is None
    labels["analysis_sha256"] = "wrong"
    with pytest.raises(ValueError, match="different exact"):
        score(raw, labels)
    assert annotation_template(raw)["labels"][0]["importance"] is None


@pytest.mark.parametrize("field", ["old_supported", "new_supported", "bridge_specific", "still_open"])
def test_unsupported_or_already_closed_chain_never_earns_valid_opportunity_gain(field):
    raw = analysis()
    labels = annotations(raw)
    labels["labels"][1][field] = False
    labels["labels"][2][field] = False
    assert score(raw, labels)["metrics"]["opportunity_precision@5"] == 0
    assert score(raw, labels)["metrics"]["false_reactivation_rate"] == 2 / 3


def test_authors_cannot_supply_their_own_independent_review():
    value = label("p")
    value["reviews"][0]["reviewer"] = "test-label-author"
    with pytest.raises(ValidationError, match="independent"):
        Label.model_validate(value)
    value = label("p")
    value["reviews"][0]["decision"] = "reject"
    with pytest.raises(ValidationError, match="adjudicator"):
        Label.model_validate(value)


def test_nested_budgets_preserve_historical_and_recent_strata():
    source = PaperImport.model_validate({"papers": [{"id": f"p-{i}", "title": str(i), "published": f"{2010+i}-01-01",
        "sections": [{"heading": "Methods", "text": "Synthetic engineering source"}]} for i in range(8)]})
    papers = nested_papers(source.papers, date(2013, 1, 1))
    assert papers[:2] == [source.papers[0], source.papers[4]]
    assert papers[:4] == nested_papers(source.papers, date(2013, 1, 1))[:4]
    assert len({p.id for p in papers}) == 8


def test_freeze_detects_tampering_and_rejects_future_material(tmp_path):
    source = {"papers": [{"id": "a", "title": "Synthetic old source", "published": "2018-01-01", "sections": [{"heading": "Methods", "text": "Old blocker"}]},
                          {"id": "b", "title": "Synthetic new source", "published": "2024-01-01", "sections": [{"heading": "Methods", "text": "New capability"}]}]}
    (tmp_path / "source.json").write_text(json.dumps(source))
    (tmp_path / "pool.json").write_text(json.dumps({"as_of": "2026-01-01", "provider": "openalex", "searches": []}))
    spec = {"mode": "smoke", "as_of": "2026-01-01", "model_training_cutoff": "2025-01-01",
            "model_training_cutoff_reference": "Synthetic fixture assumption, not model provenance", "protocol_reviewed_by": "test-only",
            "topics": [{"id": "t", "split": "development", "source": "source.json", "search_pool": "pool.json", "historical_through": "2020-01-01"}]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    freeze(path, tmp_path / "frozen")
    verify(tmp_path / "frozen")
    (tmp_path / "frozen/sources/t.json").write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        verify(tmp_path / "frozen")
    source["papers"][1]["published"] = "2026-02-01"
    (tmp_path / "source.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="post-as_of"):
        freeze(path, tmp_path / "invalid")


def test_missing_frozen_query_never_falls_back_to_live_index():
    search = replay({"as_of": "2026-01-01", "searches": []})
    with pytest.raises(OSError, match="live fallback is disabled"):
        search("unrecorded", "openalex", 5, "2026-01-01")


def test_retrieval_recall_measures_required_raw_passages():
    class Hit:
        def __init__(self, id): self.id = id
    class Index:
        def search(self, *args): return [Hit("a"), Hit("c")]
    result = score_queries(Index(), [{"id": "q", "text": "find blocker", "gold_passage_ids": ["a", "b"]}], {"a", "b", "c"})
    assert result["metrics"]["evidence_recall@5"] == 0.5
    assert result["metrics"]["mrr"] == 1
