import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.calibration import CalibrationSet, Observation, binary_scores, evaluate, wilson
from evaluation.readiness import discover, label_hash, report


def controls():
    cases = []
    for task in ("answer_correctness", "claim_support", "gap_match"):
        for split in ("development-calibration", "development-validation"):
            for label in (False, True):
                cases.append({"id": f"{task}-{split}-{label}", "task": task, "split": split,
                    "payload": {"example": f"{task} {split} {label}"}, "label": label,
                    "label_authors": ["test-author"], "reviews": [
                        {"reviewer": rater, "reviewer_type": "human", "label": label, "reason": "Test-only supplied review declaration."}
                        for rater in ("test-rater-1", "test-rater-2")]})
    return {"provenance": "Synthetic unit-test records; never independent real-world evidence.", "cases": cases}


def load(data):
    return CalibrationSet.model_validate_json(json.dumps(data))


def observations(dataset, fail_validation=False):
    return [Observation(case_id=case.id, judge="scripted", positive_probability=(
        (0.1 if case.label else 0.9) if fail_validation and case.split == "development-validation"
        else (0.9 if case.label else 0.1)), reason="scripted test") for case in dataset.cases]


def test_calibration_reports_confusion_and_separate_validation():
    dataset = load(controls())
    result = evaluate(dataset, observations(dataset), min_per_class=1)
    assert result["status"] == "passed"
    for task in result["judges"]["scripted"]["tasks"].values():
        assert task["threshold"] == 0.9
        assert task["validation"]["tp"] == task["validation"]["tn"] == 1
    # Test-only small controls fail the real minimum sample-size gate.
    assert evaluate(dataset, observations(dataset))["status"] == "failed"


def test_validation_labels_cannot_be_used_to_tune_threshold():
    dataset = load(controls())
    result = evaluate(dataset, observations(dataset, True), min_per_class=1)
    assert result["status"] == "failed"
    for task in result["judges"]["scripted"]["tasks"].values():
        assert task["threshold"] == 0.9 and task["validation"]["accuracy"] == 0


@pytest.mark.parametrize("change", [
    lambda data: data["cases"][0].update(split="held-out"),
    lambda data: data["cases"][0]["reviews"][0].update(reviewer="test-author"),
    lambda data: data["cases"][0].update(reviews=data["cases"][0]["reviews"][:1]),
    lambda data: data["cases"][0]["reviews"][0].update(reviewer_type="model"),
    lambda data: data["cases"][0]["reviews"][0].update(label=True),
])
def test_independent_review_and_development_only_contract(change):
    data = controls()
    change(data)
    with pytest.raises(ValidationError):
        load(data)


def test_disputed_controls_need_separate_adjudication():
    data = controls()
    case = data["cases"][0]
    case["reviews"][0]["label"] = not case["label"]
    case["adjudication"] = {"reviewer": "test-rater-3", "reviewer_type": "human", "label": case["label"], "reason": "Test-only resolution."}
    load(data)
    case["adjudication"]["reviewer"] = "test-rater-2"
    with pytest.raises(ValidationError, match="adjudicator"):
        load(data)


def test_missing_or_duplicate_judgments_do_not_become_negatives():
    dataset = load(controls())
    measured = observations(dataset)
    with pytest.raises(ValueError, match="every control"):
        evaluate(dataset, measured[:-1])
    with pytest.raises(ValueError, match="Duplicate judge"):
        evaluate(dataset, [*measured, measured[0]])
    data = controls()
    data["cases"][1]["payload"] = copy.deepcopy(data["cases"][0]["payload"])
    with pytest.raises(ValidationError, match="Duplicate judge control"):
        load(data)


def test_false_positive_rate_and_wilson_intervals_have_explicit_denominators():
    score = binary_scores([(True, 0.9), (False, 0.9), (False, 0.1)], 0.8)
    assert score["tp"] == score["fp"] == score["tn"] == 1 and score["fn"] == 0
    assert score["precision"] == score["false_positive_rate"] == 0.5
    assert wilson(0, 0) is None
    assert wilson(10, 10)["lower"] < 1


def test_workspace_audit_does_not_invent_a_frozen_benchmark(tmp_path):
    workspace = tmp_path / "project"
    (workspace / "data").mkdir(parents=True)
    (workspace / "evaluation" / "examples").mkdir(parents=True)
    (workspace / "data" / "plan.json").write_text(json.dumps({"mode": "smoke", "topics": [{"questions": [{"id": "q"}]}]}))
    inventory = discover(workspace)
    assert not inventory["frozen_candidates"] and inventory["plans"][0]["questions"] == 1
    result = report(workspace, tmp_path / "report")
    assert result["status"] == "blocked_by_missing_or_invalid_inputs"
    assert result["held_out_executed"] is False and result["quality_metrics"] is None
    assert "unmeasured" in (tmp_path / "report" / "HELD_OUT_REPORT.md").read_text()


def test_review_fingerprints_do_not_depend_on_set_iteration_order():
    from evaluation.metrics import Question
    first = Question(id="q", text="Q?", gold_answer="A", gold_passage_ids={"p1", "p2"}, required_paper_ids={"a", "b"})
    second = Question.model_validate_json(first.model_dump_json())
    assert label_hash("topic", first) == label_hash("topic", second)
    second.gold_answer = "Changed"
    assert label_hash("topic", first) != label_hash("topic", second)


def test_retrieval_controls_cannot_call_graph_or_sql_tools(tmp_path):
    from evaluation.baselines import RetrievalOnlyToolbox
    from gaptrace.bundle import PassageStore, read_bundle
    from gaptrace.sample import write_sample
    from gaptrace.settings import load_settings
    root = write_sample(tmp_path / "corpus")
    toolbox = RetrievalOnlyToolbox(read_bundle(root / "bundle"), PassageStore(root / "passages.jsonl"), load_settings())
    for tool in ("neighbours", "get_trail", "find_gaps", "lookup_fact"):
        assert "unavailable" in toolbox.call(tool, {})
    assert toolbox.graph.number_of_edges() == 0
    assert all(not note.links and note.fact is None for note in toolbox.notes.values())


@pytest.mark.parametrize("mode", ["full", "vector-passages", "hybrid-passages"])
def test_added_baselines_dispatch_actual_embedding_adapter(tmp_path, mode):
    from evaluation.baselines import answer
    from gaptrace.gateway import Gateway
    from gaptrace.sample import write_sample
    from gaptrace.settings import load_settings
    root = write_sample(tmp_path / "corpus")
    settings = load_settings()
    calls = []
    def embed(texts):
        calls.append(texts)
        return [[1.0, float("replay" in text.lower())] for text in texts]
    gateway = Gateway(settings, {"local": lambda *args: '{"answer":"NOT FOUND"}'})
    result = answer("replay", root, mode, gateway, settings, tmp_path / "run", embed)
    assert calls and result["baseline"] == mode
