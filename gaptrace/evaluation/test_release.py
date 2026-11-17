import copy
import json
from pathlib import Path

import pytest

from evaluation.metrics import Evaluation
from evaluation.release import Policy, assess


def inputs():
    raw = json.loads(Path(__file__).with_name("fixture.json").read_text(encoding="utf-8"))
    raw["split"] = "held-out"
    policy = Policy(description="Unit-test targets only", min_topics=1, min_questions=1,
                    min_repeats=1, min_claims_per_run=1,
                    minimums={"answer_accuracy": 0.5}, maximums={"hallucination_rate": 0.5})
    return raw, policy


def test_complete_run_meets_test_policy():
    raw, policy = inputs()
    assert assess(Evaluation.model_validate_json(json.dumps(raw)), policy, "full")["status"] == "passed"


@pytest.mark.parametrize("mode,split", [("smoke", "held-out"), ("benchmark", "development")])
def test_smoke_or_development_is_not_release_evidence(mode, split):
    raw, policy = inputs()
    raw.update(mode=mode, split=split)
    assert assess(Evaluation.model_validate_json(json.dumps(raw)), policy, "full")["status"] == "failed"


def test_missing_labels_and_empty_claims_do_not_pass():
    raw, policy = inputs()
    policy.minimums["actionable_precision@5"] = 0
    for topic in raw["runs"][0]["topics"]:
        for answer in topic["answers"]:
            answer["claims"] = []
    result = assess(Evaluation.model_validate_json(json.dumps(raw)), policy, "full")
    assert result["status"] == "failed"
    failed = [c["check"] for c in result["checks"] if not c["passed"]]
    assert any("claim_count" in name for name in failed)
    assert any("actionable_precision" in name for name in failed)


def test_bad_repeat_cannot_be_hidden_by_average():
    raw, policy = inputs()
    repeat = copy.deepcopy(raw["runs"][0])
    repeat["id"] = "bad-repeat"
    for topic in repeat["topics"]:
        for answer in topic["answers"]:
            answer["correct"] = False
    raw["runs"].append(repeat)
    assert assess(Evaluation.model_validate_json(json.dumps(raw)), policy, "full")["status"] == "failed"


def test_unknown_metric_is_configuration_error():
    raw, policy = inputs()
    policy.minimums["typo"] = 0
    with pytest.raises(ValueError, match="Unknown policy metric"):
        assess(Evaluation.model_validate_json(json.dumps(raw)), policy, "full")
