import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.metrics import Evaluation, score

HERE = Path(__file__).resolve().parent


@pytest.fixture
def manifest():
    return json.loads((HERE / "fixture.json").read_text(encoding="utf-8"))


def load(manifest):
    return Evaluation.model_validate_json(json.dumps(manifest))


def test_score_uses_confirmed_matches_and_denominators(manifest):
    result = score(load(manifest), ("full", "vector"))
    run = result["systems"]["full"]["runs"][0]
    metrics = run["metrics"]
    assert metrics["hit_rate"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert metrics["coverage"]["value"] == 0.5
    hit = run["topics"]["synthetic-a"]["confirmed_gaps"]["both-colors"]
    assert hit["paper_id"] == "future-early" and hit["lead_days"] == 90
    assert metrics["lead_time_months"]["value"] == pytest.approx(2.957, abs=0.001)
    assert metrics["answer_accuracy"]["value"] == 0.5
    assert metrics["evidence_quality"]["value"] == pytest.approx(2 / 3)
    assert metrics["cross_document_retrieval"]["value"] == 1.0
    assert metrics["hallucination_rate"]["value"] == pytest.approx(1 / 3)
    assert metrics["response_seconds"]["value"] == 4.0
    assert metrics["retrieval_steps"]["value"] == 3.0
    assert metrics["mcp_token_savings"]["value"] == 0.7
    assert run["topics"]["synthetic-a"]["efficiency"]["total_tokens"] == 8000
    assert run["topics"]["synthetic-b"]["metrics"]["evidence_quality"]["value"] is None
    paired = result["paired_comparison"]["metrics"]["answer_accuracy"]
    assert paired["topic_deltas"] == {"synthetic-a": 1.0, "synthetic-b": 0.0}
    assert paired["non_tied_pairs"] == 1 and paired["p_value"] == 1.0


def test_repeats_report_mean_and_sample_spread(manifest):
    for index, correct in enumerate([False, True], start=2):
        repeat = copy.deepcopy(manifest["runs"][0])
        repeat["id"] = f"full-{index}"
        repeat["topics"][0]["answers"][0]["correct"] = correct
        manifest["runs"].append(repeat)
    result = score(load(manifest))["systems"]["full"]
    assert result["repeat_count"] == 3
    summary = result["repeat_summary"]["answer_accuracy"]
    assert summary == {"mean": pytest.approx(1 / 3), "sample_sd": pytest.approx(0.288675), "observations": 3}


@pytest.mark.parametrize(("change", "message"), [
    (lambda data: data["runs"][0]["topics"][0]["corpus"][0].update(published="2024-02-01"), "Post-cutoff"),
    (lambda data: data["runs"][0]["topics"][0].update(build_cloud_enabled=True), "disable cloud"),
    (lambda data: data["runs"][0]["topics"][0].update(build_cloud_calls=1), "less than or equal to 0"),
    (lambda data: data["runs"][0]["topics"][0]["gaps"][0]["candidates"][0].pop("recheck"), "strict recheck"),
    (lambda data: data["runs"][0]["topics"][0]["gaps"][0]["candidates"][0].update(published="2024-01-31"),
     "outside the grading window"),
    (lambda data: data["runs"][0]["topics"][0]["answers"][0]["claims"][0].update(passage_ids=["unknown"]),
     "absent from answer"),
    (lambda data: data["runs"][0]["topics"][0].update(answers=[]), "one result per question"),
    (lambda data: data["runs"][1]["topics"][0]["corpus"][0].update(published="2023-08-31"), "different corpus"),
    (lambda data: data.update(future_through="2024-10-30"), "nine calendar months"),
    (lambda data: data.update(model_training_cutoff="2024-02-01"), "precedes the model training cutoff"),
])
def test_rejects_invalid_protocol_inputs(manifest, change, message):
    change(manifest)
    with pytest.raises(ValidationError, match=message):
        load(manifest)


def test_command_requires_explicit_held_out_split(manifest, tmp_path):
    path = tmp_path / "manifest.json"
    output = tmp_path / "report.json"
    manifest["split"] = "held-out"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    command = [sys.executable, str(HERE / "metrics.py"), str(path), "--out", str(output)]
    first = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert first.returncode == 2 and "--split held-out" in first.stderr and not output.exists()
    subprocess.run([*command, "--split", "held-out"], check=True, timeout=10)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["split"] == "held-out" and len(report["manifest_sha256"]) == 64
