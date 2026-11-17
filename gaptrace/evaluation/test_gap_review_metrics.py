import json

import pytest

from evaluation.gap_review import evaluate


def report():
    cards = [{"id": name, "statement": name, "type": "untested_combination", "status": "unchecked", "doability": None}
             for name in ("bad", "good")]
    return json.dumps({"kind": "gap-investigation", "before": cards, "ranked": list(reversed(cards)), "reviews": []}).encode()


def test_missing_independent_labels_cannot_become_quality_or_future_results():
    result = evaluate(report())
    for ranking in result["rankings"].values():
        assert ranking["metrics"]["actionable_precision@5"]["value"] is None
        assert not any(key.startswith("future_") for key in ranking["metrics"])
        assert all("future_confirmed" not in item for item in ranking["items"])
    assert result["temporal_metrics"] is None


def test_existing_quality_metrics_evaluate_actual_frozen_order():
    raw = report()
    base = evaluate(raw)
    labels = {"source_report_sha256": base["source_report_sha256"], "gaps": [
        {"id": name, "statement": name, "candidates": [], "canonical_id": name,
         "judgment": {"grader": "test-adjudicator", "real": 5 if name == "good" else 1,
                      "open": 5, "doable": 5, "reason": "Synthetic test labels, not human evidence."}}
        for name in ("bad", "good")]}
    result = evaluate(raw, labels)
    assert result["rankings"]["before"]["metrics"]["actionable_precision@1"]["value"] == 0
    assert result["rankings"]["reviewed"]["metrics"]["actionable_precision@1"]["value"] == 1
    assert result["rankings"]["reviewed"]["order"] == ["good", "bad"]
    labels["source_report_sha256"] = "tampered"
    with pytest.raises(ValueError, match="different investigation"):
        evaluate(raw, labels)


def test_dropped_candidates_cannot_inflate_ranking_metrics():
    data = json.loads(report())
    data["ranked"] = data["ranked"][:1]
    with pytest.raises(ValueError, match="exact same"):
        evaluate(json.dumps(data).encode())
