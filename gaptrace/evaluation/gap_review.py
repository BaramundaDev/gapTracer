"""Evaluate a frozen investigation order using the existing ranking metrics.

Automated operational findings and independent quality annotations stay separate.
No retrospective future-hit labels are inferred from present-day closure searches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from evaluation.metrics import Gap, GoldGap, by_id
from evaluation.ranking import score_ranking


def evaluate(raw: bytes, labels=None):
    investigation = json.loads(raw)
    if investigation.get("kind") != "gap-investigation":
        raise ValueError("Input must be a recorded gap investigation")
    digest = hashlib.sha256(raw).hexdigest()
    before = by_id([Gap.model_validate_json(json.dumps({"id": c["id"], "statement": c["statement"],
        "type": c["type"], "predicted_status": c["status"], "predicted_doability": c["doability"], "candidates": []}))
        for c in investigation["before"]], "gap before review")
    after = investigation["ranked"]
    if len({c["id"] for c in after}) != len(after) or set(before) != {c["id"] for c in after}:
        raise ValueError("Before and after lists must contain the exact same unique candidates")
    if any(c["statement"] != before[c["id"]].statement for c in after):
        raise ValueError("Gap statements changed during assessment")
    annotated = {}
    gold, mapping = None, False
    if labels is not None:
        if labels["source_report_sha256"] != digest:
            raise ValueError("Independent annotations refer to a different investigation file")
        annotated = by_id([Gap.model_validate_json(json.dumps(g)) for g in labels["gaps"]], "annotated gap")
        if any(g.candidates or g.future_probability is not None for g in annotated.values()):
            raise ValueError("Current gap review annotations must not contain retrospective future-confirmation labels")
        if annotated.keys() != before.keys() or any(g.statement != before[gid].statement for gid, g in annotated.items()):
            raise ValueError("Annotate every frozen candidate without changing its id or statement")
        if labels.get("gold_gaps") is not None:
            gold = [GoldGap.model_validate_json(json.dumps(g)) for g in labels["gold_gaps"]]
            by_id(gold, "gold gap")
        mapping = labels.get("gold_mapping_complete", False)
        if mapping and gold is None:
            raise ValueError("Complete gold mapping requires a fixed catalogue")
        if any(g.gold_id and g.gold_id not in {item.id for item in gold or []} for g in annotated.values()):
            raise ValueError("Unknown gold catalogue id")
    results = {}
    for name, cards in (("before", investigation["before"]), ("reviewed", after)):
        gaps = []
        for card in cards:
            gap = (annotated.get(card["id"]) or before[card["id"]]).model_copy(deep=True)
            gap.predicted_status, gap.predicted_doability = card["status"], card["doability"]
            gaps.append(gap)
        ranked = score_ranking(gaps, {}, gold, [1, 3, 5, 10], mapping)
        # Current closure searches are NOT a dated future-confirmation experiment.
        ranked["metrics"] = {key: value for key, value in ranked["metrics"].items() if not key.startswith("future_")}
        ranked["incomplete_metrics"] = [key for key in ranked["incomplete_metrics"] if not key.startswith("future_")]
        ranked.pop("calibration_bins", None)
        for group in ranked["by_type"].values():
            group.pop("hit_rate", None)
        for item in ranked.get("items", []):
            item.pop("future_gain", None)
            item.pop("future_confirmed", None)
        results[name] = ranked
    return {"kind": "gap-investigation-ranking-evaluation", "source_report_sha256": digest,
        "primary_quality_endpoint": "actionable_precision@5", "catalogue_endpoints": ["gold_ndcg@5", "gold_recall@5"],
        "rankings": results, "automated_status_counts": dict(Counter(c["status"] for c in after)),
        "reviewed_gaps": len(investigation["reviews"]), "temporal_metrics": None,
        "limitations": ["Quality labels are supplied independently; none are inferred from automated assessments.",
            "No future_precision or hit_rate is measured by this present-day investigation.",
            "Independent judgments, complete duplicate labels and a fixed gold catalogue are required for their respective quality metrics.",
            "One corpus comparison is descriptive; it is not a held-out multi-topic research result."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = args.report.read_bytes()
        result = evaluate(raw, json.loads(args.labels.read_bytes()) if args.labels else None)
        investigation = json.loads(raw)
        template = {"source_report_sha256": result["source_report_sha256"], "gold_gaps": None,
                    "gold_mapping_complete": False, "gaps": [
                        Gap(id=c["id"], statement=c["statement"], type=c["type"], candidates=[]).model_dump(mode="json")
                        for c in investigation["before"]]}
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "ranking-report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (args.out / "independent-labels-template.json").write_text(json.dumps(template, indent=2), encoding="utf-8")
        (args.out / "investigation.json").write_bytes(raw)
        print(f"Ranking report: {args.out / 'ranking-report.json'}")
        print("actionable_precision@5 requires independent real/open/doable judgments; missing labels remain undefined.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Gap ranking evaluation error: {exc}\n")


if __name__ == "__main__":
    main()
