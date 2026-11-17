"""Apply frozen acceptance targets to a validated recorded research manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from evaluation.metrics import Evaluation, Id, Record, Score, score

Positive = Annotated[int, Field(gt=0, strict=True)]


class Policy(Record):
    description: Id
    min_topics: Positive
    min_questions: Positive
    min_repeats: Positive
    min_claims_per_run: Positive
    minimums: dict[Id, Score]
    maximums: dict[Id, Score]

    @model_validator(mode="after")
    def nonempty(self):
        if not self.minimums or not self.maximums or self.minimums.keys() & self.maximums.keys():
            raise ValueError("Require nonempty, disjoint metric minimums and maximums")
        return self


def assess(evaluation: Evaluation, policy: Policy, system: str) -> dict:
    report = score(evaluation)
    if system not in report["systems"]:
        raise ValueError(f"Unknown system {system}")
    checks = []

    def check(name, observed, target, passed):
        checks.append({"check": name, "observed": observed, "target": target, "passed": bool(passed)})

    check("benchmark_mode", evaluation.mode, "benchmark", evaluation.mode == "benchmark")
    check("held_out_split", evaluation.split, "held-out", evaluation.split == "held-out")
    check("topic_count", len(evaluation.topics), policy.min_topics, len(evaluation.topics) >= policy.min_topics)
    questions = sum(len(t.questions) for t in evaluation.topics)
    check("question_count", questions, policy.min_questions, questions >= policy.min_questions)
    selected = report["systems"][system]
    check("repeat_count", selected["repeat_count"], policy.min_repeats, selected["repeat_count"] >= policy.min_repeats)
    for raw, run in zip([r for r in evaluation.runs if r.system == system], selected["runs"], strict=True):
        claims = sum(len(a.claims) for t in raw.topics for a in t.answers)
        check(f"{raw.id}/claim_count", claims, policy.min_claims_per_run, claims >= policy.min_claims_per_run)
        # Each repeat must pass; a good repeat cannot hide a bad one. Undefined fails.
        for direction, targets in [("min", policy.minimums), ("max", policy.maximums)]:
            for metric, target in targets.items():
                if metric not in run["metrics"]:
                    raise ValueError(f"Unknown policy metric {metric}")
                measurement = run["metrics"][metric]
                value = measurement["value"]
                good = (value is not None and not measurement.get("incomplete_labels", False)
                        and (value >= target if direction == "min" else value <= target))
                check(f"{raw.id}/{direction}/{metric}", measurement, target, good)
    return {"status": "passed" if all(c["passed"] for c in checks) else "failed", "system": system,
            "checks": checks,
            "limitations": ["Targets are acceptance policy, not estimates of future performance.",
                            "Passing needs independently audited labels and provenance; the gate cannot establish judge independence.",
                            "Three cached repeats do not provide three independent generations.",
                            "Use the scorer's paired topic comparisons and confidence intervals for improvement claims."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--system", required=True)
    parser.add_argument("--policy", type=Path, default=Path(__file__).with_name("release-policy.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        raw, policy_raw = args.manifest.read_bytes(), args.policy.read_bytes()
        result = assess(Evaluation.model_validate_json(raw), Policy.model_validate_json(policy_raw), args.system)
        result.update(manifest_sha256=hashlib.sha256(raw).hexdigest(), policy_sha256=hashlib.sha256(policy_raw).hexdigest())
        with args.out.open("x", encoding="utf-8") as output:
            output.write(json.dumps(result, indent=2) + "\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Release evaluation error: {exc}\n")
    print(f"{result['status']}: {args.out}")
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
