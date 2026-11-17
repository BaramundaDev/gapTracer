"""Calibrate binary judges on independently reviewed development controls only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import inspect
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from evaluation.metrics import Id, Record, Score, by_id

TASKS = ("answer_correctness", "claim_support", "gap_match")


class Review(Record):
    reviewer: Id
    reviewer_type: Literal["human"]
    label: bool
    reason: Id


class CalibrationCase(Record):
    id: Id
    task: Literal["answer_correctness", "claim_support", "gap_match"]
    split: Literal["development-calibration", "development-validation"]
    payload: dict
    label: bool
    label_authors: list[Id] = Field(min_length=1)
    reviews: list[Review] = Field(min_length=2)
    adjudication: Review | None = None

    @model_validator(mode="after")
    def independent_review(self):
        reviewers = by_id(self.reviews, "calibration reviewer", "reviewer")
        if set(reviewers) & set(self.label_authors):
            raise ValueError("Label authors cannot independently review their own controls")
        votes = {review.label for review in self.reviews}
        if votes == {self.label}:
            return self
        if (self.adjudication is None or self.adjudication.label != self.label
                or self.adjudication.reviewer in set(reviewers) | set(self.label_authors)):
            raise ValueError("Disputed labels need a separate independent adjudicator")
        return self


class CalibrationSet(Record):
    version: Literal[1] = 1
    provenance: Id
    cases: list[CalibrationCase] = Field(min_length=1)
    first_threshold: Score = 0.6
    recheck_threshold: Score = 0.8

    @model_validator(mode="after")
    def unique(self):
        if self.recheck_threshold <= self.first_threshold:
            raise ValueError("Strict matcher threshold must exceed the first threshold")
        by_id(self.cases, "calibration case")
        # An identical payload cannot select a threshold and validate that threshold.
        seen = {}
        for case in self.cases:
            fingerprint = hashlib.sha256(json.dumps({"task": case.task, "payload": case.payload}, sort_keys=True).encode()).hexdigest()
            if fingerprint in seen:
                raise ValueError("Duplicate judge control payloads cannot count as independent observations")
            seen[fingerprint] = case.split
        return self


class Judgment(Record):
    positive_probability: Score
    reason: Id


class Observation(Judgment):
    case_id: Id
    judge: Id


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def wilson(successes, total, z=1.959963984540054):
    if not total:
        return None
    p = successes / total
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return {"lower": max(0, center - half), "upper": min(1, center + half), "n": total}


def binary_scores(rows, threshold):
    # rows are (reference label, predicted positive probability).
    tp = sum(label and probability >= threshold for label, probability in rows)
    fp = sum(not label and probability >= threshold for label, probability in rows)
    tn = sum(not label and probability < threshold for label, probability in rows)
    fn = sum(label and probability < threshold for label, probability in rows)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": len(rows),
            "accuracy": fraction(tp + tn, len(rows)), "precision": fraction(tp, tp + fp),
            "recall": fraction(tp, tp + fn), "false_positive_rate": fraction(fp, fp + tn),
            "accuracy_95_ci": wilson(tp + tn, len(rows)), "precision_95_ci": wilson(tp, tp + fp),
            "false_positive_rate_95_ci": wilson(fp, fp + tn),
            "brier": sum((probability - label) ** 2 for label, probability in rows) / len(rows) if rows else None}


def evaluate(dataset, observations, min_per_class=10, min_accuracy=0.9, max_fpr=0.05):
    if min_per_class < 1 or not 0 <= min_accuracy <= 1 or not 0 <= max_fpr <= 1:
        raise ValueError("Invalid frozen calibration requirements")
    cases = by_id(dataset.cases, "calibration case")
    judges = {}
    for observation in observations:
        if observation.case_id not in cases:
            raise ValueError("Judgment references an unknown calibration case")
        records = judges.setdefault(observation.judge, {})
        if observation.case_id in records:
            raise ValueError("Duplicate judge observation")
        records[observation.case_id] = observation
    results = {}
    for judge, records in judges.items():
        if records.keys() != cases.keys():
            raise ValueError("Each judge must label every control, including failures")
        tasks = {}
        for task in TASKS:
            calibration = [(case.label, records[case.id].positive_probability) for case in dataset.cases
                           if case.task == task and case.split == "development-calibration"]
            validation = [(case.label, records[case.id].positive_probability) for case in dataset.cases
                          if case.task == task and case.split == "development-validation"]
            sufficient = all(sum(label == expected for label, _ in rows) >= min_per_class
                             for rows in (calibration, validation) for expected in (False, True))
            candidates = []
            if sufficient:
                for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
                    measured = binary_scores(calibration, threshold)
                    if (measured["accuracy"] >= min_accuracy and measured["false_positive_rate"] <= max_fpr
                            and measured["precision"] is not None):
                        candidates.append((measured["accuracy"], measured["precision"], threshold))
            threshold = max(candidates)[2] if candidates else None
            measured = binary_scores(validation, threshold) if threshold is not None else None
            passed = (measured is not None and measured["accuracy"] >= min_accuracy
                      and measured["false_positive_rate"] <= max_fpr and measured["precision"] is not None)
            tasks[task] = {"sufficient_controls": sufficient, "threshold": threshold,
                           "calibration": binary_scores(calibration, threshold) if threshold is not None else None,
                           "validation": measured, "passed": bool(passed)}
        results[judge] = {"passed": all(item["passed"] for item in tasks.values()), "tasks": tasks}
    return {"kind": "judge-calibration", "status": "passed" if results and all(r["passed"] for r in results.values()) else "failed",
            "requirements": {"min_per_class_per_task_per_split": min_per_class,
                             "min_accuracy": min_accuracy, "max_false_positive_rate": max_fpr},
            "judges": results, "case_count": len(cases),
            "limitations": ["Human identity and declared independence require external verification; the scorer checks the supplied records.",
                            "Thresholds use development-calibration only; development-validation is scored once without retuning.",
                            "Point-target acceptance does not mean confidence-interval bounds meet those targets.",
                            "Calibration must be frozen before the held-out run, for the same judge/prompt/model digest."]}


def run(dataset, out, model):
    from evaluation.live_services import OllamaServices, Work
    out.mkdir(parents=True, exist_ok=False)
    # Reuse production methods and exact prompts; route inference through the Gateway.
    services = object.__new__(OllamaServices)
    services.judge = model.model
    services.match_prompt = Path(__file__).with_name("matcher_prompt.txt").read_text(encoding="utf-8")
    services._judge = model.call
    observations = []
    # Neither target labels nor reviewer votes enter the model prompt.
    for case in dataset.cases:
        data = case.payload
        if case.task == "answer_correctness":
            positive, _ = services.grade_answer(data["question"], data["gold_answer"], data["answer"], data["passages"])
        elif case.task == "claim_support":
            positive = services.grade_claim(data["claim"], data["passages"]).supported
        else:
            from datetime import date
            cutoff = date.fromisoformat(data["cutoff"])
            work = Work.model_validate_json(json.dumps(data["work"]))
            if work.published <= cutoff:
                raise ValueError("Future-match calibration paper must be dated after cutoff")
            first = services.match(cutoff, data["gap"], work)
            positive = first.match and first.confidence >= dataset.first_threshold
            if positive:
                strict = services.match(cutoff, data["gap"], work, strict=True)
                positive = strict.match and strict.confidence >= dataset.recheck_threshold
        record = Observation(positive_probability=float(positive), reason="Exact production operating-point decision; this 0/1 score is not a probability estimate.",
                             case_id=case.id, judge=f"{model.model}/{model.model_digest}")
        observations.append(record)
        (out / "observations.json").write_text(json.dumps([r.model_dump() for r in observations], indent=2), encoding="utf-8")
    report = evaluate(dataset, observations)
    # Production answer/support judges expose hard decisions, not probabilities.
    # Do not report threshold optimization or Brier calibration on those 0/1 values.
    for judge in report["judges"].values():
        for task in judge["tasks"].values():
            task.pop("threshold")
            for split in ("calibration", "validation"):
                if task[split] is not None:
                    task[split].pop("brier")
    raw = dataset.model_dump_json().encode()
    report.update(dataset_sha256=hashlib.sha256(raw).hexdigest(), model_digest=model.model_digest,
                  judge_prompt_sha256=hashlib.sha256((inspect.getsource(OllamaServices.grade_answer)
                      + inspect.getsource(OllamaServices.grade_claim) + inspect.getsource(OllamaServices.match)
                      + services.match_prompt).encode()).hexdigest(),
                  usage=model.calls,
                  production_judge_compatible=True,
                  operating_point={"first_threshold": dataset.first_threshold, "recheck_threshold": dataset.recheck_threshold},
                  score_kind="hard-decision operating-point audit; no probability-calibration claim")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--observations", type=Path, help="Score pre-recorded case/judge probabilities")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        dataset = CalibrationSet.model_validate_json(args.dataset.read_bytes())
        if args.observations:
            observations = [Observation.model_validate_json(json.dumps(r)) for r in json.loads(args.observations.read_bytes())]
            report = evaluate(dataset, observations)
            report["dataset_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
            report["production_judge_compatible"] = False
            args.out.mkdir(parents=True, exist_ok=False)
            (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        else:
            if args.tokenizer is None:
                raise ValueError("Live calibration requires the judge's exact tokenizer")
            from evaluation.local_model import LocalModel
            run(dataset, args.out, LocalModel(args.tokenizer, args.model))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Calibration error: {exc}\n")


if __name__ == "__main__":
    main()
