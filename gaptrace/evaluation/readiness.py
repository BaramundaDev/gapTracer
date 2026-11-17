"""Audit frozen benchmark prerequisites and write a held-out execution status report."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.freeze import verify
from evaluation.live import Plan


def label_hash(topic_id, question):
    data = question.model_dump(mode="json")
    data["gold_passage_ids"] = sorted(question.gold_passage_ids)
    data["required_paper_ids"] = sorted(question.required_paper_ids)
    return hashlib.sha256(json.dumps({"topic": topic_id, "question": data}, sort_keys=True).encode()).hexdigest()


def discover(workspace):
    plans, snapshots, errors = [], [], []
    def walk(root):
        try:
            paths = list(root.iterdir())
        except OSError as exc:
            errors.append({"path": str(root), "error": str(exc)})
            return
        for path in paths:
            try:
                is_directory = path.is_dir()
            except OSError as exc:
                errors.append({"path": str(path), "error": str(exc)})
                continue
            if is_directory:
                if path.name not in {"baseline-workspace", "workspace", ".research-venv", "cache", "__pycache__", ".git"}:
                    walk(path)
            elif path.name == "freeze.json":
                snapshots.append(str(path.parent))
            elif path.name.endswith("plan.json") or path.name == "context-smoke.json":
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    if "topics" in raw:
                        plans.append({"path": str(path), "mode": raw.get("mode"), "split": raw.get("split"),
                                      "topics": len(raw["topics"]),
                                      "questions": sum(len(t.get("questions", [])) for t in raw["topics"])})
                except (ValueError, OSError, TypeError, AttributeError) as exc:
                    errors.append({"path": str(path), "error": str(exc)})
    for folder in (workspace / "data", workspace / "evaluation" / "examples"):
        if folder.exists():
            walk(folder)
    return {"frozen_candidates": snapshots, "plans": plans, "unreadable": errors}


def inspect_snapshot(root, reviews_path=None, calibration_path=None):
    checks, details = [], {}
    def check(name, passed, observed, expected):
        checks.append({"check": name, "passed": bool(passed), "observed": observed, "expected": expected})
    try:
        verify(root)
        check("frozen_hashes", True, "matched", "all recorded hashes match")
        plans = {split: Plan.model_validate_json((root / split / "plan.json").read_bytes())
                 for split in ("development", "held-out")}
    except (OSError, ValueError, KeyError) as exc:
        check("frozen_hashes", False, str(exc), "verified development and held-out snapshot")
        return checks, details
    topic_ids, questions, counts = set(), {}, {}
    for split, plan in plans.items():
        check(split + "_mode", plan.mode == "benchmark" and plan.split == split, {"mode": plan.mode, "split": plan.split}, "benchmark in declared split")
        check(split + "_topic_count", len(plan.topics) == (12 if split == "development" else 8), len(plan.topics), 12 if split == "development" else 8)
        for topic in plan.topics:
            check(f"{topic.id}/disjoint_topic", topic.id not in topic_ids, topic.id, "unique topic across splits")
            topic_ids.add(topic.id)
            source = json.loads((root / split / topic.source).read_bytes())
            counts[topic.id] = len(source["papers"])
            check(f"{topic.id}/paper_count", len(source["papers"]) >= 100, len(source["papers"]), "at least 100 dated papers")
            check(f"{topic.id}/future_pool", topic.future_pool_file is not None, str(topic.future_pool_file), "fixed future pool shared across comparisons")
            for question in topic.questions:
                questions[(topic.id, question.id)] = question
    check("total_topics", len(topic_ids) == 20, len(topic_ids), 20)
    check("total_questions", len(questions) == 100, len(questions), 100)
    single = sum(len(q.required_paper_ids) == 1 for q in questions.values())
    multi = sum(len(q.required_paper_ids) > 1 for q in questions.values())
    check("single_paper_questions", single == 50, single, 50)
    check("multi_paper_questions", multi == 50, multi, 50)
    review_complete = 0
    if reviews_path is not None:
        reviewed = json.loads(reviews_path.read_bytes())
        records = {}
        for record in reviewed.get("reviews", []):
            records.setdefault((record["topic_id"], record["question_id"]), []).append(record)
        for key, question in questions.items():
            votes = records.get(key, [])
            expected_hash = label_hash(key[0], question)
            reviewers = {v.get("reviewer") for v in votes if v.get("reviewer")}
            authors = set(reviewed.get("label_authors", []))
            valid = (bool(authors) and len(reviewers) >= 2 and len(votes) == len(reviewers)
                     and not reviewers & authors and all(v.get("reviewer_type") == "human"
                       and v.get("decision") == "accepted" and v.get("label_sha256") == expected_hash
                       and bool(v.get("reason", "").strip()) for v in votes))
            review_complete += valid
    check("independent_question_review", review_complete == len(questions) and bool(questions), review_complete,
          f"two independent human reviews of each of {len(questions)} exact question labels")
    if calibration_path is not None:
        calibration = json.loads(calibration_path.read_bytes())
        compatible = calibration.get("status") == "passed" and calibration.get("production_judge_compatible") is True
        check("calibrated_production_judge", compatible, calibration.get("status"), "passed exact-production-prompt audit on reviewed development controls")
        details["calibration_sha256"] = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    else:
        check("calibrated_production_judge", False, "not supplied", "reviewed controls and exact-production judge audit")
    provenance = root / "model-provenance.json"
    model_record = json.loads(provenance.read_bytes()) if provenance.exists() else {}
    documented = (all(model_record.get(key) for key in ("model", "model_digest", "tokenizer_sha256", "training_cutoff", "training_cutoff_reference"))
                  and all(model_record.get("training_cutoff") == plan.model_training_cutoff.isoformat() for plan in plans.values()))
    check("documented_training_cutoff", documented, model_record if model_record else "not supplied",
          "version-pinned model digest, matching training cutoff and supporting reference, tokenizer provenance")
    if calibration_path is not None:
        from evaluation.live_services import OllamaServices
        import inspect
        prompt = Path(__file__).with_name("matcher_prompt.txt").read_text(encoding="utf-8")
        expected = hashlib.sha256((inspect.getsource(OllamaServices.grade_answer)
            + inspect.getsource(OllamaServices.grade_claim) + inspect.getsource(OllamaServices.match) + prompt).encode()).hexdigest()
        point = calibration.get("operating_point", {})
        check("calibration_matches_frozen_protocol", calibration.get("judge_prompt_sha256") == expected
              and all(point.get("first_threshold") == p.first_threshold and point.get("recheck_threshold") == p.recheck_threshold for p in plans.values()),
              {"judge_prompt_sha256": calibration.get("judge_prompt_sha256"), "operating_point": point},
              "current exact judge prompts and the frozen matcher thresholds")
    policy = json.loads((root / "release-policy.json").read_bytes())
    held_out_questions = sum(len(t.questions) for t in plans["held-out"].topics)
    check("held_out_policy_scope", policy["min_topics"] <= 8 and policy["min_questions"] <= held_out_questions,
          {"min_topics": policy["min_topics"], "min_questions": policy["min_questions"]},
          f"policy scoped to 8 held-out topics/{held_out_questions} questions, with 20/100 verified globally")
    details.update(topic_count=len(topic_ids), question_count=len(questions), single_paper=single, multi_paper=multi,
                   source_papers=counts)
    return checks, details


def report(workspace, out, frozen=None, reviews=None, calibration=None):
    inventory = discover(workspace)
    checks, details = inspect_snapshot(frozen, reviews, calibration) if frozen is not None else ([
        {"check": name, "passed": False, "observed": "not supplied", "expected": expected}
        for name, expected in (("frozen_benchmark", "verified 12/8-topic, 100-question snapshot"),
                               ("independent_question_review", "two independent reviews per exact gold label"),
                               ("calibrated_production_judge", "validated exact-production judge receipt"),
                               ("documented_training_cutoff", "model/version-specific cutoff evidence"))], {})
    runnable = all(item["passed"] for item in checks)
    result = {"kind": "held-out-evaluation-readiness", "status": "ready" if runnable else "blocked_by_missing_or_invalid_inputs",
              "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks, "details": details,
              "inventory": inventory, "held_out_executed": False, "quality_metrics": None,
              "limitations": ["No held-out result is inferred from smoke runs or passing software tests.",
                              "This audit checks supplied review records, not human identity or the truth of those declarations.",
                              "Full baseline/ablation execution and independent generation/build repeats remain required after inputs pass."]}
    out.mkdir(parents=True, exist_ok=False)
    (out / "readiness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Held out evaluation status", "", f"Status: **{result['status']}**.", "",
             "No held-out quality result has been produced. Accuracy, gap hit rate and other quality scores are unmeasured.", "",
             "| Prerequisite | Status | Observed |", "| --- | --- | --- |"]
    lines += [f"| {c['check']} | {'passed' if c['passed'] else 'missing or invalid'} | {str(c['observed']).replace('|', '/')} |" for c in checks]
    lines += ["", f"Located {len(inventory['frozen_candidates'])} frozen snapshot candidates and {len(inventory['plans'])} plan files.",
              f"Unreadable inventory paths: {len(inventory['unreadable'])} (listed in readiness.json).", "",
              "Required next inputs: frozen topic/source/question files, independent label reviews, model cutoff provenance and exact-judge calibration controls.",
              "The current prototype release policy targets 20 topics/100 questions within a held-out manifest, while the planned split has eight held-out topics.",
              "Review and freeze a correctly scoped policy before execution; do not silently lower it after seeing results.", "",
              "After prerequisites pass: freeze the system/ablation matrix; run development comparisons and calibration; select once; execute three fresh held-out repeats; retain every outcome and provenance record."]
    (out / "HELD_OUT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = report(args.workspace.resolve(), args.out.resolve(), args.frozen, args.reviews, args.calibration)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Readiness error: {exc}\n")
    print(result["status"])
    raise SystemExit(0 if result["status"] == "ready" else 1)


if __name__ == "__main__":
    main()
