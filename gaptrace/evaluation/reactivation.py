"""Freeze, execute and independently score newly enabled research discovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaluation.ranking import dcg, bootstrap_interval
from evaluation.metrics import compare_systems


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Review(Record):
    reviewer: str = Field(min_length=1)
    reviewer_type: Literal["human"]
    decision: Literal["accept", "reject"]
    reason: str = Field(min_length=1)


class Label(Record):
    prediction_id: str
    gold_id: str | None = None
    canonical_id: str
    reference: Literal["reactivation", "already_solved", "weak_bridge", "unsupported", "uncertain"]
    old_supported: bool
    new_supported: bool
    bridge_specific: bool
    still_open: bool
    importance: int = Field(ge=0, le=4)
    consequence: int = Field(ge=0, le=4)
    information_gain: int = Field(ge=0, le=4)
    reach: int = Field(ge=0, le=4)
    reason: str = Field(min_length=1)
    label_authors: list[str] = Field(min_length=1)
    reviews: list[Review] = Field(min_length=2)
    adjudicator: str | None = None

    @model_validator(mode="after")
    def independent(self):
        reviewers = [r.reviewer for r in self.reviews]
        if len(set(reviewers)) != len(reviewers) or set(reviewers) & set(self.label_authors):
            raise ValueError("Two distinct reviewers independent of label authors are required")
        if any(r.decision == "reject" for r in self.reviews):
            if not self.adjudicator or self.adjudicator in set(reviewers) | set(self.label_authors):
                raise ValueError("Disputed labels require a separate adjudicator")
        return self


class Gold(Record):
    id: str
    statement: str = Field(min_length=1)
    importance: int = Field(ge=0, le=4)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def protocol():
    return json.loads(Path(__file__).with_name("reactivation-protocol.json").read_text(encoding="utf-8"))


def nested_papers(papers, historical_through):
    """Preserve frozen within-stratum order, interleave old/new; prefixes are nested."""
    old = [p for p in papers if p.published <= historical_through]
    new = [p for p in papers if p.published > historical_through]
    result = []
    for i in range(max(len(old), len(new))):
        if i < len(old):
            result.append(old[i])
        if i < len(new):
            result.append(new[i])
    return result


def freeze(spec_path, out):
    from gaptrace.ingest import PaperImport
    spec_path, out = Path(spec_path), Path(out)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not spec.get("protocol_reviewed_by") or not spec.get("model_training_cutoff_reference"):
        raise ValueError("Record protocol review and model training-cutoff provenance before freezing")
    if date.fromisoformat(spec["model_training_cutoff"]) > date.fromisoformat(spec["as_of"]):
        raise ValueError("The model training cutoff must not follow the evaluation as_of date")
    payloads, topics, topic_ids, paper_splits = {}, [], set(), {}
    for raw in spec["topics"]:
        tid = raw["id"]
        if not tid.replace("-", "").replace("_", "").isalnum() or tid in topic_ids:
            raise ValueError("Topic IDs must be safe and unique")
        topic_ids.add(tid)
        if raw["split"] not in {"development", "held-out"}:
            raise ValueError("Use development and held-out topic splits")
        source_bytes = (spec_path.parent / raw["source"]).read_bytes()
        source = PaperImport.model_validate_json(source_bytes)
        if any(p.published > date.fromisoformat(spec["as_of"]) for p in source.papers):
            raise ValueError("Frozen corpus contains post-as_of source material")
        if not any(p.published <= date.fromisoformat(raw["historical_through"]) for p in source.papers):
            raise ValueError("Each topic needs historical attempt papers")
        if not any(p.published > date.fromisoformat(raw["historical_through"]) for p in source.papers):
            raise ValueError("Each topic needs newer candidate developments")
        if len({p.id for p in source.papers}) != len(source.papers):
            raise ValueError("Duplicate paper identities in a topic")
        for p in source.papers:
            identity = sha(json.dumps([p.title.casefold(), [s.text for s in p.sections]], sort_keys=True).encode())
            if paper_splits.setdefault(identity, raw["split"]) != raw["split"]:
                raise ValueError("Source content overlaps development and held-out splits")
        name = f"sources/{tid}.json"
        payloads[name] = source_bytes
        # Search pools are frozen before inference; records are query-specific.
        pool_bytes = (spec_path.parent / raw["search_pool"]).read_bytes()
        pool = json.loads(pool_bytes)
        if pool["as_of"] != spec["as_of"]:
            raise ValueError("Frozen search pool has a different observation date")
        for record in pool["searches"]:
            if len(record["works"]) > 5 or any(w["published"] > spec["as_of"] for w in record["works"]):
                raise ValueError("Frozen search pool violates date or result bounds")
        pool_name = f"pools/{tid}.json"
        payloads[pool_name] = pool_bytes
        topics.append({"id": tid, "split": raw["split"], "source": name, "search_pool": pool_name,
                       "historical_through": raw["historical_through"], "paper_count": len(source.papers),
                       "source_pins": raw.get("source_pins", {}), "evidence_scope": raw.get("evidence_scope", "unreviewed")})
    if spec.get("mode", "smoke") == "benchmark":
        for split, expected in protocol()["topics"].items():
            if sum(t["split"] == split for t in topics) != expected:
                raise ValueError(f"Benchmark requires {expected} {split} topics")
        if any(t["paper_count"] < 200 or t["evidence_scope"] != "full-text" for t in topics):
            raise ValueError("Full matrix benchmark requires 200 source papers per topic with full-text provenance")
        if any(len(t["source_pins"]) < t["paper_count"] for t in topics):
            raise ValueError("Every benchmark source requires a version-pinned reference")
        for t in topics:
            source = PaperImport.model_validate_json(payloads[t["source"]])
            for p in source.papers:
                pin = t["source_pins"].get(p.id, {})
                if not str(pin.get("url", "")).startswith("https://") or not pin.get("version"):
                    raise ValueError("Every source pin needs an archived/versioned HTTPS reference")
                if all(s.heading.strip().lower() == "abstract" for s in p.sections):
                    raise ValueError("Abstract-only sources cannot satisfy full-text benchmark coverage")
    payloads["protocol.json"] = json.dumps(protocol(), indent=2).encode()
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, content in payloads.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        hashes[name] = sha(content)
    manifest = {"kind": "reactivation-benchmark-snapshot", "mode": spec.get("mode", "smoke"),
                "as_of": spec["as_of"], "model_training_cutoff": spec["model_training_cutoff"],
                "model_training_cutoff_reference": spec["model_training_cutoff_reference"],
                "protocol_reviewed_by": spec["protocol_reviewed_by"], "topics": topics, "sha256": hashes,
                "labels_in_snapshot": False}
    (out / "snapshot.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root) or sha(target.read_bytes()) != expected:
            raise ValueError("Frozen benchmark input changed: " + name)
    return manifest


def replay(pool):
    def search(query, provider, limit, as_of):
        if as_of != pool["as_of"]:
            raise ValueError("Search replay date mismatch")
        matches = [s for s in pool["searches"] if s["query"] == query and s["provider"] == provider]
        if len(matches) != 1:
            raise OSError("Frozen search pool has no unique response for this query; live fallback is disabled")
        return matches[0]["works"][:limit]
    return search


def execute(root, out, split="development", paper_budget=100, metadata_tokens=50, repeats=3, selected=None, resume=False):
    from gaptrace import tokens
    from gaptrace.discovery import run as discover
    from gaptrace.gateway import Gateway
    from gaptrace.gap_review import local_gateway
    from gaptrace.ingest import PaperImport
    from gaptrace.reactivation import analyze
    from gaptrace.settings import load_settings
    from gaptrace.workspace import open_toolbox
    from evaluation.run import fingerprints
    root, out = Path(root), Path(out)
    frozen = verify(root)
    if paper_budget not in protocol()["paper_budgets"] or metadata_tokens not in protocol()["metadata_token_budgets"] or not 1 <= repeats <= 5:
        raise ValueError("Choose a registered paper/metadata budget and 1-5 fresh repeats")
    if split == "held-out":
        if selected is None:
            raise ValueError("Freeze a development-selected operating point before held-out execution")
        selection = json.loads(Path(selected).read_text(encoding="utf-8"))
        if (selection.get("selected_on") != "development" or selection.get("snapshot_sha256") != sha((root / "snapshot.json").read_bytes())
                or selection.get("paper_budget") != paper_budget or selection.get("metadata_tokens") != metadata_tokens):
            raise ValueError("Held-out settings differ from the frozen development selection")
        for name, expected in selection.get("selection_evidence_sha256", {}).items():
            target = (Path(selected).resolve().parent / name).resolve()
            if not target.is_relative_to(Path(selected).resolve().parent) or sha(target.read_bytes()) != expected:
                raise ValueError("Development selection evidence changed")
        if not selection.get("selection_evidence_sha256") or not selection.get("selected_by"):
            raise ValueError("Held-out execution requires an audited development-selection receipt")
    before = fingerprints()
    identity = {"snapshot_sha256": sha((root / "snapshot.json").read_bytes()), "split": split,
                "paper_budget": paper_budget, "metadata_tokens": metadata_tokens, "repeats": repeats}
    out.mkdir(parents=True, exist_ok=resume)
    if resume and (out / "execution.json").exists():
        prior = json.loads((out / "execution.json").read_bytes())
        if prior.get("operating_point") != identity or prior.get("source_sha256") != before:
            raise ValueError("Resume requires identical frozen inputs, operating point and implementation")
        rows = prior["runs"]
        for row in rows:
            if sha((out / row["analysis"]).read_bytes()) != row["analysis_sha256"]:
                raise ValueError("A completed repeat's analysis changed")
    else:
        rows = []
    def checkpoint(state):
        value = {"kind": "reactivation-execution", "state": state, "split": split, "mode": frozen["mode"], "runs": rows,
                 "source_sha256": before, "operating_point": identity, "source_unchanged": before == fingerprints(),
                 "snapshot_sha256": identity["snapshot_sha256"], "quality_metrics": None, "labels_accessed": False,
                 "repeat_semantics": "Fresh builds and separate caches per repeat; resuming reuses work only within its original repeat"}
        from gaptrace.recovery import atomic_json
        atomic_json(out / "execution.json", value)
        return value
    checkpoint("running")
    settings = load_settings()
    settings.cloud.enabled = False
    tokens.configure(settings)
    for topic in frozen["topics"]:
        if topic["split"] != split:
            continue
        source = PaperImport.model_validate_json((root / topic["source"]).read_bytes())
        ordered = nested_papers(source.papers, date.fromisoformat(topic["historical_through"]))
        if len(ordered) < paper_budget:
            raise ValueError(f"{topic['id']} has fewer than the requested {paper_budget} frozen papers")
        subset = PaperImport(papers=ordered[:paper_budget])
        for repeat in range(1, repeats + 1):
            if any(row["topic"] == topic["id"] and row["repeat"] == repeat for row in rows):
                continue
            run_dir = out / f"{topic['id']}-repeat-{repeat}"
            subset_file = out / f"{topic['id']}-source-{repeat}.json"
            subset_raw = subset.model_dump_json(indent=2).encode()
            if subset_file.exists() and subset_file.read_bytes() != subset_raw:
                raise ValueError("Frozen paper subset changed on resume")
            subset_file.write_bytes(subset_raw)
            # A fresh cache and a fresh build for every repeat; not deterministic rescoring.
            gateway = local_gateway(settings)
            started = perf_counter()
            print(f"{topic['id']} repeat {repeat}/{repeats}: {paper_budget} papers, {metadata_tokens} metadata tokens", flush=True)
            prepared = discover(topic["id"], run_dir, settings, gateway, papers=paper_budget,
                                source_file=subset_file, as_of=frozen["as_of"], metadata_tokens=metadata_tokens, engine="prepare",
                                resume=run_dir.exists(), progress=lambda message: print(message, flush=True))
            pool = json.loads((root / topic["search_pool"]).read_text(encoding="utf-8"))
            tb = open_toolbox(run_dir / "corpus", settings)
            result = analyze(tb, gateway, frozen["as_of"], pool["provider"], search=replay(pool), progress=print)
            rows.append({"topic": topic["id"], "repeat": repeat, "paper_budget": paper_budget,
                         "metadata_tokens": metadata_tokens, "seconds": perf_counter() - started,
                         "analysis": str((run_dir / "corpus/reactivation.json").relative_to(out)),
                         "analysis_sha256": sha((run_dir / "corpus/reactivation.json").read_bytes()),
                         "fresh_build": True, "cache_namespace": str(gateway.cache_dir),
                         "extraction_errors": prepared["preparation_errors"],
                         "search_errors": [s["error"] for s in result["searches"] if s["error"]]})
            (out / "runs.json").write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
            checkpoint("running")
    result = checkpoint("complete")
    if not result["source_unchanged"]:
        raise ValueError("Implementation changed during execution; preserve this invalid run and rerun")
    return result


def compare(spec_path):
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    expected_repeats = spec.get("expected_repeats", 3)
    if not 1 <= expected_repeats <= 5 or len(spec["systems"]) != 2:
        raise ValueError("Compare two systems with 1-5 declared repeats")
    prepared, summary, topic_set = {}, {}, None
    metrics = protocol()["primary_endpoints"]
    for system, entries in spec["systems"].items():
        seen, topics = set(), {}
        for entry in entries:
            key = (entry["topic"], entry["repeat"])
            if key in seen:
                raise ValueError("Duplicate topic/repeat cannot count as independent evidence")
            seen.add(key)
            if not 1 <= entry["repeat"] <= expected_repeats:
                raise ValueError("Unexpected repeat id")
            raw = (spec_path.parent / entry["report"]).read_bytes()
            result = json.loads(raw)
            if result.get("kind") != "independent-reactivation-evaluation":
                raise ValueError("Compare independently annotated score reports only")
            topics.setdefault(entry["topic"], []).append(result["metrics"])
        if topic_set is None:
            topic_set = set(topics)
        if set(topics) != topic_set or any(len(rows) != expected_repeats for rows in topics.values()):
            raise ValueError("Systems must have identical topics and every declared repeat")
        point = {tid: {m: sum(row[m] for row in rows) / len(rows) if all(row.get(m) is not None for row in rows) else None for m in metrics}
                 for tid, rows in topics.items()}
        prepared[system] = [{"metrics": {m: {} for m in metrics}, "topics": {
            tid: {"metrics": {m: {"value": value} for m, value in values.items()}} for tid, values in point.items()}}]
        summary[system] = {m: {"mean": sum(values[m] for values in point.values()) / len(point) if point and all(v[m] is not None for v in point.values()) else None,
                                  "defined_topics": sum(v[m] is not None for v in point.values()), "total_topics": len(point)} for m in metrics}
    names = list(prepared)
    paired = compare_systems(prepared[names[0]], prepared[names[1]], sorted(topic_set))
    return {"kind": "paired-reactivation-comparison", "systems": names, "summary": summary, "paired": paired,
            "spec_sha256": sha(spec_path.read_bytes()), "topic_count": len(topic_set),
            "limitations": ["Bootstrap and sign tests use topics, not claims or repeated builds, as independent observations.",
                 "Incomplete primary labels withhold the all-topic mean; paired subsets retain their explicit topic coverage.",
                 "No superiority claim is justified by a tiny or selectively assembled topic sample."]}


def select(snapshot, execution_path, comparison_path, system, reviewed_by, rationale, out):
    snapshot, execution_path, comparison_path, out = map(Path, (snapshot, execution_path, comparison_path, out))
    frozen = verify(snapshot)
    execution = json.loads(execution_path.read_bytes())
    comparison = json.loads(comparison_path.read_bytes())
    if (execution.get("split") != "development" or execution.get("state") != "complete" or execution.get("source_unchanged") is not True
            or execution.get("snapshot_sha256") != sha((snapshot / "snapshot.json").read_bytes())):
        raise ValueError("Selection requires a completed unchanged development execution on this snapshot")
    expected_topics = sum(t["split"] == "development" for t in frozen["topics"])
    if (comparison.get("kind") != "paired-reactivation-comparison" or comparison.get("topic_count") != expected_topics
            or system not in comparison["summary"] or not reviewed_by.strip() or not rationale.strip()):
        raise ValueError("Record a reviewed selection with complete development topic comparisons")
    if any(v["mean"] is None for v in comparison["summary"][system].values()):
        raise ValueError("Independent primary quality labels and catalogue coverage are required before held-out selection")
    out.mkdir(parents=True, exist_ok=False)
    (out / "development-execution.json").write_bytes(execution_path.read_bytes())
    (out / "development-comparison.json").write_bytes(comparison_path.read_bytes())
    point = execution["operating_point"]
    receipt = {"kind": "frozen-development-operating-point", "selected_on": "development", "system": system,
               "paper_budget": point["paper_budget"], "metadata_tokens": point["metadata_tokens"],
               "snapshot_sha256": execution["snapshot_sha256"], "selected_by": reviewed_by, "rationale": rationale,
               "selection_evidence_sha256": {"development-execution.json": sha(execution_path.read_bytes()),
                                            "development-comparison.json": sha(comparison_path.read_bytes())}}
    (out / "selection.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def score(raw, annotations, ks=(1, 3, 5, 10)):
    report = json.loads(raw)
    if annotations["analysis_sha256"] != sha(raw):
        raise ValueError("Independent annotations refer to a different exact analysis")
    labels = [Label.model_validate(v) for v in annotations["labels"]]
    by_id = {v.prediction_id: v for v in labels}
    if len(by_id) != len(labels):
        raise ValueError("Duplicate prediction labels")
    predictions = report["opportunities"]
    prediction_ids = {p["id"] for p in predictions}
    if len(prediction_ids) != len(predictions) or not by_id.keys() <= prediction_ids:
        raise ValueError("Unknown or duplicate predictions")
    gold = [Gold.model_validate(v) for v in annotations.get("gold", [])]
    gold_by_id = {g.id: g for g in gold}
    if len(gold_by_id) != len(gold) or any(v.gold_id and v.gold_id not in gold_by_id for v in labels):
        raise ValueError("Duplicate or unknown gold catalogue identity")
    catalogue_ready = annotations.get("gold_mapping_complete") is True and annotations.get("gold") is not None
    complete = len(labels) == len(predictions)
    seen, seen_gold, good, gains, mapped = set(), set(), [], [], []
    for item in predictions:
        label = by_id.get(item["id"])
        fresh = bool(label and label.canonical_id not in seen)
        if label:
            seen.add(label.canonical_id)
        valid = bool(label and label.reference == "reactivation" and label.old_supported and label.new_supported
                     and label.bridge_specific and label.still_open)
        good.append(int(valid and fresh and label.importance >= 3))
        eligible_gold = label.gold_id if label and valid and fresh and label.gold_id not in seen_gold else None
        mapped.append(eligible_gold)
        gains.append(gold_by_id[eligible_gold].importance if eligible_gold else 0)
        if eligible_gold:
            seen_gold.add(eligible_gold)
    metrics = {"annotation_coverage": len(labels) / len(predictions) if predictions else 1,
               "candidate_count": len(predictions), "duplicate_rate": (len(labels) - len(seen)) / len(labels) if complete and labels else None}
    for k in ks:
        prefix_complete = all(p["id"] in by_id for p in predictions[:k])
        metrics[f"opportunity_precision@{k}"] = sum(good[:k]) / k if prefix_complete else None
        metrics[f"list_fill@{k}"] = min(k, len(predictions)) / k
        ideal = dcg([g.importance for g in sorted(gold, key=lambda g: -g.importance)], k)
        metrics[f"gold_importance_ndcg@{k}"] = dcg(gains, k) / ideal if complete and catalogue_ready and ideal else None
        denominator = sum(g.importance > 0 for g in gold)
        metrics[f"gold_recall@{k}"] = len({g for g in mapped[:k] if g and gold_by_id[g].importance > 0}) / denominator if complete and catalogue_ready and denominator else None
    claimed = [p for p in predictions if p["status"] == "reactivation_hypothesis"]
    metrics["false_reactivation_rate"] = (sum(by_id[p["id"]].reference != "reactivation" or not (
        by_id[p["id"]].old_supported and by_id[p["id"]].new_supported and by_id[p["id"]].bridge_specific and by_id[p["id"]].still_open)
        for p in claimed) / len(claimed)) if claimed and all(p["id"] in by_id for p in claimed) else None
    paired = [(p["importance_estimate"], by_id[p["id"]].importance) for p in predictions if p["id"] in by_id and p["importance_estimate"] is not None]
    metrics["importance_mae"] = sum(abs(a - b) for a, b in paired) / len(paired) if paired else None
    return {"kind": "independent-reactivation-evaluation", "analysis_sha256": sha(raw), "metrics": metrics,
            "provenance": annotations.get("provenance", "not supplied"),
            "limitations": ["Declared human identity and independence require external verification.",
                 "Scientific importance is separate from feasibility; no resource-cost gate is applied.",
                 "One topic is descriptive. Aggregate paired topics with uncertainty before claiming improvement.",
                 "Missing annotations and missing catalogue coverage remain undefined, not guessed from model outputs."]}


def annotation_template(raw):
    report = json.loads(raw)
    return {"analysis_sha256": sha(raw), "provenance": "Fill with independently adjudicated labels; never copy automated judgments",
            "gold": None, "gold_mapping_complete": False,
            "labels": [{"prediction_id": p["id"], "gold_id": None, "canonical_id": None, "reference": None,
                "old_supported": None, "new_supported": None, "bridge_specific": None, "still_open": None,
                "importance": None, "consequence": None, "information_gain": None, "reach": None,
                "reason": "", "label_authors": [], "reviews": [], "adjudicator": None} for p in report["opportunities"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    f = subs.add_parser("freeze")
    f.add_argument("spec", type=Path)
    f.add_argument("--out", type=Path, required=True)
    r = subs.add_parser("run")
    r.add_argument("snapshot", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--split", choices=("development", "held-out"), default="development")
    r.add_argument("--paper-budget", type=int, default=100)
    r.add_argument("--metadata-tokens", type=int, default=50)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--selected", type=Path)
    r.add_argument("--resume", action="store_true")
    a = subs.add_parser("annotate")
    a.add_argument("analysis", type=Path)
    a.add_argument("--out", type=Path, required=True)
    s = subs.add_parser("score")
    s.add_argument("analysis", type=Path)
    s.add_argument("--labels", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    c = subs.add_parser("compare")
    c.add_argument("spec", type=Path)
    c.add_argument("--out", type=Path, required=True)
    chosen = subs.add_parser("select")
    chosen.add_argument("snapshot", type=Path)
    chosen.add_argument("--execution", type=Path, required=True)
    chosen.add_argument("--comparison", type=Path, required=True)
    chosen.add_argument("--system", required=True)
    chosen.add_argument("--reviewed-by", required=True)
    chosen.add_argument("--rationale", required=True)
    chosen.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            result = freeze(args.spec, args.out)
        elif args.command == "run":
            result = execute(args.snapshot, args.out, args.split, args.paper_budget, args.metadata_tokens, args.repeats, args.selected, args.resume)
        elif args.command == "compare":
            result = compare(args.spec)
            with args.out.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
        elif args.command == "select":
            result = select(args.snapshot, args.execution, args.comparison, args.system, args.reviewed_by, args.rationale, args.out)
        else:
            raw = args.analysis.read_bytes()
            result = annotation_template(raw) if args.command == "annotate" else score(raw, json.loads(args.labels.read_bytes()))
            with args.out.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
        print(json.dumps({"kind": result.get("kind", "annotation-template"), "out": str(args.out)}))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Reactivation evaluation error: {exc}\n")


if __name__ == "__main__":
    main()
