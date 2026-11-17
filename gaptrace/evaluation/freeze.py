"""Freeze supplied, reviewed benchmark inputs without running held-out inference.

This tool snapshots inputs and records declared provenance; it cannot create
independent human labels or verify that a source archive is authentic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from evaluation.live import Plan
from evaluation.release import Policy


def freeze(spec: dict, root: Path, out: Path) -> dict:
    reviewer = spec.get("policy_reviewed_by", "").strip()
    if not reviewer:
        raise ValueError("Record the research lead's completed policy review before freezing held-out inputs")
    policy_raw = (root / spec["policy"]).read_bytes()
    Policy.model_validate_json(policy_raw)
    snapshots = {}
    topics_seen, paper_splits = set(), {}
    for split in ("development", "held-out"):
        path = root / spec["plans"][split]
        plan = Plan.model_validate_json(path.read_bytes())
        if plan.mode != "benchmark" or plan.split != split or plan.model_training_cutoff is None:
            raise ValueError(f"{split} must be a benchmark plan with a documented model training cutoff")
        from evaluation.metrics import validate_protocol
        validate_protocol(plan.mode, plan.cutoff, plan.model_training_cutoff, plan.future_through,
                          plan.first_threshold, plan.recheck_threshold)
        files = {}
        for topic in plan.topics:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", topic.id) or topic.id in topics_seen:
                raise ValueError("Topic IDs must be safe and disjoint across splits")
            topics_seen.add(topic.id)
            if topic.source is None:
                raise ValueError("Freeze source JSON, not mutable prebuilt workspace paths")
            from gaptrace.ingest import PaperImport
            source_raw = (path.parent / topic.source).read_bytes()
            papers = PaperImport.model_validate_json(source_raw)
            pins = spec["source_pins"][topic.id]
            if len({paper.id for paper in papers.papers}) != len(papers.papers):
                raise ValueError("Duplicate paper IDs in source")
            for paper in papers.papers:
                pin = pins[paper.id]
                if not pin.get("url", "").startswith("https://") or not pin.get("version", "").strip():
                    raise ValueError("Every paper needs a version-pinned or archived HTTPS source reference")
                if paper.published > plan.cutoff:
                    raise ValueError("Source contains a post-cutoff paper")
                if paper_splits.setdefault(paper.id, split) != split:
                    raise ValueError("Source papers overlap development and held-out splits")
            from gaptrace import tokens
            from gaptrace.settings import load_settings
            settings = load_settings()
            passage_ids = {f"{p.id}-{section_index}-{chunk_index}" for p in papers.papers
                           for section_index, section in enumerate(p.sections, 1)
                           for chunk_index, _ in enumerate(tokens.iter_chunks(section.text, settings.limits.passage_tokens), 1)}
            for question in topic.questions:
                if (not question.gold_passage_ids <= passage_ids
                        or not question.required_paper_ids <= {p.id for p in papers.papers}):
                    raise ValueError(f"Gold evidence is absent from frozen source for {topic.id}/{question.id}")
            name = f"sources/{topic.id}.json"
            files[name] = source_raw
            topic.source = Path(name)
            topic.data = Path("workspaces") / topic.id
            if topic.future_pool_file:
                pool_name = f"pools/{topic.id}.json"
                files[pool_name] = (path.parent / topic.future_pool_file).read_bytes()
                topic.future_pool_file = Path(pool_name)
        snapshots[split] = {"plan": plan, "files": files}
    dev, held = (snapshots[split]["plan"] for split in ("development", "held-out"))
    if (dev.cutoff, dev.model_training_cutoff, dev.future_through) != (held.cutoff, held.model_training_cutoff, held.future_through):
        raise ValueError("Splits must use the same frozen temporal protocol")
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for split, snapshot in snapshots.items():
        directory = out / split
        directory.mkdir()
        payloads = {"plan.json": (snapshot["plan"].model_dump_json(indent=2) + "\n").encode(), **snapshot["files"]}
        for name, raw in payloads.items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            hashes[f"{split}/{name}"] = hashlib.sha256(raw).hexdigest()
    (out / "release-policy.json").write_bytes(policy_raw)
    hashes["release-policy.json"] = hashlib.sha256(policy_raw).hexdigest()
    result = {"version": 1, "policy_reviewed_by": reviewer, "sha256": hashes,
              "source_pins": spec["source_pins"], "label_reviewers": spec.get("label_reviewers", []),
              "limitation": "Hashes freeze supplied inputs; they do not independently validate source authenticity, human labels, or model quality."}
    (out / "freeze.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def verify(root: Path) -> None:
    manifest = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Snapshot path escapes its root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen input changed: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, nargs="?")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
        print("Frozen input hashes match.")
    elif args.spec and args.out:
        freeze(json.loads(args.spec.read_text(encoding="utf-8")), args.spec.resolve().parent, args.out)
        print(f"Frozen inputs: {args.out}")
    else:
        parser.error("Supply SPEC --out DIRECTORY, or --verify DIRECTORY")
