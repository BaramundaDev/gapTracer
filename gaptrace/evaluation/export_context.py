"""Freeze published GapTrace bundles into the context factorial input contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from evaluation.context_experiment import ContextPlan
from evaluation.live import Plan


def export(plan, root):
    topics, fingerprints = [], {}
    for topic in plan.topics:
        directory = (root / topic.data).resolve()
        metadata = json.loads((directory / "import.json").read_bytes())
        if (metadata["cutoff"] != plan.cutoff.isoformat() or metadata["build_cloud_enabled"]
                or metadata["build_cloud_calls"] != 0):
            raise ValueError("Export requires a dated local-only corpus at the plan cutoff")
        notes, corpus, titles = [], [], {}
        paths = sorted((directory / "bundle").glob("*/*.md"))
        for path in paths:
            raw = path.read_bytes()
            fingerprints[f"{topic.id}/bundle/{path.relative_to(directory / 'bundle').as_posix()}"] = hashlib.sha256(raw).hexdigest()
            text = raw.decode("utf-8").replace("\r\n", "\n")
            if not text.startswith("---\n") or "\n---\n" not in text[4:]:
                raise ValueError("Invalid published note envelope")
            front, _, body = text[4:].partition("\n---\n")
            note = yaml.safe_load(front)
            if note["type"] == "paper":
                corpus.append({"id": note["id"], "published": str(note["paper_date"])})
                titles[note["id"]] = note["name"]
            ids = sorted(set(note.get("passage_ids", [])) | {
                link["evidence_passage_id"] for link in note.get("links", []) if link.get("evidence_passage_id")})
            if ids:
                summary = "\n".join(line for line in body.strip().splitlines() if not line.startswith("# ")).strip()
                notes.append({"id": note["id"], "type": note["type"], "title": note["name"],
                              "summary": summary or note.get("description") or note["name"],
                              "passage_ids": ids, "links": note.get("links", [])})
        source = (directory / "passages.jsonl").read_bytes()
        fingerprints[f"{topic.id}/passages.jsonl"] = hashlib.sha256(source).hexdigest()
        passages = []
        for line in source.decode("utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                passages.append({"id": item["id"], "paper_id": item["paper_id"],
                                 "paper_title": titles[item["paper_id"]], "published": item["date"], "text": item["text"]})
        topics.append({"id": topic.id, "corpus": corpus, "passages": passages, "notes": notes,
                       "questions": [question.model_dump(mode="json") for question in topic.questions]})
    data = {**plan.model_dump(mode="json", include={"mode", "split", "cutoff", "model_training_cutoff", "future_through"}),
            "provenance": "Frozen export of the published OKF-style GapTrace bundle and source passage store; see adjacent provenance JSON.",
            "topics": topics}
    result = ContextPlan.model_validate_json(json.dumps(data))
    return result, fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Existing live-evaluation plan pointing at imported corpora")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = Plan.model_validate_json(args.plan.read_bytes())
        result, hashes = export(plan, args.plan.resolve().parent)
        with args.out.open("x", encoding="utf-8") as file:
            file.write(result.model_dump_json(indent=2) + "\n")
        provenance = args.out.with_suffix(".provenance.json")
        with provenance.open("x", encoding="utf-8") as file:
            file.write(json.dumps({"source_plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                                   "source_sha256": hashes}, indent=2) + "\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Context export error: {exc}\n")


if __name__ == "__main__":
    main()
