"""Audit the OKF core envelope and GapTrace's source/relationship extensions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def audit(bundle, passages_path=None):
    paths = sorted(bundle.rglob("*.md"))
    documents, issues, reserved = [], [], []
    for path in paths:
        relative = path.relative_to(bundle).as_posix()
        if path.name in {"index.md", "log.md"}:
            reserved.append(relative)
            continue
        try:
            text = path.read_text(encoding="utf-8")
            if not text.startswith("---\n") or "\n---\n" not in text[4:]:
                raise ValueError("Missing delimited YAML frontmatter")
            front, _, body = text[4:].partition("\n---\n")
            metadata = yaml.safe_load(front)
            if not isinstance(metadata, dict) or not isinstance(metadata.get("type"), str) or not metadata["type"].strip():
                raise ValueError("A nonempty type is required")
            documents.append((relative, metadata, body))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            issues.append({"file": relative, "error": str(exc)})
    sources = None
    if passages_path is not None:
        rows = [json.loads(line) for line in passages_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        sources = {row["id"] for row in rows}
        if len(sources) != len(rows):
            issues.append({"file": str(passages_path), "error": "Duplicate source passage IDs"})
    note_ids = [metadata.get("id") for _, metadata, _ in documents if metadata.get("id")]
    known = set(note_ids)
    if len(known) != len(note_ids):
        issues.append({"file": str(bundle), "error": "Duplicate GapTrace note IDs"})
    links = [link for _, metadata, _ in documents for link in metadata.get("links", [])]
    pointers = {pid for _, metadata, _ in documents for pid in metadata.get("passage_ids", [])}
    link_pointers = {link.get("evidence_passage_id") for link in links if link.get("evidence_passage_id")}
    unresolved = sorted((pointers | link_pointers) - sources) if sources is not None else None
    dangling = sorted({link.get("target", "") for link in links} - known)
    return {"specification": "https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md",
            "profile": "OKF v0.2 core envelope plus GapTrace extension diagnostics",
            "concept_files": len(documents), "files_scanned": len(paths), "issues": issues,
            "core_envelope_passed": bool(documents) and not issues,
            "reserved_files_requiring_separate_structure_review": reserved,
            "full_conformance_claimed": False,
            "typed_links": len(links), "verified_links": sum(bool(link.get("verified")) for link in links),
            "dangling_note_targets": dangling, "source_pointers": len(pointers | link_pointers),
            "unresolved_source_pointers": unresolved,
            "standard_provenance_documents": sum(bool(metadata.get("sources")) for _, metadata, _ in documents),
            "limitations": ["This checks the core envelope; reserved-file structure and all optional standard families need separate review.",
                            "GapTrace's IDs, typed links, paper dates, summaries and passage pointers are domain extensions.",
                            "Format conformance and performance benefit are separate questions."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--passages", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.bundle, args.passages)
    with args.out.open("x", encoding="utf-8") as file:
        file.write(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["core_envelope_passed"] else 1)


if __name__ == "__main__":
    main()
