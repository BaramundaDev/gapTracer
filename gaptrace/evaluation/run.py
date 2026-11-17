"""Fast offline release gate. Exit 0 = pass, 1 = failed checks, 2 = harness error."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]


def read_junit(path: Path) -> list[dict]:
    cases = []
    for node in ET.parse(path).getroot().iter("testcase"):
        issue = next((child for child in node if child.tag in {"failure", "error", "skipped"}), None)
        cases.append({
            "id": f"{node.get('classname')}.{node.get('name')}",
            "status": {"failure": "failed", "error": "error", "skipped": "skipped"}.get(
                issue.tag if issue is not None else "", "passed"),
            "seconds": float(node.get("time", "0")),
            "detail": (issue.text or issue.get("message", "")) if issue is not None else "",
            "measurements": {p.get("name"): p.get("value") for p in node.findall("properties/property")},
        })
    if not cases:
        raise ValueError("No tests were collected; an empty suite cannot pass")
    return cases


def gate_status(cases: list[dict], returncode: int) -> str:
    if not cases or returncode not in {0, 1} or any(c["status"] == "error" for c in cases):
        return "error"
    return "passed" if returncode == 0 and all(c["status"] == "passed" for c in cases) else "failed"


def fingerprints() -> dict[str, str]:
    files = [ROOT / "pyproject.toml", ROOT / "DEPENDENCIES.md"]
    for folder in ("src", "prompts", "config", "tests", "evaluation"):
        files += [p for p in (ROOT / folder).rglob("*")
                  if p.is_file() and p.suffix in {".py", ".json", ".yaml", ".txt"}
                  and "__pycache__" not in p.parts]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def run(out: Path, timeout: int = 180) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    before = fingerprints()
    xml = out / "junit.xml"
    command = [sys.executable, "-m", "pytest", "tests", "evaluation", "-q", "--tb=short",
               "-p", "no:cacheprovider", "-o", "junit_family=xunit1", f"--junitxml={xml}"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    start = monotonic()
    problem = None
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
        (out / "pytest.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        cases = read_junit(xml)
        status = gate_status(cases, result.returncode)
        returncode = result.returncode
    except (subprocess.TimeoutExpired, OSError, ValueError, ET.ParseError) as exc:
        cases, status, returncode = [], "error", None
        problem = f"{type(exc).__name__}: {exc}"
    after = fingerprints()
    changed_sources = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    if before != after:
        status, problem = "error", "Source files changed while evaluation was running; rerun on a stable snapshot"
    counts = dict(Counter(case["status"] for case in cases))
    report = {
        "version": "1.0", "kind": "offline-contracts", "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": monotonic() - start, "counts": counts, "pytest_exit_code": returncode,
        "error": problem, "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("pytest", "pydantic", "numpy", "networkx", "rank-bm25", "pyyaml")},
        "source_sha256": before, "cases": cases,
        "changed_sources": changed_sources,
        "limitations": ["Scripted model outputs measure enforcement, not live-model quality.",
                        "Retrieval controls use BM25; production embeddings require a live run.",
                        "Token checks use the default estimate or a deterministic boundary counter; real tokenizer and server usage require live measurements.",
                        "Skipped checks fail the gate. No external services are required."],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failures = [f"- `{c['id']}` ({c['status']})" for c in cases if c["status"] != "passed"]
    summary = [f"# Offline evaluation: {status}", "", f"Counts: {counts}. Elapsed: {report['seconds']:.2f}s.", "",
               "Scripted enforcement checks; this is not a live-model benchmark.", "", *failures]
    if problem:
        summary += ["", problem]
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory; existing artifacts are never overwritten")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    try:
        report = run(args.out.resolve(), args.timeout)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Evaluation error: {exc}\n")
    print(f"{report['status']}: {report['counts']} ({report['seconds']:.2f}s); {args.out / 'report.json'}")
    raise SystemExit({"passed": 0, "failed": 1, "error": 2}[report["status"]])


if __name__ == "__main__":
    main()
