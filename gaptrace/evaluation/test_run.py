import xml.etree.ElementTree as ET
import json
import subprocess
from types import SimpleNamespace

import pytest

from evaluation.run import gate_status, read_junit
from evaluation import run as runner


@pytest.mark.parametrize("code,statuses,expected", [
    (0, ["passed"], "passed"),
    (0, [], "error"),
    (0, ["skipped"], "failed"),
    (1, ["failed"], "failed"),
    (1, ["error"], "error"),
    (2, ["passed"], "error"),
    (1, ["passed"], "failed"),
])
def test_gate_fails_closed(code, statuses, expected):
    assert gate_status([{"status": s} for s in statuses], code) == expected


def test_junit_preserves_failures_and_measurements(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text('<testsuites><testsuite><testcase classname="suite" name="case" time="0.1">'
                    '<properties><property name="recall" value="0.5"/></properties>'
                    '<failure message="failure">Missing gold passage</failure></testcase></testsuite></testsuites>')
    case = read_junit(path)[0]
    assert case["status"] == "failed" and case["detail"] == "Missing gold passage"
    assert case["measurements"] == {"recall": "0.5"}


@pytest.mark.parametrize("text,error", [("<testsuite/>", ValueError), ("broken", ET.ParseError)])
def test_bad_junit_cannot_pass(tmp_path, text, error):
    path = tmp_path / "bad.xml"
    path.write_text(text)
    with pytest.raises(error):
        read_junit(path)


@pytest.mark.parametrize("changed", [False, True])
def test_runner_records_artifacts_and_rejects_changed_sources(tmp_path, monkeypatch, changed):
    versions = iter([{"source.py": "before"}, {"source.py": "after" if changed else "before"}])
    monkeypatch.setattr(runner, "fingerprints", lambda: next(versions))

    def fake_run(command, **kwargs):
        xml = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml="))
        from pathlib import Path
        Path(xml).write_text('<testsuite><testcase classname="check" name="passes"/></testsuite>')
        return SimpleNamespace(stdout="1 passed", stderr="", returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    out = tmp_path / "artifacts"
    result = runner.run(out)
    assert result["status"] == ("error" if changed else "passed")
    assert result["changed_sources"] == (["source.py"] if changed else [])
    assert json.loads((out / "report.json").read_text())["counts"] == {"passed": 1}
    assert (out / "summary.md").exists() and (out / "pytest.log").exists()
    with pytest.raises(FileExistsError):
        runner.run(out)


def test_runner_timeout_is_an_error_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "fingerprints", lambda: {})

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    result = runner.run(tmp_path / "timeout")
    assert result["status"] == "error" and result["counts"] == {}
    assert "TimeoutExpired" in result["error"]
