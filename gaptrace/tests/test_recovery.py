import json
import subprocess
import sys

import pytest

from gaptrace import recovery
from gaptrace.gateway import Gateway
from gaptrace.schema import VerifierOutput


def test_interrupted_atomic_save_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    recovery.atomic_json(path, {"completed": 4})
    def fail(*args):
        raise OSError("write interrupted")
    monkeypatch.setattr(recovery.os, "replace", fail)
    with pytest.raises(OSError, match="write interrupted"):
        recovery.atomic_json(path, {"completed": 5})
    assert json.loads(path.read_text()) == {"completed": 4}
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_checksum_rejects_damage_and_preserves_bad_file(tmp_path):
    path = tmp_path / "checkpoint.json"
    recovery.write_checkpoint(path, {"completed": 4})
    value = json.loads(path.read_text())
    value["payload"]["completed"] = 999
    path.write_text(json.dumps(value))
    assert recovery.read_checkpoint(path) is None
    assert not path.exists() and len(list(tmp_path.glob("*.corrupt-*"))) == 1


def test_model_response_cache_repairs_only_damaged_entry(tmp_path, settings):
    calls = []
    def backend(*args):
        calls.append(1)
        return '{"supported": true}'
    gateway = Gateway(settings, {"local": backend}, tmp_path)
    gateway.call("local", "evidence", VerifierOutput)
    path = next(tmp_path.glob("*.json"))
    path.write_text("{partial")
    assert gateway.call("local", "evidence", VerifierOutput)[0].supported
    assert len(calls) == 2 and len(list(tmp_path.glob("*.corrupt-*"))) == 1
    gateway.call("local", "evidence", VerifierOutput)
    assert len(calls) == 2


def test_lock_rejects_concurrent_writer_and_releases_on_failure(tmp_path):
    root = tmp_path / "research"
    with pytest.raises(ValueError):
        with recovery.run_lock(root):
            with pytest.raises(RuntimeError, match="already running"):
                with recovery.run_lock(root):
                    pytest.fail("Second writer acquired the lock")
            raise ValueError("simulated run failure")
    with recovery.run_lock(root):
        pass


def test_recovery_retries_transient_failure_with_backoff_but_not_bad_config(monkeypatch):
    events, delays, calls = [], [], []
    monkeypatch.setattr(recovery, "sleep", delays.append)
    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise recovery.TransientFailure("connection lost")
        return "recovered"
    assert recovery.recover(operation, "extraction", on_retry=events.append) == "recovered"
    assert delays == [2, 4] and [e["attempt"] for e in events] == [1, 2]
    with pytest.raises(ValueError, match="bad configuration"):
        recovery.recover(lambda: (_ for _ in ()).throw(ValueError("bad configuration")), "setup")


def test_abrupt_process_exit_preserves_checkpoint_and_releases_lock(tmp_path):
    script = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from gaptrace.recovery import run_lock, write_checkpoint\n"
        "root = Path(sys.argv[1])\n"
        "with run_lock(root):\n"
        "    write_checkpoint(root / 'passage.json', {'completed': 1})\n"
        "    os._exit(7)\n"
    )
    root = tmp_path / "crashed-run"
    result = subprocess.run([sys.executable, "-c", script, str(root)], timeout=10,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), capture_output=True)
    assert result.returncode == 7, result.stderr
    assert recovery.read_checkpoint(root / "passage.json") == {"completed": 1}
    with recovery.run_lock(root):
        pass
