import io

import pytest
from rich.console import Console
from typer.testing import CliRunner

from gaptrace.cli import app
from gaptrace.terminal_ui import Activity, emit


def test_request_estimate_uses_only_matching_measured_calls():
    now = [0.0]
    stream = io.StringIO()
    activity = Activity("Test", clock=lambda: now[0], console=Console(file=stream, width=160))
    with activity:
        emit("First call", kind="request_start", schema="extract", model="local")
        assert "learning" in activity.estimate()
        now[0] = 20
        emit("Response", kind="request_done", seconds=20)
        emit("Second call", kind="request_start", schema="extract", model="local")
        now[0] = 25
        assert "15s left" in activity.estimate()
        now[0] = 41
        assert "remaining unknown" in activity.estimate()
        emit("Response", kind="request_done", seconds=21)
        emit("Different task", kind="request_start", schema="verify", model="local")
        assert "learning" in activity.estimate()
        emit("Failure", kind="request_error")
        assert "unknown" in activity.estimate()
    assert "Completed in 41s" in stream.getvalue()


def test_failure_restores_listener_and_never_reports_success():
    stream = io.StringIO()
    with pytest.raises(RuntimeError):
        with Activity("Test", console=Console(file=stream)):
            emit("[source-id] literal evidence")
            raise RuntimeError("unavailable")
    before = stream.getvalue()
    emit("Must not leak into a finished activity")
    assert stream.getvalue() == before
    assert "[source-id] literal evidence" in before
    assert "Failed in" in before and "Completed in" not in before


def test_nested_activity_restores_parent_listener():
    outer, inner = io.StringIO(), io.StringIO()
    with Activity("Outer", console=Console(file=outer)):
        with Activity("Inner", console=Console(file=inner)):
            emit("child event")
        emit("parent event")
    assert "child event" in inner.getvalue() and "child event" not in outer.getvalue()
    assert "parent event" in outer.getvalue() and "parent event" not in inner.getvalue()


def test_bare_command_opens_branded_shell_without_creating_a_corpus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, [], input="exit\n")
    assert result.exit_code == 0, result.output
    assert "GapTracer" in result.output and "No corpus loaded" in result.output
    assert not (tmp_path / "data").exists()


def test_live_render_updates_elapsed_time_and_stops_on_interrupt():
    stream = io.StringIO()
    now = [0.0]
    activity = Activity("Test", clock=lambda: now[0], console=Console(file=stream, force_terminal=True))
    with pytest.raises(KeyboardInterrupt):
        with activity:
            now[0] = 64
            assert "elapsed 1m 04s" in activity.__rich__().plain
            raise KeyboardInterrupt
    assert "Interrupted in" in stream.getvalue()
    assert not activity.live.is_started
