import asyncio
import json

from typer.testing import CliRunner

from gaptrace.cli import app

runner = CliRunner()


def test_cli_end_to_end(tmp_path):
    data = str(tmp_path / "data")
    assert runner.invoke(app, ["demo", "--data", data]).exit_code == 0
    out = runner.invoke(app, ["rebuild", "--data", data])
    assert out.exit_code == 0 and "notes 14" in out.stdout and "facts 2" in out.stdout
    out = runner.invoke(app, ["gaps", "--data", data])
    assert out.exit_code == 0 and "untested_combination" in out.stdout and "conflicting_results" in out.stdout
    out = runner.invoke(app, ["fact", "--method", "replay", "--data", data])
    assert "71.2" in out.stdout and "65" in out.stdout
    assert "Replay buffer" in runner.invoke(app, ["search", "replay buffer", "--data", data]).stdout


def test_ingest_validates_cutoff_and_publishes_json_inventory(tmp_path, monkeypatch):
    from gaptrace import gateway

    calls = []

    def backend(prompt, schema, limit):
        calls.append(prompt)
        return '{"notes": [], "links": []}'

    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: backend)
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"papers": [{"id": "paper-1", "title": "First", "published": "2024-02-01",
                                             "sections": [{"heading": "Abstract", "text": "Example text."}]}]}),
                      encoding="utf-8")
    data = str(tmp_path / "corpus")
    base = ["ingest", str(source), "--data", data, "--local-only", "--cutoff"]
    rejected = runner.invoke(app, [*base, "2024-01-01"])
    assert rejected.exit_code != 0 and "published after cutoff" in rejected.output and not calls
    result = runner.invoke(app, [*base, "2024-02-01"])
    assert result.exit_code == 0, result.output
    assert "Extracting passage 1" in result.stderr and "Saved" in result.stderr
    assert json.loads(result.stdout)["stats"]["passages"] == 1
    metadata = json.loads(runner.invoke(app, ["corpus", "--data", data]).stdout)
    assert metadata["papers"] == [{"id": "paper-1", "published": "2024-02-01"}]
    assert metadata["import"]["build_cloud_enabled"] is False and len(calls) == 1
    passage = runner.invoke(app, ["passage", "paper-1-1-1", "--data", data, "--json"])
    assert json.loads(passage.stdout)["text"] == "Example text."
    assert json.loads(runner.invoke(app, ["gaps", "--data", data, "--json"]).stdout) == []


def test_terminal_session_switches_corpora_and_recovers_from_errors(data_dir, tmp_path):
    other = str(tmp_path / "other corpus")
    assert runner.invoke(app, ["demo", "--data", other]).exit_code == 0
    result = runner.invoke(app, ["shell", "--data", str(data_dir)], input=(
        'search "replay buffer"\nunknown-command\nsearch "unclosed\nuse missing-corpus\n'
        f'use "{other}"\nstatus\ngaps\nexit\n'
    ))
    assert result.exit_code == 0, result.output
    assert "Replay buffer" in result.output
    assert "No such command" in result.output and "No closing quotation" in result.output
    assert "No corpus bundle" in result.output and f"Corpus: {other}" in result.output
    assert "Notes 14" in result.output and "untested_combination" in result.output


def test_status_reports_evidence_and_rejects_missing_corpus(data_dir, tmp_path):
    result = runner.invoke(app, ["status", "--data", str(data_dir), "--json"])
    assert result.exit_code == 0, result.output
    status = json.loads(result.stdout)
    assert status["notes"] == 14 and status["facts"] == 2
    assert 0 < status["verified_links"] <= status["links"]
    missing = tmp_path / "missing"
    result = runner.invoke(app, ["status", "--data", str(missing)])
    assert result.exit_code != 0 and "No corpus" in result.output and not missing.exists()


def test_terminal_eof_exits_cleanly(data_dir):
    result = runner.invoke(app, ["shell", "--data", str(data_dir)], input="")
    assert result.exit_code == 0, result.output


def test_ask_json_includes_passages_fetched_by_final_validation(data_dir, monkeypatch):
    from gaptrace import gateway
    from gaptrace.schema import make_id
    def backend(prompt, schema, limit):
        if "answer" in schema["properties"]:
            return json.dumps({"answer": "Replay buffer reaches 71.2% and 65.0% accuracy on Split CIFAR-100.",
                               "cited": [make_id("method", "Replay buffer")]})
        return json.dumps({"supported": True})
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: backend)
    out = runner.invoke(app, ["ask", "Replay buffer accuracy?", "--data", str(data_dir), "--json", "--local-only"])
    assert out.exit_code == 0, out.output
    result = json.loads(out.stdout)
    assert "Searching the active corpus" in out.stderr
    assert "Checking answer claims" in out.stderr
    assert "Completed in" in out.stderr
    assert set(result["cited_passage_ids"]) == {"a-1", "b-1"}
    assert {"pap-a", "pap-b"} <= set(result["retrieved_paper_ids"])


def test_api(data_dir):
    from fastapi.testclient import TestClient
    from gaptrace.api import create_app

    client = TestClient(create_app(str(data_dir)))
    gaps = client.get("/gaps").json()
    assert len(gaps) == 5 and all("trail" in g and "verified_share" in g for g in gaps)
    assert client.get(f"/gaps/{gaps[0]['id']}").status_code == 200
    assert client.get("/notes/search", params={"q": "replay buffer", "note_type": "method"}).json()[0]["name"] == "Replay buffer"
    assert client.get("/passages/a-1").json()["paper_id"] == "pap-a"
    assert client.get("/notes/nope").status_code == 404


def test_mcp_server_lists_and_runs_tools(data_dir):
    from gaptrace.mcp_server import build_server

    server = build_server(str(data_dir))
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {"search_notes", "get_note", "neighbours", "get_passage", "lookup_fact", "find_gaps", "get_trail",
                     "search_literature", "review_gaps", "search_passages"}
    result = asyncio.run(server.call_tool("find_gaps", {}))
    assert "untested_combination" in str(result)
