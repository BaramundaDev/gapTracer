import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gaptrace import cli
from gaptrace.discovery import run
from gaptrace.gateway import Gateway
from gaptrace.ingest import PaperImport
from gaptrace.knowledge import load, passage_index
from gaptrace.settings import load_settings
from gaptrace.workspace import open_toolbox


OLD = "We attempted precise molecular simulation but failed because storing all pairwise interactions exceeded available memory."
NEW = "Our sparse interaction method reduces memory requirements while preserving interaction accuracy."


def source(path):
    value = {"papers": [{"id": "old", "title": "Old ambitious attempt", "published": "2018-01-01",
                         "sections": [{"heading": "Limitations", "text": OLD}]},
                        {"id": "new", "title": "New enabling method", "published": "2024-01-01",
                         "sections": [{"heading": "Method", "text": NEW}]}]}
    path.write_text(json.dumps(value))
    return path


def scripted(prompt, schema, cap):
    fields = schema["properties"]
    if "notes" in fields:
        return '{"notes":[],"links":[]}'
    if "context" in fields:
        old = '"passage_id": "old-' in prompt
        return json.dumps({"context": "Pairwise interaction memory for molecular simulation" if old else "Sparse interaction memory reduction",
            "observations": [{"role": "blocked_attempt" if old else "new_capability", "goal": "precise molecular simulation",
                "mechanism": "pairwise memory requirement" if old else "sparse interaction memory reduction", "quote": OLD if old else NEW}]})
    if "supported" in fields:
        return '{"supported":true}'
    if "mechanism_plausible" in fields:
        return json.dumps({"mechanism_plausible": True, "already_solved": False, "quote": NEW,
            "bridge": "Sparse storage may remove the old memory bottleneck", "assumptions": ["Sparse approximation preserves required precision"],
            "experiment": "Compare precise simulation against a sparse baseline on a frozen system", "falsifier": "Sparse errors exceed the fixed precision tolerance",
            "importance": 4, "importance_reason": "Would enable the stated previously impossible simulation"})
    raise AssertionError(schema)


@pytest.fixture
def settings():
    s = load_settings()
    s.switches.hybrid_search = False
    s.cloud.enabled = False
    return s


def test_local_preparation_completes_before_engine_choice_and_never_calls_cloud(tmp_path, settings):
    gateway = Gateway(settings, {"local": scripted})
    root = tmp_path / "discovery"
    def choose(manifest):
        assert manifest["state"] == "awaiting_engine_choice"
        assert (root / "corpus/knowledge.json").exists()
        knowledge = json.loads((root / "corpus/knowledge.json").read_text())
        assert len(knowledge["observations"]) == 2
        assert knowledge["cloud_calls"] == 0
        return "connector"
    result = run("molecular simulation", root, settings, gateway, source_file=source(tmp_path / "papers.json"), choose_engine=choose)
    assert result["state"] == "awaiting_external_analysis"
    assert result["external_model_called"] is False
    packet = json.loads((root / "connector-packet.json").read_text())
    assert len(packet["task"]["attempts"]) == 1
    assert all(c["tokens"] <= 50 and c["evidence"] is False for c in packet["knowledge"]["contexts"].values())
    assert not any(entry.get("tier") != "local" for entry in gateway.log)


def test_prepared_context_is_retrieval_only_and_sources_are_unchanged(tmp_path, settings):
    root = tmp_path / "d"
    run("topic", root, settings, Gateway(settings, {"local": scripted}), source_file=source(tmp_path / "papers.json"), engine="prepare")
    tb = open_toolbox(root / "corpus", settings)
    knowledge = load(tb)
    hits = passage_index(tb, knowledge).search("sparse interaction", 2)
    assert hits[0].id == "new-1-1"
    assert tb.passages.get("old-1-1").text == OLD
    assert "not evidence" in tb.search_passages("sparse interaction")


def test_local_reactivation_uses_time_order_and_requires_rechecks(tmp_path, settings, monkeypatch):
    from gaptrace import reactivation
    monkeypatch.setattr(reactivation.literature, "search", lambda *args: [])
    root = tmp_path / "d"
    result = run("topic", root, settings, Gateway(settings, {"local": scripted}),
                 source_file=source(tmp_path / "papers.json"), engine="local")
    assert result["state"] == "analysis_complete"
    report = json.loads((root / "corpus/reactivation.json").read_text())
    opportunity = report["opportunities"][0]
    assert opportunity["status"] == "reactivation_hypothesis" and opportunity["recheck"]
    assert opportunity["novelty"] == "unchecked" and opportunity["feasibility"] is None
    assert opportunity["old_source"]["published"] < opportunity["new_source"]["published"]


def test_changed_source_cannot_resume_as_same_experiment(tmp_path, settings):
    root = tmp_path / "d"
    run("topic", root, settings, Gateway(settings, {"local": scripted}), source_file=source(tmp_path / "papers.json"), engine="prepare")
    path = root / "papers.json"
    value = json.loads(path.read_text())
    value["papers"][0]["sections"][0]["text"] = "Changed"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Source files changed"):
        run("topic", root, settings, Gateway(settings, {"local": scripted}), resume=True)


def test_cli_prompts_after_extraction_and_supports_noninteractive_prepare(tmp_path, monkeypatch):
    from gaptrace import gateway
    monkeypatch.setattr(gateway, "litellm_backend", lambda *args, **kwargs: scripted)
    path = source(tmp_path / "papers.json")
    result = CliRunner().invoke(cli.app, ["discover", "topic", "--source", str(path), "--data", str(tmp_path / "d")], input="connector\n")
    assert result.exit_code == 0, result.output
    assert "Local extraction and knowledge preparation are saved" in result.output
    assert "awaiting_external_analysis" in result.stdout
