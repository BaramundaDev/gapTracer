import asyncio
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest

from gaptrace import connector_protocol
from gaptrace.connector_protocol import Connector, digest
from gaptrace.discovery import run
from gaptrace.gateway import Gateway
from gaptrace.settings import load_settings


@pytest.fixture
def packet(tmp_path):
    # Explicitly synthetic engineering controls, never independent research labels.
    from test_discovery import source, scripted
    s = load_settings()
    s.switches.hybrid_search = False
    s.cloud.enabled = False
    root = tmp_path / "research"
    run("Synthetic test", root, s, Gateway(s, {"local": scripted}), source_file=source(tmp_path / "papers.json"), engine="connector")
    return root / "connector-packet.json"


def assessment():
    from test_discovery import NEW
    return {"mechanism_plausible": True, "already_solved": False, "quote": NEW, "bridge": "Sparse storage may address the old memory blocker",
            "assumptions": ["Precision survives sparse approximation"], "experiment": "Compare dense and sparse simulation",
            "falsifier": "Required precision is lost", "importance": 4, "importance_reason": "Could unlock the stated simulation objective"}


def test_connector_snapshot_source_integrity_and_retrieval(packet):
    data = json.loads(packet.read_text())
    core = Connector(data)
    assert core.search("sparse")["results"][0]["id"] == "new-1-1"
    assert core.fetch("new-1-1")["text"] in data["passages"][1]["text"]
    assert core.get_research_task()["attempts"]
    data["passages"][0]["text"] += "Modified source"
    with pytest.raises(ValueError, match="differs"):
        Connector(data)


def test_frontier_analysis_has_same_provenance_requirements_and_no_model_api(packet):
    data = json.loads(packet.read_text())
    saved = []
    core = Connector(data, commit=saved.append)
    old = core.task["attempts"][0]
    item = {"attempt_id": old["id"], "new_source_id": "new-1-1", "assessment": assessment(), "recheck": assessment()}
    value = core.submit_analysis(core.task["task_sha256"], "external-test-model", [item])
    assert value["analysis"]["opportunities"][0]["status"] == "reactivation_hypothesis"
    assert saved[0]["engine"] == "connector" and saved[0]["opportunities"][0]["novelty"] == "unchecked"
    item["recheck"]["quote"] = "Made up quotation"
    with pytest.raises(ValueError, match="not in"):
        core.submit_analysis(core.task["task_sha256"], "external-test-model", [item])


def test_source_that_predates_attempt_cannot_enable_it(packet):
    core = Connector(json.loads(packet.read_text()))
    item = {"attempt_id": core.task["attempts"][0]["id"], "new_source_id": "old-1-1", "assessment": assessment(), "recheck": assessment()}
    with pytest.raises(ValueError, match="postdate"):
        core.submit_analysis(core.task["task_sha256"], "external-test", [item])


def test_official_mcp_sdk_exposes_research_task_search_fetch_and_submission(packet):
    from gaptrace.connector import build_server
    server = build_server(packet)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"search", "fetch", "get_research_task", "get_attempt", "search_literature", "submit_analysis"}
    value = asyncio.run(server.call_tool("fetch", {"id": "new-1-1"}))
    assert "sparse interaction" in str(value)


@pytest.fixture
def aws_handler(monkeypatch):
    from gaptrace import literature
    monkeypatch.setitem(sys.modules, "connector_protocol", connector_protocol)
    monkeypatch.setitem(sys.modules, "literature", literature)
    path = Path(__file__).resolve().parents[1] / "deploy/aws/handler.py"
    spec = importlib.util.spec_from_file_location("gaptrace_aws_handler_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("RESOURCE_URL", "https://example.execute-api.test/mcp")
    monkeypatch.setenv("OAUTH_ISSUER", "https://cognito-idp.test/pool")
    monkeypatch.setenv("KNOWLEDGE_BUCKET", "private-test-bucket")
    return module


def event(method="initialize", scopes=None):
    return {"rawPath": "/mcp", "headers": {}, "requestContext": {"http": {"method": "POST"},
        "authorizer": {"jwt": {"scopes": scopes or ["gaptrace/read"], "claims": {
            "sub": "test-user", "iss": "https://cognito-idp.test/pool", "aud": "https://example.execute-api.test/mcp", "token_use": "access"}}}},
        "body": json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": {"protocolVersion": "2025-11-25"}})}


def test_aws_protocol_initialization_and_resource_bound_auth(aws_handler):
    request = event()
    result = aws_handler.handler(request, None)
    assert result["statusCode"] == 200 and json.loads(result["body"])["result"]["protocolVersion"] == "2025-11-25"
    request["requestContext"].pop("authorizer")
    assert aws_handler.handler(request, None)["statusCode"] == 401
    request = event()
    request["requestContext"]["authorizer"]["jwt"]["claims"]["aud"] = "https://other-resource.test"
    assert aws_handler.handler(request, None)["statusCode"] == 401


def test_aws_write_scope_is_enforced_on_invocation_not_just_tool_listing(aws_handler):
    request = event("tools/call")
    request["body"] = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "submit_analysis", "arguments": {}}})
    assert aws_handler.handler(request, None)["statusCode"] == 403
    request = event("tools/list")
    tools = json.loads(aws_handler.handler(request, None)["body"])["result"]["tools"]
    assert "submit_analysis" not in {t["name"] for t in tools}


def test_aws_origin_validation_and_public_oauth_discovery(aws_handler):
    request = event()
    request["headers"]["origin"] = "https://malicious.test"
    assert aws_handler.handler(request, None)["statusCode"] == 403
    request = {"rawPath": "/.well-known/oauth-protected-resource", "requestContext": {"http": {"method": "GET"}}}
    result = aws_handler.handler(request, None)
    assert result["statusCode"] == 200
    assert json.loads(result["body"])["resource"] == "https://example.execute-api.test/mcp"


def test_aws_malformed_rpc_is_a_bounded_error(aws_handler):
    request = event()
    request["body"] = '{"jsonrpc":"2.0","id":1}'
    result = json.loads(aws_handler.handler(request, None)["body"])
    assert result["error"]["code"] == -32602
