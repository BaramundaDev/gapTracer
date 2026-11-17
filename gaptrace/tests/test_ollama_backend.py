import io
import json
from urllib.error import HTTPError

import pytest

from gaptrace import ollama_backend
from gaptrace.gateway import litellm_backend


def stream(*items):
    return io.BytesIO(b"".join(json.dumps(item).encode() + b"\n" for item in items))


def test_local_backend_uses_native_schema_stream_without_litellm(monkeypatch):
    calls = []
    def open_request(request, timeout):
        calls.append((request, timeout))
        return stream({"message": {"content": '{"supported":'}},
                      {"message": {"content": "true}"}, "done": True})
    monkeypatch.setattr(ollama_backend, "urlopen", open_request)
    schema = {"type": "object", "properties": {"supported": {"type": "boolean"}}}
    run = litellm_backend("ollama_chat/qwen3.5:4b", 8192, api_base="http://localhost:11434")
    assert json.loads(run("Read the source", schema, 1200)) == {"supported": True}
    request, timeout = calls[0]
    payload = json.loads(request.data)
    assert request.full_url == "http://localhost:11434/api/chat" and timeout == 180
    assert payload["format"] == schema and payload["stream"] and payload["think"] is False
    assert payload["options"] == {"num_ctx": 8192, "num_predict": 1200, "temperature": 0}


@pytest.mark.parametrize("first", [TimeoutError("stalled"),
    [{"message": {"content": "partial output"}}]])
def test_retry_discards_timed_out_or_incomplete_output(monkeypatch, first):
    calls, progress = [], []
    def open_request(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            if isinstance(first, Exception):
                raise first
            return stream(*first)
        return stream({"message": {"content": "{}"}, "done": True})
    monkeypatch.setattr(ollama_backend, "urlopen", open_request)
    assert ollama_backend.backend("ollama_chat/qwen3.5:4b", progress=progress.append)("p", {}, 100) == "{}"
    assert len(calls) == 2 and "retrying this request once" in progress[0]


def test_retry_is_bounded_and_stall_is_actionable(monkeypatch):
    calls = []
    def open_request(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("read timed out")
    monkeypatch.setattr(ollama_backend, "urlopen", open_request)
    with pytest.raises(RuntimeError, match="after one retry"):
        ollama_backend.backend("ollama_chat/qwen3.5:4b", progress=lambda _: None)("p", {}, 100)
    assert len(calls) == 2


def test_missing_model_is_not_retried(monkeypatch):
    calls = []
    def open_request(*args, **kwargs):
        calls.append(1)
        raise HTTPError("http://localhost", 404, "Missing", {}, io.BytesIO(b'{"error":"model missing"}'))
    monkeypatch.setattr(ollama_backend, "urlopen", open_request)
    with pytest.raises(RuntimeError, match="HTTP 404.*model missing"):
        ollama_backend.backend("ollama_chat/qwen3.5:4b")("p", {}, 100)
    assert len(calls) == 1


def test_generation_has_a_total_deadline_even_with_active_stream(monkeypatch):
    monkeypatch.setattr(ollama_backend, "urlopen", lambda *args, **kwargs:
                        stream({"message": {"content": "{}"}, "done": True}))
    times = iter([0, 11, 20, 31])
    monkeypatch.setattr(ollama_backend, "monotonic", lambda: next(times))
    with pytest.raises(RuntimeError, match="exceeded 10s"):
        ollama_backend.backend("ollama_chat/qwen3.5:4b", total_timeout=10, progress=lambda _: None)("p", {}, 100)


def test_native_backend_preserves_measurement_callback(monkeypatch):
    events = []
    monkeypatch.setattr(ollama_backend, "urlopen", lambda *args, **kwargs:
                        stream({"message": {"content": "{}"}, "done": True, "eval_count": 2,
                                "eval_duration": 100_000_000}))
    backend = litellm_backend("ollama_chat/qwen3.5:4b", logger_fn=events.append)
    assert backend("p", {}, 100) == "{}"
    assert events[0]["log_event_type"] == "post_api_call"
    assert json.loads(events[0]["original_response"])["eval_count"] == 2


def test_streamed_model_crash_retries_then_can_recover(monkeypatch):
    calls = []
    def open_request(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return stream({"error": "llama-server process terminated"})
        return stream({"message": {"content": "{}"}, "done": True})
    monkeypatch.setattr(ollama_backend, "urlopen", open_request)
    assert ollama_backend.backend("ollama_chat/qwen3.5:4b", progress=lambda _: None)("p", {}, 100) == "{}"
    assert len(calls) == 2
