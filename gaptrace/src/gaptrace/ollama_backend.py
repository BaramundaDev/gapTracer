"""Streaming local inference with bounded retries and explicit stall detection."""
from __future__ import annotations

import json
import sys
from http.client import IncompleteRead
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .recovery import TransientFailure

from .terminal_ui import emit, _listener


def backend(model, num_ctx=None, api_base="http://127.0.0.1:11434", timeout=180,
            total_timeout=600, progress=None, logger_fn=None):
    def report(message):
        if _listener.get() is not None:
            emit(message)
        else:
            print(message, file=sys.stderr, flush=True)
    progress = progress or report

    def run(prompt, schema, max_tokens):
        options = {"temperature": 0, "num_predict": max_tokens}
        if num_ctx:
            options["num_ctx"] = num_ctx
        payload = {"model": model.split("/", 1)[1], "messages": [{"role": "user", "content": prompt}],
                   "format": schema, "stream": True, "options": options, "keep_alive": "5m"}
        if payload["model"].split(":", 1)[0] == "qwen3.5":
            payload["think"] = False
        for attempt in range(2):
            started = last_update = monotonic()
            pieces, size = [], 0
            request = Request(api_base.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
            try:
                with urlopen(request, timeout=timeout) as response:
                    while True:
                        line = response.readline(1_000_001)
                        elapsed = monotonic() - started
                        if elapsed > total_timeout:
                            raise TimeoutError(f"generation exceeded {total_timeout}s")
                        if not line:
                            raise IncompleteRead(b"", None)
                        if len(line) > 1_000_000:
                            raise RuntimeError("Ollama returned an oversized response chunk")
                        item = json.loads(line)
                        if item.get("error"):
                            raise TransientFailure(f"Ollama inference failed: {item['error']}")
                        content = item.get("message", {}).get("content", "")
                        pieces.append(content)
                        size += len(content)
                        if size > 2_000_000:
                            raise RuntimeError("Ollama response exceeded its size limit")
                        if monotonic() - last_update >= 30:
                            progress(f"Local model is still generating ({elapsed:.0f}s elapsed)")
                            last_update = monotonic()
                        if item.get("done"):
                            if logger_fn:
                                logger_fn({"log_event_type": "post_api_call", "original_response": json.dumps(item)})
                            # An incomplete JSON document is repaired by the schema gateway.
                            return "".join(pieces)
            except HTTPError as exc:
                detail = exc.read(2000).decode("utf-8", errors="replace")
                retryable = exc.code in {408, 429} or 500 <= exc.code < 600
                if not retryable or attempt:
                    if retryable:
                        raise TransientFailure(f"Ollama HTTP {exc.code}: {detail}") from exc
                    raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
                reason = f"HTTP {exc.code}: {detail}"
            except (TransientFailure, TimeoutError, URLError, ConnectionError, IncompleteRead) as exc:
                if attempt:
                    raise TransientFailure(f"Ollama stalled or disconnected after one retry: {exc}. "
                                       "Completed work is checkpointed; resume this research run.") from exc
                reason = str(exc)
            progress(f"Ollama request interrupted ({reason}); retrying this request once.")

    return run
