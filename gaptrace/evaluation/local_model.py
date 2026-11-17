"""Loopback-only research inference through the product's existing token gateway."""
from __future__ import annotations

import json
from pathlib import Path
from time import monotonic
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from gaptrace import tokens
from gaptrace.gateway import Gateway
from gaptrace.settings import load_settings


class LocalModel:
    def __init__(self, tokenizer: Path, model="qwen3.5:4b", base="http://127.0.0.1:11434",
                 settings_path=None, timeout=120):
        from tokenizers import Tokenizer

        url = urlparse(base)
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
                or url.username or url.password or url.path not in {"", "/"} or url.query or url.fragment):
            raise ValueError("Research inference requires a loopback Ollama URL")
        if "/" in model or "-cloud" in model or model.endswith(":cloud"):
            raise ValueError("Use a local Ollama model tag")
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.model, self.base, self.timeout = model, base.rstrip("/"), timeout
        self.settings = load_settings(settings_path)
        if self.settings.budget.total != 8000:
            raise ValueError("Research runs preserve the 8,000-token policy")
        self.settings.cloud.enabled = False
        self.settings.model.local = "ollama_chat/" + model
        self.calls = []
        with urlopen(self.base + "/api/tags", timeout=min(10, timeout)) as response:
            inventory = json.load(response)
        matches = [item for item in inventory["models"] if item["name"] == model or item.get("model") == model]
        if not matches:
            raise ValueError("The requested local model is not installed on this Ollama server")
        self.model_metadata = matches[0]
        if matches[0].get("remote_host") or matches[0].get("remote_model"):
            raise ValueError("Cloud-proxied Ollama models cannot generate research experiments")
        self.model_digest = matches[0].get("digest")
        self.deadline = None
        self.gateway = Gateway(self.settings, {"local": self._backend})

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def clip(self, text, limit):
        # Character-prefix search preserves Unicode and does not depend on token decode.
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.count(text[:mid]) <= limit:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo]

    def _backend(self, prompt, schema, max_tokens):
        # Include structured-output schema and reserve chat framing overhead up front.
        limit = self.settings.budget.total - max_tokens - self.settings.budget.margin
        if self.count(prompt) + self.count(json.dumps(schema)) + 256 > limit:
            raise ValueError("Prompt plus output schema exceeds the research input allowance")
        payload = {"model": self.model, "stream": False, "think": False, "format": schema,
                   "messages": [{"role": "user", "content": prompt}],
                   "options": {"num_ctx": 8192, "num_predict": max_tokens, "temperature": 0}}
        request = Request(self.base + "/api/chat", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        start = monotonic()
        timeout = self.timeout if self.deadline is None else min(self.timeout, self.deadline - monotonic())
        if timeout <= 0:
            raise TimeoutError("Local research inference deadline exhausted")
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        usage = {"prompt_tokens": result["prompt_eval_count"], "output_tokens": result["eval_count"],
                 "seconds": monotonic() - start, "model": result["model"],
                 "done_reason": result.get("done_reason")}
        self.calls.append(usage)
        if usage["prompt_tokens"] + usage["output_tokens"] > 8000:
            raise ValueError("Server reported a call exceeding the 8,000-token policy")
        if result.get("done_reason") == "length":
            raise ValueError("Research model exhausted its output allowance")
        return result["message"]["content"]

    def call(self, prompt, schema):
        # The gateway and its schema retry use the exact tokenizer for this model.
        previous = tokens._counter
        tokens.set_counter(self.count)
        try:
            result, tier = self.gateway.call("local", prompt, schema)
            assert tier == "local"
            return result
        finally:
            tokens.set_counter(previous)
