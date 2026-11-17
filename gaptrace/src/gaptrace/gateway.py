"""One gateway for every model call: token gate, disk cache, schema check,
tier routing with local fallback, and the per-topic cloud call budget."""
from __future__ import annotations

import hashlib
import json
from time import perf_counter
from pathlib import Path
from typing import Callable, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

from .context import assert_within_budget
from . import tokens
from .terminal_ui import emit
from .recovery import read_checkpoint, write_checkpoint

Backend = Callable[[str, dict, int], str]      # (prompt, json_schema, max_output_tokens) -> raw text

_ACTIVITIES = {
    "StepOutput": "selecting the next evidence action",
    "ExtractorOutput": "extracting research notes from a source passage",
    "VerifierOutput": "checking a claim against source evidence",
    "ClosureOutput": "checking whether later research addresses a gap",
    "DoabilityOutput": "assessing experiment resources and scope",
    "ConfirmOutput": "checking whether notes describe the same concept",
    "Direction": "proposing an experiment from reviewed evidence",
}


def _activity(schema):
    return _ACTIVITIES.get(schema.__name__, "preparing a structured response")


class ModelOutputInvalid(Exception):
    pass


def _extract_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):]
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start:end + 1] if start != -1 and end != -1 else raw


class Gateway:
    """backends: {"local": fn, "cloud_light": fn, "cloud_strong": fn}; only
    "local" is required."""

    def __init__(self, settings, backends: Dict[str, Backend], cache_dir: Optional[Path] = None):
        self.s = settings
        self.backends = backends
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cloud_calls = 0
        self.log: list = []

    def _cached(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.json"
        def validate(value):
            if not isinstance(value["raw"], str):
                raise ValueError("Invalid cached model response")
            return value["raw"]
        return read_checkpoint(p, validate)

    def _store(self, key: str, raw: str, tier: str) -> None:
        if self.cache_dir:
            write_checkpoint(self.cache_dir / f"{key}.json", {"tier": tier, "raw": raw})

    def _pick(self, tier: str) -> str:
        if tier == "local":
            return "local"
        usable = (self.s.cloud.enabled and tier in self.backends and
                  self.cloud_calls < self.s.limits.max_cloud_calls_per_topic)
        return tier if usable else "local"

    def _attempt(self, tier: str, prompt: str, schema: Type[BaseModel], repair: bool) -> str:
        if tier == "local":
            assert_within_budget(prompt, self.s)
        else:
            if self.cloud_calls >= self.s.limits.max_cloud_calls_per_topic:
                raise RuntimeError("Cloud topic call cap exhausted")
            # Dispatches that raise can still incur provider costs.
            self.cloud_calls += 1
        started = perf_counter()
        entry = {"tier": tier, "model": getattr(self.s.model, tier, ""), "dispatch": True,
                 "schema": schema.__name__, "repair": repair, "cached": False,
                 "input_tokens": tokens.count(prompt), "token_count_kind": "configured counter"}
        emit(f"Calling {entry['model']}: {_activity(schema)}"
             f" | {entry['input_tokens']} input tokens" + (" | schema repair" if repair else ""),
             kind="request_start", schema=schema.__name__, model=entry["model"])
        try:
            raw = self.backends[tier](prompt, schema.model_json_schema(), self.s.budget.output)
        except Exception as exc:
            entry.update(error=str(exc)[:200], seconds=perf_counter() - started)
            if tier != "local":
                entry["fallback"] = tier
            self.log.append(entry)
            emit(f"Model request failed: {type(exc).__name__}", kind="request_error")
            raise
        entry.update(seconds=perf_counter() - started, output_tokens=tokens.count(raw))
        self.log.append(entry)
        emit(f"Model response received in {entry['seconds']:.1f}s | {entry['output_tokens']} output tokens",
             kind="request_done", seconds=entry["seconds"])
        return raw

    def call(self, tier: str, prompt: str, schema: Type[BaseModel]) -> tuple[BaseModel, str]:
        """One logical request, at most one schema repair, with bounded fallback."""
        used = tier if tier == "local" or (self.s.cloud.enabled and tier in self.backends) else "local"
        repaired = False
        while True:
            request = prompt + ("\n\nReturn valid JSON only, matching the schema exactly." if repaired else "")
            if used == "local":
                assert_within_budget(request, self.s)
            model = getattr(self.s.model, used, "")
            context_size = self.s.model.local_num_ctx if used == "local" else None
            fields = (used, model, context_size, self.s.budget.output, schema.model_json_schema(), prompt)
            key = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
            raw = self._cached(key) if not repaired else None
            cached = raw is not None
            if cached:
                emit(f"Reusing cached response: {_activity(schema)}", kind="cache_hit")
            if raw is None:
                if used != "local" and self.cloud_calls >= self.s.limits.max_cloud_calls_per_topic:
                    self.log.append({"fallback": used, "reason": "topic call cap", "repair": repaired})
                    emit("Cloud call cap reached; switching to local model")
                    used = "local"
                    continue
                try:
                    raw = self._attempt(used, request, schema, repaired)
                except Exception as exc:
                    if used == "local":
                        raise
                    emit("Cloud request failed; switching to local model")
                    used = "local"
                    continue
            try:
                out = schema.model_validate_json(_extract_json(raw))
            except ValidationError as exc:
                if repaired or not self.s.switches.constrained_output:
                    raise ModelOutputInvalid(f"{schema.__name__}: {exc.errors()[:2]}") from exc
                repaired = True
                emit("Model response did not match the required structure; requesting one repair")
                continue
            if not cached:
                self._store(key, raw, used)
            self.log.append({"tier": used, "schema": schema.__name__, "cached": cached, "repair": repaired})
            return out, used

def litellm_backend(model: str, num_ctx: Optional[int] = None, **extra) -> Backend:
    """Use native streaming for local Ollama; LiteLLM for other providers."""
    if model.startswith("ollama_chat/") and not model.endswith((":cloud", "-cloud")):
        from .ollama_backend import backend
        return backend(model, num_ctx, **extra)

    def run(prompt: str, json_schema: dict, max_tokens: int) -> str:
        if model.startswith("ollama_chat/"):
            import os
            os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        import litellm  # imported lazily so the core works without it

        kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}],
                      max_tokens=max_tokens, temperature=0,
                      response_format={"type": "json_schema",
                                       "json_schema": {"name": "output", "schema": json_schema}})
        if num_ctx:
            kwargs["num_ctx"] = num_ctx
        if model.partition(":")[0] == "ollama_chat/qwen3.5":
            kwargs["reasoning_effort"] = "none"  # Keep the generation budget available for JSON.
        kwargs.update(extra)
        return litellm.completion(**kwargs).choices[0].message.content or ""
    return run
