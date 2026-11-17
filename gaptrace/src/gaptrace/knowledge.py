"""Locally prepared, source-backed observations and contextual retrieval sidecars."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import tokens
from .pipeline import fill, load_prompt
from .recovery import atomic_json, read_checkpoint, write_checkpoint
from .schema import Note, VerifierOutput
from .search import HybridSearch


class ObservationDraft(BaseModel):
    role: Literal["blocked_attempt", "new_capability"]
    goal: str = Field(min_length=1, max_length=400)
    mechanism: str = Field(min_length=1, max_length=700)
    quote: str = Field(min_length=1, max_length=1800)
    outcome: Literal["failed", "limited", "unresolved", "not_applicable"] = "unresolved"


class PassageKnowledge(BaseModel):
    context: str = Field(default="", max_length=2000)
    observations: list[ObservationDraft] = Field(default_factory=list, max_length=2)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_fingerprint(tb):
    return digest({"passages": [p.model_dump(mode="json") for p in sorted(tb.passages.all(), key=lambda p: p.id)],
                   "papers": [{"id": n.id, "title": n.name, "published": str(n.paper_date)}
                              for n in sorted(tb.notes.values(), key=lambda n: n.id) if n.type == "paper"]})


def excerpt(text, budget):
    return next(tokens.iter_chunks(text, budget), "")


def prepare(tb, gateway, metadata_tokens=50, progress=None, checkpoint_dir=None):
    if metadata_tokens not in {0, 25, 50, 100}:
        raise ValueError("Metadata budget must be 0, 25, 50 or 100 tokens; select it on development data")
    if gateway.s.cloud.enabled or gateway.cloud_calls:
        raise ValueError("Knowledge preparation requires cloud inference disabled")
    progress = progress or (lambda message: None)
    protocol = digest({"source": source_fingerprint(tb), "metadata_tokens": metadata_tokens,
                       "model": vars(tb.settings.model), "tokenizer": tokens.description(),
                       "prompts": {p: load_prompt(p) for p in ("knowledge_context", "knowledge_verify")}})
    contexts, observations, errors = {}, [], []
    passages = sorted(tb.passages.all(), key=lambda p: (str(p.date), p.paper_id, p.id))
    by_paper = {}
    for p in passages:
        by_paper.setdefault(p.paper_id, []).append(p.text)
    paper_excerpts = {pid: excerpt("\n".join(texts), 800) for pid, texts in by_paper.items()}
    for index, passage in enumerate(passages, 1):
        progress(f"Preparing knowledge {index}/{len(passages)}: {passage.id}")
        checkpoint = Path(checkpoint_dir) / f"{digest([protocol, passage.id])}.json" if checkpoint_dir else None
        saved = read_checkpoint(checkpoint) if checkpoint else None
        if saved is not None:
            contexts[passage.id] = saved["context"]
            observations.extend(saved["observations"])
            errors.extend(saved["errors"])
            continue
        paper = tb.notes.get(passage.paper_id)
        paper_context = {"title": paper.name if paper else passage.paper_id, "published": str(passage.date),
                         "section": passage.section,
                         "paper_excerpt": paper_excerpts[passage.paper_id]}
        payload = {"paper": paper_context, "passage_id": passage.id, "passage": passage.text}
        # Existing passages are already bounded; reject unbounded imported stores.
        if tokens.count(passage.text) > tb.settings.limits.passage_tokens:
            errors.append({"passage_id": passage.id, "error": "Passage exceeds configured chunk budget; re-ingest source"})
            continue
        result = {"context": {"text": "", "tokens": 0, "evidence": False}, "observations": [], "errors": []}
        try:
            draft, _ = gateway.call("local", fill(load_prompt("knowledge_context"),
                metadata_tokens=metadata_tokens, payload=json.dumps(payload)), PassageKnowledge)
            context = excerpt(draft.context, metadata_tokens) if metadata_tokens else ""
            if context:
                verdict, _ = gateway.call("local", fill(load_prompt("knowledge_verify"), payload=json.dumps({
                    "claim": context, "source": payload})), VerifierOutput)
                if verdict.supported:
                    result["context"] = {"text": context, "tokens": tokens.count(context), "evidence": False}
                else:
                    result["errors"].append({"passage_id": passage.id, "error": "Retrieval metadata unsupported", "reason": verdict.reason})
            for item in draft.observations:
                if item.role == "new_capability":
                    item.outcome = "not_applicable"
                if not item.quote.strip() or item.quote not in passage.text:
                    result["errors"].append({"passage_id": passage.id, "error": "Observation quote absent from raw passage"})
                    continue
                verdict, _ = gateway.call("local", fill(load_prompt("knowledge_verify"), payload=json.dumps({
                    "claim": item.model_dump(), "source": passage.text})), VerifierOutput)
                if not verdict.supported:
                    result["errors"].append({"passage_id": passage.id, "error": "Observation not entailed", "reason": verdict.reason})
                    continue
                value = {**item.model_dump(), "paper_id": passage.paper_id, "passage_id": passage.id,
                         "published": str(passage.date) if passage.date else None, "verified": True}
                value["id"] = "obs-" + digest(value)[:20]
                result["observations"].append(value)
        except (ValueError, RuntimeError, OSError) as exc:
            result["errors"].append({"passage_id": passage.id, "error": f"{type(exc).__name__}: {exc}"})
        contexts[passage.id] = result["context"]
        observations.extend(result["observations"])
        errors.extend(result["errors"])
        # Failed passages are retried on resume, rather than frozen as completed.
        if checkpoint and not result["errors"]:
            write_checkpoint(checkpoint, result)
    report = {"kind": "prepared-knowledge", "version": 1, "source_sha256": source_fingerprint(tb),
              "protocol_sha256": protocol, "metadata_budget": metadata_tokens, "model": tb.settings.model.local,
              "prepared_utc": datetime.now(timezone.utc).isoformat(), "contexts": contexts, "observations": observations,
              "coverage": {"passages": len(passages), "contextualized": sum(bool(c["text"]) for c in contexts.values()),
                           "roles": dict(Counter(o["role"] for o in observations))}, "errors": errors,
              "cloud_calls": gateway.cloud_calls, "limitations": ["Generated context is retrieval metadata, never primary evidence.",
                   "Metadata budget is provisional until a development retrieval comparison selects it.",
                   "Verified observations are local-model assessments, not independent human labels."]}
    if tb.data_dir:
        atomic_json(tb.data_dir / "knowledge.json", report)
    return report


def load(tb):
    if not tb.data_dir or not (tb.data_dir / "knowledge.json").exists():
        raise ValueError("Prepare this knowledge base before selecting an analysis engine")
    report = json.loads((tb.data_dir / "knowledge.json").read_text(encoding="utf-8"))
    if report.get("kind") != "prepared-knowledge" or report["source_sha256"] != source_fingerprint(tb):
        raise ValueError("Prepared knowledge does not match the current source corpus")
    return report


def passage_index(tb, report=None, embed=None):
    report = report or load(tb)
    if report["source_sha256"] != source_fingerprint(tb):
        raise ValueError("Stale contextual index")
    notes = []
    for p in tb.passages.all():
        context = report["contexts"].get(p.id, {}).get("text", "")
        paper = tb.notes.get(p.paper_id)
        notes.append(Note(id=p.id, type="paper", name=paper.name if paper else p.paper_id,
                          l0=p.text, l1=context, paper_id=p.paper_id, passage_ids=[p.id]))
    return HybridSearch(notes, embed=embed, hybrid=embed is not None, text_fn=lambda n: " ".join([n.l1, n.name, n.l0]))


def contextual_embedder(tb):
    from .runtime import OllamaEmbedder, create_embedder
    model = tb.settings.model.embedding
    if model.startswith("ollama/"):
        # Prefix metadata is retained if MiniLM truncates its much shorter input
        # window. BM25 still indexes the complete passage; evidence stays intact.
        return OllamaEmbedder(model.split("/", 1)[1], tb.settings.model.local_api_base,
                              tb.data_dir / "cache/contextual-embeddings" if tb.data_dir else None, truncate=True)
    return create_embedder(tb.settings, tb.data_dir / "cache" if tb.data_dir else None)
