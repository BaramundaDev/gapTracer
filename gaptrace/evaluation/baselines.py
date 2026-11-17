"""Executable retrieval baselines; research-only, with the same agent/checker.

lexical/vector/hybrid retrieve note summaries. raw-passages retrieves full
passages using BM25. All variants use the unchanged context packer and gateway.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from gaptrace.agent import Agent
from gaptrace.bundle import PassageStore, read_bundle
from gaptrace.gateway import Gateway
from gaptrace.schema import Note
from gaptrace.search import VectorIndex
from gaptrace.settings import load_settings
from gaptrace.tools import Toolbox
from gaptrace import tokens


MODES = ("full", "lexical", "vector", "hybrid", "raw-passages", "vector-passages", "hybrid-passages")


class RetrievalOnlyToolbox(Toolbox):
    """A retrieval baseline cannot secretly use graph, gap, or SQL-fact tools."""
    def __init__(self, notes, *args, **kwargs):
        # get_note itself exposes relationships and exact structured facts.
        clean = [note.model_copy(update={"links": [], "fact": None}, deep=True) for note in notes]
        super().__init__(clean, *args, **kwargs)

    def call(self, tool, args):
        if tool not in {"search_notes", "get_note", "get_passage"}:
            return f"Tool {tool} is unavailable in this retrieval-only baseline."
        return super().call(tool, args)


def answer(question: str, data: Path, mode: str, gateway: Gateway, settings, run_dir: Path, embed=None) -> dict:
    if mode not in MODES:
        raise ValueError(f"Unknown baseline {mode}")
    if mode in {"full", "vector", "hybrid", "vector-passages", "hybrid-passages"} and embed is None:
        raise ValueError(f"{mode} requires actual embeddings")
    store = PassageStore(data / "passages.jsonl")
    notes = read_bundle(data / "bundle")
    if not notes or not store.all():
        raise ValueError("Baseline requires a populated corpus")
    if mode in {"raw-passages", "vector-passages", "hybrid-passages"}:
        papers = {n.id: n.name for n in notes if n.type == "paper"}
        notes = [Note(id=f"{p.id}-raw-{index}", type="paper", name=papers.get(p.paper_id, p.paper_id),
                      l0=text, l1=text, paper_id=p.paper_id, passage_ids=[p.id]) for p in store.all()
                 for index, text in enumerate(tokens.iter_chunks(p.text, settings.limits.passage_tokens))]
    toolbox = Toolbox if mode == "full" else RetrievalOnlyToolbox
    tb = toolbox(notes, store, settings, embed=embed if mode in {"full", "hybrid", "hybrid-passages"} else None, run_dir=run_dir)
    if mode in {"vector", "vector-passages"}:
        index = VectorIndex(notes, embed)
        def search(query, k, note_type=None):
            return [tb.notes[nid] for nid in index.search(query, len(notes))
                    if note_type is None or tb.notes[nid].type == note_type][:k]
        tb.search = SimpleNamespace(search=search)
    agent = Agent(tb, gateway, settings, run_dir)
    started = perf_counter()
    result = agent.ask(question)
    shown = {nid for entry in agent.trail.read() if entry["kind"] == "step"
             for nid in [*entry["shown"], *entry["stubs"]]}
    validated = {pid for entry in agent.trail.read() if entry["kind"] == "answer_validation"
                 and entry["supported"] for pid in entry["passage_ids"]}
    retrieved = {store.get(pid).paper_id for nid in shown
                 for pid in tb.notes[nid].passage_ids if store.get(pid)}
    retrieved.update(store.get(pid).paper_id for pid in validated if store.get(pid))
    result.update(response_seconds=perf_counter() - started, baseline=mode,
                  cited_passage_ids=sorted(validated), retrieved_paper_ids=sorted(retrieved),
                  gateway_log=gateway.log)
    return result


def main():
    from evaluation.live_services import request_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mode", choices=(*MODES, "all"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", choices=("disabled", "warm"), default="disabled")
    parser.add_argument("--tokenizer", type=Path, help="Use the actual Qwen tokenizer instead of the default estimate")
    parser.add_argument("--repeats", type=int, default=1, help="Actual answer dispatches; shared corpus, not repeated builds")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    settings = load_settings()
    settings.cloud.enabled = False
    if args.tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        tokens.set_counter(lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids))
    def embed(texts):
        return request_json("http://127.0.0.1:11434/api/embed", {
            "model": "all-minilm:l6-v2", "input": texts, "truncate": True})["embeddings"]
    files = [args.data / "passages.jsonl", *sorted((args.data / "bundle").glob("*/*.md"))]
    provenance = {"mode": args.mode, "question": args.question, "model": settings.model.local, "cache": args.cache,
                  "repeats": args.repeats, "repeat_semantics": "Actual answer runs on a shared corpus; warm cache may reuse outputs; no independent builds.",
                  "corpus_sha256": {str(p.relative_to(args.data)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    from evaluation.run import fingerprints
    provenance["source_sha256"] = fingerprints()
    if args.tokenizer:
        provenance["tokenizer_sha256"] = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    results = []
    modes = MODES if args.mode == "all" else (args.mode,)
    for repeat, mode in [(i, mode) for i in range(1, args.repeats + 1) for mode in modes]:
        usage = []
        def backend(prompt, schema, max_tokens):
            response = request_json("http://127.0.0.1:11434/api/chat", {
                "model": settings.model.local.split("/", 1)[1], "messages": [{"role": "user", "content": prompt}],
                "stream": False, "think": False, "format": schema,
                "options": {"num_ctx": settings.model.local_num_ctx, "num_predict": max_tokens, "temperature": 0}})
            usage.append({k: response.get(k) for k in ("prompt_eval_count", "eval_count", "total_duration")})
            if response["prompt_eval_count"] + response["eval_count"] > settings.budget.total:
                raise ValueError("Baseline server usage exceeded the unchanged token policy")
            return response["message"]["content"]
        gateway = Gateway(settings, {"local": backend},
                          args.data / "cache" / "baselines" / mode if args.cache == "warm" else None)
        result = answer(args.question, args.data, mode, gateway, settings, args.out / f"{mode}-{repeat}", embed)
        result["server_usage"] = usage
        result["repeat"] = repeat
        results.append(result)
        (args.out / "generation.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"{mode}: {result['answer']}", flush=True)
    tokens.set_counter(None)


if __name__ == "__main__":
    main()
