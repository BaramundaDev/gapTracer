"""Measure contextual metadata budgets against independently labeled retrieval queries."""
import argparse
import json
from pathlib import Path
from time import perf_counter


def score_queries(index, queries, known_ids, ks=(5, 10, 20)):
    if not queries:
        raise ValueError("Supply independently labeled retrieval queries")
    ids, rows = set(), []
    for query in queries:
        if query["id"] in ids or not query["gold_passage_ids"] or not set(query["gold_passage_ids"]) <= known_ids:
            raise ValueError("Queries need unique ids and real nonempty gold source passages")
        ids.add(query["id"])
        hits = [n.id for n in index.search(query["text"], max(ks))]
        gold = set(query["gold_passage_ids"])
        first = next((rank for rank, pid in enumerate(hits, 1) if pid in gold), None)
        rows.append({"query_id": query["id"], "hits": hits, "mrr": 1 / first if first else 0,
                     **{f"evidence_recall@{k}": len(set(hits[:k]) & gold) / len(gold) for k in ks}})
    keys = ["mrr", *(f"evidence_recall@{k}" for k in ks)]
    return {"query_count": len(rows), "metrics": {key: sum(row[key] for row in rows) / len(rows) for key in keys}, "queries": rows}


def run(data, queries_path, out, budgets=(0, 25, 50, 100), retrieval="hybrid"):
    from gaptrace.knowledge import prepare, passage_index, contextual_embedder, digest
    from gaptrace.gap_review import local_gateway
    from gaptrace.workspace import open_toolbox
    from gaptrace.recovery import atomic_json
    data, out = Path(data), Path(out)
    queries = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    if queries.get("split") != "development":
        raise ValueError("Tune metadata budgets only on development queries")
    out.mkdir(parents=True, exist_ok=False)
    tb = open_toolbox(data)
    rows = []
    original = (data / "knowledge.json").read_bytes() if (data / "knowledge.json").exists() else None
    try:
        for budget in budgets:
            gateway = local_gateway(tb.settings, out / f"cache-{budget}")
            started = perf_counter()
            prepared = prepare(tb, gateway, budget, progress=print)
            atomic_json(out / f"knowledge-{budget}.json", prepared)
            embed = contextual_embedder(tb) if retrieval == "hybrid" else None
            result = score_queries(passage_index(tb, prepared, embed), queries["queries"], {p.id for p in tb.passages.all()})
            rows.append({"metadata_tokens": budget, "retrieval": retrieval, "seconds": perf_counter() - started,
                         "preparation_errors": prepared["errors"], "gateway_log": gateway.log, **result})
    finally:
        if original is not None:
            (data / "knowledge.json").write_bytes(original)
        elif (data / "knowledge.json").exists():
            (data / "knowledge.json").unlink()
    result = {"kind": "development-contextual-retrieval-comparison", "query_sha256": digest(queries), "rows": rows,
              "selection": None, "limitations": ["Budget selection requires independently reviewed development queries, costs and adequate topic coverage.",
                   "Same frozen passage corpus; metadata is regenerated per budget. This is not independent corpus-build replication.",
                   "Hybrid MiniLM may truncate passage tails; contextual prefixes are first and BM25 retains all source text."]}
    atomic_json(out / "report.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--queries", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--retrieval", choices=("bm25", "hybrid"), default="hybrid")
    args = p.parse_args()
    try:
        run(args.data, args.queries, args.out, retrieval=args.retrieval)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, f"Contextual retrieval evaluation error: {exc}\n")
