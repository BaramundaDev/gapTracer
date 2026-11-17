import json
from pathlib import Path

import pytest

from gaptrace.schema import Note
from gaptrace.search import HybridSearch

DATA = json.loads(Path(__file__).with_name("retrieval.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("query", DATA["queries"], ids=lambda q: q["id"])
def test_gold_retrieval_at_three(query, record_property):
    notes = [Note.model_validate(n) for n in DATA["notes"]]
    search = HybridSearch(notes)
    hits = search.search(query["text"], 3, query["type"])
    ids = [n.id for n in hits]
    gold = set(query["gold"])
    recall = len(set(ids) & gold) / len(gold) if gold else None
    mrr = next((1 / i for i, nid in enumerate(ids, 1) if nid in gold), 0) if gold else None
    record_property("recall_at_3", recall)
    record_property("reciprocal_rank", mrr)
    record_property("retrieved_ids", json.dumps(ids))
    assert len(ids) == len(set(ids))
    assert all(n.type == query["type"] for n in hits) if query["type"] else True
    if gold:
        assert recall == 1, f"Gold {sorted(gold)} missing from {ids}"
        assert mrr == 1, f"First result is irrelevant: {ids}"
    else:
        assert ids == [], "Unanswerable lexical query returned unrelated evidence"
