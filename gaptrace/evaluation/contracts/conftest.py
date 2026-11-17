"""Independent, invented corpus; no network, embedding downloads, or GPU."""
import json

import pytest

from gaptrace import tokens
from gaptrace.bundle import PassageStore
from gaptrace.schema import Fact, Link, Note, Passage
from gaptrace.settings import load_settings


class Script:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def __call__(self, prompt, schema, max_tokens):
        self.calls.append((prompt, schema, max_tokens))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


@pytest.fixture(autouse=True)
def counter():
    tokens.set_counter(None)
    yield
    tokens.set_counter(None)


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def corpus(tmp_path):
    def edge(target, kind, pid, sentence):
        return Link(target=target, type=kind, evidence_passage_id=pid,
                    evidence_sentence=sentence, verified=True, confidence=1)

    texts = {
        "p1": "Method Alpha targets drift. Alpha scores 81% accuracy on Atlas. Vision only remains a limitation.",
        "p2": "Method Beta targets drift and is evaluated on Atlas. Alpha scores 75% accuracy on Atlas.",
        "p3": "Method Gamma targets drift and is evaluated on Atlas. Does Alpha transfer to speech?",
    }
    notes = [
        Note(id="paper1", type="paper", name="Alpha study", l0="Alpha study",
             links=[edge("lim", "states", "p1", "Vision only remains a limitation."),
                    edge("r1", "reports", "p1", "Alpha scores 81% accuracy on Atlas.")]),
        Note(id="paper2", type="paper", name="Beta study", l0="Beta study",
             links=[edge("r2", "reports", "p2", "Alpha scores 75% accuracy on Atlas.")]),
        Note(id="paper3", type="paper", name="Gamma study", l0="Gamma study",
             links=[edge("q", "states", "p3", "Does Alpha transfer to speech?")]),
        Note(id="problem", type="problem", name="Drift", l0="Drift"),
        Note(id="atlas", type="dataset", name="Atlas", l0="Atlas benchmark"),
        Note(id="alpha", type="method", name="Alpha", l0="Alpha addresses drift", passage_ids=["p1"],
             links=[edge("problem", "targets", "p1", "Method Alpha targets drift.")]),
        Note(id="beta", type="method", name="Beta", l0="Beta addresses drift", passage_ids=["p2"],
             links=[edge("problem", "targets", "p2", "Method Beta targets drift and is evaluated on Atlas."),
                    edge("atlas", "evaluated_on", "p2", "Method Beta targets drift and is evaluated on Atlas.")]),
        Note(id="gamma", type="method", name="Gamma", l0="Gamma addresses drift", passage_ids=["p3"],
             links=[edge("problem", "targets", "p3", "Method Gamma targets drift and is evaluated on Atlas."),
                    edge("atlas", "evaluated_on", "p3", "Method Gamma targets drift and is evaluated on Atlas.")]),
        Note(id="lim", type="limitation", name="Vision only", l0="Vision only remains a limitation",
             passage_ids=["p1"], paper_id="paper1"),
        Note(id="q", type="open_question", name="Speech transfer", l0="Does Alpha transfer to speech?",
             passage_ids=["p3"], paper_id="paper3"),
        *[Note(id=f"r{i}", type="result", name=f"Alpha accuracy study {i}", l0=f"Alpha accuracy {v}%",
               paper_id=f"paper{i}", passage_ids=[f"p{i}"],
               fact=Fact(method="Alpha", dataset="Atlas", metric="accuracy", value=v, unit="%"))
          for i, v in [(1, 81), (2, 75)]],
    ]
    store = PassageStore(tmp_path / "passages.jsonl")
    store.add([Passage(id=pid, paper_id=f"paper{i}", text=text)
               for i, (pid, text) in enumerate(texts.items(), 1)])
    return notes, store
