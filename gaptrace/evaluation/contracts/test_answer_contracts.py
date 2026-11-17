"""Adversarial backend responses test enforcement, not model intelligence."""
import pytest

from gaptrace.agent import Agent, NOT_FOUND
from gaptrace.gateway import Gateway
from gaptrace.tools import Toolbox

from .conftest import Script


@pytest.mark.parametrize("cited", [[], ["invented-note"], ["paper2"]])
def test_missing_unknown_or_unseen_citation_abstains(corpus, settings, tmp_path, cited):
    notes, store = corpus
    tb = Toolbox(notes, store, settings)
    tb.search.search = lambda *a, **kw: [next(n for n in notes if n.id == "alpha")]
    backend = Script([{"answer": "Alpha scores 999%.", "cited": cited}])
    result = Agent(tb, Gateway(settings, {"local": backend}), settings, tmp_path).ask("Alpha?")
    assert result["answer"] == NOT_FOUND and result["cited"] == []


def test_wrong_fact_with_real_citation_is_rejected(corpus, settings, tmp_path):
    notes, store = corpus
    backend = Script([{"answer": "Alpha scores 999% accuracy on Atlas.", "cited": ["alpha"]}])
    agent = Agent(Toolbox(notes, store, settings), Gateway(settings, {"local": backend}), settings, tmp_path)
    result = agent.ask("Alpha accuracy?")
    assert result["answer"] == NOT_FOUND, "A real note ID authenticated a contradicted number"


def test_claim_without_source_passage_is_rejected(corpus, settings, tmp_path):
    notes, store = corpus
    note = next(n for n in notes if n.id == "alpha")
    note.passage_ids = []
    note.links = []
    backend = Script([{"answer": "Alpha addresses drift.", "cited": ["alpha"]}])
    agent = Agent(Toolbox(notes, store, settings), Gateway(settings, {"local": backend}), settings, tmp_path)
    assert agent.ask("Alpha drift?")["answer"] == NOT_FOUND


def test_explicit_refusal_removes_citations(corpus, settings, tmp_path):
    notes, store = corpus
    backend = Script([{"answer": "NOT FOUND in this corpus.", "cited": ["alpha"]}])
    agent = Agent(Toolbox(notes, store, settings), Gateway(settings, {"local": backend}), settings, tmp_path)
    result = agent.ask("Alpha?")
    assert result["answer"] == NOT_FOUND and not result["cited"]


def test_loop_exhaustion_is_bounded_and_abstains(corpus, settings, tmp_path):
    notes, store = corpus
    backend = Script([{"tool_request": {"tool": "search_notes", "args": {"query": "Alpha"}}}]
                     * settings.limits.max_steps)
    result = Agent(Toolbox(notes, store, settings), Gateway(settings, {"local": backend}), settings, tmp_path).ask("Alpha?")
    assert result["answer"] == NOT_FOUND
    assert result["steps"] == len(backend.calls) == settings.limits.max_steps


def test_supported_answer_after_fetch_is_preserved(corpus, settings, tmp_path):
    notes, store = corpus
    backend = Script([
        {"tool_request": {"tool": "get_passage", "args": {"passage_id": "p1"}}},
        {"answer": "Alpha scores 81% accuracy on Atlas.", "cited": ["alpha"]},
        {"supported": True, "confidence": 1, "reason": "The complete claim matches the passage."},
    ])
    result = Agent(Toolbox(notes, store, settings), Gateway(settings, {"local": backend}), settings, tmp_path).ask("Alpha accuracy?")
    assert result["answer"] == "Alpha scores 81% accuracy on Atlas."
    assert result["cited"] == ["alpha"]
