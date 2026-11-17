"""End-to-end controls for search-backed gap assessment, ranking and provenance."""
import json
from datetime import date

import pytest
from typer.testing import CliRunner

from gaptrace.agent import Agent
from gaptrace.bundle import PassageStore
from gaptrace.cli import app
from gaptrace.gap_review import GapReviewer, load, render, save
from gaptrace.gateway import Gateway
from gaptrace.literature import search
from gaptrace.schema import GapCard
from gaptrace.tools import Toolbox


def work(pid="outside"):
    return {"id": pid, "title": "Replay experiments on language models", "abstract": "We evaluate replay on language models and publish usable code.",
            "url": "https://example.org/" + pid, "published": "2024-02-01", "evidence_level": "abstract"}


def backend(verdict="does_not", bad_quote=False, fail_recheck=False, feasibility=False):
    def run(prompt, schema, limit):
        fields = schema["properties"]
        if "supported" in fields:
            return json.dumps({"supported": True, "reason": "Relation supported by source"})
        if "verdict" in fields:
            strict = "STRICT RECHECK" in prompt
            return json.dumps({"verdict": "does_not" if fail_recheck and strict else verdict,
                "quote": "Invented quote" if bad_quote else "We evaluate replay on language models", "reason": "Test-only decision"})
        return json.dumps({axis: {"score": 2 if feasibility else None, "source_id": "outside", "quote": "invented feasibility evidence"}
                           for axis in ("data", "compute", "blocks", "scope")})
    return run


@pytest.fixture
def tb(corpus, settings, tmp_path):
    settings.switches.hybrid_search = False
    notes, passages = corpus
    store = PassageStore(tmp_path / "passages.jsonl")
    store.add(passages)
    return Toolbox(notes, store, settings)


def review(tb, **kwargs):
    return GapReviewer(tb, Gateway(tb.settings, {"local": backend(**kwargs)}),
                       search=lambda *args: [work()])


def open_question(tb):
    return next(c for c in tb.gaps() if c.type == "unanswered_open_question")


def test_confirmed_closure_demotes_gap_and_retains_quotes(tb, tmp_path):
    before = tb.gaps()
    report = review(tb, verdict="fills").run(before, gap_id=open_question(tb).id)
    result = next(r for r in report["reviews"] if r["gap_id"] == open_question(tb).id)
    assert result["card"]["status"] == "closed"
    assert result["card"]["decided_by"] == "outside"
    assert report["ranked"][-1]["id"] == open_question(tb).id
    candidate = result["candidates"][0]
    assert candidate["first"]["quote"] in candidate["text"] and candidate["recheck"]["verdict"] == "fills"
    assert open_question(tb).status == "unchecked"  # Original predictions remain frozen.
    path = tmp_path / "review.json"
    save(report, path)
    assert load(path, tb)["ranked"] == report["ranked"]
    tb.notes["pap-a"].l0 = "Changed content"
    with pytest.raises(ValueError, match="does not match"):
        load(path, tb)


@pytest.mark.parametrize("kwargs", [{"verdict": "fills", "bad_quote": True}, {"verdict": "fills", "fail_recheck": True}])
def test_unsupported_positive_or_disagreement_never_closes_gap(tb, kwargs):
    record = review(tb, **kwargs).review(open_question(tb))
    assert record["card"]["status"] == "unchecked" and record["errors"]


def test_search_failure_preserves_uncertainty_even_with_negative_judgments(tb):
    calls = []
    def searcher(*args):
        calls.append(args)
        if len(calls) == 1:
            return [work()]
        raise OSError("Provider unavailable")
    reviewer = GapReviewer(tb, Gateway(tb.settings, {"local": backend()}), search=searcher)
    # Two formulations guaranteed for the numeric-conflict control.
    card = next(c for c in tb.gaps() if c.type == "conflicting_results")
    record = reviewer.review(card)
    assert record["card"]["status"] == "unchecked"
    assert any("Provider unavailable" in error for error in record["errors"])
    assert len(calls) == 2


def test_no_candidates_and_no_local_evidence_remains_unchecked(tb):
    tb.search.search = lambda *args: []
    reviewer = GapReviewer(tb, Gateway(tb.settings, {"local": backend()}), search=lambda *args: [])
    assert reviewer.review(open_question(tb))["card"]["status"] == "unchecked"


def test_successful_negative_search_is_only_bounded_open(tb):
    record = review(tb).review(open_question(tb))
    assert record["card"]["status"] == "open"
    assert "bounded" in record["scope"]


def test_feasibility_without_actual_quotes_remains_unknown(tb):
    record = review(tb, feasibility=True).review(open_question(tb))
    assert record["card"]["doability"] is None
    assert all(axis["score"] is None for axis in record["feasibility"].values())


def test_search_response_bounds_dates_and_abstracts():
    seen = []
    def read(url, headers):
        seen.append(url)
        return json.dumps({"results": [{"id": "https://openalex.org/W123", "publication_date": "2024-01-01",
            "display_name": "Paper", "abstract_inverted_index": {"replay": [1], "We": [0]}}]}).encode()
    papers = search("replay", as_of="2024-02-01", read=read)
    assert papers[0]["abstract"] == "We replay" and papers[0]["id"] == "oa-W123"
    assert "to_publication_date" in seen[0]
    with pytest.raises(ValueError, match="out-of-scope"):
        search("replay", as_of="2023-01-01", read=read)
    with pytest.raises(ValueError, match="between"):
        search("replay", limit=100, read=read)
    with pytest.raises(ValueError, match="future"):
        search("replay", as_of="2999-01-01", read=read)


def test_assistant_has_recorded_review_action_and_does_not_invent_answer(tb, tmp_path):
    tb.reviewer = review(tb, verdict="fills")
    gateway = Gateway(tb.settings, {"local": lambda *args: json.dumps({"tool_request": {"tool": "review_gaps", "args": {"gap_id": open_question(tb).id}}})})
    result = Agent(tb, gateway, tb.settings, tmp_path).ask("Please validate the open gap using current literature")
    assert "automated assessments" in result["answer"]
    assert "outside" in result["answer"]
    assert result["gap_review"] == str(tmp_path / "gap-review.json")
    assert json.loads((tmp_path / "gap-review.json").read_text())["reviews"]


def test_explicit_review_runs_even_when_planning_model_would_abstain(tb, tmp_path):
    tb.reviewer = review(tb, verdict="fills")
    gateway = Gateway(tb.settings, {"local": lambda *args: '{"answer":"NOT FOUND"}'})
    result = Agent(tb, gateway, tb.settings, tmp_path).ask("Validate and rank the research gaps using current literature")
    assert "automated assessments" in result["answer"] and "gap_review" in result
    assert not gateway.log  # Dispatch does not rely on a model volunteering a tool call.


def test_external_search_results_are_separate_and_fetchable(tb):
    original_count = len(tb.notes)
    tb.reviewer = review(tb)
    digest = tb.search_literature("replay")
    assert "[outside]" in digest and len(tb.notes) == original_count
    assert "We evaluate replay" in tb.get_passage("outside-abstract")
    assert "Replay experiments" in tb.get_note("outside")


def test_cli_review_and_saved_report(tb, data_dir, monkeypatch):
    import gaptrace.gap_review as module
    def noisy_backend(*args):
        print("provider diagnostic")
        return backend()(*args)
    monkeypatch.setattr(module, "local_gateway", lambda *args: Gateway(tb.settings, {"local": noisy_backend}))
    monkeypatch.setattr(module.literature, "search", lambda *args: [work()])
    runner = CliRunner()
    result = runner.invoke(app, ["review-gaps", "--data", str(data_dir), "--k", "1", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert len(report["reviews"]) == 1 and report["searches"]
    result = runner.invoke(app, ["review-gaps", "--data", str(data_dir), "--saved"])
    assert result.exit_code == 0 and "automated assessments" in result.stdout


def test_assistant_external_claims_still_require_actual_abstract_validation(tb, tmp_path):
    tb.reviewer = review(tb)
    count = []
    def answer_backend(prompt, schema, limit):
        if "answer" in schema["properties"]:
            count.append(1)
            if len(count) == 1:
                return json.dumps({"tool_request": {"tool": "search_literature", "args": {"query": "replay"}}})
            return json.dumps({"answer": "The paper evaluates replay on language models.", "cited": ["outside"]})
        assert "We evaluate replay on language models" in prompt
        return json.dumps({"supported": True, "reason": "Exact abstract supports the claim"})
    result = Agent(tb, Gateway(tb.settings, {"local": answer_backend}), tb.settings, tmp_path).ask("Search the literature for replay")
    assert result["cited"] == ["outside"] and "evaluates replay" in result["answer"]
    events = [json.loads(line) for line in (tmp_path / "trail.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "answer_validation" and e["supported"] for e in events)


def test_fully_evidenced_feasibility_retains_each_axis_and_total(tb):
    quotes = {"data": "Ready-to-use public data is available.", "compute": "Training needs one consumer GPU.",
              "blocks": "Public usable code is available.", "scope": "The project takes two months."}
    paper = work()
    paper["abstract"] = " ".join(quotes.values())
    def run(prompt, schema, limit):
        if "data" in schema["properties"]:
            return json.dumps({axis: {"score": 2, "source_id": "outside", "quote": quote} for axis, quote in quotes.items()})
        return backend()(prompt, schema, limit)
    reviewer = GapReviewer(tb, Gateway(tb.settings, {"local": run}), search=lambda *args: [paper])
    record = reviewer.review(open_question(tb))
    assert record["card"]["doability"] == 8
    assert all(v["score"] == 2 and v["quote"] in paper["abstract"] for v in record["feasibility"].values())


def test_historical_investigation_rejects_future_corpus_content(tb):
    with pytest.raises(ValueError, match="precedes"):
        GapReviewer(tb, Gateway(tb.settings, {"local": backend()}), as_of="2023-01-01")
