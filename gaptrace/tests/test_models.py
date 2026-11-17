"""Model-facing code, exercised with fake backends."""
import json
from datetime import date

import pytest

from gaptrace import tokens
from gaptrace.agent import NOT_FOUND, Agent
from gaptrace.closure import check_closure, inside_candidates, score_doability
from gaptrace.context import ContextBudgetExceeded
from gaptrace.gaps import find_gaps
from gaptrace.gateway import Gateway, ModelOutputInvalid
from gaptrace.pipeline import build_notes, merge_notes
from gaptrace.schema import ClosureOutput, Note, Passage, VerifierOutput, make_id
from gaptrace.tools import Toolbox

from conftest import FakeBackend, fake_embed


def test_gateway_cache_retry_and_budget(settings, tmp_path):
    replies = iter(["not json at all", '{"supported": true, "confidence": 0.8, "reason": "ok"}'])
    backend = FakeBackend([("Link:", lambda p: next(replies))])
    gw = Gateway(settings, {"local": backend}, tmp_path)
    out, tier = gw.call("local", "Link: a --targets--> b", VerifierOutput)
    assert out.supported and tier == "local" and len(backend.calls) == 2        # one retry
    gw.call("local", "Link: a --targets--> b", VerifierOutput)
    assert len(backend.calls) == 2                                               # served from cache
    with pytest.raises(ContextBudgetExceeded):
        gw.call("local", "word " * 9000, VerifierOutput)
    with pytest.raises(ModelOutputInvalid):
        Gateway(settings, {"local": FakeBackend([], default="nope")}).call("local", "x", VerifierOutput)


def test_gateway_blocks_retry_that_would_exceed_budget(settings):
    settings.budget.total = settings.budget.output + settings.budget.margin + 1
    backend = FakeBackend([], default="invalid json")
    gw = Gateway(settings, {"local": backend})
    with pytest.raises(ContextBudgetExceeded):
        gw.call("local", "", VerifierOutput)
    assert backend.calls == [""]


def test_gateway_cloud_routing_and_fallback(settings):
    ok = '{"verdict": "does_not", "reason": "r"}'
    local, cloud = FakeBackend([], default=ok), FakeBackend([], default=ok)
    gw = Gateway(settings, {"local": local, "cloud_light": cloud})
    assert gw.call("cloud_light", "p1", ClosureOutput)[1] == "local"            # cloud is off by default
    settings.cloud.enabled = True
    settings.limits.max_cloud_calls_per_topic = 2
    assert gw.call("cloud_light", "p2", ClosureOutput)[1] == "cloud_light"
    assert gw.call("cloud_light", "p3", ClosureOutput)[1] == "cloud_light"
    assert gw.call("cloud_light", "p4", ClosureOutput)[1] == "local"            # budget of 2 spent

    def broken(prompt, schema, max_tokens):
        raise RuntimeError("429 rate limited")
    gw2 = Gateway(settings, {"local": local, "cloud_light": broken})
    assert gw2.call("cloud_light", "p5", ClosureOutput)[1] == "local"           # falls back, marked


def test_build_pipeline_extracts_verifies_and_rejects(settings):
    paper = Note(id="pap-z", type="paper", name="Sparse replay", l0="Paper Z", paper_date=date(2024, 5, 1))
    passage = Passage(id="z-1", paper_id="pap-z", section="limitations",
                      text="We propose sparse replay for forgetting in sequential fine-tuning. "
                           "Sparse replay was tested only on small models. It reaches 80.5% accuracy on Split CIFAR-100.")
    extraction = {
        "notes": [
            {"type": "method", "name": "Sparse replay", "l0": "Replays a sparse subset", "l1": ""},
            {"type": "problem", "name": "Forgetting in sequential fine-tuning", "l0": "Forgetting", "l1": ""},
            {"type": "limitation", "name": "Tested only on small models", "l0": "Tested only on small models", "l1": ""},
            {"type": "result", "name": "Sparse replay accuracy", "l0": "80.5% on Split CIFAR-100", "l1": "",
             "fact": {"method": "Sparse replay", "dataset": "Split CIFAR-100", "metric": "accuracy", "value": 80.5, "unit": "%"}},
            {"type": "result", "name": "No numbers", "l0": "better", "l1": ""}],
        "links": [
            {"from": "PAPER", "to": "Sparse replay", "type": "proposes",
             "evidence_sentence": "We propose sparse replay for forgetting in sequential fine-tuning."},
            {"from": "Sparse replay", "to": "Forgetting in sequential fine-tuning", "type": "targets",
             "evidence_sentence": "We propose sparse replay for forgetting in sequential fine-tuning."},
            {"from": "Sparse replay", "to": "Forgetting in sequential fine-tuning", "type": "targets",
             "evidence_sentence": "This sentence is not in the passage at all."},
            {"from": "PAPER", "to": "Forgetting in sequential fine-tuning", "type": "targets",
             "evidence_sentence": "We propose sparse replay for forgetting in sequential fine-tuning."}]}
    backend = FakeBackend([
        ("You read one passage", extraction),
        ("--states-->", {"supported": False, "confidence": 0.1, "reason": "test rejects this one"}),
        ("Does the sentence support", {"supported": True, "confidence": 0.9, "reason": "ok"})])
    notes, stats = build_notes([paper], [passage], Gateway(settings, {"local": backend}), settings)
    by_type = {}
    for n in notes:
        by_type.setdefault(n.type, []).append(n)
    assert len(by_type["result"]) == 1                      # the result without numbers was dropped
    assert stats["links_dropped"] == 2                      # invented quote, and a link the schema forbids
    paper_out = by_type["paper"][0]
    kinds = sorted(l.type for l in paper_out.links)
    assert kinds == ["proposes", "reports", "states"]       # reports/states were added by code
    states = next(l for l in paper_out.links if l.type == "states")
    assert not states.verified and "small models" in states.evidence_sentence
    assert stats["links"]["rejected"] == 1 and stats["links"]["verified"] == 3
    assert stats["links"]["rejected_examples"]
    # an unverified 'states' link is not evidence, so this limitation is not reported as a gap
    assert not any("small models" in c.statement for c in find_gaps(notes, settings))


def test_build_reads_the_end_of_an_oversized_passage(settings):
    paper = Note(id="pap-long", type="paper", name="Long paper", l0="Long paper")
    quote = "The study only evaluated English."
    text = "This is background about the study. " * 65 + quote
    passage = Passage(id="long-1", paper_id=paper.id, section="limitations", text=text)
    extraction = {"notes": [{"type": "limitation", "name": "English only", "l0": quote}],
                  "links": [{"from": "PAPER", "to": "English only", "type": "states",
                             "evidence_sentence": quote}]}
    backend = FakeBackend([("Does the sentence support", {"supported": True, "confidence": 0.9}),
                           (quote, extraction)], default='{"notes": [], "links": []}')
    notes, stats = build_notes([paper], [passage], Gateway(settings, {"local": backend}), settings)
    prompts = [p for p in backend.calls if p.startswith("You read one passage")]
    assert len(prompts) > 1
    assert all(tokens.count(p.split("):\n", 1)[-1]) <= settings.limits.passage_tokens for p in prompts)
    assert len(notes) == 2 and notes[1].passage_ids == [passage.id]
    assert paper.links[0].evidence_passage_id == passage.id and paper.links[0].verified
    assert paper.links[0].evidence_sentence == quote and stats["passages"] == 1


def test_merge_joins_duplicates_and_retargets_links(corpus):
    notes, _ = corpus
    dup = Note(id=make_id("method", "Experience replay buffer"), type="method", name="Experience replay buffer",
               l0="Stores past examples and replays them", passage_ids=["b-1"])
    pointer = Note(id="pap-q", type="paper", name="Q", l0="q", links=[
        {"target": dup.id, "type": "proposes", "evidence_passage_id": "b-1", "evidence_sentence": "s", "verified": True}])
    merged, remap = merge_notes(notes + [dup, pointer], fake_embed, threshold=0.6, confirm=lambda a, b: True)
    m1 = make_id("method", "Replay buffer")
    assert remap.get(dup.id) == m1
    kept = next(n for n in merged if n.id == m1)
    assert "Experience replay buffer" in kept.aliases and "b-1" in kept.passage_ids
    assert next(n for n in merged if n.id == "pap-q").links[0].target == m1
    merged2, remap2 = merge_notes(notes + [dup], fake_embed, threshold=0.6, confirm=lambda a, b: False)
    assert dup.id not in remap2                              # the confirm step can veto a merge


def test_closure_and_doability(corpus, passages, settings):
    notes, _ = corpus
    tb = Toolbox(notes, passages, settings, embed=fake_embed)
    card = next(c for c in tb.gaps() if c.type == "unaddressed_limitation")
    outside = lambda statement: [{"id": "ext-1", "title": "Replay beyond vision", "year": 2025,
                                  "abstract": "We test continual learning methods on language tasks."}]
    judge = FakeBackend([("Replay beyond vision", {"verdict": "fills", "reason": "tests language tasks"}),
                         ("Score this gap", {"data": 2, "compute": 2, "blocks": 1, "scope": 2, "reasons": {}})],
                        default='{"verdict": "does_not", "reason": "different"}')
    gw = Gateway(settings, {"local": judge})
    check_closure(card, gw, [inside_candidates(tb), outside])
    assert card.status == "closed" and card.decided_by == "ext-1" and card.judged_by == "local"
    other = next(c for c in tb.gaps() if c.type == "unanswered_open_question")
    check_closure(other, gw, [inside_candidates(tb)])
    assert other.status == "open"
    score_doability(other, gw, [tb.notes[i] for i in other.note_ids])
    assert other.doability == 7


def test_agent_loop_uses_tools_writes_trail_and_refuses(corpus, passages, settings, tmp_path):
    notes, _ = corpus
    m1 = make_id("method", "Replay buffer")
    steps = iter([
        {"answer": "", "cited": [], "tool_request": {"tool": "lookup_fact", "args": {"method": "replay"}},
         "state_update": "Need the reported accuracy for replay."},
        {"answer": "Replay buffer is reported at 71.2% and 65.0% on Split CIFAR-100.", "cited": [m1, "made-up-id"],
         "tool_request": None, "state_update": "Two papers disagree on replay accuracy."}])
    backend = FakeBackend([("STEP QUESTION", lambda p: json.dumps(next(steps))),
                           ("Check whether the complete claim", {"supported": True, "confidence": 1})])
    tb = Toolbox(notes, passages, settings, embed=fake_embed, run_dir=tmp_path)
    agent = Agent(tb, Gateway(settings, {"local": backend}), settings, tmp_path)
    result = agent.ask("What accuracy does the replay buffer reach?")
    assert result["steps"] == 2 and "71.2" in result["answer"]
    assert result["cited"] == [m1]                                       # the invented id was removed
    assert "TOOL RESULT (lookup_fact)" in backend.calls[1] and "71.2" in backend.calls[1]
    assert "Need the reported accuracy" in backend.calls[1]              # state carried by the file, not a transcript
    kinds = [r["kind"] for r in agent.trail.read()]
    assert kinds == ["step", "tool", "step", "answer_validation", "answer_validation_summary", "final"]
    assert all(r["input_tokens"] <= 5300 for r in agent.trail.read() if r["kind"] == "step")
    assert (tmp_path / "state.md").exists()

    silent = FakeBackend([], default='{"answer": "It is probably 90%.", "cited": [], "tool_request": null, "state_update": ""}')
    agent2 = Agent(tb, Gateway(settings, {"local": silent}), settings, tmp_path / "r2")
    assert agent2.ask("What is the best method?")["answer"] == NOT_FOUND  # an answer with no citation is refused
