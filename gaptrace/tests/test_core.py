from datetime import date

import pytest

from gaptrace import tokens
from gaptrace.bundle import read_bundle, read_note, write_note
from gaptrace.context import ContextBudgetExceeded, State, assert_within_budget, pack
from gaptrace.facts import FactsTable
from gaptrace.gaps import find_gaps
from gaptrace.schema import Note, link_allowed, make_id
from gaptrace.search import HybridSearch, rrf
from gaptrace.tools import Toolbox

from conftest import fake_embed


def test_settings_budget_sums(settings):
    b = settings.budget
    assert b.instructions + b.state + b.notes + b.passage + b.output + b.margin == b.total == 8000


def test_schema_rules():
    assert link_allowed("targets", "method", "problem")
    assert not link_allowed("targets", "paper", "problem")
    assert make_id("method", "Replay Buffer") == make_id("method", "replay buffer ")
    with pytest.raises(Exception):
        Note(id="x", type="idea", name="bad", l0="bad")           # type outside the schema


def test_bundle_roundtrip(tmp_path, corpus):
    notes, _ = corpus
    for n in notes:
        assert read_note(write_note(tmp_path, n)) == n
    assert len(read_bundle(tmp_path)) == len(notes)
    text = (tmp_path / "paper" / "pap-a.md").read_text()
    assert text.startswith("---\n") and "description:" in text       # OKF-style front matter


def test_truncate_respects_limit():
    long = "word " * 5000
    assert tokens.count(tokens.truncate(long, 120)) <= 120


def test_token_chunks_preserve_text_and_fit_the_limit():
    text = "Lead sentence. Next sentence.\n\n" + "x" * 134 + " Last sentence."
    chunks = list(tokens.iter_chunks(text, 12))
    assert chunks[0].rstrip().endswith(".")
    assert "".join(chunks) == text
    assert all(0 < tokens.count(chunk) <= 12 for chunk in chunks)


def test_rrf_prefers_items_ranked_by_both():
    merged = [i for i, _ in rrf([["a", "b", "c"], ["c", "a", "d"]])]
    assert merged[0] == "a" and set(merged) == {"a", "b", "c", "d"}


def test_hybrid_search(corpus):
    notes, _ = corpus
    hs = HybridSearch(notes, fake_embed)
    assert "Replay buffer" in [h.name for h in hs.search("replay buffer", 5)]
    assert hs.search("replay buffer", 5, "method")[0].name == "Replay buffer"
    assert all(n.type == "dataset" for n in hs.search("CIFAR MNIST", 5, "dataset"))


def test_facts_exact_lookup(corpus):
    notes, _ = corpus
    t = FactsTable()
    assert t.rebuild(notes) == 2
    rows = t.lookup(method="replay", dataset="cifar", metric="accuracy")
    assert sorted(r["value"] for r in rows) == [65.0, 71.2]
    assert t.lookup(method="nothing") == []


def test_four_gap_patterns(corpus, settings):
    notes, _ = corpus
    cards = find_gaps(notes, settings)
    kinds = {c.type for c in cards}
    assert kinds == {"unaddressed_limitation", "unanswered_open_question", "untested_combination",
                     "conflicting_results"}
    lim = [c for c in cards if c.type == "unaddressed_limitation"]
    assert len(lim) == 1 and "vision" in lim[0].statement and lim[0].support_count == 2
    # the limitation that paper C addresses is not reported as a gap
    assert not any("task boundaries" in c.statement for c in cards)
    combo = [c for c in cards if c.type == "untested_combination"]
    statements = " | ".join(c.statement for c in combo)
    assert len(combo) == 2 and "Adapter isolation has not been evaluated on Split CIFAR-100" in statements
    assert "Replay buffer has not been evaluated on Permuted MNIST" in statements
    assert all(c.trail and c.verified_share == 1.0 for c in cards)
    assert cards[0].support_count >= cards[-1].support_count          # ranked by support


def test_unverified_links_are_not_evidence(corpus, settings):
    notes, _ = corpus
    for n in notes:
        for l in n.links:
            if l.type == "addresses":
                l.verified = False                                     # paper C's claim now unverified
    cards = find_gaps(notes, settings)
    flagged = [c for c in cards if "task boundaries" in c.statement]
    assert len(flagged) == 1 and flagged[0].needs_check               # shown, but flagged for a check


def test_pack_stays_in_budget_with_many_notes(corpus, settings):
    notes, _ = corpus
    many = [Note(id=f"met-x-{i}", type="method", name=f"Method {i}", l0="A short headline about a method " * 2,
                 l1="A longer summary sentence about what this method does and why. " * 6) for i in range(400)]
    packed = pack("Answer the question.", "What reduces forgetting?", "GOAL: test", many, "passage " * 3000, settings)
    b = settings.budget
    assert packed.slot_tokens["notes"] <= b.notes + 10
    assert tokens.count(packed.prompt) <= b.total - b.output - b.margin
    assert packed.full_ids and packed.stub_ids and packed.omitted > 0
    assert len(packed.full_ids) + len(packed.stub_ids) + packed.omitted == 400
    assert "[...]" in packed.prompt                                    # the passage was cut to its slot


def test_pack_without_stubs_switch(settings):
    settings.switches.pointer_stubs = False
    many = [Note(id=f"met-x-{i}", type="method", name=f"M{i}", l0="headline", l1="summary " * 60) for i in range(200)]
    packed = pack("Do it.", "Q?", "", many, None, settings)
    assert packed.stub_ids == [] and packed.omitted > 0


def test_budget_gate_refuses_oversized_prompt(settings):
    with pytest.raises(ContextBudgetExceeded):
        assert_within_budget("word " * 9000, settings)
    with pytest.raises(ContextBudgetExceeded):
        pack("instruction " * 2000, "q", "", [], None, settings)


def test_state_drops_oldest_findings_to_ids(tmp_path, settings):
    s = State(goal="find what reduces forgetting")
    for i in range(200):
        s.add_finding(f"finding number {i} with some explanatory words attached to it", [f"note-{i}"])
    text = s.render(settings.budget.state)
    assert tokens.count(text) <= settings.budget.state
    assert "finding number 199" in text and "finding number 0 " not in text
    assert "EARLIER" in text and "note-" in text                        # pointers survive
    s.save(tmp_path, settings.budget.state)
    assert State.load(tmp_path).findings == s.findings                  # nothing is lost on disk


def test_tools_return_capped_digests_and_write_full_results(corpus, passages, settings, tmp_path):
    notes, _ = corpus
    tb = Toolbox(notes, passages, settings, embed=fake_embed, run_dir=tmp_path)
    caps = settings.tool_caps
    out = tb.search_notes("forgetting replay consolidation adapter")
    assert tokens.count(out) <= caps.search_notes and "[met-" in out
    assert tokens.count(tb.get_passage("a-1")) <= caps.get_passage
    assert "71.2" in tb.lookup_fact("replay", "cifar", "accuracy")
    m1 = make_id("method", "Replay buffer")
    assert "targets" in tb.neighbours(m1) and "evaluated_on" in tb.get_note(m1)
    gaps_text = tb.find_gaps()
    assert "untested_combination" in gaps_text
    gid = tb.gaps()[0].id
    assert "verified" in tb.get_trail(gid)
    assert tb.call("delete_everything", {}).startswith("Unknown tool")
    assert len(list((tmp_path / "tool_results").glob("*.json"))) >= 6
