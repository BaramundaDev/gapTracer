import pytest

from gaptrace.gateway import Gateway
from gaptrace.pipeline import extract_passage, merge_notes, verify_links
from gaptrace.schema import Link, Note, Passage

from .conftest import Script


@pytest.mark.parametrize("sentence,expected", [
    ("Alpha targets drift.", 1),
    ("Alpha cures cancer.", 0),
    ("Alpha", 0),
    ("", 0),
])
def test_extraction_rejects_fabricated_and_short_quotes(settings, sentence, expected):
    paper = Note(id="paper", type="paper", name="Study", l0="Study")
    passage = Passage(id="p", paper_id="paper", text="Alpha targets drift.")
    reply = {"notes": [{"type": "method", "name": "Alpha", "l0": "Alpha"}],
             "links": [{"from": "PAPER", "to": "Alpha", "type": "proposes", "evidence_sentence": sentence}]}
    _, stats = extract_passage(Gateway(settings, {"local": Script([reply])}), passage, paper, settings)
    assert stats["links"] == expected
    assert stats["links_dropped"] == 1 - expected


def test_invalid_extractor_is_counted_and_skipped(settings):
    paper = Note(id="paper", type="paper", name="Study", l0="Study")
    notes, stats = extract_passage(Gateway(settings, {"local": Script(["invalid", "invalid"])}),
                                   Passage(id="p", paper_id="paper", text="Study."), paper, settings)
    assert not notes and stats["invalid"] == 1


def test_rejected_verification_does_not_keep_stale_verified_flag(settings):
    a = Note(id="a", type="method", name="Alpha", l0="Alpha",
             links=[Link(target="b", type="targets", verified=True, evidence_sentence="Alpha targets drift.")])
    b = Note(id="b", type="problem", name="Drift", l0="Drift")
    stats = verify_links(Gateway(settings, {"local": Script([{"supported": False}])}), [a, b], settings)
    assert not a.links[0].verified and stats["rejected"] == 1


def test_merge_preserves_distinct_paper_results():
    # Identical names/IDs from extraction must not erase disagreement between papers.
    from gaptrace.schema import Fact
    notes = [Note(id="same-result", type="result", name="Alpha accuracy", l0="Accuracy",
                  paper_id=pid, passage_ids=[pid],
                  fact=Fact(method="Alpha", dataset="Atlas", metric="accuracy", value=value, unit="%"))
             for pid, value in [("paper1", 81), ("paper2", 75)]]
    merged, _ = merge_notes(notes)
    assert {(n.paper_id, n.fact.value) for n in merged} == {("paper1", 81), ("paper2", 75)}, \
        "Cross-paper result collision erased a measured value"
