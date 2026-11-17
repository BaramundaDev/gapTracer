"""Gold graph mutations exercise all four patterns and their negative controls."""
import pytest

from gaptrace.gaps import find_gaps
from gaptrace.schema import Link


def select(notes, settings, kind):
    return [c for c in find_gaps(notes, settings) if c.type == kind]


@pytest.mark.parametrize("kind,ids", [
    ("unaddressed_limitation", {"lim", "paper1"}),
    ("unanswered_open_question", {"q", "paper3"}),
    ("untested_combination", {"alpha", "atlas", "problem"}),
    ("conflicting_results", {"r1", "r2"}),
])
def test_exact_positive_patterns(corpus, settings, kind, ids):
    notes, _ = corpus
    cards = select(notes, settings, kind)
    assert len(cards) == 1
    assert set(cards[0].note_ids) == ids
    assert cards[0].trail and all(t.verified for t in cards[0].trail)


@pytest.mark.parametrize("target,kind", [("lim", "unaddressed_limitation"), ("q", "unanswered_open_question")])
@pytest.mark.parametrize("verified", [True, False])
def test_addressed_gap_negative_control(corpus, settings, target, kind, verified):
    notes, _ = corpus
    notes[1].links.append(Link(target=target, type="addresses", verified=verified))
    cards = select(notes, settings, kind)
    assert len(cards) == (0 if verified else 1)
    if cards:
        assert cards[0].needs_check


@pytest.mark.parametrize("verified", [True, False])
def test_evaluation_link_closes_combination_only_when_verified(corpus, settings, verified):
    notes, _ = corpus
    next(n for n in notes if n.id == "alpha").links.append(
        Link(target="atlas", type="evaluated_on", verified=verified))
    cards = select(notes, settings, "untested_combination")
    assert len(cards) == (0 if verified else 1)
    if cards:
        assert cards[0].needs_check


def test_below_independent_dataset_support_threshold(corpus, settings):
    notes, _ = corpus
    next(n for n in notes if n.id == "gamma").links = []
    assert select(notes, settings, "untested_combination") == []


@pytest.mark.parametrize("mutation", ["same_paper", "equal_values", "within_tolerance", "different_dataset", "different_metric"])
def test_non_conflicting_numbers(corpus, settings, mutation):
    notes, _ = corpus
    result = next(n for n in notes if n.id == "r2")
    if mutation == "same_paper":
        result.paper_id = "paper1"
    elif mutation == "equal_values":
        result.fact.value = 81
    elif mutation == "within_tolerance":
        result.fact.value = 81 - settings.gaps.conflict_tolerance
    elif mutation == "different_dataset":
        result.fact.dataset = "Other"
    else:
        result.fact.metric = "recall"
    assert select(notes, settings, "conflicting_results") == []


def test_unverified_reports_cannot_create_evidence(corpus, settings):
    notes, _ = corpus
    for n in notes:
        for link in n.links:
            if link.type == "reports":
                link.verified = False
    assert select(notes, settings, "conflicting_results") == [], "Unverified numeric reports became a gap"


def test_unverified_statements_cannot_create_gaps(corpus, settings):
    notes, _ = corpus
    for n in notes:
        for link in n.links:
            if link.type == "states":
                link.verified = False
    assert select(notes, settings, "unaddressed_limitation") == []
    assert select(notes, settings, "unanswered_open_question") == []


def test_insertion_order_does_not_change_rank_or_ids(corpus, settings):
    notes, _ = corpus
    before = [c.model_dump() for c in find_gaps(notes, settings)]
    after = [c.model_dump() for c in find_gaps(list(reversed(notes)), settings)]
    assert [c["id"] for c in before] == [c["id"] for c in after]
