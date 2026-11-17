import pytest

from gaptrace.closure import check_closure
from gaptrace.gateway import Gateway
from gaptrace.schema import GapCard

from .conftest import Script


def card():
    return GapCard(id="gap", type="unaddressed_limitation", statement="Speech remains untested", note_ids=["origin"])


def paper(pid):
    return {"id": pid, "title": pid, "abstract": "A controlled abstract."}


@pytest.mark.parametrize("replies,status", [
    ([{"verdict": "does_not"}], "open"),
    ([{"verdict": "partly"}], "partly"),
    ([{"verdict": "fills"}], "closed"),
])
def test_successful_closure_verdicts(settings, replies, status):
    backend = Script(replies)
    result = check_closure(card(), Gateway(settings, {"local": backend}), [lambda _: [paper("candidate")]])
    assert result.status == status
    assert result.judged_by == "local"


def test_no_candidates_remains_unchecked(settings):
    result = check_closure(card(), Gateway(settings, {"local": Script([])}), [lambda _: []])
    assert result.status == "unchecked", "No search evidence was promoted to an open verdict"


def test_invalid_judge_remains_unchecked(settings):
    result = check_closure(card(), Gateway(settings, {"local": Script(["invalid", "invalid"])}),
                           [lambda _: [paper("candidate")]])
    assert result.status == "unchecked", "Failed judgment was promoted to an open verdict"


def test_exclusions_deduplication_and_candidate_cap(settings):
    backend = Script([{"verdict": "does_not"}] * 2)
    result = check_closure(card(), Gateway(settings, {"local": backend}),
                           [lambda _: [paper("origin"), paper("a"), paper("a"), paper("b"), paper("c")]],
                           max_papers=2)
    assert len(backend.calls) == 2 and result.status == "open"


def test_closed_verdict_stops_further_judging(settings):
    backend = Script([{"verdict": "fills"}])
    result = check_closure(card(), Gateway(settings, {"local": backend}),
                           [lambda _: [paper("a"), paper("b")]])
    assert result.status == "closed" and result.decided_by == "a"
    assert len(backend.calls) == 1
