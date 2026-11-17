import json
from datetime import date

import pytest

from evaluation import live, live_services
from evaluation.live_services import OllamaServices, Work
from evaluation.metrics import Claim, Verdict


def test_live_runner_freezes_product_outputs_before_future_search(tmp_path, monkeypatch):
    plan = live.Plan.model_validate_json(json.dumps({
        "mode": "smoke", "cutoff": "2024-01-01", "future_through": "2025-01-01",
        "topics": [{"id": "topic", "data": "corpus", "source": "input.json", "future_query": "topic query",
                    "questions": [{"id": "question", "text": "Question?", "gold_answer": "Answer.",
                                   "gold_passage_ids": ["p-1"], "required_paper_ids": ["p"]}]}],
    }))
    inventory = {"papers": [{"id": "p", "published": "2023-01-01"}],
                 "passages": [{"id": "p-1", "paper_id": "p", "date": "2023-01-01"}],
                 "import": {"cutoff": "2024-01-01", "build_cloud_enabled": False, "build_cloud_calls": 0,
                            "model": "fixture"}}
    events = []
    actions = {"ingest": {}, "corpus": inventory,
               "gaps": [{"id": "gap", "statement": "Topic gap", "trail": [{"passage_id": "p-1"}]}],
               "ask": {"answer": "Answer.", "cited_passage_ids": ["p-1"], "retrieved_paper_ids": ["p"],
                       "tool_calls": 1, "response_seconds": 2.0, "model": "fixture"},
               "passage": {"id": "p-1", "paper_id": "p", "date": "2023-01-01", "text": "Answer."}}

    def product(command, *args):
        events.append(command)
        return actions[command]

    def search(query, start, end, limit):
        events.append("future")
        assert (query, start, end, limit) == ("topic query", date(2024, 1, 2), date(2025, 1, 1), 50)
        return [Work(id="f", published=start, title="Future work", abstract="Fills missing piece.")]

    class Grader:
        judge = "fixture"

        def shortlist(self, gaps, works):
            return [[(works[0], 0.9)]]

        def match(self, cutoff, statement, paper, *, strict=False):
            events.append("strict" if strict else "first")
            return Verdict(match=True, confidence=0.9, grader=self.judge)

        def grade_answer(self, question, gold, answer, passages):
            assert passages == {"p-1": "Answer."}
            return True, [Claim(text="Answer", supported=True, passage_ids={"p-1"})]

    monkeypatch.setattr(live, "product", product)
    monkeypatch.setattr(live, "future_works", search)
    result = live.run(plan, tmp_path, tmp_path / "results", Grader(), repeats=2)
    assert events == ["ingest", "corpus", "gaps", "ask", "passage", "ask", "future", "first", "strict"]
    assert result["systems"]["gaptrace-local"]["repeat_summary"]["hit_rate"]["mean"] == 1
    assert (tmp_path / "results" / "generation.json").exists()


def test_openalex_reconstructs_abstracts_and_rejects_leaked_date(monkeypatch):
    paper = {"id": "https://openalex.org/W1", "publication_date": "2024-01-02",
             "display_name": "Future paper", "abstract_inverted_index": {"was": [1], "It": [0], "done.": [2]}}
    monkeypatch.setattr(live_services, "request_json", lambda *args, **kwargs: {"results": [paper]})
    found = live_services.future_works("query", date(2024, 1, 2), date(2025, 1, 1), 10)
    assert found[0].id == "oa-W1" and found[0].abstract == "It was done."
    paper["publication_date"] = "2024-01-01"
    with pytest.raises(ValueError, match="outside the future window"):
        live_services.future_works("query", date(2024, 1, 2), date(2025, 1, 1), 10)


def test_support_requires_model_quote_to_be_in_the_cited_passage(monkeypatch):
    services = object.__new__(OllamaServices)
    replies = iter([live_services.AnswerGrade(correct=True, claims=["A claim"]),
                    live_services.SupportGrade(supported=True, quote="Invented quote")])
    monkeypatch.setattr(services, "_judge", lambda *args: next(replies))
    correct, claims = services.grade_answer("Q", "A", "A", {"p-1": "Real quote"})
    assert correct and claims[0].supported is False
