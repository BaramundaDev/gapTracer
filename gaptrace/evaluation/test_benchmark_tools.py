import json
from pathlib import Path

import pytest

from evaluation.baselines import answer
from evaluation.live import Plan, grade_gaps
from evaluation.live_services import Work
from evaluation.metrics import Question
from gaptrace.gateway import Gateway
from gaptrace.sample import write_sample
from gaptrace.settings import load_settings


def test_unanswerable_questions_have_explicit_empty_evidence():
    question = Question(id="q", text="Unanswerable?", gold_answer="Not found in these papers.", answerable=False)
    assert not question.gold_passage_ids and not question.required_paper_ids
    with pytest.raises(ValueError, match="require gold"):
        Question(id="q", text="Answerable?", gold_answer="A")
    with pytest.raises(ValueError, match="fictitious"):
        Question(id="q", text="Unanswerable?", gold_answer="A", answerable=False, gold_passage_ids={"fake"})


@pytest.mark.parametrize("mode", ["lexical", "vector", "hybrid", "raw-passages"])
def test_real_baseline_adapters_dispatch_retrieval_and_keep_grounding(tmp_path, mode):
    data = write_sample(tmp_path / "corpus")
    settings = load_settings()
    embed_calls = []
    def embed(texts):
        embed_calls.append(texts)
        return [[1.0, float("replay" in text.lower())] for text in texts]
    def backend(prompt, schema, limit):
        return json.dumps({"answer": "NOT FOUND"})
    result = answer("replay accuracy", data, mode, Gateway(settings, {"local": backend}), settings,
                    tmp_path / mode, embed)
    assert result["baseline"] == mode and "pap-a" in result["retrieved_paper_ids"]
    assert bool(embed_calls) == (mode in {"vector", "hybrid"})
    assert result["cited"] == []


def test_future_pool_is_frozen_before_judging_and_reused_without_search(tmp_path, monkeypatch):
    from evaluation import live
    plan = Plan.model_validate_json(json.dumps({"mode": "smoke", "cutoff": "2024-01-01", "future_through": "2025-01-01",
        "topics": [{"id": "t", "data": "data", "future_query": "topic", "questions": [{"id": "q", "text": "Q",
            "gold_answer": "A", "gold_passage_ids": ["p"], "required_paper_ids": ["paper"]}]}]}))
    topic = plan.topics[0]
    pool = tmp_path / "pool.json"
    work = Work.model_validate_json('{"id":"future","published":"2024-02-01","title":"Future","abstract":"Relevant text"}')
    calls = []
    monkeypatch.setattr(live, "future_works", lambda *args: calls.append("search") or [work])
    class Services:
        def shortlist(self, gaps, works):
            assert pool.exists()  # Freeze before any model judgment.
            return [[] for _ in gaps]
    inventory = {"papers": [{"id": "paper"}]}
    grade_gaps([{"id": "gap", "statement": "Gap"}], inventory, topic, plan, Services(), pool, tmp_path)
    assert calls == ["search"]
    topic.future_pool_file = Path("pool.json")
    grade_gaps([{"id": "gap", "statement": "Gap"}], inventory, topic, plan, Services(), tmp_path / "copy.json", tmp_path)
    assert calls == ["search"]
    frozen = json.loads(pool.read_text())
    frozen["scope"]["query"] = "different"
    pool.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="scope differs"):
        grade_gaps([{"id": "gap", "statement": "Gap"}], inventory, topic, plan, Services(), root=tmp_path)


def dataset_spec(tmp_path):
    spec = {"plans": {}, "source_pins": {}, "label_reviewers": []}
    for split, pid in [("development", "a"), ("held-out", "b")]:
        paper = {"papers": [{"id": pid, "title": pid, "published": "2024-01-01",
                             "sections": [{"heading": "Abstract", "text": "Real source content supplied by a curator."}]}]}
        source = tmp_path / f"{pid}.json"
        source.write_text(json.dumps(paper))
        plan = {"mode": "benchmark", "split": split, "cutoff": "2024-02-01", "model_training_cutoff": "2024-01-01",
                "future_through": "2025-02-01", "topics": [{"id": pid, "data": "unused", "source": source.name,
                  "future_query": pid, "questions": [{"id": "q", "text": "Q", "gold_answer": "Not found",
                                                        "answerable": False}]}]}
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps(plan))
        spec["plans"][split] = path.name
        spec["source_pins"][pid] = {pid: {"url": f"https://example.org/{pid}/v1", "version": "v1"}}
    return spec


def test_dataset_freeze_keeps_split_and_detects_changed_inputs(tmp_path):
    from evaluation.freeze import freeze, verify
    spec = dataset_spec(tmp_path)
    spec["policy_reviewed_by"] = "test reviewer"
    frozen = tmp_path / "frozen"
    policy = (Path(__file__).parent / "release-policy.json").read_bytes()
    spec["policy"] = "policy.json"
    (tmp_path / "policy.json").write_bytes(policy)
    result = freeze(spec, tmp_path, frozen)
    assert result["policy_reviewed_by"] == "test reviewer"
    verify(frozen)
    target = frozen / "held-out" / "sources" / "b.json"
    target.write_text("changed")
    with pytest.raises(ValueError, match="Frozen input changed"):
        verify(frozen)


def test_dataset_freeze_requires_policy_review_before_opening_plans(tmp_path):
    from evaluation.freeze import freeze
    with pytest.raises(ValueError, match="review"):
        freeze({"policy_reviewed_by": ""}, tmp_path, tmp_path / "out")
