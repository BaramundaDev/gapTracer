import copy
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.autoresearch import (Change, SearchPlan, acceptance, apply_change, bounded_process,
                                    freeze_inputs, run, validate_change)
from evaluation.context_experiment import (AnswerGrade, ContextPlan, FinalAnswer, MemoryUpdate,
                                           grade_answer, records_for, run as run_context, run_question)
from evaluation.metrics import Evaluation
from evaluation.okf_audit import audit
from evaluation.export_context import export as export_context
from evaluation.live import Plan
from evaluation.local_model import LocalModel

HERE = Path(__file__).parent


def context_plan():
    return ContextPlan.model_validate_json((HERE / "examples" / "context-smoke.json").read_bytes())


def search_plan(**extra):
    return SearchPlan(engine="context", input_plan="context.json", system="typed-managed",
                      objective="answer_accuracy", rounds=1, repeats=1, **extra)


def change(**extra):
    return Change(kind="context_policy", target="context-policy.md", content="Retain exact facts and IDs.",
                  hypothesis="E1 step 1 shows source IDs should stay in memory.", **extra)


def evaluation(correct=False, unsupported=False):
    data = json.loads((HERE / "fixture.json").read_bytes())
    data["mode"] = "smoke"
    data["runs"] = [data["runs"][0]]
    data["runs"][0]["system"] = "typed-managed"
    data["runs"][0]["topics"][0]["answers"][0]["correct"] = correct
    if unsupported:
        data["runs"][0]["topics"][0]["answers"][0]["claims"][0]["supported"] = False
    return Evaluation.model_validate_json(json.dumps(data))


class FakeModel:
    model = "scripted-test"

    def __init__(self, memory="Source transformer-v1-conclusion: make generation less sequential.", invented_quote=False):
        self.calls = []
        self.memory = memory
        self.invented_quote = invented_quote

    def count(self, text):
        return len(text)

    def clip(self, text, limit):
        return text[:limit]

    def call(self, prompt, schema):
        self.calls.append({"prompt_tokens": 20, "output_tokens": 10, "seconds": 0.01})
        if schema is Change:
            return change()
        if schema is MemoryUpdate:
            return MemoryUpdate(memory=self.memory)
        if schema is FinalAnswer:
            return FinalAnswer(answer="Make generation less sequential.",
                               cited_passage_ids=["transformer-v1-conclusion", "unknown-id"])
        if schema is AnswerGrade:
            if "Not found in these papers." in prompt:
                return AnswerGrade(correct=False, claims=[])
            return AnswerGrade(correct=True, claims=[{
                "text": "The goal is less sequential generation.", "passage_ids": ["transformer-v1-conclusion"],
                "quote": "invented" if self.invented_quote else "Making generation less sequential", "supported": True}])
        raise AssertionError(schema)


def test_guardrail_rejects_hallucination_increase_even_with_accuracy_gain():
    result = acceptance(evaluation(False), evaluation(True, True), "typed-managed", search_plan())
    assert result["keep"] is False
    assert next(check for check in result["checks"] if check["check"] == "objective_improvement")["passed"]
    assert not next(check for check in result["checks"] if check["check"] == "hallucination_rate")["passed"]


def test_keep_requires_strict_improvement_and_unchanged_protocol():
    assert acceptance(evaluation(False), evaluation(True), "typed-managed", search_plan())["keep"] is True
    assert acceptance(evaluation(True), evaluation(True), "typed-managed", search_plan())["keep"] is False
    altered = evaluation(True)
    altered.topics[0].questions[0].gold_answer = "Changed scoring target"
    with pytest.raises(ValueError, match="frozen"):
        acceptance(evaluation(False), altered, "typed-managed", search_plan())


@pytest.mark.parametrize("target, value", [("budget.total", 9000), ("model.local", 1),
                                          ("retrieval.top_k", 0), ("retrieval.top_k", True),
                                          ("limits.max_steps", 100), ("retrieval.top_k", 1.5),
                                          ("../schema.py", 1)])
def test_proposals_cannot_change_frozen_or_out_of_range_settings(target, value):
    proposal = Change(kind="setting", target=target, value=value, hypothesis="E1 step 1 test")
    with pytest.raises(ValueError):
        validate_change(proposal, "live")


def test_one_allowed_change_in_private_copy_preserves_budget(tmp_path):
    import yaml
    settings = {"budget": {"total": 8000}, "retrieval": {"top_k": 20}}
    (tmp_path / "config").mkdir()
    path = tmp_path / "config" / "settings.yaml"
    path.write_text(yaml.safe_dump(settings))
    proposal = Change(kind="setting", target="retrieval.top_k", value=12, hypothesis="E1 step 1 test")
    apply_change(tmp_path, proposal, "live")
    actual = yaml.safe_load(path.read_text())
    assert actual["retrieval"]["top_k"] == 12 and actual["budget"] == settings["budget"]
    with pytest.raises(ValueError, match="does not change"):
        apply_change(tmp_path, proposal, "live")


def test_malformed_unrelated_prompt_and_unsupported_evidence_are_rejected():
    invalid = change().model_copy(update={"target": "../prompts/step_agent.txt"})
    with pytest.raises(ValueError):
        validate_change(invalid, "context")
    invalid = change().model_copy(update={"hypothesis": "This might help."})
    with pytest.raises(ValueError, match="episode and step"):
        validate_change(invalid, "context")
    with pytest.raises(ValidationError, match="mandatory"):
        search_plan(guardrails={})


def test_held_out_plans_are_never_available_to_proposer(tmp_path):
    plan = context_plan().model_dump(mode="json")
    plan["split"] = "held-out"
    path = tmp_path / "held-out.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="held-out"):
        freeze_inputs(search_plan(), path, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()


def test_controller_runs_keeps_logs_and_resumes_without_repeating_trials(tmp_path):
    path = tmp_path / "context.json"
    path.write_text(context_plan().model_dump_json())
    config = tmp_path / "search.json"
    config.write_text(search_plan().model_dump_json())
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("scripted-tokenizer-identity")
    calls = []
    def worker(workspace, trial, *args):
        calls.append(workspace)
        result = evaluation(correct=len(calls) > 1)
        (trial / "output").mkdir(parents=True)
        (trial / "output" / "manifest.json").write_text(result.model_dump_json())
        return result
    out = tmp_path / "run"
    result = run(search_plan(), config, out, tokenizer, "scripted-test", proposer_model=FakeModel(), evaluator=worker)
    assert result["incumbent"] == "round-0001"
    assert len(calls) == 2
    assert (out / "round-0001" / "decision.json").exists()
    assert "source IDs" in (out / "results.tsv").read_text()
    assert json.loads((out / "best.json").read_text())["production_files_modified"] is False
    resumed = run(search_plan(), config, out, tokenizer, "scripted-test", proposer_model=FakeModel(), evaluator=worker)
    assert len(calls) == 2 and resumed["incumbent"] == result["incumbent"]
    (out / "inputs" / "plan.json").write_text("tampered")
    with pytest.raises(ValueError, match="input material"):
        run(search_plan(), config, out, tokenizer, "scripted-test", proposer_model=FakeModel(), evaluator=worker)


def test_controller_records_failed_trial_without_promoting_it(tmp_path):
    path = tmp_path / "context.json"
    path.write_text(context_plan().model_dump_json())
    config = tmp_path / "search.json"
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("test")
    def worker(workspace, trial, *args):
        if workspace.name == "workspace":
            raise TimeoutError("test deadline")
        (trial / "output").mkdir(parents=True)
        result = evaluation(False)
        (trial / "output" / "manifest.json").write_text(result.model_dump_json())
        return result
    result = run(search_plan(), config, tmp_path / "run", tokenizer, "scripted-test",
                 proposer_model=FakeModel(), evaluator=worker)
    assert result["incumbent"] == "baseline"
    assert "TimeoutError" in result["history"][0]["error"]


def test_bounded_process_stops_its_own_child(tmp_path):
    with pytest.raises(TimeoutError, match="exceeded"):
        bounded_process([sys.executable, "-c", "import time;time.sleep(30)"], tmp_path, tmp_path / "worker.log", 0.2)


def test_representations_have_same_source_ids_and_control_structure():
    topic = context_plan().topics[0]
    records = {name: records_for(topic, name) for name in ("raw", "flat", "typed", "plain")}
    for values in records.values():
        assert {pid for record in values for pid in record["passage_ids"]} == {"transformer-v1-conclusion"}
    assert "links" not in records["flat"][0]["text"] and "links" in records["typed"][0]["text"]
    assert "states" in records["typed"][0]["text"] and "states" in records["plain"][0]["text"]


def test_managed_context_really_reloads_the_file_and_filters_citations(tmp_path):
    plan = context_plan()
    result, diagnostics = run_question(plan.topics[0], plan.topics[0].questions[0], "typed", "managed", FakeModel(), tmp_path / "trial", settings=plan)
    assert result["correct"] and result["cited_passage_ids"] == ["transformer-v1-conclusion"]
    assert diagnostics["memory_edits"] == 1 and diagnostics["generation_tokens"] == 60
    trajectory = json.loads((tmp_path / "trial" / "trajectory.json").read_text())
    assert trajectory["turns"][-1]["memory_after"] == (tmp_path / "trial" / "memory.md").read_text()


def test_discarded_source_ids_force_abstention_and_invented_quotes_fail_support(tmp_path):
    plan = context_plan()
    topic, question = plan.topics[0], plan.topics[0].questions[0]
    result, _ = run_question(topic, question, "flat", "managed", FakeModel(memory="Forgot all IDs"), tmp_path / "trial", settings=plan)
    assert result["answer"] == "Not found in these papers." and not result["correct"]
    answer = FinalAnswer(answer="Make generation less sequential.", cited_passage_ids=["transformer-v1-conclusion"])
    correct, claims = grade_answer(FakeModel(invented_quote=True), question, answer,
                                  {"transformer-v1-conclusion": topic.passages[0].text})
    assert not correct and not claims[0]["supported"]


def test_factorial_runner_writes_scorable_manifest_and_separates_usage(tmp_path):
    result = run_context(context_plan(), tmp_path / "experiment", FakeModel(), ("typed", "flat"), ("managed",), 1)
    assert set(result["systems"]) == {"typed-managed", "flat-managed"}
    assert len(result["measurements"]) == 2
    assert result["systems"]["typed-managed"]["repeat_summary"]["answer_accuracy"]["mean"] == 1
    assert result["measurements"][0]["generation_calls"] == 2


def test_context_plan_rejects_missing_or_future_sources():
    data = context_plan().model_dump(mode="json")
    data["topics"][0]["passages"][0]["published"] = "2026-01-01"
    with pytest.raises(ValidationError, match="date or paper"):
        ContextPlan.model_validate_json(json.dumps(data))
    data = context_plan().model_dump(mode="json")
    data["topics"][0]["notes"][0]["passage_ids"] = ["unknown"]
    with pytest.raises(ValidationError, match="unknown passage"):
        ContextPlan.model_validate_json(json.dumps(data))


def test_okf_audit_checks_core_envelope_and_extension_resolvability(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "note.md").write_text("---\ntype: limitation\nid: gap\npassage_ids: [p]\nlinks: [{target: missing, verified: true}]\n---\nSource-backed limitation\n")
    sources = tmp_path / "passages.jsonl"
    sources.write_text(json.dumps({"id": "p", "text": "source"}) + "\n")
    result = audit(bundle, sources)
    assert result["core_envelope_passed"] and not result["full_conformance_claimed"]
    assert result["unresolved_source_pointers"] == [] and result["dangling_note_targets"] == ["missing"]
    (bundle / "bad.md").write_text("No frontmatter")
    assert not audit(bundle)["core_envelope_passed"]


def test_export_reads_windows_bundle_and_preserves_source_hashes(tmp_path):
    import yaml
    dataset = context_plan()
    topic = dataset.topics[0]
    corpus = tmp_path / "corpus"
    (corpus / "bundle" / "paper").mkdir(parents=True)
    front = {"id": "transformer-v1", "type": "paper", "name": "Attention Is All You Need",
             "description": "Making generation less sequential is a research goal.", "paper_date": "2017-06-12",
             "passage_ids": ["transformer-v1-conclusion"], "links": []}
    content = "---\n" + yaml.safe_dump(front) + "---\n\n# Attention Is All You Need\n\nGoal: less sequential generation.\n"
    (corpus / "bundle" / "paper" / "transformer.md").write_bytes(content.replace("\n", "\r\n").encode())
    (corpus / "import.json").write_text(json.dumps({"cutoff": "2017-07-01", "build_cloud_enabled": False, "build_cloud_calls": 0}))
    (corpus / "passages.jsonl").write_text(json.dumps({"id": "transformer-v1-conclusion", "paper_id": "transformer-v1",
                                                       "date": "2017-06-12", "text": topic.passages[0].text}) + "\n")
    plan = Plan.model_validate_json(json.dumps({"mode": "smoke", "cutoff": "2017-07-01", "future_through": "2018-07-01",
        "topics": [{"id": topic.id, "data": "corpus", "future_query": "unused", "questions": [q.model_dump(mode="json") for q in topic.questions]}]}))
    exported, fingerprints = export_context(plan, tmp_path)
    assert exported.topics[0].passages[0].paper_title == "Attention Is All You Need"
    assert exported.topics[0].notes[0].summary == "Goal: less sequential generation."
    assert len(fingerprints) == 2


def test_unanswerable_abstention_has_no_invented_claims():
    from evaluation.metrics import Question
    question = Question(id="absent", text="An absent question?", gold_answer="Not found in these papers.", answerable=False)
    correct, claims = grade_answer(FakeModel(), question,
        FinalAnswer(answer="Not found in these papers.", cited_passage_ids=[]), {})
    assert correct and claims == []


def test_research_adapter_accounts_for_schema_before_calling_server():
    from gaptrace.settings import load_settings
    model = object.__new__(LocalModel)
    model.settings = load_settings()
    model.count = len
    with pytest.raises(ValueError, match="schema"):
        model._backend("a" * 5300, {"schema": "large"}, 1200)


def test_research_adapter_rejects_reported_token_overflow(monkeypatch):
    from gaptrace.settings import load_settings
    import io
    model = object.__new__(LocalModel)
    model.settings = load_settings()
    model.model, model.base, model.timeout, model.deadline = "scripted", "http://127.0.0.1:11434", 1, None
    model.count = len
    model.calls = []
    payload = {"model": "scripted", "prompt_eval_count": 7500, "eval_count": 1000,
               "done_reason": "stop", "message": {"content": "{}"}}
    monkeypatch.setattr("evaluation.local_model.urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode()))
    with pytest.raises(ValueError, match="8,000"):
        model._backend("small", {}, 1200)
