"""Run GapTrace and grade the resulting evidence against real literature."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Literal
from urllib.error import URLError

from pydantic import Field

from evaluation.live_services import OllamaServices, Work, arxiv_future_works, future_works
from evaluation.metrics import Evaluation, Id, Question, Record, Score, by_id, score, validate_protocol


class TopicPlan(Record):
    id: Id
    data: Path
    source: Path | None = None
    future_query: Id
    future_provider: Literal["openalex", "arxiv"] = "openalex"
    future_pool_file: Path | None = None
    questions: list[Question]


class Plan(Record):
    mode: Literal["benchmark", "smoke"] = "benchmark"
    split: Literal["development", "held-out"] = "development"
    cutoff: date
    model_training_cutoff: date | None = None
    future_through: date
    first_threshold: Score = 0.6
    recheck_threshold: Score = 0.8
    future_pool: Annotated[int, Field(ge=5, le=200)] = 50
    gap_limit: Annotated[int, Field(ge=1, le=100)] = 10
    system: Id = "gaptrace-local"
    topics: list[TopicPlan] = Field(min_length=1)


def product(*args: str) -> dict | list:
    completed = subprocess.run([sys.executable, "-m", "gaptrace.cli", *args], capture_output=True,
                               encoding="utf-8", errors="replace", timeout=3600)
    if completed.returncode:
        raise ValueError(f"GapTrace {args[0]} failed: {completed.stderr.strip() or completed.stdout.strip()}")
    return json.loads(completed.stdout)


def verify_inventory(inventory: dict, cutoff: date) -> set[str]:
    provenance = inventory["import"]
    if (not provenance or provenance["cutoff"] != cutoff.isoformat()
            or provenance["build_cloud_enabled"] or provenance["build_cloud_calls"] != 0):
        raise ValueError("Corpus must be imported with the current cutoff and cloud disabled")
    papers = {paper["id"]: paper["published"] for paper in inventory["papers"]}
    if len(papers) != len(inventory["papers"]) or not papers:
        raise ValueError("Corpus must contain unique paper IDs")
    for paper_id, published in papers.items():
        if not published or date.fromisoformat(published) > cutoff:
            raise ValueError(f"Undated or post-cutoff corpus paper {paper_id}")
    passages = {passage["id"] for passage in inventory["passages"]}
    if len(passages) != len(inventory["passages"]) or not passages:
        raise ValueError("Corpus must contain unique passage IDs")
    for passage in inventory["passages"]:
        if passage["paper_id"] not in papers or passage["date"] != papers[passage["paper_id"]]:
            raise ValueError(f"Passage {passage['id']} disagrees with its dated source paper")
    return passages


def generate_topic(topic: TopicPlan, plan: Plan, root: Path, repeats: int) -> tuple[dict, list[dict], list[list[dict]]]:
    data = str((root / topic.data).resolve())
    if Path(data).exists() and not Path(data).is_dir():
        raise ValueError(f"Corpus path is not a directory: {data}")
    if not Path(data).is_dir():
        if topic.source is None:
            raise ValueError(f"Topic {topic.id} needs an imported corpus or a source file")
        product("ingest", str((root / topic.source).resolve()), "--cutoff", plan.cutoff.isoformat(),
                "--data", data, "--local-only")
    inventory = product("corpus", "--data", data)
    passage_ids = verify_inventory(inventory, plan.cutoff)
    paper_ids = {paper["id"] for paper in inventory["papers"]}
    for question in by_id(topic.questions, f"question in {topic.id}").values():
        if not question.gold_passage_ids <= passage_ids or not question.required_paper_ids <= paper_ids:
            raise ValueError(f"Gold evidence for {topic.id}/{question.id} is absent from its corpus")

    gaps = product("gaps", "--data", data, "--k", str(plan.gap_limit), "--json")
    for gap in gaps:
        if any(step["passage_id"] not in passage_ids for step in gap["trail"] if step["passage_id"]):
            raise ValueError(f"Gap {gap['id']} points outside its corpus")
    answer_runs = []
    passages = {}
    for _ in range(repeats):
        answers = []
        for question in topic.questions:
            answer = product("ask", question.text, "--data", data, "--json", "--local-only")
            if answer["model"] != inventory["import"]["model"]:
                raise ValueError(f"The answering model for {topic.id} differs from the import model")
            if not set(answer["cited_passage_ids"]) <= passage_ids or not set(answer["retrieved_paper_ids"]) <= paper_ids:
                raise ValueError(f"Question {topic.id}/{question.id} returned evidence outside its corpus")
            for pid in answer["cited_passage_ids"]:
                if pid not in passages:
                    record = product("passage", pid, "--data", data, "--json")
                    if (record["id"] != pid or record["paper_id"] not in paper_ids or not record["date"]
                            or date.fromisoformat(record["date"]) > plan.cutoff):
                        raise ValueError(f"Question {topic.id}/{question.id} cites an invalid passage")
                    passages[pid] = record["text"]
            answers.append({"question_id": question.id, **answer,
                            "passage_texts": {pid: passages[pid] for pid in answer["cited_passage_ids"]}})
        answer_runs.append(answers)
    return inventory, gaps, answer_runs


def grade_gaps(gaps: list[dict], inventory: dict, topic: TopicPlan, plan: Plan, services: OllamaServices,
               pool_out: Path | None = None, root: Path | None = None) -> list[dict]:
    if not gaps:
        return []
    source_ids = {paper["id"] for paper in inventory["papers"]}
    search = arxiv_future_works if topic.future_provider == "arxiv" else future_works
    scope = {"query": topic.future_query, "provider": topic.future_provider,
             "start": (plan.cutoff + timedelta(days=1)).isoformat(), "end": plan.future_through.isoformat(),
             "limit": plan.future_pool}
    if topic.future_pool_file is not None:
        frozen = json.loads(((root or Path.cwd()) / topic.future_pool_file).read_text(encoding="utf-8"))
        if frozen["scope"] != scope:
            raise ValueError(f"Frozen future pool scope differs for {topic.id}")
        works = [Work.model_validate_json(json.dumps(work)) for work in frozen["works"]]
    else:
        works = search(topic.future_query, plan.cutoff + timedelta(days=1), plan.future_through, plan.future_pool)
    if len(works) > plan.future_pool or len({w.id for w in works}) != len(works):
        raise ValueError("Future pool must contain unique papers within the requested limit")
    if any(not plan.cutoff < work.published <= plan.future_through for work in works):
        raise ValueError("Future pool contains a paper outside the grading window")
    if pool_out is not None:
        payload = {"scope": scope, "works": [w.model_dump(mode="json") for w in works]}
        with pool_out.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2) + "\n")
    if any(work.id in source_ids for work in works):
        raise ValueError(f"A future search result duplicates a source paper in {topic.id}")
    if not works:
        raise ValueError(f"No dated future papers with abstracts found for {topic.id}")
    result = []
    for gap, ranked in zip(gaps, services.shortlist(gaps, works), strict=True):
        candidates = []
        for work, similarity in ranked:
            first = services.match(plan.cutoff, gap["statement"], work)
            second = (services.match(plan.cutoff, gap["statement"], work, strict=True)
                      if first.match and first.confidence >= plan.first_threshold else None)
            candidates.append({**work.model_dump(mode="json"), "similarity": similarity,
                               "first": first.model_dump(), "recheck": second.model_dump() if second else None})
        result.append({"id": gap["id"], "statement": gap["statement"], "candidates": candidates,
                       "type": gap.get("type"), "predicted_status": gap.get("status"),
                       "predicted_doability": gap.get("doability")})
    return result


def grade_answers(answers: list[dict], topic: TopicPlan, services: OllamaServices) -> list[dict]:
    questions = by_id(topic.questions, f"question in {topic.id}")
    graded = []
    for answer in answers:
        question = questions[answer["question_id"]]
        correct, claims = services.grade_answer(question.text, question.gold_answer, answer["answer"],
                                                answer["passage_texts"])
        graded.append({key: answer[key] for key in ("question_id", "answer", "cited_passage_ids",
                                                   "retrieved_paper_ids", "tool_calls", "response_seconds")}
                      | {"correct": correct, "claims": [claim.model_dump(mode="json") for claim in claims],
                         "grader": f"ollama/{services.judge}"})
    return graded


def run(plan: Plan, root: Path, out: Path, services: OllamaServices, repeats: int) -> dict:
    from evaluation.run import fingerprints
    before = fingerprints()
    header = plan.model_dump(mode="json", include={"mode", "split", "cutoff", "model_training_cutoff",
                                                   "future_through", "first_threshold", "recheck_threshold"})
    validate_protocol(plan.mode, plan.cutoff, plan.model_training_cutoff, plan.future_through,
                      plan.first_threshold, plan.recheck_threshold)
    topics = list(by_id(plan.topics, "topic").values())
    generated = {}
    for topic in topics:
        print(f"Generating from the dated corpus for {topic.id}", file=sys.stderr, flush=True)
        inventory, gaps, answers = generate_topic(topic, plan, root, repeats)
        generated[topic.id] = {"inventory": inventory, "gaps": gaps, "answers": answers}
    out.mkdir(parents=True, exist_ok=False)
    (out / "plan.json").write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (out / "generation.json").write_text(json.dumps(generated, indent=2, ensure_ascii=False), encoding="utf-8")
    if before != fingerprints():
        raise ValueError("Source changed during generation; rerun on a stable snapshot")
    provenance = {"source_sha256": before, "generation_sha256": hashlib.sha256(
        (out / "generation.json").read_bytes()).hexdigest(),
        "repeat_semantics": "Answer requests on a shared corpus and deterministic gaps; product cache may reuse outputs. Not independent builds.",
        "judge": services.judge, "independent_human_validation": False}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    runs = [{"id": f"{plan.system}-{i}", "system": plan.system, "topics": []} for i in range(1, repeats + 1)]
    for topic in topics:
        print(f"Grading {topic.id} against the future window and gold evidence", file=sys.stderr, flush=True)
        source = generated[topic.id]
        inventory = source["inventory"]
        # Use an index rather than an untrusted topic ID as a filename.
        gaps = grade_gaps(source["gaps"], inventory, topic, plan, services,
                          out / f"future-pool-{topics.index(topic) + 1}.json", root)
        for current, answers in zip(runs, source["answers"], strict=True):
            current["topics"].append({"id": topic.id, "corpus": inventory["papers"], "gaps": gaps,
                                      "build_cloud_enabled": inventory["import"]["build_cloud_enabled"],
                                      "build_cloud_calls": inventory["import"]["build_cloud_calls"],
                                      "answers": grade_answers(answers, topic, services)})
    manifest = {**header, "runs": runs, "topics": [{"id": topic.id, "questions": [q.model_dump(mode="json")
                                                                                for q in topic.questions]}
                                                   for topic in topics]}
    raw = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode(encoding="utf-8")
    report = score(Evaluation.model_validate_json(raw))
    report["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    (out / "manifest.json").write_bytes(raw)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="New directory for the generated artifacts")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Qwen3.5 tokenizer JSON")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--judge", default="qwen3.5:4b")
    parser.add_argument("--embed", default="all-minilm:l6-v2")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--split", choices=("development", "held-out"), default="development")
    args = parser.parse_args()
    try:
        plan = Plan.model_validate_json(args.plan.read_bytes())
        if plan.split != args.split:
            raise ValueError(f"Plan split is {plan.split}; use --split {plan.split} to run it")
        repeats = args.repeats if args.repeats is not None else (1 if plan.mode == "smoke" else 3)
        if repeats < 1:
            raise ValueError("Repeats must be positive")
        if args.out.exists():
            raise ValueError(f"Choose a new output directory. {args.out} already exists.")
        services = OllamaServices(args.ollama_url, args.judge, args.embed, args.tokenizer)
        run(plan, args.plan.resolve().parent, args.out, services, repeats)
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired, URLError) as exc:
        parser.exit(2, f"Evaluation run failed: {exc}\n")
    print(f"Wrote {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
