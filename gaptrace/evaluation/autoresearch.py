"""Bounded local-model prompt/setting experiments with immutable candidate copies."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
from pathlib import Path
from time import monotonic
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import Field, model_validator

from evaluation.context_experiment import ContextPlan
from evaluation.local_model import LocalModel
from evaluation.metrics import Evaluation, Id, Record, score
from evaluation.ranking import metric_direction
from evaluation.vendor.context_language_models.evolve import Score as UpstreamScore, diff_se
from evaluation.vendor.context_language_models.note import build_note
from evaluation.vendor.context_language_models.propose import parse_proposal
from evaluation.vendor.context_language_models.tasks import Episode

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = "Preserve exact facts, uncertainty, and passage IDs before discarding context."
SETTING_RANGES = {"retrieval.top_k": (1, 100), "retrieval.rrf_k": (1, 200), "limits.max_steps": (1, 16),
                  "gaps.min_dataset_users": (1, 20), "gaps.conflict_tolerance": (0, 100), "gaps.top_n": (1, 50)}
SWITCHES = {"stateless_steps", "three_note_sizes", "tool_results_to_disk", "pointer_stubs", "write_before_drop",
            "exact_fact_route", "constrained_output", "link_verifier", "hybrid_search"}


class SearchPlan(Record):
    engine: Literal["context", "live"]
    input_plan: Id
    system: Id
    objective: Id = "hit_rate"
    guardrails: dict[Id, Literal["higher", "lower"]] = Field(default_factory=lambda: {
        "hallucination_rate": "lower", "evidence_quality": "higher"})
    rounds: int = Field(default=5, ge=1, le=1000)
    repeats: int = Field(default=3, ge=1, le=20)
    trial_seconds: int = Field(default=900, ge=1, le=86400)
    total_seconds: int = Field(default=7200, ge=1, le=604800)
    minimum_gain: float = Field(default=0, ge=0, allow_inf_nan=False)
    seed: int = 1729
    context_representation: Literal["raw", "flat", "typed", "plain"] = "typed"
    context_management: Literal["static", "managed"] = "managed"

    @model_validator(mode="after")
    def check(self):
        if metric_direction(self.objective) != "higher":
            raise ValueError("Autoresearch objective must be a higher-is-better quality metric")
        if self.guardrails.get("hallucination_rate") != "lower" or self.guardrails.get("evidence_quality") != "higher":
            raise ValueError("Hallucination and evidence guardrails are mandatory")
        if self.engine == "context" and self.system != f"{self.context_representation}-{self.context_management}":
            raise ValueError("Context system must match the selected representation-management arm")
        return self


class Change(Record):
    kind: Literal["context_policy", "prompt", "setting"]
    target: Id
    content: str | None = Field(default=None, max_length=16000)
    value: bool | int | float | None = None
    hypothesis: Id = Field(pattern=r"^E[1-9][0-9]* step [1-9][0-9]* .+")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def product_hashes(root):
    hashes = {p.relative_to(root).as_posix(): sha(p.read_bytes())
            for directory in ("src", "prompts", "config", "evaluation")
            for p in sorted((root / directory).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".txt", ".yaml", ".md", ".json"}}
    for filename in ("context-policy.md", "pyproject.toml"):
        if (root / filename).exists():
            hashes[filename] = sha((root / filename).read_bytes())
    return hashes


def file_hashes(root):
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in sorted(root.rglob("*")) if p.is_file()}


def snapshot(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for directory in ("src", "prompts", "config", "evaluation"):
        shutil.copytree(source / directory, destination / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "ranking-report.json"))
    shutil.copy2(source / "pyproject.toml", destination / "pyproject.toml")
    policy = source / "context-policy.md"
    (destination / "context-policy.md").write_text(policy.read_text(encoding="utf-8") if policy.exists()
                                                  else DEFAULT_POLICY, encoding="utf-8")


def validate_change(change, engine):
    if not re.search(r"\bE\d+\b", change.hypothesis) or not re.search(r"\bstep\s*\d+\b", change.hypothesis, re.I):
        raise ValueError("Hypothesis must cite an observed episode and step")
    if change.kind == "setting":
        if engine != "live" or change.content is not None:
            raise ValueError("Setting changes are available only to the live product adapter")
        if change.target.startswith("switches.") and change.target.partition(".")[2] in SWITCHES:
            if type(change.value) is not bool:
                raise ValueError("Switch changes require a boolean")
        elif change.target in SETTING_RANGES:
            lower, upper = SETTING_RANGES[change.target]
            if type(change.value) not in (int, float) or not math.isfinite(change.value) or not lower <= change.value <= upper:
                raise ValueError("Setting value outside its fixed experiment range")
            if change.target != "gaps.conflict_tolerance" and type(change.value) is not int:
                raise ValueError("This setting requires an integer")
        else:
            raise ValueError("Setting is outside the experiment allowlist; model, budget, corpus and scorer are frozen")
        return
    expected = ("context_policy", "context-policy.md") if engine == "context" else ("prompt", "prompts/step_agent.txt")
    if (change.kind, change.target) != expected or change.value is not None or not change.content or not change.content.strip():
        raise ValueError("Only the selected engine's single prompt target can be rewritten")
    # Reuse the upstream parser's evidence and nonempty policy validation.
    front = yaml.safe_dump({"name": "candidate", "description": change.hypothesis}, sort_keys=False)
    response = f"<<<ANALYSIS>>>\n{change.hypothesis}\n<<<END>>>\n<<<SKILL name=candidate>>>\n---\n{front}---\n{change.content}\n<<<END>>>"
    parsed = parse_proposal(response, max_skills=1)
    if not parsed.skills or parsed.rejected:
        raise ValueError(f"Upstream proposal validation failed: {parsed.rejected}")


def apply_change(workspace, change, engine):
    validate_change(change, engine)
    if change.kind == "setting":
        path = workspace / "config" / "settings.yaml"
        settings = yaml.safe_load(path.read_text(encoding="utf-8"))
        section, _, key = change.target.partition(".")
        if settings[section][key] == change.value:
            raise ValueError("Proposal does not change the incumbent setting")
        original_budget = copy.deepcopy(settings["budget"])
        settings[section][key] = change.value
        if settings["budget"] != original_budget:
            raise ValueError("Budget changed")
        path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    else:
        path = workspace / change.target
        if path.read_text(encoding="utf-8").strip() == change.content.strip():
            raise ValueError("Proposal repeats the incumbent prompt")
        path.write_text(change.content.strip() + "\n", encoding="utf-8")


def comparable(left, right):
    def protocol(evaluation):
        # Keep Python sets so citation-set iteration order cannot change equality
        # after a JSON checkpoint or a new worker process.
        header = evaluation.model_dump(exclude={"runs"})
        header["topics"] = sorted(header["topics"], key=lambda t: t["id"])
        for topic in header["topics"]:
            topic["questions"] = sorted(topic["questions"], key=lambda question: question["id"])
        corpora = [{t.id: sorted((p.id, p.published.isoformat()) for p in t.corpus) for t in r.topics}
                   for r in evaluation.runs]
        return header, corpora[0]
    if protocol(left) != protocol(right):
        raise ValueError("Candidate changed frozen questions, corpus, cutoffs, thresholds, or ranking protocol")


def acceptance(incumbent, candidate, system, plan):
    comparable(incumbent, candidate)
    reports = [score(e)["systems"].get(system) for e in (incumbent, candidate)]
    if None in reports:
        raise ValueError("Requested system is missing from an experiment")
    old, new = reports
    checks = []
    for name, result in (("incumbent", old), ("candidate", new)):
        checks.append({"check": name + "_repeats", "passed": result["repeat_count"] >= plan.repeats})
    a, b = [result["repeat_summary"].get(plan.objective) for result in reports]
    if a is None or b is None:
        raise ValueError("Unknown objective metric")
    defined = a["observations"] == old["repeat_count"] and b["observations"] == new["repeat_count"]
    gain = b["mean"] - a["mean"] if defined else None
    spread = max(a["sample_sd"] or 0, b["sample_sd"] or 0, plan.minimum_gain)
    checks.append({"check": "objective_improvement", "passed": defined and gain > spread + 1e-12,
                   "delta": gain, "required_exclusive_gain": spread})
    for metric, direction in plan.guardrails.items():
        previous, current = [result["repeat_summary"].get(metric) for result in reports]
        if previous is None or current is None:
            raise ValueError(f"Unknown guardrail {metric}")
        valid = previous["observations"] == old["repeat_count"] and current["observations"] == new["repeat_count"]
        passed = valid and (current["mean"] <= previous["mean"] + 1e-12 if direction == "lower"
                            else current["mean"] >= previous["mean"] - 1e-12)
        checks.append({"check": metric, "passed": passed, "before": previous["mean"], "after": current["mean"]})
    # Meta's task-paired SE is retained as a diagnostic; GapTrace uses the stricter frozen gate above.
    topic_scores = []
    for result in reports:
        values = {tid: statistics.mean([run["topics"][tid]["metrics"][plan.objective]["value"]
                    for run in result["runs"] if run["topics"][tid]["metrics"][plan.objective]["value"] is not None])
                  for tid in incumbent_topic_ids(incumbent)
                  if all(run["topics"][tid]["metrics"][plan.objective]["value"] is not None for run in result["runs"])}
        topic_scores.append(UpstreamScore(len(values), len(values), statistics.mean(values.values()) if values else None,
                                         None, None, values))
    se, kind = diff_se(topic_scores[1], topic_scores[0])
    return {"keep": all(item["passed"] for item in checks), "checks": checks,
            "paired_topic_se": se, "se_kind": kind,
            "before": old["repeat_summary"], "after": new["repeat_summary"]}


def incumbent_topic_ids(evaluation):
    return [topic.id for topic in evaluation.topics]


def proposer_note(evaluation, system, objective):
    report = score(evaluation)["systems"][system]
    episodes, digests = [], {}
    source_runs = {run.id: run for run in evaluation.runs}
    for repeat, run in enumerate(report["runs"]):
        for topic in source_runs[run["id"]].topics:
            reward = run["topics"][topic.id]["metrics"][objective]["value"]
            episode = Episode(topic.id, repeat, reward,
                              cost=run["topics"][topic.id]["metrics"]["response_seconds"]["value"], cost_metric="seconds")
            episodes.append(episode)
            # Do not disclose future candidate titles/abstracts, held-out material, or test labels.
            digests[id(episode)] = "\n".join(
                [f"step {i}: Q={answer.question_id}; answer={answer.answer}; correct={answer.correct}"
                 for i, answer in enumerate(topic.answers, 1)]
                + [f"step {len(topic.answers) + i}: proposed gap={gap.statement}" for i, gap in enumerate(topic.gaps, 1)])
    return build_note(episodes, lambda episode: digests[id(episode)], k=2), len(episodes)


def propose(model, current_workspace, evaluation, search, artifacts=None, history=()):
    note, episode_count = proposer_note(evaluation, search.system, search.objective)
    if search.engine == "context":
        targets = "Only kind=context_policy, target=context-policy.md, content=a complete replacement discipline; value=null."
        current = (current_workspace / "context-policy.md").read_text(encoding="utf-8")
    else:
        targets = ("Choose ONE: kind=prompt target=prompts/step_agent.txt content=complete replacement value=null; "
                   "OR kind=setting target=one of " + str(SETTING_RANGES) + " content=null value=the new scalar. "
                   "Switches allowed: " + str(sorted(SWITCHES)) + ". Do not change the model, budget, scorer, or data.")
        current = ((current_workspace / "prompts" / "step_agent.txt").read_text(encoding="utf-8") + "\n"
                   + (current_workspace / "config" / "settings.yaml").read_text(encoding="utf-8"))
    prompt = (f"Propose exactly one small experiment to improve {search.objective}. Preserve evidence quality and do not increase hallucinations. "
              "Start hypothesis with a real citation, exactly like 'E1 step 1 shows ...'. Cite only evidence actually provided. " + targets
              + "\nCurrent incumbent:\n" + model.clip(current, 1300)
              + "\nDevelopment evidence only:\n" + model.clip(note, 1700)
              + "\nRecent experiment decisions; avoid repeating failed or unchanged proposals:\n"
              + model.clip(json.dumps([{key: item.get(key) for key in ("proposal", "keep", "error", "checks")}
                                       for item in history[-10:]]), 500)
              + "\nReturn the required JSON only. Prefer a short discipline and retain source IDs.")
    if artifacts is not None:
        (artifacts / "proposer-request.txt").write_text(prompt, encoding="utf-8")
        (artifacts / "development-note.md").write_text(note, encoding="utf-8")
    change = model.call(prompt, Change)
    if artifacts is not None:
        dump(artifacts / "raw-proposal.json", change.model_dump())
    validate_change(change, search.engine)
    if any(int(number) < 1 or int(number) > episode_count for number in re.findall(r"\bE(\d+)\b", change.hypothesis)):
        raise ValueError("Proposal cites an episode that was not provided")
    return change


def bounded_process(command, cwd, log, seconds):
    env = dict(os.environ, PYTHONPATH=str(cwd / "src"), PYTHONIOENCODING="utf-8")
    kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, **kwargs)
        try:
            return process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=15)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
            raise TimeoutError(f"Experiment exceeded {seconds:.1f} seconds")


def freeze_inputs(search, source_path, destination):
    raw = source_path.read_bytes()
    if search.engine == "context":
        parsed = ContextPlan.model_validate_json(raw)
    else:
        from evaluation.live import Plan
        parsed = Plan.model_validate_json(raw)
    if parsed.split != "development":
        raise ValueError("Autoresearch cannot read or tune held-out plans")
    if parsed.mode == "benchmark" and search.repeats < 3:
        raise ValueError("Benchmark autoresearch requires at least three repeats")
    destination.mkdir(parents=True, exist_ok=False)
    if search.engine == "live":
        if parsed.system != search.system:
            raise ValueError("Search system does not match live plan system")
        data = parsed.model_dump(mode="json")
        for index, topic in enumerate(data["topics"]):
            corpus = (source_path.parent / topic["data"]).resolve()
            frozen = destination / f"corpus-{index}"
            if corpus.is_dir():
                shutil.copytree(corpus, frozen, ignore=shutil.ignore_patterns("cache", "runs", "__pycache__"))
                topic["data"] = str(frozen)
            elif topic.get("source"):
                source = (source_path.parent / topic["source"]).resolve()
                frozen_source = destination / f"source-{index}.json"
                frozen_source.write_bytes(source.read_bytes())
                topic["source"], topic["data"] = str(frozen_source), str(frozen)
            else:
                raise ValueError("Live plan needs a source or an existing corpus")
        raw = (json.dumps(data, indent=2) + "\n").encode()
    (destination / "plan.json").write_bytes(raw)
    return sha(source_path.read_bytes())


def evaluate(workspace, trial, search, frozen_plan, tokenizer, model, base, seconds):
    trial.mkdir(parents=True, exist_ok=False)
    plan_path = frozen_plan
    if search.engine == "live":
        data = json.loads(frozen_plan.read_bytes())
        for index, topic in enumerate(data["topics"]):
            frozen = Path(topic["data"])
            corpus = workspace / "corpora" / str(index)
            if frozen.is_dir():
                shutil.copytree(frozen, corpus)
            topic["data"] = str(corpus)
        plan_path = trial / "plan.json"
        dump(plan_path, data)
        command = [sys.executable, "-m", "evaluation.live", str(plan_path), "--out", str(trial / "output"),
                   "--tokenizer", str(tokenizer), "--judge", model, "--ollama-url", base, "--repeats", str(search.repeats)]
    else:
        command = [sys.executable, "-m", "evaluation.context_experiment", str(plan_path), "--out", str(trial / "output"),
                   "--tokenizer", str(tokenizer), "--model", model, "--ollama-url", base, "--repeats", str(search.repeats),
                   "--representations", search.context_representation, "--managements", search.context_management,
                   "--policy", str(workspace / "context-policy.md")]
    before = product_hashes(workspace)
    returncode = bounded_process(command, workspace, trial / "worker.log", seconds)
    if product_hashes(workspace) != before:
        raise ValueError("Experiment worker changed source, scoring, settings or prompts during evaluation")
    if returncode:
        raise ValueError(f"Experiment process exited {returncode}; inspect {trial / 'worker.log'}")
    result = Evaluation.model_validate_json((trial / "output" / "manifest.json").read_bytes())
    if result.split != "development":
        raise ValueError("Worker produced held-out results")
    frozen = json.loads(frozen_plan.read_bytes())
    for field in ("mode", "split", "cutoff", "future_through"):
        actual = result.model_dump(mode="json")[field]
        if actual != frozen.get(field, "development" if field == "split" else None):
            raise ValueError(f"Worker changed frozen plan field {field}")
    expected_questions = {t["id"]: t["questions"] for t in frozen["topics"]}
    actual_questions = {t.id: [q.model_dump(mode="json") for q in t.questions] for t in result.topics}
    # Sets in gold IDs are order independent; parse through Question before comparing.
    from evaluation.metrics import Question
    if {key: [Question.model_validate_json(json.dumps(q)) for q in questions] for key, questions in expected_questions.items()} != {
        key: [Question.model_validate_json(json.dumps(q)) for q in questions] for key, questions in actual_questions.items()}:
        raise ValueError("Worker changed frozen questions")
    return result


def run(search, config_path, out, tokenizer, model, base="http://127.0.0.1:11434", proposer_model=None,
        evaluator=evaluate):
    """Resume only on identical plans/tokenizer/model and the frozen source snapshot."""
    out = out.resolve()
    if search.engine == "live" and urlparse(base).port != 11434:
        raise ValueError("The public product CLI uses Ollama port 11434; live experiments must use the same server")
    if any(out == ROOT / part or (ROOT / part) in out.parents for part in ("src", "config", "prompts", "evaluation")):
        raise ValueError("Experiment outputs must be outside the source and evaluation directories")
    input_path = (config_path.parent / search.input_plan).resolve()
    identity = {"search": search.model_dump(), "input_sha256": sha(input_path.read_bytes()),
                "tokenizer_sha256": sha(tokenizer.read_bytes()), "model": model, "ollama_url": base,
                "evaluation_code_sha256": {name: sha((ROOT / "evaluation" / name).read_bytes()) for name in
                    ("autoresearch.py", "context_experiment.py", "local_model.py", "metrics.py", "ranking.py")}}
    start = monotonic()
    if out.exists():
        state = json.loads((out / "state.json").read_bytes())
        if state["identity"] != identity:
            raise ValueError("Resume configuration differs from the frozen experiment")
        if product_hashes(out / "baseline-workspace") != state["baseline_hashes"]:
            raise ValueError("Frozen baseline source was modified")
        if file_hashes(out / "inputs") != state["input_hashes"]:
            raise ValueError("Frozen input material was modified")
    else:
        out.mkdir(parents=True)
        freeze_inputs(search, input_path, out / "inputs")
        snapshot(ROOT, out / "baseline-workspace")
        if search.engine == "live":
            settings_path = out / "baseline-workspace" / "config" / "settings.yaml"
            settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
            settings["model"]["local"] = "ollama_chat/" + model
            settings["cloud"]["enabled"] = False
            settings_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
        state = {"identity": identity, "baseline_hashes": product_hashes(out / "baseline-workspace"),
                 "input_hashes": file_hashes(out / "inputs"),
                 "round": 0, "incumbent": "baseline", "elapsed_seconds": 0, "history": []}
        dump(out / "state.json", state)
    base_elapsed = state["elapsed_seconds"]
    remaining = lambda: search.total_seconds - base_elapsed - (monotonic() - start)
    model_runner = proposer_model or LocalModel(tokenizer, model, base, timeout=min(120, max(1, remaining())))
    digest = getattr(model_runner, "model_digest", None)
    if "model_digest" in state and state["model_digest"] != digest:
        raise ValueError("Local model tag now refers to a different model digest")
    state["model_digest"] = digest
    dump(out / "state.json", state)
    def current_workspace():
        return out / "baseline-workspace" if state["incumbent"] == "baseline" else out / state["incumbent"] / "workspace"
    if not (out / "baseline-manifest.json").exists():
        if remaining() <= 0:
            raise TimeoutError("Total experiment budget is exhausted")
        baseline = evaluator(out / "baseline-workspace", out / "baseline-trial", search,
                             out / "inputs" / "plan.json", tokenizer, model, base, min(search.trial_seconds, remaining()))
        if search.system not in score(baseline)["systems"]:
            raise ValueError("Baseline did not produce the requested system")
        baseline_scores = score(baseline)["systems"][search.system]
        for metric in [search.objective, *search.guardrails]:
            summary = baseline_scores["repeat_summary"].get(metric)
            if summary is None or summary["observations"] != baseline_scores["repeat_count"]:
                raise ValueError(f"Baseline metric {metric} is missing or undefined; do not tune on an incomplete scorecard")
        dump(out / "baseline-manifest.json", baseline.model_dump(mode="json"))
        state["elapsed_seconds"] = base_elapsed + monotonic() - start
        dump(out / "state.json", state)
    incumbent = Evaluation.model_validate_json((out / "baseline-manifest.json" if state["incumbent"] == "baseline"
                else out / state["incumbent"] / "trial" / "output" / "manifest.json").read_bytes())
    for number in range(state["round"] + 1, search.rounds + 1):
        if remaining() <= 0:
            break
        label = f"round-{number:04d}"
        folder = out / label
        folder.mkdir(exist_ok=True)
        record = {"round": number, "incumbent_before": state["incumbent"], "keep": False}
        try:
            print(f"Autoresearch {label}: proposing one change", flush=True)
            if (folder / "proposal.json").exists():
                change = Change.model_validate_json((folder / "proposal.json").read_bytes())
            else:
                model_runner.timeout = max(1, min(120, remaining()))
                model_runner.deadline = monotonic() + remaining()
                change = propose(model_runner, current_workspace(), incumbent, search, folder, state["history"])
                dump(folder / "proposal.json", change.model_dump())
                dump(folder / "proposer-usage.json", model_runner.calls)
            record["proposal"] = change.model_dump()
            workspace = folder / "workspace"
            if not workspace.exists():
                snapshot(current_workspace(), workspace)
                apply_change(workspace, change, search.engine)
                dump(folder / "candidate-hashes.json", product_hashes(workspace))
            elif not (folder / "candidate-hashes.json").exists() or product_hashes(workspace) != json.loads((folder / "candidate-hashes.json").read_bytes()):
                raise ValueError("Candidate snapshot is incomplete or was modified")
            if remaining() <= 0:
                raise TimeoutError("Total budget exhausted before candidate evaluation")
            trial = folder / "trial"
            if (trial / "output" / "manifest.json").exists():
                candidate = Evaluation.model_validate_json((trial / "output" / "manifest.json").read_bytes())
            else:
                if trial.exists():
                    raise ValueError("Incomplete previous candidate trial; retain artifacts and start a new run")
                candidate = evaluator(workspace, trial, search, out / "inputs" / "plan.json", tokenizer,
                                      model, base, min(search.trial_seconds, remaining()))
            decision = acceptance(incumbent, candidate, search.system, search)
            record.update(decision)
            if decision["keep"]:
                state["incumbent"] = label
                incumbent = candidate
        except (OSError, ValueError, TimeoutError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            dump(folder / "proposer-usage.json", model_runner.calls)
        record["incumbent_after"] = state["incumbent"]
        dump(folder / "decision.json", record)
        state["history"].append(record)
        state["round"] = number
        state["elapsed_seconds"] = base_elapsed + monotonic() - start
        dump(out / "state.json", state)
        with (out / "results.tsv").open("a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            if file.tell() == 0:
                writer.writerow(["round", "keep", "incumbent", "objective", "error", "hypothesis"])
            writer.writerow([number, record["keep"], state["incumbent"], search.objective,
                             record.get("error", ""), record.get("proposal", {}).get("hypothesis", "")])
        print(f"Autoresearch {label}: {'kept candidate' if record['keep'] else 'retained incumbent'}", flush=True)
    state["elapsed_seconds"] = base_elapsed + monotonic() - start
    dump(out / "state.json", state)
    dump(out / "best.json", {"workspace": str(current_workspace()), "incumbent": state["incumbent"],
                             "production_files_modified": False, "rounds_completed": state["round"]})
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    try:
        search = SearchPlan.model_validate_json(args.plan.read_bytes())
        run(search, args.plan.resolve(), args.out, args.tokenizer.resolve(), args.model, args.ollama_url)
    except (OSError, ValueError, TimeoutError) as exc:
        parser.exit(2, f"Autoresearch error: {exc}\n")


if __name__ == "__main__":
    main()
