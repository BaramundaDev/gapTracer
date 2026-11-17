"""Score recorded GapTrace runs without importing the product or calling a model."""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import statistics
from datetime import date
from pathlib import Path
from typing import Annotated, Iterable, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

if __package__:
    from .ranking import score_ranking, human_agreement, bootstrap_interval, metric_direction
else:
    from ranking import score_ranking, human_agreement, bootstrap_interval, metric_direction

Id = Annotated[str, Field(min_length=1)]
Count = Annotated[int, Field(ge=0, strict=True)]
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
T = TypeVar("T", bound=BaseModel)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Paper(Record):
    id: Id
    published: date


class Verdict(Record):
    match: bool
    confidence: Score
    grader: Id
    reason: str = ""


class Candidate(Paper):
    first: Verdict
    recheck: Verdict | None = None
    title: str = ""
    abstract: str = ""
    similarity: Annotated[float, Field(ge=-1, le=1)] | None = None


Rating = Annotated[int, Field(ge=1, le=5, strict=True)]
RubricScore = Annotated[int, Field(ge=0, le=2, strict=True)]


class HumanRating(Record):
    rater: Id
    real: Rating
    open: Rating
    doable: Rating
    reason: Id


class DoabilityRubric(Record):
    data: RubricScore
    compute: RubricScore
    blocks: RubricScore
    scope: RubricScore


class GapJudgment(Record):
    """Adjudicated labels, independent of generation and future matching."""
    grader: Id
    real: Rating
    open: Rating
    doable: Rating
    reason: Id
    doability_rubric: DoabilityRubric | None = None


class EvidenceLink(Record):
    passage_id: Id
    paper_id: Id
    # Exact source text allows deterministic quote checking; entailment is a judgment.
    passage_text: Id
    quote: Id
    supported: bool
    grader: Id

    @model_validator(mode="after")
    def check_quote(self) -> EvidenceLink:
        if self.supported and self.quote not in self.passage_text:
            raise ValueError("Supported gap evidence quote is absent from its source passage")
        return self


class GoldGap(Record):
    id: Id
    statement: Id
    relevance: Annotated[int, Field(ge=0, le=3, strict=True)]


class Gap(Record):
    id: Id
    statement: Id
    candidates: list[Candidate] = Field(max_length=5)
    type: Literal["unaddressed_limitation", "unanswered_open_question", "untested_combination",
                  "conflicting_results"] | None = None
    canonical_id: Id | None = None
    gold_id: Id | None = None
    judgment: GapJudgment | None = None
    ratings: list[HumanRating] = Field(default_factory=list)
    evidence: list[EvidenceLink] | None = None
    predicted_status: Literal["open", "partly", "closed", "unchecked"] | None = None
    reference_status: Literal["open", "partly", "closed"] | None = None
    future_probability: Score | None = None
    predicted_doability: Annotated[int, Field(ge=0, le=8, strict=True)] | None = None

    @model_validator(mode="after")
    def check_raters(self) -> Gap:
        by_id(self.ratings, "rater", "rater")
        return self


class Question(Record):
    id: Id
    text: Id
    gold_answer: Id
    answerable: bool = True
    gold_passage_ids: set[Id] = Field(default_factory=set)
    required_paper_ids: set[Id] = Field(default_factory=set)

    @model_validator(mode="after")
    def evidence_contract(self):
        if self.answerable and (not self.gold_passage_ids or not self.required_paper_ids):
            raise ValueError("Answerable questions require gold passages and source papers")
        if not self.answerable and (self.gold_passage_ids or self.required_paper_ids):
            raise ValueError("Unanswerable questions must not contain fictitious gold evidence")
        return self


class Claim(Record):
    text: Id
    supported: bool
    passage_ids: set[Id] = Field(default_factory=set)
    quote: str = ""


class TokenComparison(Record):
    mcp_tokens: Count
    raw_passage_tokens: Annotated[int, Field(gt=0, strict=True)]


class Answer(Record):
    question_id: Id
    answer: str
    grader: Id
    correct: bool
    cited_passage_ids: set[Id]
    retrieved_paper_ids: set[Id]
    claims: list[Claim]
    tool_calls: Count
    response_seconds: Seconds
    token_comparison: TokenComparison | None = None

    @model_validator(mode="after")
    def check_claim_evidence(self) -> Answer:
        for claim in self.claims:
            if not claim.passage_ids <= self.cited_passage_ids:
                raise ValueError(f"Claim references a passage absent from answer {self.question_id}'s citations")
            if claim.supported and not claim.passage_ids:
                raise ValueError(f"Supported claim has no citation in answer {self.question_id}")
        return self


class Topic(Record):
    id: Id
    questions: list[Question]
    gold_gaps: list[GoldGap] | None = None


class Efficiency(Record):
    total_tokens: Count
    cloud_calls: Count
    largest_context_tokens: Count
    peak_gpu_mib: Count
    build_seconds: Seconds


class TopicRun(Record):
    id: Id
    corpus: list[Paper] = Field(min_length=1)
    build_cloud_enabled: bool
    build_cloud_calls: Annotated[Count, Field(le=0)]
    gaps: list[Gap]
    answers: list[Answer]
    efficiency: Efficiency | None = None
    gold_mapping_complete: bool = False


class Run(Record):
    id: Id
    system: Id
    topics: list[TopicRun]


def by_id(items: Iterable[T], label: str, field: str = "id") -> dict[str, T]:
    keyed = {}
    for item in items:
        key = getattr(item, field)
        if key in keyed:
            raise ValueError(f"Duplicate {label} ID {key}")
        keyed[key] = item
    return keyed


def validate_protocol(mode: str, cutoff: date, training_cutoff: date | None, future_through: date,
                      first_threshold: float, recheck_threshold: float) -> None:
    if mode == "benchmark" and training_cutoff is None:
        raise ValueError("A benchmark requires the model training cutoff")
    if training_cutoff and cutoff < training_cutoff:
        raise ValueError("The corpus cutoff precedes the model training cutoff")
    year_offset, month_index = divmod(cutoff.month - 1 + 9, 12)
    year, month = cutoff.year + year_offset, month_index + 1
    day = min(cutoff.day, calendar.monthrange(year, month)[1])
    if future_through < date(year, month, day):
        raise ValueError("The future window must span at least nine calendar months")
    if recheck_threshold <= first_threshold:
        raise ValueError("The recheck threshold must be stricter than the first threshold")


class Evaluation(Record):
    mode: Literal["benchmark", "smoke"] = "benchmark"
    split: Literal["development", "held-out"]
    cutoff: date
    model_training_cutoff: date | None = None
    future_through: date
    first_threshold: Score = 0.6
    recheck_threshold: Score = 0.8
    ranking_ks: list[Annotated[int, Field(ge=1, le=100, strict=True)]] = Field(default_factory=lambda: [1, 3, 5, 10], min_length=1)
    topics: list[Topic] = Field(min_length=1)
    runs: list[Run] = Field(min_length=1)

    @model_validator(mode="after")
    def check_protocol(self) -> Evaluation:
        validate_protocol(self.mode, self.cutoff, self.model_training_cutoff, self.future_through,
                          self.first_threshold, self.recheck_threshold)

        expected = {topic.id: by_id(topic.questions, f"question in {topic.id}")
                    for topic in by_id(self.topics, "topic").values()}
        if len(set(self.ranking_ks)) != len(self.ranking_ks):
            raise ValueError("ranking_ks must be unique")
        gold = {topic.id: by_id(topic.gold_gaps or [], f"gold gap in {topic.id}") for topic in self.topics}
        reference_corpora: dict[str, dict[str, date]] = {}
        for run in by_id(self.runs, "run").values():
            actual = by_id(run.topics, f"topic in run {run.id}")
            if actual.keys() != expected.keys():
                raise ValueError(f"Run {run.id} must include every declared topic exactly once")
            for topic in actual.values():
                self.check_topic(run.id, topic, expected[topic.id])
                if topic.gold_mapping_complete and next(t for t in self.topics if t.id == topic.id).gold_gaps is None:
                    raise ValueError(f"Complete gold mapping requires a gold catalogue for {topic.id}")
                for gap in topic.gaps:
                    if gap.gold_id is not None and gap.gold_id not in gold[topic.id]:
                        raise ValueError(f"Unknown gold gap {gap.gold_id} in {topic.id}")
                    for link in gap.evidence or []:
                        if link.paper_id not in {paper.id for paper in topic.corpus}:
                            raise ValueError(f"Gap evidence paper {link.paper_id} is outside the corpus")
                corpus = {paper.id: paper.published for paper in topic.corpus}
                if reference_corpora.setdefault(topic.id, corpus) != corpus:
                    raise ValueError(f"Run {run.id} uses a different corpus for topic {topic.id}")
        return self

    def check_topic(self, run_id: str, topic: TopicRun, questions: dict[str, Question]) -> None:
        label = f"{run_id}/{topic.id}"
        if topic.build_cloud_enabled:
            raise ValueError(f"Time-split build must disable cloud access in {label}")
        corpus = by_id(topic.corpus, f"corpus paper in {label}")
        for paper in corpus.values():
            if paper.published > self.cutoff:
                raise ValueError(f"Post-cutoff corpus paper {paper.id} in {label}")

        answers = by_id(topic.answers, f"answer in {label}", "question_id")
        if answers.keys() != questions.keys():
            raise ValueError(f"{label} must include one result per question, including abstentions")
        for key, answer in answers.items():
            if not (questions[key].required_paper_ids | answer.retrieved_paper_ids) <= corpus.keys():
                raise ValueError(f"Question {key} in {label} refers to papers outside the recorded corpus")

        for gap in by_id(topic.gaps, f"gap in {label}").values():
            for candidate in by_id(gap.candidates, f"candidate for {label}/{gap.id}").values():
                if candidate.id in corpus or not self.cutoff < candidate.published <= self.future_through:
                    raise ValueError(f"Future candidate {candidate.id} for {label}/{gap.id} is outside the grading window")
                if candidate.first.match and candidate.first.confidence >= self.first_threshold:
                    if candidate.recheck is None:
                        raise ValueError(f"Candidate {candidate.id} for {label}/{gap.id} is missing its strict recheck")


def ratio(numerator: int | float, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def score_topic(topic: TopicRun, questions: dict[str, Question], evaluation: Evaluation) -> dict:
    hits = {}
    for gap in topic.gaps:
        confirmed = [candidate for candidate in gap.candidates
                     if candidate.first.match and candidate.first.confidence >= evaluation.first_threshold
                     and candidate.recheck and candidate.recheck.match
                     and candidate.recheck.confidence >= evaluation.recheck_threshold]
        if confirmed:
            earliest = min(confirmed, key=lambda paper: (paper.published, paper.id))
            hits[gap.id] = {"statement": gap.statement, "paper_id": earliest.id,
                            "published": earliest.published.isoformat(),
                            "lead_days": (earliest.published - evaluation.cutoff).days}

    claims = [claim for answer in topic.answers for claim in answer.claims]
    multi_paper = [answer for answer in topic.answers
                   if len(questions[answer.question_id].required_paper_ids) > 1]
    comparisons = [answer.token_comparison for answer in topic.answers if answer.token_comparison is not None]
    raw_tokens = sum(pair.raw_passage_tokens for pair in comparisons)
    days = sum(hit["lead_days"] for hit in hits.values())
    metrics = {
        "hit_rate": ratio(len(hits), len(topic.gaps)),
        "coverage": ratio(int(bool(hits)), 1),
        "lead_time_months": ratio(days / (365.2425 / 12), len(hits)),
        "answer_accuracy": ratio(sum(answer.correct for answer in topic.answers), len(topic.answers)),
        "evidence_quality": ratio(sum(len(answer.cited_passage_ids & questions[answer.question_id].gold_passage_ids)
                                      for answer in topic.answers),
                                  sum(len(answer.cited_passage_ids) for answer in topic.answers)),
        "cross_document_retrieval": ratio(sum(questions[answer.question_id].required_paper_ids
                                              <= answer.retrieved_paper_ids for answer in multi_paper),
                                          len(multi_paper)),
        "hallucination_rate": ratio(sum(not claim.supported for claim in claims), len(claims)),
        "retrieval_steps": ratio(sum(answer.tool_calls for answer in topic.answers), len(topic.answers)),
        "response_seconds": ratio(math.fsum(answer.response_seconds for answer in topic.answers), len(topic.answers)),
        "mcp_token_savings": ratio(raw_tokens - sum(pair.mcp_tokens for pair in comparisons), raw_tokens),
    }
    definition = next(t for t in evaluation.topics if t.id == topic.id)
    ranking = score_ranking(topic.gaps, hits, definition.gold_gaps, evaluation.ranking_ks,
                            topic.gold_mapping_complete)
    metrics.update(ranking.pop("metrics"))
    return {"metrics": metrics, "confirmed_gaps": hits, "mcp_question_count": len(comparisons),
            "ranking": ranking,
            "efficiency": topic.efficiency.model_dump() if topic.efficiency else None}


def summarize(values: Iterable[float | None]) -> dict:
    defined = [value for value in values if value is not None]
    return {"mean": statistics.mean(defined) if defined else None,
            "sample_sd": statistics.stdev(defined) if len(defined) > 1 else None,
            "observations": len(defined)}


def compare_systems(left: list[dict], right: list[dict], topic_ids: Iterable[str]) -> dict:
    """Two-sided paired sign tests on within-topic means across repeats."""
    comparisons = {}
    for metric in left[0]["metrics"]:
        deltas = {}
        for topic_id in topic_ids:
            means = [summarize(r["topics"][topic_id]["metrics"][metric]["value"] for r in runs)["mean"]
                     for runs in (left, right)]
            if None not in means:
                deltas[topic_id] = means[0] - means[1]
        positive = sum(delta > 1e-12 for delta in deltas.values())
        negative = sum(delta < -1e-12 for delta in deltas.values())
        n = positive + negative
        tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1))
        comparisons[metric] = {"left_minus_right": summarize(deltas.values()),
                               "topic_deltas": deltas, "non_tied_pairs": n,
                               "direction": metric_direction(metric),
                               "topic_bootstrap_95_ci": bootstrap_interval(list(deltas.values())),
                               "p_value": min(1.0, 2 * tail / 2 ** n) if n else None}
    # Holm family-wise correction over every defined comparison in this report.
    ordered = sorted((item["p_value"], name) for name, item in comparisons.items() if item["p_value"] is not None)
    previous = 0.0
    for index, (p_value, name) in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - index) * p_value))
        comparisons[name]["holm_p_value"] = previous
    for item in comparisons.values():
        item.setdefault("holm_p_value", None)
    return comparisons


def score(evaluation: Evaluation, compare: tuple[str, str] | None = None) -> dict:
    question_sets = {topic.id: by_id(topic.questions, "question") for topic in evaluation.topics}
    runs_by_system: dict[str, list[dict]] = {}
    for run in evaluation.runs:
        topics = {topic.id: score_topic(topic, question_sets[topic.id], evaluation) for topic in run.topics}
        metrics = {}
        for name in next(iter(topics.values()))["metrics"]:
            parts = [topic["metrics"][name] for topic in topics.values()]
            metrics[name] = ratio(sum(part["numerator"] for part in parts),
                                  sum(part["denominator"] for part in parts))
            if any(name in topic["ranking"]["incomplete_metrics"] for topic in topics.values()):
                metrics[name] = {**ratio(0, 0), "incomplete_labels": True,
                                 "available_subset": metrics[name]}
        runs_by_system.setdefault(run.system, []).append({"id": run.id, "metrics": metrics, "topics": topics,
            "metric_topic_coverage": {name: {"defined": sum(t["metrics"][name]["value"] is not None for t in topics.values()),
                                             "total": len(topics)} for name in metrics}})

    systems = {}
    for name, runs in runs_by_system.items():
        systems[name] = {"repeat_count": len(runs), "runs": runs,
                         "topic_macro_summary": {metric: summarize(
                             summarize(run["topics"][tid]["metrics"][metric]["value"] for run in runs)["mean"]
                             for tid in question_sets) for metric in runs[0]["metrics"]},
                         "repeat_summary": {metric: summarize(run["metrics"][metric]["value"] for run in runs)
                                            for metric in runs[0]["metrics"]}}
    report = {"metric_version": "2.0.0", "mode": evaluation.mode, "split": evaluation.split, "cutoff": evaluation.cutoff.isoformat(),
              "model_training_cutoff": (evaluation.model_training_cutoff.isoformat()
                                        if evaluation.model_training_cutoff else None),
              "future_through": evaluation.future_through.isoformat(),
              "thresholds": {"first": evaluation.first_threshold, "recheck": evaluation.recheck_threshold},
              "ranking_ks": evaluation.ranking_ks,
              "metric_directions": {name: metric_direction(name) for name in next(iter(systems.values()))["runs"][0]["metrics"]},
              "human_agreement": human_agreement(evaluation),
              "warnings": (["Smoke results are integration checks, not research evidence."] if evaluation.mode == "smoke" else [])
                          + [f"{name} has {len(runs)} repeats; research reporting requires at least three."
                             for name, runs in runs_by_system.items() if len(runs) < 3]
                          + ["Unmatched future gaps are unconfirmed within the search window, not proven invalid.",
                             "Within-list ranking metrics are conditional on the submitted candidate set; use fixed gold catalogues for discovery comparisons.",
                             "Repeated cached outputs measure reproducibility, not independent model uncertainty."],
              "systems": systems}
    if compare:
        left, right = compare
        if left == right or left not in systems or right not in systems:
            raise ValueError("--compare requires two different system names present in the manifest")
        report["paired_comparison"] = {"left": left, "right": right,
                                       "metrics": compare_systems(runs_by_system[left], runs_by_system[right], question_sets)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, help="Write the JSON report to this path")
    parser.add_argument("--compare", nargs=2, metavar=("SYSTEM", "BASELINE"))
    parser.add_argument("--split", choices=("development", "held-out"), default="development")
    args = parser.parse_args()
    try:
        manifest = args.manifest.read_bytes()
        evaluation = Evaluation.model_validate_json(manifest)
        if evaluation.split != args.split:
            raise ValueError(f"Manifest split is {evaluation.split}; use --split {evaluation.split} to score it")
        comparison = tuple(args.compare) if args.compare else None
        report = score(evaluation, comparison)
        report["manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
        output = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.out:
            args.out.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (OSError, ValidationError, ValueError) as exc:
        parser.exit(2, f"Evaluation error: {exc}\n")


if __name__ == "__main__":
    main()
