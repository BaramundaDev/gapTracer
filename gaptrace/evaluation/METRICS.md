# Gap ranking evaluation specification

Version 2.0.0. This is the evaluation contract for GapTrace's ranked gap lists.
It implements the research requirements in the root project DOCX, the agent
brief, and this folder's existing README. The product remains unchanged.
Scoring code is in `metrics.py` and `ranking.py`; `review.py` handles blinded
rating forms. No additional packages are required.

## Decisions the metrics support

Evaluate discovery, order, validity, openness, feasibility, and source support
separately. A later matching paper confirms that the proposed missing work was
pursued within the observed window. A failure to find a matching paper does
not establish that a gap is invalid, closed, unimportant, or infeasible.
Search coverage, the five-paper shortlist, publication delays, and incomplete
indexing all affect that outcome. Human judgments address this blind spot.

Use `future_precision@5` as the primary rank-sensitive temporal endpoint and
retain the brief's original `hit_rate` as the primary historical endpoint.
Use `actionable_precision@5` as the primary quality endpoint on independently
adjudicated complete lists. These are design choices to freeze before tuning,
not measured claims about which metric best predicts research success.
Use `gold_ndcg@5` and `gold_recall@5` for discovery comparisons only when a fixed,
independently assembled catalogue has been reviewed for every system.
Report support, closure, duplicates, annotation coverage, and list fill with
these endpoints. Do not average them into one opaque overall score.

## Input and provenance

Run from the repository root:

```powershell
python evaluation/metrics.py evaluation/examples/ranking-fixture.json --out ranking-report.json
python evaluation/metrics.py evaluation/fixture.json --compare full vector --out comparison.json
python evaluation/metrics.py held-out.json --split held-out --out held-out-report.json
```

`manifest.schema.json` is generated from `Evaluation.model_json_schema()`.
Pydantic additionally enforces relationships not expressible in the schema.
Unknown fields, duplicate IDs, nonfinite numbers, invalid ranges, and missing
required fields fail validation. IDs are case sensitive. Dates are ISO dates.
The report records a SHA-256 of the exact CLI input bytes and metric version.
Programmatic callers of `score()` must separately preserve/hash their input.

The top-level manifest records split, cutoff, model training cutoff, future
window, thresholds, topic definitions, runs, and `ranking_ks` (default
`[1,3,5,10]`, unique integers 1 through 100). Set these before generation.
Each run includes every topic; every question needs one answer, even refusals.
An empty gap list is allowed and is measured as zero yield.

Each topic run records the complete dated corpus, cloud-disabled build
provenance, an ordered gap list, answers, and optional efficiency measurements.
All systems and repeats must have identical corpus paper IDs and dates per
topic. The recorder must also preserve source revisions, full corpus/index
hashes, model versions, prompt/settings hashes, seed, cache identity, hardware,
and generation logs alongside the manifest. Paper IDs and dates alone cannot
prove equal content or absence of contamination in a model or index.

The recorder must freeze gap statements, order, status, doability, and any
probability before future search or grading. A score is meaningless if labels
are used to reorder outputs. Equal product scores retain the product's stable
tie order. There is no evaluator-side tie averaging or relevance sorting.

### Optional per-gap annotations

| Field | Contract and purpose |
| --- | --- |
| `type` | One of the product's four gap patterns; used for diagnostic slices. |
| `canonical_id` | Independent duplicate cluster. Assign singleton clusters too. Only the first occurrence earns rank gain. Unknown clusters fall back to gap identity, and coverage reveals this limitation. |
| `gold_id` | A match to a fixed topic catalogue entry; `null` means no match only after mapping is declared complete. |
| `judgment` | Adjudicated grader, real/open/doable integers 1..5, reason, optional doability rubric. Required for quality endpoints. |
| `ratings` | Independent human rater IDs, three 1..5 labels, reasons. Used for agreement, never silently converted into adjudication. |
| `evidence` | All claimed evidence links with source paper/passage IDs, exact passage text, quote, support label, grader. `null` means unreviewed; `[]` means reviewed with no evidence. |
| `predicted_status` | Frozen product open/partly/closed/unchecked prediction. |
| `reference_status` | Independent as-of-cutoff open/partly/closed label. |
| `predicted_doability` | Frozen product score 0..8. |
| `future_probability` | Frozen probability in [0,1] of confirmation under this exact future-search protocol. Optional; verification share and matcher confidence are different quantities. |

The topic's `gold_gaps` catalogue contains unique IDs, statements, and fixed
relevance 0..3. The topic run's `gold_mapping_complete` defaults to false;
true requires a catalogue. Every prediction must then have been matched or
explicitly rejected. No automatic semantic matching is assumed by the scorer.

## Temporal confirmation protocol

A benchmark requires a documented model training cutoff, a corpus cutoff T
on or after it, and an observation window ending at least nine calendar months
after T. The same generation-side restriction applies to every generating
model and external resource; the scalar cutoff in the manifest is an audit
assertion, not proof. Smoke mode permits historical replay without a training
cutoff but its results cannot support research claims.

All corpus papers must be dated on or before T. Future candidates must be
outside the corpus and have T < publication date <= future_through. Build
cloud access and build cloud calls must both be zero. The live runner records
all generation before future retrieval. Retain the provider, query, bounded
pool, search time, retrieved abstracts and versions, embedding model, similarity
scores, matcher prompt/version, raw decisions and rechecks as external logs.

Shortlist at most five future papers per gap. A paper confirms a gap only if
both `first.match` and `recheck.match` are true and their confidences meet the
fixed thresholds (defaults 0.60 and 0.80). The recheck threshold must be strictly
higher. Every qualifying first pass requires a recheck. A missing strict
recheck is an error, not a negative. The same judge at a stricter threshold is
not an independent human confirmation. Check 50 hits manually as required by
the brief and report their error rate separately from the automated score.

Each proposed gap counts once in the original hit rate, regardless of how
many future papers fill it. Its lead time uses the earliest confirmed
shortlisted paper, with paper ID breaking date ties. It is not necessarily
the earliest filling paper in all literature.

## Ranking definitions

Let G be a topic's ordered list of n predictions. Rank i starts at 1. A missing
position after n earns zero. Let u_i be 1 at a canonical cluster's first
occurrence and 0 thereafter. Let h_i be 1 when gap i is temporally confirmed.
The rank gain is y_i = h_i u_i. Canonical identities must be labeled completely
to establish a reliable duplicate penalty; unknown identities do not license
a claim of uniqueness. Original `hit_rate` stays unchanged for compatibility.

For a nonnegative grade g define gain(g) = 2^g - 1 and
DCG@k = sum over i<=min(k,n) of gain(g_i)/log2(i+1).
IDCG@k uses the ideal descending reference grades. NDCG = DCG/IDCG.
Zero ideal gain produces `null`, because there is no relevance to rank.
This implementation explicitly uses exponential gain; library defaults may
use linear gain. The [official NDCG definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
explains discounting and ideal normalization.

| Report metric | Exact definition and interpretation |
| --- | --- |
| `gap_yield` | n per topic. Diagnostic, neither more nor fewer is intrinsically better. |
| `list_fill@k` | min(k,n)/k. Reveals short output lists. |
| `future_precision@k` | sum of y_i through k / k. Unfilled slots and repeated clusters earn zero. |
| `future_success@k` | 1 if any y_i through k is positive, else 0. Topic-level hit coverage at a review budget. |
| `future_mrr` | Reciprocal of first positive y_i rank, or 0 if none. |
| `future_recall_within_list@k` | Positive y_i through k / all positive y_i in G. Conditional on submitted predictions, not recall of all research opportunities. |
| `future_ap_within_list@k` | Sum of precision@i at positive y_i ranks through k / min(k, number of positive y_i in G). `null` if no observed positives. |
| `future_ndcg_within_list@k` | NDCG of binary y_i relative to ideal ordering of G. Measures order conditional on candidate generation. |
| `actionable_precision@k` | Count through k with u_i=1 and adjudicated real>=4, open>=4, doable>=4, divided by k. |
| `real_utility@k`, `open_utility@k`, `doable_utility@k` | Sum through k of u_i*(axis_label-1)/4, divided by k. Fixed-budget utility in [0,1]. |
| `quality_ndcg_within_list@k` | NDCG with grade 0 if real<4 or open<4; otherwise grade 1 for doable<=2, 2 for doable=3, 3 for doable>=4. Multiply by u_i. Ideal order uses the same submitted list. |
| `gold_recall@k` | Number of distinct positive-relevance catalogue entries matched through k / all positive catalogue entries. Duplicated gold IDs earn no further gain. |
| `gold_ndcg@k` | Gain from the fixed catalogue relevance of each matched gold ID; first match only. Ideal uses the entire fixed catalogue, including unreturned entries. |
| `duplicate_rate` | (n - distinct canonical clusters)/n; withheld unless every prediction has a cluster label. |

Within-list recall/AP/NDCG can be perfect on a single good prediction while
missing most opportunities. Never use them alone to compare systems returning
different candidates. Fixed-k precision penalizes short lists; catalogue
recall and catalogue NDCG measure misses against a common reference. The
catalogue is a reviewed reference set, not a claim to exhaust all possible gaps.
Multiple gaps matching the same future paper are not automatically duplicates;
the paper may address distinct missing pieces.

## Support and judgment metrics

| Report metric | Definition |
| --- | --- |
| `gap_evidence_precision` | Supported links / all reviewed claimed links. |
| `gap_unsupported_link_rate` | Unsupported links / all reviewed claimed links. |
| `fully_supported_gap_rate` | Gaps with a nonempty evidence list and every link supported / n. Requires evidence review of all gaps. |
| `independent_support_papers` | Mean number of distinct supporting paper IDs per reviewed gap, including zero-evidence reviewed gaps. Distinct IDs are a proxy, not proof of independent authors or experiments. |
| `closure_accuracy` | Exact predicted/reference status matches / paired status labels. `unchecked` is a prediction, never a correct reference class. |
| `false_open_rate` | Reference-closed gaps predicted open / reference-closed gaps with a prediction. Partly/unchecked predictions are not counted as false open. Inspect the full confusion table too. |
| `doability_mae` | Mean absolute error between predicted 0..8 score and sum of reference data/compute/blocks/scope 0..2 ratings. |
| `future_brier` | Mean (frozen probability - h_i)^2 on gaps with probabilities, range 0..1, lower is better. |
| `future_ece` | Ten fixed bins [0,.1), ... [.9,1], sum bin_count*abs(mean_probability-observed_hit_fraction)/all probability predictions. |

Brier uses the [binary probability scoring definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html).
The probability target is observed confirmation in the bounded search window,
not universal scientific validity. ECE is descriptive, sample dependent, and
must be read with the counts and reliability bins; it is not a proper scoring
rule. Calibration uses each supplied gap prediction, without canonical
deduplication; report the duplicate rate beside it.

Supported evidence must have its exact quote present in recorded passage text,
and its paper must belong to the pre-cutoff corpus. Quote presence is necessary
but does not prove semantic support. The support flag is an external judgment.
The scorer cannot authenticate submitted text against an absent passage store.
Review every trail link; do not omit failed links or count only verified ones.

Coverage metrics (`gap_judgment_coverage`, `gap_type_coverage`,
`canonical_label_coverage`, `gap_evidence_coverage`, `closure_label_coverage`,
`doability_label_coverage`, `future_probability_coverage`, and
`judgment_coverage@k`) retain labeled/eligible denominators. Missing evidence,
status, rubric, probability, or type is not filled in by model guesses.

The report includes each gap's original rank, temporal gain, quality grade,
canonical ID, catalogue ID/gain, and confirmation status. `by_type` reports
gap counts and original hit rate for all four patterns. `closure_confusion`
records reference classes by predicted classes. Small slices are diagnostic.

## Original scorecard retained

| Metric | Definition |
| --- | --- |
| `hit_rate` | Confirmed proposed gaps / all proposed gaps; unlike top-k metrics, no new canonical deduplication is applied. Empty output is `null`, accompanied by zero yield and zero fixed-k precision. |
| `coverage` | Topics with >=1 confirmed gap / all topics. |
| `lead_time_months` | Mean exact days from T to each hit's earliest confirmed paper, divided by 365.2425/12. Conditional on hits; diagnostic. |
| `answer_accuracy` | Correct answers / all questions, including abstentions. Correctness is an external gold-answer judgment. |
| `evidence_quality` | Gold cited passage IDs / all cited passage IDs, distinct within each answer. Citation precision, not semantic entailment or recall. |
| `cross_document_retrieval` | Multi-paper questions retrieving every required paper / all multi-paper questions. |
| `hallucination_rate` | Unsupported factual claims / all labeled factual claims, including uncited claims. |
| `retrieval_steps` | Total tool calls / questions. |
| `response_seconds` | Total answer-loop elapsed seconds / questions. |
| `mcp_token_savings` | Sum(raw tokens - MCP tokens)/sum(raw tokens) for matched question measurements. Can be negative. |

An answer with no factual claims contributes no hallucination denominator;
this does not make abstention accurate. Correctness and claim completeness
are supplied judgments. The scorer cannot discover omitted claims. Supported
claims must cite passages present among the answer's citations. Efficiency
records preserve tokens, cloud calls, largest context, peak GPU MiB, and
build seconds per topic; they are measurements, not inferred values.

## Missing values and aggregation

Every metric stores numerator, denominator, and value. A zero denominator
produces JSON `null`. Scores must not substitute zero for missing annotations.
Fixed-k precision of an empty output is zero. No-hit recall/NDCG is undefined.
Quality precision/utilities require judgments for every existing top-k item;
quality NDCG requires the entire submitted list labeled. Catalogue metrics
require a catalogue and complete mappings. Full-support and duplicate rates
require complete evidence/cluster labels respectively.

For these completeness-dependent metrics, any topic missing required labels
withholds the run-level value. `incomplete_labels: true` and `available_subset`
expose what could be calculated without presenting it as complete.
`ranking.incomplete_metrics` lists the affected per-topic metrics. A topic
with zero ideal NDCG gain is undefined rather than incompletely labeled.
Evidence precision, closure, rubric error, and calibration deliberately score
their reviewed subset; always report their coverage.

Within each run, pool ratio numerators and denominators across topics.
NDCG, reciprocal rank, success, and yield use one observation per eligible
topic, so this gives a topic mean. Brier, MAE, and ECE pool their recorded
observations; ECE therefore averages topic-local bin errors by prediction
count, not a single globally rebinned reliability diagram. The report preserves
the local bins so this distinction can be checked.

`repeat_summary` reports mean, sample SD, and defined-run count for pooled run
metrics. `topic_macro_summary` first averages repeats within each topic, then
summarizes those topic means; it may describe a labeled subset and must be
read with its observation count. `metric_topic_coverage` reports defined/total
topics per metric in each run. These two weighting schemes answer different
questions; do not mix their numbers in one unlabeled table.

## System comparisons and uncertainty

`--compare LEFT RIGHT` pairs topics by ID. For each metric, average defined
repeats within each topic/system, compute left-minus-right differences, and
exclude pairs undefined on either side. Deltas within 1e-12 of zero are ties.
The two-sided exact sign test uses the positive and negative non-tied counts:
p = min(1, 2*sum(C(n,j), j=0..min(positive,negative))/2^n).
No non-tied pairs gives `null`, not evidence of equivalence.

The confidence interval resamples paired topic deltas with replacement 2,000
times using seed 1729, then reports interpolated 2.5 and 97.5 percentiles of
the mean. Fewer than two paired topics gives `null`. This is a percentile
topic-bootstrap interval, not a gap-level binomial interval or an interval
for independent repeat randomness. With few topics or rare hits, precision
is poor. Report sample sizes and raw deltas.

Holm adjustment covers all defined p-values in this comparison report,
including diagnostics. It is conservative; do not select the smallest raw
p-value across many k values and call it a predeclared primary result.
`metric_directions` specifies higher/lower/diagnostic; a positive delta is
worse for a lower-is-better metric. Lead time and support counts are diagnostic.
There is no automatic winner, deployment decision, or keep/revert action.

Three repeats are the brief's minimum; fewer produce a report warning.
Cached identical outputs demonstrate reproducibility. Zero repeat SD is not
evidence of zero model or labeling uncertainty. Topics, not individual gaps,
claims, questions, or cache replays, are the independent paired-test units.

## Human agreement

Two experienced raters independently score 50 held-out gaps. The scorer groups
reviews by exact (topic ID, statement) across systems/repeats, so reused cached
gaps do not inflate agreement sample size. Near-duplicates with different
statements remain separate; curate the sample accordingly. Conflicting repeated
labels by the same rater on an identical review unit raise an error.

For each rater pair and axis, report paired gaps, exact agreement, within-one
agreement, and linear-weighted Cohen kappa on the fixed 1..5 scale. Disagreement
weight is abs(a-b)/4. Kappa = 1 - observed weighted disagreement / expected
weighted disagreement under independent rater marginals. If expected
disagreement is zero, kappa is `null`, even if constant ratings agree perfectly.
Agreement is reported before adjudication; it measures reliability, not truth.
`human_agreement.rating_summary` and `rating_summary_by_system` additionally
report the raw 1..5 score distributions and mean quality ratings. Raters are
averaged within each reviewed gap before averaging across gaps. The SD is
across gap means, not across correlated rater responses or cache repeats.
Report reviewed counts and sampling design with these summaries; an overall
union sample is not automatically balanced across systems.

## Acceptance and reporting contract

Before research reporting, freeze 20 topics (12 development, 8 held out),
approximately 100 dated pre-cutoff papers per topic, 100 reviewed questions
(50 single-paper and 50 multi-paper over the same corpora), and the matched
future-query protocol. These dataset targets are not silently fabricated or
enforced on tiny scorer fixtures. Verify overall dataset balance separately
across split manifests. Never tune on held-out labels.

Run the brief's vector-only, hybrid-no-graph, context-off, verifier-off, and
full-local systems on the same material. Cloud-assisted and large-hosted
references belong to Test 2 only under this time-split protocol. Freeze one
ablation switch at a time. Preserve all outcomes, including failures and
negative comparisons. ContextBench 32k/8k measurements remain a distinct
external task, not a proxy for gap quality.

Publish the primary endpoints, denominators, topic macro results, repeat
spread, paired deltas/intervals, adjusted tests, support/hallucination rates,
label coverage, list fill, per-pattern slices, human agreement, and efficiency.
The original under-5% unsupported-claim goal and 8,000-token local-call limit
remain project targets, not new quality scores or claims of achieved results.
Do not claim full evaluation is finished until the real corpora, comparisons,
50 manual hit checks, and 50 double-rated held-out gaps have actually run.

The tuning agent's `program.md` keeps scoring frozen. This engineering task
explicitly authorizes constructing the scorer; it does not authorize changing
gold labels to improve an experiment. After acceptance, version scorer changes
and rerun affected systems rather than comparing incompatible score versions.
