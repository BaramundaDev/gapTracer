# Gap evaluation annotation guide

This guide fixes the judgments required by the metric specification. Reviewers
assess the missing work as of the corpus cutoff, using version-pinned sources.
They do not use the later-paper match to decide whether a gap was valid at T.
Keep temporal grading, source support, quality adjudication, and independent
human ratings as separate records. Record reasons and source IDs for decisions.

## Freeze and review workflow

1. Freeze topic split, source corpus, cutoffs, generating model/settings/prompts,
   candidate order, and any predicted probability/status/doability. Archive the
   raw generation. No future paper is available during this step.
2. Review the entire claimed evidence trail. Attach `evidence` records for
   every link, including unsupported links. Use `[]` when reviewed but empty;
   use `null` when not yet reviewed. Validate exact source text against the
   passage store independently of the scorer.
3. Search for closure as of T, including nearest papers inside and outside
   the corpus, with date restrictions. Record search queries, providers,
   dates, coverage limits, and deciding citations in the annotation log.
   Absence from one search is not proof of global novelty.
4. Independently cluster semantically equivalent gaps. Assign every gap a
   `canonical_id`, including unique singletons. Distinct wording is not proof
   of distinct missing work. Do not cluster merely because two gaps match the
   same later paper. Retain disagreements for adjudication.
5. Have two experienced raters independently review a frozen sample using the
   anchors below. Keep system identity, original rank, automated support
   verdicts, product scores, future matches, and the other rater's labels hidden.
6. Calculate agreement from the independent ratings. An adjudicator resolves
   disagreements separately in `judgment`; preserve both original ratings.
   A mean of two ordinal labels is not an adjudicated integer judgment.
7. Separately grade the future shortlist and recheck every first-pass hit.
   Manually check 50 automated hits (or all when fewer exist, stating the
   reduced count). Record errors and disagreements, never silently overwrite
   an earlier published score.

Do not use a sample of 50 quality ratings as if every returned gap was judged.
Full-list quality endpoints remain undefined on unreviewed lists. A random
sample can support descriptive human-quality reporting and agreement; a
complete top-k review per evaluated topic/system is needed for quality
precision, and the complete candidate set for quality NDCG.

## Human rating anchors

Ratings are integers from 1 to 5. Use the same field-specific interpretation
for every system. Explain each decision. If sources are insufficient to rate,
resolve missing material before marking the form complete; do not substitute
a midpoint for missing information. Hold a small development calibration
session, revise ambiguous instructions, and freeze the guide before held-out
review. Do not calibrate using the final held-out sample.

| Score | Real | Open as of T | Doable for a small team |
| --- | --- | --- | --- |
| 1 | Claim is contradicted, fabricated, or not a coherent research gap. | Specific missing work is already fully addressed. | Not feasible with the stated resources or timeframe. |
| 2 | Serious mismatch between the statement and its evidence. | Substantially addressed; little missing work remains. | Major unavailable resources or unresolved prerequisites. |
| 3 | Plausible, but evidence or formulation leaves material uncertainty. | Partly addressed or novelty remains materially uncertain after review. | Feasible after substantial preparation, narrowing, or access work. |
| 4 | Clear missing piece supported by relevant sources. | Specific missing work remains after a documented closure search. | A credible bounded plan with accessible resources and manageable preparation. |
| 5 | Specific, well-supported missing piece with compelling corroboration. | Strong documented evidence of a clearly defined unresolved missing piece within the reviewed literature. | Immediately actionable bounded experiment with ready data/code and suitable compute. |

The labels assess the statement actually returned, not a better proposal the
reviewer could invent. Popularity, eloquence, model confidence, and the number
of citations are not substitutes for support. A graph's missing edge alone
does not prove a research gap; it may reflect extraction failure.

`actionable_precision@k` counts real>=4 AND open>=4 AND doable>=4. The quality
NDCG grade is 0 when real<4 OR open<4; otherwise 1 for doable 1..2, 2 for
doable 3, and 3 for doable 4..5. Freeze these thresholds before inspecting
system results. Report axis utilities too so a threshold does not hide nuance.

## Doability reference rubric

The product's predicted total is 0..8. For a reference total, score each
component independently using the original brief's rubric; sum the four.
Save reasons and supporting resource links in the annotation log. This
reference is optional and distinct from the human 1..5 holistic doable label.

| Component | 0 | 1 | 2 |
| --- | --- | --- | --- |
| `data` | No public data. | Public data needs preparation. | Public data is ready. |
| `compute` | Cluster required. | Rented GPU required. | One consumer GPU suffices. |
| `blocks` | No public method code. | Partial code. | Required method code is public. |
| `scope` | More than one year. | Three to twelve months. | Under three months for a small team. |

Fix the reference resource envelope (team size, GPU, access assumptions) in
the run's annotation log. Source availability can change; use availability
as of T for historical claims and record retrieval/archive dates.

## Closure and source support

`reference_status` has three possible labels. `open` means the specific work
is unaddressed within the documented review scope as of T; `partly` means
existing work fills part of it; `closed` means the described missing work has
already been done. Future papers after T cannot establish closed-at-T status.
The holistic openness rating also captures uncertainty and strength of review;
do not mechanically derive it from the status alone.

For each evidence link, check that its exact quote exists in the supplied
pre-cutoff passage and entails the claimed relationship. Mark unsupported for
a relevant-sounding quote that does not entail that relationship. Keep
unsupported links in the denominator. A copied quote can still be
unsupported. The scorer checks literal substring presence for positive
support labels, not semantic entailment. It checks source paper membership,
not authenticity of supplied passage text.

For conflicting results, check method, dataset, metric, units, evaluation
conditions, and whether variance/protocol differences explain the values.
For an untested combination, verify that the combination is scientifically
meaningful and genuinely absent from the review scope. For limitations and
open questions, confirm the paper actually states the missing piece and that
later pre-cutoff work has not addressed it. Record counterevidence.

## Fixed gold catalogue

Build a topic-level reference pool independently, using source review and a
blinded union of system candidates if necessary. A union pool must be fixed
and judged before comparisons are scored; disclose that it may omit gaps no
system generated. Give each distinct missing piece a gold ID and relevance
0..3 under the same quality-grade definition. Include known non-relevant
entries at grade 0 when useful for mapping audits.

Review every returned gap against this same pool. Match only the same
specific missing piece; broad topical overlap is insufficient. A many-to-one
mapping is allowed and receives one catalogue gain. Keep disputed mappings
unresolved and leave `gold_mapping_complete` false until adjudicated. A null
gold ID after complete review is an explicit nonmatch. The scorer does not
invent semantic matches or treat an unfinished mapping as complete.

## Blinded review commands

Export a uniform random sample without replacement from unique exact
(topic ID, statement) units. Cached repeats and identical statements across
systems share one review unit. Near-duplicates with different wording remain
separate, so inspect population composition before freezing the sample.
The exporter records seed, population size, sample size, and manifest hash.
It is not stratified by system, gap type, or topic; do not describe it as
balanced. For a specifically balanced study, construct a predeclared population
manifest and document its selection independently.

```powershell
python evaluation/review.py export held-out.json --split held-out --sample-size 50 --seed 1729 --out human-review
```

The output contains:

- `items.json`: review IDs, topics, statements, and supplied source passages
  and quotes. It omits ranks, system names, original gap IDs, predictions,
  automatic labels, and later-paper candidates.
- `rater-1.json` and `rater-2.json`: independently shuffled forms. Set a stable
  pseudonymous rater ID and fill each real/open/doable score and reason.
- `private-key.json`: mapping from review IDs to every original run/topic/gap.
  The coordinator retains this file. Do not give it to the raters.

Only include pre-cutoff material with raters' items; provide a separate dated
closure-review dossier when needed. The exporter cannot produce source text
that has not been attached to the manifest. Statements can contain stylistic
identity clues, so this is identity-masked review, not a guarantee that raters
cannot guess the generating system. Avoid system names embedded in topic IDs.

After both raters finish independently:

```powershell
python evaluation/review.py merge held-out.json --split held-out --key human-review/private-key.json --ratings human-review/rater-1.json human-review/rater-2.json --out held-out-rated.json
python evaluation/metrics.py held-out-rated.json --split held-out --out held-out-rated-report.json
```

Import requires the exact original manifest hash, distinct raters, known
unique review IDs, and a completed rating for every sampled item in each
supplied form. Scores must be integers 1..5. It refuses an existing output
file or duplicate existing rater label. Ratings propagate to identical units
across repeats; agreement counts each exact unit once. The original gap order
is preserved. Human ratings do not alter future matcher verdicts or add
adjudicated quality labels automatically.

Read `human_agreement` in the report for per-axis weighted kappa and exact/
within-one agreement. All-constant identical labels have undefined kappa;
this is reported as null, not a perfect reliability estimate. Preserve the
private key and original forms with the evaluation archive, and share
pseudonymized records according to the project's data policy.
