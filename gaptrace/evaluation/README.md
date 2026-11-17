# Research track

## Evaluate searched and reviewed gap rankings

The product's `review-gaps` command now checks source relations, searches
scholarly abstracts and assesses closure/feasibility before applying the
existing deterministic ordering. The research adapter compares its original
and revised orders using the existing ranking scorer:

```powershell
python -m evaluation.gap_review data/terminal-demo/gap-review.json --out data/evals/gap-ranking-new
```

The export includes a hash-bound independent annotation template. Supply it
with `--labels` after review to measure actionable precision and catalogue
NDCG/recall. Missing quality labels remain undefined; current closure results
do not become future-confirmation labels. See [the workflow](../docs/GAP_REVIEW.md).

## Autoresearch and context experiments

The implemented local-model loop, upstream CLM code reuse, OKF dataflow,
representation/memory comparisons and runnable commands are described in
[AUTORESEARCH.md](AUTORESEARCH.md). Executed smoke checks are recorded in
[AUTORESEARCH_RESULTS.md](AUTORESEARCH_RESULTS.md). These checks establish
wiring, not improved real gap discovery or an OKF performance benefit.

```powershell
python -m evaluation.autoresearch evaluation/examples/autoresearch-context-smoke.json --tokenizer "$env:TEMP/gaptrace-qwen35-tokenizer.json" --out data/autoresearch-001
python -m pytest evaluation/test_autoresearch.py -q
```

The pinned CLM subset retains its CC BY-NC 4.0 license in the research folder.

## Core-fix validation and reproducible inputs

The original seven failing product contracts have been fixed. Implementation
details and development observations are in [CORE_FIXES.md](../docs/CORE_FIXES.md).
Run the full offline gate into a new directory:

```powershell
python -m evaluation.run --out data/evals/my-new-audit
```

The live driver now saves each topic's normalized future-search pool before
shortlisting or judging (`future-pool-1.json`, etc.). Set a topic's
`future_pool_file` to that file to replay it without searching again; query,
provider, date window and pool limit must match. Generation artifacts include
source fingerprints and explicit repeat/cache limitations. A stricter prompt
on the same local judge remains a correlated recheck.

Unanswerable evaluation questions use `"answerable": false`, an expected
abstention in `gold_answer`, and empty `gold_passage_ids` and
`required_paper_ids`. Answerable questions still require both evidence sets.
This changes only the research manifest, not the product schema.

Actual retrieval baselines are available independently of the product CLI:

```powershell
python -m evaluation.baselines "Your question" --data data/my-topic --mode all --out data/evals/baselines-new --tokenizer PATH_TO_QWEN_TOKENIZER
python -m evaluation.grounding_probe --out data/evals/probe-new --tokenizer PATH_TO_QWEN_TOKENIZER
```

Install the documented `tokenizers` package for exact counts. Baselines use
the local Ollama API and the existing 4B model; vector and hybrid modes use
the existing MiniLM embedding model. Lexical/vector/hybrid rank note summaries;
raw-passages ranks source chunks with BM25. All share the same agent, final
answer checker, context packer and gateway. Their distinction is retrieval and
representation, not a different judge. `--cache disabled` is the default;
`--cache warm` permits cache reuse. `--repeats` makes real answer requests on
a shared corpus, not independent corpus builds. Outputs contain source hashes,
answers, trails, gateway attempts, latency, and actual Ollama input/output counts.
Baseline outputs still require independent grading before scoring research quality.
`full` retains the product tools. The retrieval controls disable graph, gap and
SQL tools and remove structured links/facts from notes. `vector-passages` and
`hybrid-passages` add embedding and hybrid retrieval over source chunks.

`python -m evaluation.freeze SPEC.json --out NEW_DIRECTORY` freezes supplied
development/held-out plans, source JSON, optional search pools and a reviewed
release policy. The spec contains `policy`, `policy_reviewed_by`,
`plans: {development: PATH, held-out: PATH}`, `source_pins`, and optional
`label_reviewers`. Each `source_pins[topic_id][paper_id]` needs an archived or
version-pinned HTTPS `url` and `version`. Paths resolve relative to the spec
and each original plan. Both plans must be benchmark plans using the same
documented training/corpus cutoffs and future window, disjoint topics and
papers, and explicit source JSON. Source passages and gold IDs are checked.
Use `python -m evaluation.freeze --verify FROZEN_DIRECTORY` before a run to
detect changed files. Record an actual completed research-policy review;
the tool does not supply, authenticate or substitute for that review.

The frozen 20-topic/100-question dataset, independently adjudicated labels,
calibrated judge, repeated builds, ContextBench comparison and held-out report
remain research work. No held-out scores or human ratings have been invented.

## Judge calibration and held-out readiness

Audit the workspace without inference:

```powershell
python -m evaluation.readiness --out data/evals/readiness-new
```

Exit 1 means prerequisites are missing or invalid; exit 2 means the audit could
not complete. The output includes `readiness.json` and `HELD_OUT_REPORT.md`,
with unreadable inventory paths recorded explicitly. No quality score is
inferred from smoke results. Supply a frozen snapshot and review/calibration
records to check the intended 12-development/8-held-out topic split,
100 questions (50 single-paper and 50 multi-paper), and at least 100 papers
per topic:

```powershell
python -m evaluation.readiness --frozen FROZEN_DIRECTORY --reviews QUESTION_REVIEWS.json --calibration CALIBRATION_REPORT.json --out data/evals/readiness-reviewed
```

Question reviews contain `label_authors` and `reviews`. Each review records
`topic_id`, `question_id`, `reviewer`, `reviewer_type: "human"`,
`decision: "accepted"`, `reason`, and `label_sha256` calculated with
`evaluation.readiness.label_hash(topic_id, question)`. Each question needs two
distinct reviewers independent of its authors. Declarations do not authenticate
human identity. A snapshot also needs `model-provenance.json` recording `model`,
`model_digest`, `tokenizer_sha256`, `training_cutoff` and
`training_cutoff_reference`; its cutoff must match both plans.

Calibration controls use the `CalibrationSet` schema in
[calibration.py](calibration.py). Cover answer correctness, claim support and
future-gap matching, with separate `development-calibration` and
`development-validation` controls. Every control requires two independent human
reviews; disputed labels require a separate adjudicator. Labels and reviews
never enter judge prompts. Live calibration reuses the production grading
methods and matcher prompt, recording model digest, prompt hash, operating
thresholds and inference usage:

```powershell
python -m evaluation.calibration CONTROLS.json --tokenizer PATH_TO_QWEN_TOKENIZER --out data/evals/calibration-new
```

Payloads contain `question`, `gold_answer`, `answer`, `passages` for correctness;
`claim`, `passages` for support; and `cutoff`, `gap`, `work` for matching.
The corresponding production methods in `live_services.py` define their nested
formats. Minimum coverage is ten positive and ten negative controls per task
per development split. Reports retain confusion counts and Wilson intervals;
passing point targets does not establish that the interval bounds pass them.
`--observations FILE` scores supplied observations without live inference, but
that report cannot satisfy the production-judge compatibility check.

The prototype policy currently expects 20 topics/100 questions in a held-out
manifest. Review a policy scoped to the eight held-out topics before freezing
and executing the protocol. Keep the global 20-topic/100-question requirement
in the readiness audit. Independent builds/repeats and complete system and
ablation comparisons remain necessary after prerequisite checks pass.

## Gap ranking evaluation

The scorer now evaluates the frozen ranked list as well as the original
time-split and question scorecards. Start with the complete
[metric specification](METRICS.md) for equations, denominators, aggregation,
missing-label rules, statistical tests, and interpretation. The
[annotation guide](ANNOTATION.md) defines the real/open/doable anchors,
evidence review, closure labels, and blinded two-rater workflow.
[Evaluation status](results.md) separates tested scorer behavior from research
results that still require a benchmark dataset and independent ratings.

```powershell
python evaluation/metrics.py evaluation/examples/ranking-fixture.json --out ranking-report.json
python evaluation/review.py export evaluation/examples/ranking-fixture.json --sample-size 3 --out review-example
python -m pytest evaluation/test_metrics.py evaluation/test_ranking.py evaluation/test_live.py -q
```

The example and its labels are synthetic. The JSON Schema for the full input
contract is [manifest.schema.json](manifest.schema.json). Existing manifests
remain accepted. Optional annotations unlock additional metrics; absent
annotations produce `null`, never invented quality labels. Set
`gold_mapping_complete: true` for a topic run only after all of its predictions
have been matched or rejected against the fixed gold catalogue.

List order is rank. The scorer does not replace or tune the product's ranking
function. It uses no model calls or new dependencies. The live runner preserves
gap type, predicted closure status, and predicted doability when provided by
the public CLI; independent annotation is a separate evaluation step.

Nothing in this folder is imported by the product. The harness drives the
finished product through its command line and API, using the date-range
option and config/settings.yaml like any user.

## Fast release checks

Run from the repository root with the existing development dependencies:

```powershell
python -m evaluation.run --out data/evals/check-001
```

Choose a new output directory each time. The runner executes the existing
product tests, scorer tests, and independent adversarial contracts in one
invocation, with a 180-second outer timeout. It produces `report.json`,
`junit.xml`, `pytest.log`, and `summary.md`. The report records case-level
failures, retrieval measurements, elapsed time, package versions, and SHA-256
hashes of product code, prompts, settings, tests, and evaluation inputs. A
source change during the run invalidates the result. Exit codes are **0**
for all checks passed, **1** for failed or skipped checks, and **2** for a
harness error. Setup errors and an empty suite cannot pass. Existing outputs
are never overwritten. `python -m pytest -q` also discovers evaluation tests.

The additional `contracts/` suite uses a separate invented corpus and fixed
gold retrieval labels. It covers all four gap patterns and negative controls,
verified versus unverified evidence, exact conflict boundaries, insertion-order
stability, citations, fabricated numbers, missing source passages, bounded
agent loops, extraction quotes, cross-paper result identity, closure failures,
cloud retry limits, cache invalidation, and budget checks before initial and
retry calls. Eight lexical retrieval controls record recall@3 and reciprocal
rank, including an absent query and a query requiring two notes.

Scripted backends deliberately emit bad outputs to test product enforcement.
These results do **not** estimate model accuracy, embedding quality, prompt
injection resistance, or performance on real scientific topics. Budget boundary
checks use a deterministic counter; a live run must also inspect the actual
model tokenizer and server token usage. Known product violations remain failing
assertions, with assignments in [improvement.md](../../improvement.md).
Do not skip or relabel those cases to turn the release gate green.

For targeted debugging:

```powershell
python -m pytest evaluation/contracts/test_answer_contracts.py -q
python -m pytest evaluation/contracts/test_gap_contracts.py -q
python -m pytest evaluation/test_run.py evaluation/test_release.py -q
```

## Gate a recorded research run

Keep the existing live runner and scorer below for expensive model evaluation.
After collecting and independently reviewing the held-out labels, apply the
frozen acceptance policy:

```powershell
python -m evaluation.release path/to/manifest.json --system gaptrace-local --out data/release-check.json
```

`release-policy.json` contains **initial proposed targets**, not measured
performance: at least 20 topics, 100 distinct question definitions, three
repeats, and 100 labeled claims per repeat. It requires complete gap judgments,
canonical labels and evidence annotations, fully supported gap evidence,
answer accuracy >= 0.80, citation precision >= 0.95, multi-paper retrieval
>= 0.80, actionable precision@5 >= 0.60, duplicate rate <= 0.10, and zero
observed unsupported claims. The research lead should approve and freeze
these values before using held-out data. `--policy` selects a different
versioned policy. Both the manifest and policy hashes are retained.

Every repeat must meet every target. An undefined metric fails, so refusing
everything or omitting labels cannot pass. Development and historical smoke
manifests fail even when their reported metrics are perfect. The release gate
is an acceptance check; use paired topic comparisons and topic bootstrap
intervals from `metrics.py` for improvement claims. Repeated cached generations
do not estimate sampling uncertainty. Independently review judge decisions;
an exact quote establishes provenance but cannot by itself establish entailment.

Use the cheap offline gate for every change. Run the fixed development live
set after prompt, model, retrieval or extraction changes, using the same corpus,
questions and labels across systems. Keep held-out topics untouched during
tuning and run them for a release decision. A real benchmark still needs the
topic collection, baselines, and blinded human review described below; invented
contracts and the saved smoke run cannot replace that work.

## Score recorded runs

The first implementation scores recorded results for the time-split gap
benchmark and the retrieval question set. It takes a JSON manifest and emits
a machine-readable report. The fixture is synthetic and exercises both
scorecards. Its scores are for checking the scorer.

From the repository root, with the project's Python dependencies installed:

```powershell
python evaluation/metrics.py evaluation/fixture.json --compare full vector --out evaluation-report.json
python -m pytest evaluation/test_metrics.py -q
```

Use [`fixture.json`](fixture.json) as the input example. Keep separate manifests
for development and held-out topics. Scoring the latter requires `--split held-out`.
Record three runs per system for benchmark reporting. Each run uses the same
questions and corpus per topic. Include an answer record for every question,
even for refusals, and label every factual claim, including uncited claims.
Record the grader and passage IDs used for claim support. Answer correctness
and claim support are supplied judgments. The scorer computes the metrics from
them.

| Metric | Calculation |
| --- | --- |
| Hit rate | Confirmed gaps divided by all proposed gaps |
| Coverage | Topics with a confirmed gap divided by all topics |
| Lead time | Mean elapsed time from cutoff to the earliest confirmed candidate per hit. Exact days are kept per hit and months use 365.2425 days per year. |
| Answer accuracy | Correct answers divided by all questions |
| Evidence quality | Gold citations divided by all cited passages, counting distinct IDs within each answer |
| Cross-document retrieval | Multi-paper questions that retrieved every required paper divided by all multi-paper questions |
| Hallucination rate | Unsupported claims divided by all labeled factual claims |
| Retrieval steps and response time | Mean tool calls and elapsed seconds per question |
| MCP token savings | Tokens saved divided by raw-passage tokens across questions with a matched measurement |

A gap counts once if one of its five or fewer shortlisted future papers
passes both judgments. The first-pass and strict-recheck confidence thresholds
default to 0.60 and 0.80 and can be fixed in the manifest before a run.
The scorer rejects post-cutoff papers in the recorded corpus, out-of-window
future candidates, cloud-enabled time-split builds, and missing strict rechecks
for first-pass hits. The date window must span at least nine calendar months,
and the corpus cutoff must be on or after the model training cutoff.

The report preserves denominators, uses `null` for undefined metrics, and
aggregates ratios by pooling counts across topics within each run. NDCG,
success, and reciprocal rank are topic-level observations, averaged over
defined topics. Label-dependent primary metrics are withheld if required
labels are missing in any topic; `available_subset` retains diagnostics.
Repeat summaries
report the mean and sample standard deviation across runs. `--compare` pairs
topics by ID, averages repeats within each topic and system, and runs a
two-sided paired sign test on non-tied topic differences for each metric.
Comparisons also include seeded topic-bootstrap intervals and Holm-adjusted
p-values. Metric directions and annotation coverage are explicit in the report.
The report includes any supplied efficiency measurements per topic and a
SHA-256 of the input manifest.

## Benchmark runs on the finished product

### Run a real-paper smoke check

With Ollama running, install LiteLLM and the Qwen tokenizer using the main
README. Pull the small embedding model, then run the included plan from the
repository root. It imports a dated excerpt from the first version of the
Transformer paper, asks one gold question, and searches a separate year of
later literature. This historical replay checks the wiring. The cutoff
predates the tested model, so use its results for integration testing.
[Source paper, version 1](https://arxiv.org/pdf/1706.03762v1)

```powershell
ollama pull all-minilm:l6-v2
python -m evaluation.live evaluation/examples/smoke-plan.json --tokenizer "$env:TEMP\qwen35-tokenizer.json" --out data/evaluation-smoke-run
```

Choose a new `--out` directory for each run. The command writes `plan.json`,
`generation.json` with the product's original outputs and cited text,
`manifest.json` with the graded records, and `report.json` with the metrics.
`--ollama-url` selects the loopback judge and embedding server. GapTrace's
own local-only CLI uses its default loopback server on port 11434.

### Prepare benchmark inputs

Use [the smoke plan](examples/smoke-plan.json) as the input format. For a
benchmark, set `mode` to `benchmark`, supply a documented
`model_training_cutoff`, and set the corpus cutoff on or after it. Point each
topic at a new `data` directory and a dated `source` JSON with sectioned text.
Paths are relative to the plan. Use version-pinned or archived pre-cutoff
text for the source files. The driver can also reuse a corpus imported
through `gaptrace ingest` at the matching cutoff. Review the gold questions
and record their exact passage and paper IDs before running it.

The runner calls the public GapTrace CLI and verifies the imported paper and
passage inventory. It freezes all gap and answer outputs before contacting
the future index. Each topic specifies a query and either `openalex` or
`arxiv` as `future_provider`. OpenAlex accepts `OPENALEX_API_KEY` when
available. The arXiv adapter uses submission dates and skips abstracts whose
returned revision falls after the window. Both providers are queried for a
bounded pool. The driver reranks titles and abstracts by MiniLM cosine
similarity and sends up to five per gap to a first judge and a stricter
recheck for first-pass hits.
[OpenAlex date filters](https://help.openalex.org/api/filtering/) · [arXiv API manual](https://info.arxiv.org/help/api/user-manual.html)

Answer grading compares with the gold answer and checks each claim against
the passages backing the cited note IDs. A claim marked supported must
include a quote found in a cited passage. Retrieved paper IDs are derived
from the notes and tool results shown to the agent. Answer times cover the
agent loop, including tool calls. The runner defaults to three answer
repeats for a benchmark and one for a smoke check. Repeat runs share the
same imported corpus and deterministic gap list. Keep the gold labels,
corpora and future-query setup fixed across system comparisons.

The full 20-topic corpus, 100-question set, product baseline adapters,
ContextBench runs, and blinded human rating still need to be assembled
before reporting research results.

R1. Harness: build each topic with a cutoff date and `cloud.enabled: false`;
fail if any paper after the cutoff is present; match proposed gaps to later
papers (embedding search, then the matcher prompt); re-judge every hit at a
stricter threshold; report hit rate, coverage and lead time per topic. Add
the 100-question comparison (accuracy, evidence quality, cross-document
retrieval, hallucination rate, steps, response time) and the baselines.

R2. Runs: all systems on development topics, three repeats; ablations by
flipping one switch in settings.yaml at a time; ContextBench adapter; export
of the human-rating sample; results.md.

R3. Improvement loop: `autoresearch.py` and `program.md` in this folder.
