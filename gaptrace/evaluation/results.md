# Evaluation engineering status

## November 17 core-fix update

The seven original product-contract failures listed later in this document
have been fixed. The latest complete offline audit has **214 passing checks**
and no failed/skipped checks, with a stable source snapshot. The final local
4B checker development probe passed 9/9 cases, and four executable retrieval
baselines returned supported answers on the one-excerpt real-paper smoke.
See [CORE_FIXES.md](../docs/CORE_FIXES.md) for artifacts, observed costs and
limitations. These results do not complete the independent research benchmark.
The earlier status and counts below are historical snapshots.

The subsequent autoresearch and context experiment implementation is described
in [AUTORESEARCH.md](AUTORESEARCH.md), with executed local-model checks in
[AUTORESEARCH_RESULTS.md](AUTORESEARCH_RESULTS.md). The historical scorer
validation snapshot below predates that additional work.

The gap-ranking metric implementation is complete for recorded manifests.
The code, input contract, independent-label workflow, statistical comparisons,
documentation, and synthetic validation are available in this folder.
Real benchmark findings have not been measured by this work.

## Delivered

| Component | Location |
| --- | --- |
| Existing temporal and question metrics, strict input validation, report CLI | `metrics.py` |
| Frozen-order ranking metrics, gold catalogue scoring, evidence/closure/doability/calibration metrics, human summaries and agreement | `ranking.py` |
| Blinded sample export and validated two-rater import | `review.py` |
| Complete equations, aggregation and interpretation contract | `METRICS.md` |
| Rating anchors, feasibility rubric, evidence/closure review, sampling workflow | `ANNOTATION.md` |
| Machine-readable input contract | `manifest.schema.json` |
| Annotated synthetic two-system, three-repeat manifest | `examples/ranking-fixture.json` |
| Reproducible full synthetic report | `examples/ranking-report.json` |
| Fixture/schema regeneration | `examples/make_ranking_fixture.py` |
| Hand-calculated and adversarial scorer/review tests | `test_ranking.py`, alongside existing scorer/live tests |

No product ranking algorithm, product schema, model budget, prompts, or
dependencies were changed for this work. The live evaluator now retains
available product type, predicted status, and predicted doability metadata.

## Validation

On 16 November 2026, this command passed **74 tests**, including 44 scorer,
review, and live-harness tests and 30 existing product regression tests:

```powershell
python -m pytest evaluation/test_metrics.py evaluation/test_ranking.py evaluation/test_live.py tests -q
```

The checks cover hand-calculated rank discounting, rank sensitivity,
duplicate penalties, incomplete labels, empty lists, fixed catalogue misses,
zero ideal gain, unsupported evidence, out-of-corpus sources, probability
boundaries and nonfinite input, closure confusion, feasibility error,
weighted kappa, repeat deduplication, exact sign-test probabilities, Holm
correction, deterministic bootstrap intervals, blinded sampling, corrupted
review forms, CLI import/export/scoring, and held-out split selection.

A final broader discovery run (`python -m pytest evaluation tests -q`) found
**141 passed and 7 failed**. All failures were in product contract tests being
added concurrently under `evaluation/contracts/`, covering these behaviors:

- Incorrect numeric answers accepted with real citations.
- Cited notes accepted without source passages.
- Closure marked open when no candidate was judged.
- Closure marked open after invalid judge responses.
- Unverified numeric report links creating conflict gaps.
- Cloud schema retries exceeding the per-topic call cap.
- Result merging erasing a distinct paper's measured value.

Those product fixes are outside this scoring implementation. The targeted
74-test success does not assert that those separate contracts or a full live
benchmark pass. These counts are a snapshot of a workspace with concurrent
engineering changes.

## Synthetic example only

Reproduce the example report from the repository root:

```powershell
python -m evaluation.examples.make_ranking_fixture
python evaluation/metrics.py evaluation/examples/ranking-fixture.json --compare full vector --out evaluation/examples/ranking-report.json
```

The fixture contains two synthetic topics, three unique statements, two
systems, and three identical recorded repeats per system. All labels and
source text are invented solely to exercise evaluation. The names `full`
and `vector` do not represent executed system comparisons here.

| Metric | Synthetic full | Synthetic vector |
| --- | ---: | ---: |
| Original hit rate | 0.50 | 0.00 |
| Future precision at 5 | 0.10 | 0.00 |
| Future reciprocal rank averaged over topics | 0.50 | 0.00 |
| Actionable precision at 5 | 0.10 | 0.00 |
| Fixed catalogue recall at 5 | 1.00 | 0.00 |
| Fixed catalogue NDCG at 5 | 1.00 | 0.00 |
| Evidence link precision | 1.00 | 0.00 |
| False-open rate | undefined, no reference-closed gaps | 1.00 |
| Brier loss on supplied gap probabilities | 0.05125 | 0.49 |

Fixed-k precision includes the empty topic's five review slots, which is why
it differs from hit rate. The gold catalogue is empty for that topic, so it
does not add catalogue-recall denominator or ideal gain. This deliberate
example demonstrates differing denominators, not a performance claim.
All repeat SDs for defined example metrics are zero because the records are
copies. Human agreement counts three unique reviewed gaps, not nine repeated
copies. The example's sample is too small to establish real system quality.

## Inputs and runs still required

Other engineering and research work must provide the fixed development and
held-out topic lists, version-pinned corpora, documented generating-model
training cutoffs, gold questions/passages, independently reviewed catalogue,
complete source/closure/quality labels, and comparable system runs. The brief
calls for 20 topics split 12/8, about 100 papers per topic, 100 questions split
equally between single- and multi-paper retrieval, three repeats per system,
50 manual hit checks, and 50 held-out gaps reviewed by two experienced raters.

Baseline execution, ablation execution, ContextBench integration, and live
research findings remain separate deliverables. Their absence does not prevent
using these metrics on recorded runs; it prevents claiming the complete
research experiment has been performed. Preserve negative results and report
coverage, denominators, uncertainty, annotation reliability, and provenance
with every real comparison.
