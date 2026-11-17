# Core correctness fixes

Implemented against [improvement.md](../../improvement.md), November 17, 2026.
The public product schema, configured token slots, Qwen3.5 4B selection,
four gap patterns and ranking formula are unchanged. No cloud inference was
used for development validation. Existing corpora and pre-fix artifacts were
retained.

## Implemented behavior

| Area | Result |
| --- | --- |
| Result identity | Result IDs include source paper, normalized name and full fact tuple. Different measurements survive same-name extraction and merging. Reports links are repaired using paper, passage and quote provenance; ambiguous links are withheld. |
| Conflict candidates | Only compatible-unit observations with verified reports from their actual source papers participate. Orphans, missing quotes and unverified extreme values cannot create or inflate a conflict. Ingestion and workspace loading check passage ownership and quote/value integrity. |
| Final answers | Every returned sentence is checked against actual source passages using a dedicated local entailment prompt. Unsupported sentences and unused citations are removed. No supported sentence means the existing abstention response. Multi-paper premises can be checked together; short fragments retain question context. |
| Gateway attempts | Every dispatched cloud attempt counts, including exceptions and JSON repair. The existing cap applies to each attempt; fallback uses the local gate and serving-tier cache. A logical request has at most one schema repair across fallback. |
| Closure | Empty or failed judgments remain unchecked. Negative judgments establish bounded openness only when the configured searches/judgments complete without failure. Partial/full positive evidence is retained; diagnostics record failures and search scope. |

The checker has a provisional limit of 24 sentence/source checks, at most
eight candidate chunks per sentence, and at most four chunks in a combined
comparison. The gateway's existing repair rule bounds dispatches to twice
the logical check count. Missing evidence, invalid verification output and
exhausted allowance withhold the affected sentence. These limits need broader
development calibration before being treated as an optimal operating point.
The real tokenizer is used in the development probes and ordinary product
commands. The bundled Qwen tokenizer is selected in settings; entry points
report an error if it cannot be loaded instead of silently estimating.

`answer_validation` and `answer_validation_summary` trail events retain decisions,
source IDs, rejected text, added calls, cache hits, repairs, token counts and
latency. Gateway token counts exclude server framing; the live probes separately
record actual Ollama counts. CLI JSON citations include the passages actually
used by successful validation, including evidence fetched after the draft answer.

## Verification artifacts

| Check | Recorded result | Artifact |
| --- | --- | --- |
| Fresh pre-fix complete audit | 144 passed, 7 failed | [Baseline](../data/evals/before-core-fixes-001/report.json) |
| Final complete offline audit | 214 passed, no failures, no skips; unchanged source during execution | [Report](../data/evals/after-core-fixes-002/report.json), [summary](../data/evals/after-core-fixes-002/summary.md), [JUnit](../data/evals/after-core-fixes-002/junit.xml) |
| Research-tooling continuation audit | 230 passed, no failures or skips; unchanged source during execution | [Report](../data/evals/research-tooling-002/report.json), [summary](../data/evals/research-tooling-002/summary.md) |
| Final local 4B development controls | 9/9; cache disabled; 7 actual verifier calls; 2,936 server input and 606 output tokens | [Probe](../data/evals/grounding-probe-003/report.json), [per-case records](../data/evals/grounding-probe-003/records.jsonl) |
| Real-paper retrieval baseline smoke | All four modes returned supported answers citing the source passage | [Outputs](../data/evals/live-baselines-002/generation.json), [provenance](../data/evals/live-baselines-002/provenance.json) |

The nine development controls cover supported claims, invented numbers,
misattributed methods, unsupported qualifiers, paper instructions, equivalent
numeric formatting, mixed sentences, absent evidence and short answer fragments.
Median case latency was 4.79 seconds; the observed p95/max was 16.47 seconds,
including the first model request/load. Cases with missing evidence or invented
numbers needed no model dispatch. This is a tiny engineering sample, not a
population latency estimate or independent research-quality measurement.

The first prompt version rejected equivalent number formatting. A later raw-text
smoke answer was also falsely rejected for lacking context. Both failures and
their original artifacts remain available in `grounding-probe-001` and
`live-baselines-001`; the revised prompt explicitly allows equivalent formatting
and interprets fragments in their question context.

The real-paper smoke corpus contains one Transformer-paper excerpt and one
question. All baseline answer runs used the same local 4B model/checker with
disk response caching disabled. They are functional checks, not a retrieval
quality comparison:

| Mode | Answer + validation seconds | Model calls | Server input/output tokens |
| --- | ---: | ---: | ---: |
| Lexical notes | 8.49 | 2 | 760 / 150 |
| Vector notes | 8.00 | 2 | 788 / 145 |
| Hybrid notes | 8.55 | 2 | 788 / 145 |
| Raw passages with BM25 | 7.03 | 2 | 738 / 126 |

## Research tooling and unfinished acceptance work

Executable baseline adapters, source fingerprints, frozen future-search pool
replay, explicit unanswerable questions, dataset/policy snapshots and hash
verification are implemented under `evaluation/`. See its README for commands.
The existing independent two-rater export/import workflow and scorer remain
available; no synthetic label was relabeled as human or independent evidence.

The continuation also verifies development-only judge calibration, independent
review declarations, stable question-label fingerprints and restricted retrieval
baselines. Commands and input requirements are documented in the
[evaluation guide](../evaluation/README.md#judge-calibration-and-held-out-readiness).
The [workspace readiness audit](../data/evals/held-out-readiness-001/HELD_OUT_REPORT.md)
found no frozen snapshot candidates among 14 discovered plan files; two inventory
paths were unreadable and are listed in its JSON. Frozen benchmark inputs,
independent reviews, compatible calibration and model-cutoff provenance were
not supplied. No held-out inference or quality score was produced. The first
continuation audit encountered pytest temporary-directory permission errors;
its original artifacts remain in `research-tooling-001`. The complete rerun
with the required filesystem access passed all 230 checks.

The independent research benchmark is not complete. It still needs a curated,
version-pinned 20-topic/100-question collection; a documented model training
cutoff; independent adjudication and judge calibration; reviewed release targets;
independent build/generation repeats; a full ContextBench comparison; and a
held-out report. GPU memory and cold/warm-cache measurements also need a larger
controlled run. No held-out inference was performed and release targets were
not weakened. The saved historical smoke manifest remains unsuitable for
research-release validation.

For existing workspaces, re-ingest the original paper text into a new directory
when checking the result-identity fix. Rebuilding SQLite or indexes cannot
recover a measured value already lost from an old bundle. The regression suite
covers fresh builds and saved/reloaded bundles while preserving the originals.
