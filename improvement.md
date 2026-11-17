# GapTrace improvements to assign

## Implementation update — November 17, 2026

The five localized product fixes described below are now implemented. A fresh
baseline reproduced **144 passed / 7 failed**; the final complete offline audit
records **214 passed, zero failures or skips**, with no source changes during
execution. The final local Qwen 4B development probe passed **9/9** controls.
Lexical, vector, hybrid and raw-passage adapters were also executed against
the existing real-paper smoke corpus. These checks establish engineering
behavior on their inputs, not broad research accuracy.

See [implementation details and remaining work](gaptrace/docs/CORE_FIXES.md),
[final offline report](gaptrace/data/evals/after-core-fixes-002/report.json),
[local model probe](gaptrace/data/evals/grounding-probe-003/report.json), and
[baseline outputs](gaptrace/data/evals/live-baselines-002/generation.json).

Research infrastructure now includes frozen search-pool replay, source hashes,
executable baselines, explicit unanswerable-question records and dataset/policy
snapshot verification. The independently curated benchmark, human adjudication,
reviewed policy, full comparisons and held-out research report remain unfinished.
The assignment and pre-fix findings below are retained as historical context.

## Chief engineer recommendation and scope

Fix these as localized correctness defects in the existing pipeline. Keep the
small local model, gateway, graph-based four-pattern miner, hybrid retrieval,
SQLite facts, note bundles, passage store, stateless agent steps and code-only
ranking. The fixes below need no replacement framework, vector database,
larger context window, fine-tuning job, new service or mandatory cloud judge.
Keep the public schemas, CLI/API response shapes and configured token slots.
Private helpers and prompt text files may be added within the existing modules.
Keep benchmark logic in `evaluation/`.

These are implementation recommendations for assignment, not completed fixes.
The seven failures cited below are the recorded audit results; this document
update does not claim a new product run or an observed production failure rate.

| Order | Localized change | Engineer owner | Completion evidence |
| --- | --- | --- | --- |
| 1 | Preserve result identity before merge | Ingestion/data pipeline engineer | Both paper measurements and correctly targeted reports links survive |
| 2 | Filter conflict inputs by verified provenance | Knowledge graph/backend engineer | Valid conflict remains; unverified or orphan observations cannot create one |
| 3 | Validate final answer sentences against passages | RAG/applied ML engineer | False claims removed, correct answers retained, measured local inference cost |
| 4 | Apply attempt accounting to gateway retries | Inference platform engineer | Provider attempts never exceed the existing topic cap |
| 5 | Make closure uncertainty explicit | Research reasoning/backend engineer | Empty, failed, partial and complete searches follow the decision table below |
| 6 | Independently validate model quality | Evaluation engineer + research scientist | Frozen development/held-out corpora, audited labels and reproducible comparisons |

Ship small changes independently after their focused checks pass. Complete
result identity before assessing conflict recall: values lost during merging
cannot be recovered by changing the miner. The answer, gateway and closure
fixes can otherwise be developed independently. The evaluation owner reviews
all patches against both negative and positive controls.

The project is a local-model research-gap finder. Its core promises are an
8,000-token local-call ceiling and source-supported evidence. I reviewed the
design document, project brief, source, prompts, settings, existing research
scorer/live runner, and saved smoke artifacts before adding evaluations.

Product fixes below are intentionally left to the assigned engineers. No
product schema, token budget, model configuration or runtime code was changed
for this evaluation work. Failing contract assertions are release blockers,
not expected failures to skip. Priorities are proposed triage, not measured
incidence in live traffic.

Latest stable audit: **144 passed, 7 failed in 9.27 seconds**, with no source
changes detected during execution. All seven failures map to the first five
tickets below. The runner and research-policy checks themselves have **19
passing tests**. Applying the research gate to the saved real-paper smoke
manifest correctly rejected it as release evidence.

Artifacts: [summary](gaptrace/data/evals/contract-audit-verified/summary.md),
[full JSON report](gaptrace/data/evals/contract-audit-verified/report.json),
[JUnit](gaptrace/data/evals/contract-audit-verified/junit.xml), and
[smoke release check](gaptrace/data/evals/smoke-release-check.json).

Run commands from `gaptrace/`. The complete command is:

```powershell
python -m evaluation.run --out data/evals/after-fixes-001
```

## P0 — Validate answer claims against fetched passages

**Assign to: RAG / applied ML engineer responsible for answer grounding.**

`src/gaptrace/agent.py:Agent.ask` filters cited IDs for visibility but accepts
answer text when at least one ID remains. It does not verify claim entailment
or require that the cited note has source text. A scripted model saying
“Alpha scores 999% accuracy on Atlas” passes despite the source reporting 81%.
A note with no passages can also authenticate a final answer.

Implement the claim checker already required by the project brief: use actual
fetched source passages, check every factual sentence, remove unsupported
claims and abstain when none survive. Numbers must agree with their source
and fact table. Keep the gateway and token limits intact. Keep supported
answers working; do not solve this by refusing everything.

### Recommended implementation

1. Add a private final-answer validation helper called by `Agent.ask` before
   the final answer is recorded or returned. Preserve the existing return
   dictionary. Keep the draft in the step trail; record validation decisions
   and rejected sentences as separate trail events.
2. Resolve surviving cited note IDs to actual `PassageStore` records through
   their `passage_ids` and applicable verified evidence links. Missing notes,
   missing passages, or unverified links supply no supporting evidence. A
   summary/headline or a `verified` flag alone does not establish entailment.
   Retrieve relevant source text through the existing passage/tool path and
   retain the exact source IDs; the helper must work even when the model did
   not request a passage during its earlier steps.
3. Check each factual sentence, including all of its factual clauses, against
   its candidate sources. Reuse `VerifierOutput` and `Gateway.call("local", ...)`
   with a dedicated prompt in `prompts/` for claim entailment. Do not force a
   factual sentence into a relation-link prompt with an invented link type.
   Treat paper text as data. A real citation ID, shared keywords or a quoted
   number is insufficient support for the sentence's asserted meaning.
4. Prefer one sentence and one bounded passage per verification call, stopping
   when support is established. Resolve a sentence containing several claims
   into independently checked clauses only when their meaning is preserved;
   otherwise drop the whole unsupported sentence. A multi-paper comparison
   must have support for each premise; do not demand that one paper contain
   both papers' results. Keep source attribution and comparison direction.
5. For numeric statements, compare the complete method/dataset/metric/value/
   unit/paper association with source-backed fact rows or explicit passage
   text. Normalize equivalent number formatting without inventing unit
   conversions. A matching number elsewhere in the corpus is not support.
   Validate semantic qualifiers as well as the number itself.
6. Remove unsupported sentences and citations unused by the remaining answer.
   On invalid checker output, missing evidence or exhausted verification
   allowance, withhold the unverified sentence. If nothing factual survives,
   return the existing `NOT_FOUND` response. Do not regenerate answers in an
   unbounded repair loop or send this work to the cloud.
7. Keep source candidates deduplicated and deterministically ordered. Use the
   existing gateway cache, bound total sentence/source checks explicitly, and
   apply the unchanged token gate to every call. Record added calls, tokens
   and latency on the fixed development set before choosing that bound.

This completes the answer checker already in the design. It adds verification
at the final-answer boundary and preserves the agent's retrieval and state loop.

Acceptance: both failing cases in `evaluation/contracts/test_answer_contracts.py`
(`test_wrong_fact_with_real_citation_is_rejected` and
`test_claim_without_source_passage_is_rejected`) pass, alongside the supported
answer positive control. Add mixed supported/unsupported sentence and injected
paper-instruction cases when implementing the checker.

## P0 — Require verified numeric provenance for conflict gaps

**Assign to: knowledge graph / retrieval backend engineer.**

`src/gaptrace/gaps.py:find_conflicts` builds its graph with
`verified_only=False` and uses fact rows without requiring a verified reports
link. Setting both reports links to unverified still produces a conflict card
whose evidence trail is unverified. This violates the verified-evidence rule.

Require independently sourced, verified reports for the compared facts.
Retain traceability to the exact passages and preserve valid conflicts.
Consider orphan facts and mixed verified/unverified reports explicitly.

Acceptance: `test_unverified_reports_cannot_create_evidence` in
`evaluation/contracts/test_gap_contracts.py` passes, while the conflict positive
control, independent-paper control, metric/dataset controls and tolerance
boundary controls remain passing.

### Recommended implementation

Build a verified graph inside `find_conflicts`, then admit a result observation
only when a verified incoming `reports` edge comes from an actual paper note
whose ID agrees with the result's `paper_id`. Require a nonempty source-passage
ID and quote on that edge. The ingestion/verifier boundary should check that
the passage exists, belongs to the reporting paper and contains the evidence
quote. Where runtime gap generation has no passage-store argument, preserve
the existing public signature and enforce passage integrity during ingestion
or workspace loading; do not treat a flag on a broken link as sufficient.

Construct comparison groups, values, support counts, note IDs and trails from
these eligible observations only. Require two distinct reporting papers after
filtering. A third unverified observation must neither create a conflict nor
inflate its support. Preserve the current method/dataset/metric matching and
configured numeric tolerance; compare compatible units and withhold ambiguous
comparisons rather than infer conversions. Sort retained observations and
trails deterministically. Keep the facts table available for exact lookup;
eligibility for a gap is an additional evidence requirement at the miner.

Extend the regression checks with one verified plus one unverified report,
orphan results, mismatched reporting-paper ownership, and a valid two-paper
conflict plus an unverified extreme value. Validate passage ownership at the
boundary that has access to the store. These checks protect the evidence rule
without changing the four patterns or ranking formula.

## P0 — Preserve distinct results when extraction IDs collide

**Assign to: ingestion / data pipeline engineer.**

`src/gaptrace/pipeline.py:merge_notes` merges identical IDs before considering
note type. Extraction derives IDs from type and name, so two papers naming
their result “Alpha accuracy” can collide. Merging keeps the first fact and
paper ID, attaches both passage IDs, and loses the second measured value.
The reproduction uses 81% and 75%; only 81% remains afterward.

Preserve per-paper numeric observations and their provenance. Coordinate with
the schema owner if changing schema is necessary; a schema change has not
been authorized by this task. Make result identity stable under input order
and retries. Do not merge different paper measurements merely because their
names match.

Acceptance: `test_merge_preserves_distinct_paper_results` in
`evaluation/contracts/test_pipeline_contracts.py` passes; both values remain
queryable and can produce a valid conflict after link verification.

### Recommended implementation

Introduce a private result-ID helper in `pipeline.py`; keep `make_id` unchanged
for all other note types and preserve the existing `Note`/`Fact` fields. Derive
result IDs deterministically from source paper ID, normalized result name and
the full fact tuple (method, dataset, metric, value, unit), using canonical
serialization before hashing. Including the fact tuple preserves differing
measurements within one paper as well as across papers. Identical repeated
observations can merge their source passage IDs without losing their fact.

Assign this identity in `_extract_chunk` before its result dictionary can
overwrite a same-name observation, and before paper `reports` links are
constructed. Guard the chunk accumulator and `merge_notes` identical-ID path
too: merge result objects only when paper and fact identity agree. Do not let
`_merge_into` silently choose the first measurement. Keep results excluded
from embedding-based semantic merging, as the current design intends.

Retarget each reports link using its source paper and passage provenance. A
global `{old_id: new_id}` map cannot represent an old ID that referred to two
different papers; use a private provenance-scoped mapping while repairing
inputs. If an old ambiguous link cannot be assigned safely, leave it
unverified and report the issue. Preserve the public merge return shape for
its existing callers. Within one extractor output, a same-name link may also
be ambiguous: resolve it using the supporting quote/fact, or leave it
unverified rather than guessing which observation it describes.

For existing workspaces, build a new corpus from original passages and keep
the old corpus intact for comparison. Rebuilding SQLite/indexes alone cannot
restore a fact already lost from the bundle. Test reversed paper order,
repeat imports, duplicate observations and differing values within one paper.
Check that every reports link still reaches the correct observation and that
saved/reloaded bundles retain both facts. This is a correction to identity
generation and deduplication, not a schema or storage redesign.

## P1 — Preserve uncertainty when closure was not judged

**Assign to: research reasoning / closure engineer.**

`src/gaptrace/closure.py:check_closure` initializes status to `open`. It returns
`open` when the candidate finder returns nothing or all judge outputs fail
validation, with `judged_by="none"`. An incomplete search is not a completed
open verdict. The existing `unchecked` status can express this uncertainty.

Keep `unchecked` until there is a successful applicable judgment. Define how
partial search failures affect the final verdict and retain diagnostic reasons.
Avoid presenting a bounded search as proof that no later work exists.

Acceptance: `test_no_candidates_remains_unchecked` and
`test_invalid_judge_remains_unchecked` in
`evaluation/contracts/test_closure_contracts.py` pass. Successful open/partly/
closed judgments, exclusions, deduplication and early stopping remain passing.

### Recommended implementation

Track candidate search completion, attempted judgments, valid judgments and
failures separately within `check_closure`. Initialize the verdict to the
existing `unchecked` status. Use this reduction over the bounded search:

| Observed result | Returned status | Deciding paper |
| --- | --- | --- |
| A valid `fills` judgment | `closed` | That paper; stop early |
| At least one valid `partly`, with no valid `fills` | `partly` | The first applicable partial paper in deterministic order |
| At least one valid judgment, all `does_not`, and required configured searches/judgments completed without failure | `open` | Keep current empty deciding-paper convention |
| No eligible candidates, no valid judgments, or only negative judgments with an unresolved search/judgment failure | `unchecked` | Empty |

An unavailable source is a failure; a successfully queried source with zero
results is an empty search. A successful source can still yield candidates
when another successful source is empty. Preserve candidate exclusions,
deduplication, the existing `max_papers` cap and local fallback. `open` means
the evaluated bounded search found no closure; it does not establish absence
from all literature. A partial verdict records positive evidence of partial
coverage even when remaining checks are incomplete; retain that incompleteness
in diagnostics. Count only valid judgments in `judged_by`.

Use existing run diagnostics for search errors, validation failures and search
scope rather than adding fields to `GapCard`. Catch known recoverable provider
errors at their boundary; log unexpected errors and surface them instead of
silently claiming openness. Extend tests with a negative judgment followed by
a failed judgment, mixed partial/failed results, and a later valid full closure.

## P1 — Count and cap every cloud backend attempt

**Assign to: inference platform / model gateway engineer.**

`src/gaptrace/gateway.py:Gateway.call` increments `cloud_calls` only after the
initial successful backend return. The schema-repair retry calls the cloud
backend directly without rechecking the cap or incrementing the counter.
With a cap of one, an invalid response followed by valid JSON makes two cloud
calls. Failed cloud attempts also deserve a documented accounting policy;
their cost cannot be assumed zero.

Apply the budget and fallback policy to every backend attempt, including
schema retries. Handle retry exceptions consistently with initial exceptions.
Keep cached calls distinct from actual provider attempts.

Acceptance: `test_cloud_retry_cannot_exceed_topic_call_cap` in
`evaluation/contracts/test_gateway_contracts.py` passes; cloud-disabled,
provider-failure fallback, cache, and local budget checks remain passing.

### Recommended implementation

Refactor the actual backend invocation into one private attempt path inside
the gateway. Every initial request, JSON-repair request and fallback must use
that path. For an uncached cloud request: check the remaining topic allowance,
increment `cloud_calls` immediately before dispatch, then invoke the backend.
Count exceptions as dispatched attempts because the provider may have already
processed them. A validated cache hit consumes no provider attempt.

If the cap is exhausted or a recoverable cloud error occurs, route to local
through the same gateway and apply the local prompt/token checks again,
including any repair suffix. Keep the configured one-repair limit explicit
across the logical request; recursive fallbacks must not reset it and produce
an unbounded chain. Transport fallback is distinct from JSON repair, and both
must have a bounded control path. Use the actual serving tier/model when
reading or writing its cache; never store a local fallback under a cloud key.

Log tier, cache use, dispatch attempt, repair and fallback reason using the
existing gateway log. Do not alter `max_cloud_calls_per_topic`, its default,
the output budget, local-model choice, or caller-facing result tuple. Add
checks for cap exhaustion before dispatch, a cloud exception consuming an
attempt, an exception on the repair call, successful cached reuse, and local
fallback whose repaired prompt exceeds the token gate.

## P1 — Assemble an independent real benchmark and calibrate graders

**Assign to: evaluation/data engineer, paired with a research scientist.**

The available real-paper results are historical smoke checks. Their cutoff
predates the model and they do not establish research quality. The project
guide explicitly says that the full 20-topic/100-question collection, baseline
adapters, ContextBench runs and blinded human ratings remain to be assembled.
The current live runner repeats answers on a shared corpus and deterministic
gap list; repeats can reuse cached outputs.

Build version-pinned, dated corpora with a documented model training cutoff;
freeze topic-level development/held-out splits and dataset hashes. Include
negative gaps, already-closed gaps, partial closure, multi-paper evidence,
numeric contradictions, unanswerable questions and malicious instructions
inside paper text. The current Question schema requires gold passages, so
coordinate a deliberate representation for truly unanswerable questions
instead of supplying fictitious evidence.

Use a separate blinded judge or adjudicated human labels for correctness,
atomic claims, gap validity/openness/doability, canonical duplicates and gold
gap catalogues. A stricter prompt on the same model is a correlated recheck,
not independent validation. Exact quote presence is necessary provenance but
does not prove entailment. Audit omitted claims and all-refusal answers. Freeze
future search pools, not just the generated outputs, for comparable trials.

Run actual lexical/vector/hybrid and raw-passage baselines plus one-switch
ablations. Measure real tokenizer/server usage, per-schema retries/cache hits,
build time, p50/p95 latency, GPU memory and total model tokens. Separate cold
and warm caches. If estimating run-to-run variation, perform independent
generations/builds rather than duplicating cached deterministic results.

Acceptance: publish the frozen dataset manifest and labeling rubric, judge
calibration against blinded human labels, three clearly characterized runs,
paired topic comparisons with uncertainty, and a held-out report. Review and
freeze `evaluation/release-policy.json` before accessing held-out labels; its
numbers are proposed acceptance targets, not established performance. Both
the offline gate and the recorded research gate must pass before claiming a
release is validated.

### Recommended implementation and review sequence

Keep the present scorer, live driver and release-policy file. Add the missing
data and baseline adapters under `evaluation/`; avoid embedding research
labels, metric targets or baseline branches in product code. Store frozen
search responses and generation provenance with each run so future-index
changes cannot change the comparison after the fact. Define actual build and
generation repeats separately from deterministic rescoring.

Use the existing seven failing assertions as the minimum reproductions for
the first five fixes. The owning engineer should add the edge cases named in
their ticket, run the focused suite, then run the complete offline gate to a
new artifact directory. Do not weaken thresholds or remove positive controls
to obtain a pass. The evaluation owner reviews counterexamples and retained
supported outputs, then compares development-set accuracy, unsupported-claim
rate, gap precision/recall, tokens and latency against the frozen pre-fix run.
Report undefined and underpowered metrics explicitly. The proposed release
targets require research-lead review before any held-out evaluation.

For identity changes, retain the original workspace and compare a freshly
ingested workspace. For prompt changes, preserve versioned prompts and cached
run artifacts. Promote a fix only after correctness checks pass and added
inference cost is measured; no gate score alone establishes scientific utility.
