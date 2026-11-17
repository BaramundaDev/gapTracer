# Investigate and rank gaps

GapTrace can now search scholarly literature, check the original evidence,
assess whether papers address a gap, and rerank the candidates. All model
judgments use the configured local Qwen model and the existing token gateway.
The original notes and mined candidates remain intact.

From PowerShell in the project directory:

```powershell
.\.research-venv\Scripts\python.exe -m gaptrace.cli review-gaps --data data/terminal-demo --k 3
```

The command prints progress for evidence checks, searches and candidate
judgments. Each reviewed gap makes at most two index searches, retrieves at
most five works per search, checks at most three external abstracts plus two
local source passages, and rechecks positive closure findings. Up to eight
original evidence links are checked per gap. Default review budget is three
gaps; `--k` permits 1 through 10. Model response repair remains bounded by the
gateway's existing one-repair rule. A full review can take several minutes.
Feasibility can add up to four source-entailment checks, one for each proposed
non-null axis; matching text alone cannot establish a resource claim.

Searches use OpenAlex by default; `--provider arxiv` selects arXiv. An optional
`OPENALEX_API_KEY` increases the provider's allowance. `--as-of YYYY-MM-DD`
bounds publication dates and must not precede the corpus. Results include
links, publication dates, abstracts, queries, retrieved pools, quotes, model
decisions, failures and gateway usage in `data/terminal-demo/gap-review.json`.
A separate copy is saved in the new run directory. The search providers'
official documentation is available at
[OpenAlex authentication](https://help.openalex.org/api/authentication/),
[OpenAlex search](https://help.openalex.org/api/searching/), and the
[arXiv API manual](https://info.arxiv.org/help/api/user-manual.html).

Review one candidate or redisplay a saved result without inference:

```powershell
.\.research-venv\Scripts\python.exe -m gaptrace.cli review-gaps --data data/terminal-demo --gap-id gap-208f2a44b0
.\.research-venv\Scripts\python.exe -m gaptrace.cli review-gaps --data data/terminal-demo --saved
```

Saved reports record their assessment date. Reload rejects a changed corpus,
model configuration or review prompt; run a new investigation for newer
literature. Plain `gaps` still shows the original mined list, preserving its
use in frozen evaluation runs.

## Assistant tools

Inside `gaptrace shell`, try:

```text
ask "Validate and rank the research gaps using current literature" --local-only
review-gaps --k 3
review-gaps --saved
literature "experience replay language models" --k 5
```

The assistant can call `search_literature(query, k)` and
`review_gaps(k, gap_id, provider, as_of)`. Explicit requests beginning with
validate, review, investigate, rank or check and naming gaps dispatch the
investigation directly; they do not depend on Qwen volunteering a tool call.
Search abstracts remain separate from the imported
corpus and can be fetched by passage ID. External factual answers still pass
the final source checker. A completed `review_gaps` tool action returns a
deterministic rendering of the recorded assessment rather than asking the
model to invent a ranking explanation. The MCP server exposes both tools;
the HTTP API exposes `POST /gap-reviews?k=3` for the same operation.

## How ranking works

The existing code-only ordering is retained: status (open, unchecked, partly,
closed), possible evidence problems, descending support, feasibility and
verified share, then stable ID. The report preserves the original and revised
orders over the exact same candidates. Unreviewed candidates remain unchecked.
Closed candidates are retained in the list so their evidence can be inspected.

The investigation checks each quoted source relation. A positive closure
finding needs an exact quote and a successful stricter recheck. A failed
search, failed judgment or contradictory recheck preserves uncertainty unless
there is positive evidence of partial/full closure. A bounded open finding
requires source support, at least one completed negative judgment and no
unresolved search or closure-check failure.

Feasibility uses the existing four axes: data, compute, building blocks and
scope, each 0 through 2. Each non-null axis needs an exact source quote. An
unknown axis remains null, and the total is withheld unless all four are known.
The report retains the individual scores, reasons and source IDs.

These are automated assessments from a bounded search over corpus passages
and external abstracts. A quote plus model recheck is useful evidence, but is
not independent adjudication or exhaustive full-paper review. The same model
performs the recheck, so its errors may be correlated. No-result searches do
not establish universal novelty. Feasibility is often unknown from abstracts.

## Use the existing evaluation system

```powershell
.\.research-venv\Scripts\python.exe -m evaluation.gap_review data/terminal-demo/gap-review.json --out data/evals/gap-ranking-001
```

This calls the existing `evaluation.ranking.score_ranking` for both frozen
orders. It produces `ranking-report.json`, an exact copy of the investigation,
and `independent-labels-template.json` bound to its SHA-256. The primary quality
endpoint remains **actionable_precision@5**: independently adjudicated real,
open and doable labels must each be at least 4 out of 5. A fixed independently
assembled catalogue enables **gold_ndcg@5** and **gold_recall@5**. Complete
canonical cluster labels enable duplicate penalties. These definitions are
unchanged from [METRICS.md](../evaluation/METRICS.md).

Have independent reviewers complete the template's `judgment`, `canonical_id`
and optional gold catalogue fields using the existing
[annotation guide](../evaluation/ANNOTATION.md). Preserve the report hash,
candidate IDs and statements. Keep `candidates` empty: current closure papers
are not retrospective future-confirmation labels. Then score both orders:

For this current-literature investigation, assess both orders against the same
report `as_of` date and the same catalogue. The separate historical benchmark
continues to use its original corpus cutoff and future observation window.

```powershell
.\.research-venv\Scripts\python.exe -m evaluation.gap_review data/terminal-demo/gap-review.json --labels independent-labels.json --out data/evals/gap-ranking-reviewed-001
```

Without those labels, quality and catalogue endpoints remain undefined.
Automated judgments are never copied into independent quality labels. Temporal
metrics such as future_precision@5 and hit_rate are withheld entirely here;
the existing dated benchmark runner remains responsible for that experiment.
This comparison is descriptive for the selected corpus. Independent multi-topic
and held-out evaluation is still required to establish a quality improvement.

## Executed verification

On November 17, 2026, live searches succeeded against both OpenAlex and arXiv.
A complete local investigation of the invented demo's language-model replay
question recorded an automated closed verdict backed by a verbatim external
abstract quote and a stricter recheck. Feasibility remained unknown. The
natural-language `ask` request also executed the review action and saved its
report; that one-gap request took about 90 seconds. This exercises the workflow
and is not independent scientific validation of that verdict.

The earlier live run's altered recheck quote was rejected and its artifacts
were retained. This led to stricter contiguous-quote instructions, clearer
open-question relation semantics, and resource-specific feasibility entailment.
The final offline audit passed **248 tests**, with no failures or skips and no
source changes during execution:
[audit summary](../data/evals/gap-investigation-003/summary.md).
The [live ranking export](../data/evals/gap-ranking-live-002/ranking-report.json)
correctly withholds quality metrics without independent labels and does not
report retrospective temporal metrics. The original and revised orders in this
single-gap demonstration happen to be identical because the affected candidate
was already last; no ranking improvement is claimed.
