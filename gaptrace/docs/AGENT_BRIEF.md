# Brief for the coding agent

Read this file and README.md first. Do not re-derive the design: the schema,
the token budget, the tools, the gap patterns and the prompts already exist
and are tested. Your job is the remaining parts below. Reuse open-source
packages (DEPENDENCIES.md lists the checked ones); write only glue.

## Rules that outrank everything

1. No call to the local model exceeds 8,000 tokens. `gateway.Gateway.call`
   enforces it through `context.assert_within_budget`. Never bypass the gateway.
2. Nothing is evidence unless a source passage supports it. Only links with
   `verified: true` are used; answers without citations become "Not found".
3. This is a product. No benchmark code, scores or comparison wording in
   `src/`. Evaluation lives in `evaluation/` and drives the product from outside.
4. Do not change `schema.py` or the budget in `settings.yaml` without asking.
5. Check a package's licence before adding it and record it in DEPENDENCIES.md.
6. Work one phase at a time: code, tests, run, show results, wait.

## Implemented core

The initial list below is supplemented by [CORE_FIXES.md](CORE_FIXES.md).
The expanded suite covers product contracts and research tools; its latest
offline artifact is `data/evals/after-core-fixes-002/`. Tests do not establish
independent research accuracy.

- Data model, note bundle, passage store, facts table, graph
- Hybrid search (BM25 + pluggable embeddings + rank fusion)
- Context manager: slot budget, summary/stub packing, state file, trail
- Tools with size caps and full results written to disk
- Four gap patterns and ranking
- Gateway: token gate, disk cache, schema validation with one retry, cloud
  routing with call budget and local fallback
- Pipeline: extraction to notes, quote-in-passage check, merge, link verification
- Closure check and doability scoring (candidate finders are injected)
- Agent step loop
- Command line, MCP server (mcp v2 `MCPServer`), HTTP API (read endpoints)

## Validation scope and remaining integration

- `gateway.litellm_backend` was measured in the hardware smoke tests. The
  answer checker and four retrieval baselines also have local 4B smoke runs.
- `search.sentence_transformer_embedder`
- Broad extraction and answer quality across topics still require independent
  real-paper evaluation; the smoke runs are too small to establish this.
- `tokens.count` is a pessimistic estimate. Replace it with the model's real
  tokenizer via `tokens.set_counter` and keep the 8,000-token test.

## Left to build

Terminal-first direction (2026-11-16): prioritize core behavior before the
web app. `gaptrace shell --data PATH` now provides a persistent command
session, corpus switching, and command error recovery. `gaptrace status`
reports corpus and verification counts; `ingest` reports foreground progress.
The CLI also supports dated sectioned-JSON ingestion and JSON inventories.
Automated collection and resumable builds remain unfinished. Saved hardware
measurements and a real-paper evaluation smoke run are present; the older
"not yet run" list above should not be treated as a current validation inventory.

Phase 0. `scripts/measure.py`: peak GPU memory and tokens per second at
1,000 / 4,000 / 8,000-token prompts; write docs/hardware.md. Verify the
litellm backend here.

Phase 1. `collect.py`: topic, paper count, optional date range -> paper notes
(type "paper", `paper_date` set) and passages (<= 400 tokens, section tagged,
limitations / future work / conclusion marked priority). Use arxiv, pyalex,
semanticscholar and docling. Respect rate limits. Then call
`pipeline.build_notes` and `bundle.write_bundle`. Add `gaptrace build`
with progress, pause and resume.

Phase 2-3. Mostly done in `pipeline.py`. Add: batching and resume, a report
of proposed / verified / rejected links with 20 rejected examples (the stats
dict already has them), and `l0`/`l1` rewriting for merged notes.

Phase 4. The final-answer checker now resolves source passages and verifies
each sentence through a dedicated entailment prompt and the local gateway.
It preserves supported sentences, trims citations and records decisions and
costs. Ordinary product entry points now require the real tokenizer configured
in settings (the Qwen tokenizer is bundled); evaluation probes and baseline
commands also accept an explicit tokenizer path. See TERMINAL.md for the
prepared PowerShell walkthrough and interactive shell.

Phase 5. Search-backed investigation is implemented through `review-gaps` and
the assistant's `review_gaps` tool. OpenAlex/arXiv search, source-relation checks,
quoted closure judgments/rechecks and evidence-gated feasibility feed the
existing deterministic ranker. Original/reviewed orders and full decisions are
persisted per corpus and run. See GAP_REVIEW.md for the research metric adapter.
Search scope is bounded to corpus passages and external abstracts.

Phase 6. Add write endpoints and background jobs to `api.py` (build, status),
and MCP tools `build_corpus`, `corpus_status`, `check_closure`.

Phase 7. Web app (Next.js, Tailwind, shadcn/ui; components from 21st.dev;
React Flow for the graph; assistant-ui for chat). Screens: topic and build,
gap list, gap detail with trail, graph explorer, ask, library, settings.
No benchmark screen.

Research (only after phase 7): see evaluation/README.md.

## Long papers and local inference

Split collected text by section into stable, token-bounded passages. Process
priority sections first and checkpoint by passage ID. The extractor also
splits oversized passages already in a bundle and checks quotes against the
chunk the model saw. Links retain the original passage ID so the full source
is still available. Use one inference worker on the 4 GB GPU and queue parsed
passages ahead of it. Keep the queue bounded so PDF parsing can pause when
inference falls behind.

For model-guided context, use the existing `state_update` and retrieval tools
as the starting point. A follow-up experiment can let the model choose which
note IDs to retain or retrieve while the context packer enforces the budget
and the trail retains evidence. Test this on long papers before adopting it.
[Context Language Models](https://arxiv.org/abs/2609.37725)

If builds need durable branching or tool interrupts, evaluate wrapping the
existing tools and gateway in LangGraph nodes with checkpoints.
[LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)

Use the existing evaluation program for Karpathy-style tuning. Change one
setting or prompt per trial, score it on the same development cases, and
keep the change only when evidence quality holds. Use Qwen3.5 4B as the
local language model. The 2B model was removed by user request; do not
reinstall it or route extraction to it.
[Karpathy autoresearch program](https://github.com/karpathy/autoresearch/blob/master/program.md)
