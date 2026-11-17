# Bounded autoresearch and context experiments

The working local-model controller follows the bounded experiment pattern in
[Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch) and
reuses pinned code from the official
[Context Language Models repository](https://github.com/facebookresearch/context-language-models).
No weights are trained. The model proposes a policy, prompt or scalar setting;
the fixed evaluator measures it; a fixed gate keeps or rejects it. A working
loop does not establish that a small model reliably proposes improvements.

## Meta paper and code reuse

[Context Language Models](https://arxiv.org/abs/2609.37725), by researchers at
UW, Meta, MIT and Trillium Labs, treats active context as a model-editable
file and studies learning its management. Its official repository includes
in-context skill evolution. The original project brief already cites this
work. Earlier product code used state files and packing but did not import
the upstream implementation or expose unrestricted context editing.

The new controller directly imports upstream episode records, contrastive
notes, proposal validation and paired-task SE. The six unmodified files,
commit, copyright headers and CC BY-NC 4.0 notices are retained in
`vendor/context_language_models/`. They remain noncommercial research code.
The full upstream `Evolution` class is preserved as a reference; our executed
controller adds fixed quality guardrails and bounded private experiment copies.
Upstream unrestricted Bash edits, Harbor, RL training and serving/cache
optimizations are not executed. The memory-file experiment is a constrained
adaptation, not an exact reproduction. Published larger-model results cannot
establish a benefit for our 4B model.

## Where OKF is used

The [official OKF specification](https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md)
defines a Markdown/YAML interchange format. GapTrace stores persistent
scientific notes in that style, adding its own typed relationships and source
pointers. The product is a scientific domain consumer, not a general OKF reader.

| Code | Role |
| --- | --- |
| `src/gaptrace/bundle.py` | Writes `bundle/<type>/<id>.md`; YAML holds IDs, type, links, paper dates, facts and passage pointers. The body holds L1. Reads these back into Note objects. |
| `ingest.py`, `pipeline.py` | Extracts and verifies notes/evidence before publishing the bundle and passage store. |
| `workspace.py`, `tools.py` | Loads the bundle and exposes note lookup, passages, neighbors, facts and trails. |
| `search.py`, `graph.py`, `facts.py` | Builds retrieval, relationship and numeric views of parsed notes. |
| `gaps.py` | Finds missing-work patterns in typed relationships and ranks candidates. |
| `context.py`, `agent.py` | Packs summaries and stubs, persists findings with IDs, and rebuilds model steps. |

The model normally sees selected summaries, IDs and tool digests, not the
original YAML bytes. Markdown supplies portable storage. Typed information,
source links, retrieval and memory policy are separate possible benefits.
Equivalent JSON parsed into identical Note objects should not by itself
change inference; do not attribute graph or memory gains to a file extension.

`okf_audit.py` checks core envelopes and extension resolvability, not every
optional OKF feature. The existing Transformer bundle passed: two concepts,
one verified typed link, no dangling note targets or passage pointers. Its
source provenance uses domain fields rather than standard `sources` entries.
No full conformance certification is claimed.

## Context experiment design

`context_experiment.py` crosses four representations with two memory policies.

| Representation | Information |
| --- | --- |
| raw | Exact passages with paper IDs and titles. |
| flat | Frozen note titles/summaries and source pointers, without types/relations. |
| typed | The same notes with their types and relationships. |
| plain | Equivalent typed fields and relations rendered in prose. |

`static` appends observations and drops oldest memory on overflow. `managed`
lets the model replace the complete memory file. That literal bounded file
is reloaded on the next turn. Both use a 600-token memory, the same budget,
same source inventory, same batch schedule within a representation, same
answer schema and grader. Each source batch gets one update call; a final
answer call follows. Source batches above 2,400 tokens fail explicitly:
reduce `records_per_step` or split passages rather than silently omit them.

Raw/flat changes extraction and compression together. Flat/typed tests type
and relation information together. Typed/plain tests presentation with the
same fields, not storage alone. Static/managed tests policy. None alone tests
the product's graph traversal or its complete gap-generation pipeline.

All source records are streamed; cross-document retrieval metrics therefore
reflect source exposure, not selective retriever quality. Final citations
must have been seen and remain named in memory. Supported claim labels require
exact quotes in cited source text. Literal abstention has no factual claims
and is correct only for explicitly unanswerable questions. The same local
model grades answers; independent human checks are required for quality claims.
Generation and grading usage are recorded separately.

`local_model.py` uses the existing Gateway, exact tokenizer, loopback-only
Ollama, output-schema and chat-overhead reserves, and checks actual server
usage against 8,000 tokens. No model-generated shell command is executed.

## Run the context comparison

From the repository root, with the project and documented tokenizer installed:

```powershell
python -m evaluation.context_experiment evaluation/examples/context-smoke.json --tokenizer "$env:TEMP/gaptrace-qwen35-tokenizer.json" --out data/context-001 --repeats 1
```

For actual published product notes rather than the manually supplied control:

```powershell
python -m evaluation.export_context evaluation/examples/smoke-plan.json --out data/context-from-bundle.json
python -m evaluation.context_experiment data/context-from-bundle.json --tokenizer "$env:TEMP/gaptrace-qwen35-tokenizer.json" --out data/context-bundle-001 --repeats 3
python -m evaluation.okf_audit data/evaluation-smoke-transformer/bundle --passages data/evaluation-smoke-transformer/passages.jsonl --out data/okf-audit.json
```

The exporter reads published bundle/passages, checks cutoff/cloud provenance,
normalizes Windows newlines, hashes sources and validates matching passage
coverage. It does not change the product corpus. Review extraction fidelity:
source pointers alone do not prove the summaries follow from the passages.
Freeze the entire extracted note set across representation arms.

Outputs include plan, scorable manifest, per-question memory/trajectories,
actual usage, metrics and `factorial_contrasts`. Contrasts pair topic means
across repeats, with sign tests, topic-bootstrap intervals and Holm correction
within a comparison. Multiple contrast families remain exploratory. Predeclare
primary endpoints and use enough independent topics for significance claims.

A benchmark needs a documented model training cutoff on or before the corpus
cutoff. The historical Transformer example is explicitly smoke mode and does
not establish future-blind performance. Held-out context comparison requires
`--split held-out`; tuning refuses held-out plans entirely.

## Run autoresearch

```powershell
python -m evaluation.autoresearch evaluation/examples/autoresearch-context-smoke.json --tokenizer "$env:TEMP/gaptrace-qwen35-tokenizer.json" --out data/autoresearch-001
```

This workspace also has `.research-venv/Scripts/python.exe`. Set
`$env:PYTHONPATH` to the repository's `src` directory if the project is not
installed in that interpreter. To create it elsewhere, run
`python -m venv --system-site-packages .research-venv`, then
`.research-venv/Scripts/python.exe -m pip install tokenizers==0.23.3`.
No mandatory product dependencies were added.

The search plan fixes engine, input plan, system, objective, guardrails,
repeats, per-trial and cumulative time limits, and round count. The same local
model proposes and executes our smoke experiment. This is self-proposal,
not self-training or demonstrated autonomous scientific competence.

The machine-readable contracts are [autoresearch.schema.json](autoresearch.schema.json)
and [context.schema.json](context.schema.json). Cross-record checks in the
Python models enforce additional cutoff, corpus, and evidence relationships.

For the context engine, the sole editable target is its policy text. For the
live engine, the runner uses the existing CLI-driven `evaluation.live` harness
inside a private copied workspace. A candidate changes one allowed setting
or rewrites `prompts/step_agent.txt`. The live example is
`examples/autoresearch-gap-smoke.json`; it depends on the main live harness's
optional dependencies and future-index access. It has not been run as a full
gap-discovery improvement experiment in this work.

Imported corpora are frozen and copied per trial; source-based plans rebuild
into private data directories. The generation model is fixed in the baseline
snapshot and cloud generation is disabled. Budget, scorer, schema, questions,
cutoffs, matching thresholds and corpus identity cannot be proposed as changes.

The gate requires the same protocol, all requested repeats, a defined primary
metric improving strictly beyond both recorded sample SDs and the configured
minimum gain, no mean hallucination increase, and no mean evidence-quality
decrease. The two evidence guardrails are mandatory and must be defined for
every repeat. Additional guards can be fixed in the plan. Benchmark mode needs
at least three repeats. Paired-topic SE from upstream code is diagnostic and
does not override this gate. Ties are rejected even if text is shorter.

The default objective is `hit_rate`; the context-only example explicitly uses
`answer_accuracy`. Do not claim better gap discovery from answer-only scores.

Some existing switches are partial ablations: `three_note_sizes: false` still
supplies summaries, `stateless_steps: false` uses a bounded transcript view,
and current CLI construction supplies no vector embedder. Disabling hybrid
search there cannot establish vector-versus-hybrid performance. Settings
without an actual effect should tie and be rejected. The factorial runner
makes explicit interventions instead of assuming switches remove mechanisms.

Candidates are immutable private copies. Rejection leaves the incumbent
pointer unchanged; acceptance moves it. This workspace has no Git repository,
so snapshots and content hashes implement keep/revert rather than Git commits.
Production files remain untouched. `best.json` names the selected workspace.

Per-round requests, raw proposals, development notes, usage, decisions, worker
logs, `results.tsv`, and `state.json` preserve outcomes. Resume checks config,
evaluation code, tokenizer, frozen material, source snapshots and available
model digest, then reuses completed results. Failed partial worker directories
are retained; start a new run rather than silently reuse them. Completed
checkpoints accumulate elapsed time; externally killing the controller before
a checkpoint cannot perfectly account for lost wall time. One worker runs at
a time. Trial timeouts stop its own child process tree. Local proposer retries
share a remaining-time deadline. Nothing deploys a winner or edits labels.

## Evidence so far

The local Qwen3.5 4B completed all eight historical arms. After equalizing
paper-title metadata, every arm answered the single question correctly. This
ceiling result on a short stream does not show an OKF or memory-policy benefit.
The earlier unequal-metadata diagnostic is preserved but invalid for that
comparison. Larger streams must exceed the 600-token memory to test retention.

In a live one-round self-proposal run, the model required multiple corroborating
passages. On the one-passage task the candidate abstained: accuracy fell from
1 to 0 and evidence guardrails became undefined. The controller rejected it
and retained the baseline. An earlier malformed proposal was also rejected.
These establish loop execution, not successful improvement. The proposer's
claimed failure was not supported by its successful baseline trace; syntactic
citations do not make a hypothesis sound.

See AUTORESEARCH_RESULTS.md for artifacts. Establish benefit using the real
frozen dataset, long streams, independent labels, at least three repeats and
paired topic comparisons. Keep matched plain-data controls and report negative
findings. The code supports this investigation; a one-question replay cannot
answer the scientific question alone.
