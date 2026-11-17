# Executed autoresearch and context checks

The bounded local-model loop, actual-bundle export, OKF core audit, and eight
representation/context arms have been executed in this workspace. The current
full suite passes **214 tests** (`python -m pytest evaluation tests -q`). The
targeted scorer/controller/product command passed **100 tests**. These are
engineering checks and historical smoke experiments, not a real gap benchmark.
The earlier product-contract failures in results.md are a historical snapshot;
the latest full-suite run passed after concurrent product engineering changes.

## Implemented components

| File | Purpose |
| --- | --- |
| `autoresearch.py` | One-change local proposals, immutable copied trials, fixed acceptance gate, deadlines, results journal, checked resume. |
| `context_experiment.py` | Four representations crossed with static/managed memory; scorable manifests, usage and paired contrasts. |
| `local_model.py` | Loopback inference through the existing Gateway with exact tokenizer, schema overhead, usage and model-digest checks. |
| `export_context.py` | Reads actual published bundle/passages into frozen experiment inputs with source hashes. |
| `okf_audit.py` | Core Markdown/YAML envelope and domain-extension diagnostics. |
| `vendor/context_language_models/` | Pinned upstream code and original license/notices. |
| `test_autoresearch.py` | Scripted tests of acceptance/rejection, isolation, resume, forbidden changes, deadlines, citations, support, export and token boundaries. |

Input contracts: `autoresearch.schema.json`, `context.schema.json`, and the
main `manifest.schema.json`. Procedure and interpretation: AUTORESEARCH.md.

## Model and research code provenance

Live generation and proposal model: local Ollama `qwen3.5:4b`, Q4_K_M,
model digest `d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04`.
Tokenizer SHA-256:
`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.
The latest run records evaluation-code hashes, frozen inputs, source copies
and model digest in its state. Neither weights nor production settings changed.

Upstream CLM commit: `483198229ab7b90c0eed4651306ab3b3505aaf50`.
`upstream.json` lists file hashes and original paths. The executed imports are
episode records, contrastive notes, proposal validation and paired-task SE.
The CC BY-NC 4.0 research subset remains separate from product source.

## Local-model autoresearch outcomes

| Artifact directory under `data/` | Outcome |
| --- | --- |
| `autoresearch-context-live-001` | Malformed evidence citation rejected before candidate evaluation. Early diagnostic; no improvement claim. |
| `autoresearch-context-live-002` | Model proposed requiring multiple corroborating sources. On one source, the candidate abstained: accuracy 1 to 0; hallucination/evidence guards undefined. Candidate rejected. |
| `autoresearch-bundle-live-001` | Final controller executed on exported actual product notes. Candidate proposed stricter literal-source verification. Accuracy stayed 1, evidence precision stayed 1, unsupported-claim rate stayed 0. No strict improvement; candidate rejected and baseline retained. |

The final actual-bundle round took 55.875 seconds in the recorded run. Its
seven recorded calls comprise baseline generation/grading, the proposal,
and candidate generation/grading. The largest reported prompt-plus-output
usage was **619 tokens**, below the 8,000-token ceiling. This short run does
not demonstrate performance at a full context window; token boundary behavior
is separately tested.

Inspect `data/autoresearch-bundle-live-001/round-0001/decision.json`, the raw
proposal, proposer request, usage, worker logs, copied workspaces and
`results.tsv`. `best.json` points to the baseline. Unit tests additionally
exercise keeping a genuinely improved scripted candidate. No live run here
has demonstrated a successful improvement.

The most recent run can be checked without rerunning the model by reading
its state, decisions and manifests. Re-running the same controller command
requires the recorded code/config/input/model identities to match; it will
refuse a changed experiment rather than combine incompatible runs.

## OKF audit and actual-bundle export

`data/okf-audit-transformer.json` reports the existing Transformer corpus:

- Two Markdown/YAML concept files passed the core-envelope check.
- One typed link, marked verified by the product.
- No dangling note targets or unresolved source pointers.
- No standard `sources` fields; passage provenance is stored in domain fields.
- No full optional-family or general-consumer conformance certification.

`data/context-export-transformer.json` contains the actual extracted notes,
paper title and exact source passage. Its adjacent provenance JSON hashes
the original source files. The exporter required a matching cutoff and
local-only import. It also exposed and corrected a Windows CRLF parsing
issue before the successful export.

## Representation and context outcomes

All runs below replay a single question about the Transformer authors' goal
to make generation less sequential. The source is a dated excerpt from
Attention Is All You Need v1. The 2017 cutoff predates this model's training,
so these are integration replays, not future-blind evaluations.

`context-factorial-live-001` was an early diagnostic with missing paper-title
metadata in the raw arm. Do not use its apparent raw/notes difference as
evidence. The corrected manually supplied control is
`context-factorial-live-002`: all eight arms scored accuracy 1, citation
precision 1 and unsupported-claim rate 0 on its single question.

The actual exported product notes were then used in
`data/context-bundle-live-001`. These are the recorded results for one repeat:

| Arm | Answer accuracy | Citation precision | Unsupported-claim rate | Generation tokens |
| --- | ---: | ---: | ---: | ---: |
| raw-static | 1 | 1 | 0 | 390 |
| raw-managed | 1 | 1 | 0 | 409 |
| flat-static | 0 | undefined | undefined | 720 |
| flat-managed | 1 | 1 | 0 | 691 |
| typed-static | 0 | undefined | undefined | 771 |
| typed-managed | 1 | 1 | 0 | 986 |
| plain-static | 1 | 1 | 0 | 607 |
| plain-managed | 1 | 1 | 0 | 968 |

The two zero-accuracy cases abstained after source IDs were not retained in
final memory. Undefined claim/citation rates are not zero hallucinations.
Tokens above sum generation input/output; grading overhead is separate in
the report. The report contains ten paired contrast records and full traces.

This result does not establish that OKF or typed structure helps. Raw passages
already answered the question correctly and used fewer tokens. Plain prose
also succeeded. Static and managed instructions can elicit different outputs,
but this one-batch stream never overflowed the 600-token memory, so the run
cannot establish superior long-context retention. Local same-model grades
need independent checks. One question, one topic and one repeat cannot support
generalized performance or significance claims.

## What remains a research question

The implemented pipeline can now test whether the source-linked typed
representation, graph tools, or learned memory discipline improve a 4B model.
That requires the frozen real corpus, questions and gap labels described in
METRICS.md, long source streams, three repeats and independent topic-level
comparisons. The full temporal gap autoresearch adapter is implemented but
has not been run as a complete future-blind optimization study. No weight
training, full upstream unrestricted-CLM reproduction, ContextBench result,
or measured novel-gap improvement is claimed.
