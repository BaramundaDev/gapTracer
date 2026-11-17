# GapTrace

Finds research gaps in a set of papers, with an evidence trail for each one.
Built to run on a small local model: every model call fits in 8,000 tokens,
and nothing counts as evidence unless a source passage supports it.

## Terminal-first workflow

Start from a topic and let the local model build a research map:

```powershell
python -m gaptrace.cli research
python -m gaptrace.cli research "retrieval augmented generation" --papers 30 --k 3
```

The first command offers suggested topics and accepts your own text. The second
collects up to 30 unique papers from OpenAlex, extracts and verifies notes with
the configured local Qwen model, investigates the top three gap candidates,
and brainstorms testable experiments for supported candidates left open by the
investigation. Use `--provider arxiv` for arXiv or `--as-of YYYY-MM-DD` for a
publication cutoff. Internet access is needed for scholarly searches; model
inference stays local. Ollama and the configured models must be installed.

This workflow reads **abstracts**, not full PDFs. Its gap and experiment reports
are candidates for further research, not proof of novelty. `--papers` accepts
1–100; the default is 30. Fewer papers may be available for a narrow topic.

Each run creates a new directory under `data/research/` with `report.md`,
`research.json`, the downloaded `papers.json`, and a reusable `corpus/`.
Use `--data NEW_DIRECTORY` to choose the run location and `--json` for machine
output. Existing directories are never overwritten. Downloaded papers and
completed passages and model responses are checkpointed if inference fails.
Continue the same run without downloading again:

```powershell
python -m gaptrace.cli research --resume "data/research/YOUR_RUN_DIRECTORY"
```

Runs created before checkpoint support keep their downloaded papers but must
extract the unsaved passages once again. Local Ollama calls now use its native
streaming API, report generation progress every 30 seconds while receiving
output, and retry a stalled or disconnected request once. If that is not enough,
the research workflow automatically resumes the interrupted stage twice, with
a short increasing delay and reuse of completed passages and responses.
Use `--auto-retries 0` to disable stage recovery or choose a value up to 5.
Checkpoint files are flushed to disk and checked for damage; damaged derived
caches are preserved and rebuilt automatically. Completed gap reviews are saved
individually too. Ctrl+C saves the interrupted status and stops immediately.
After closing the program or restarting the computer, use `--resume` above.
Only one process can write to a given research directory at a time.
Run `research` inside the
interactive shell to switch automatically to the newly built corpus.

Gap candidates can now be investigated with scholarly search and local source
checks, then reranked using the existing evidence/closure/feasibility ordering:

```powershell
.\.research-venv\Scripts\python.exe -m gaptrace.cli review-gaps --data data/terminal-demo --k 3
```

See [gap investigation and ranking evaluation](docs/GAP_REVIEW.md) for assistant
tools, saved evidence, and comparisons using the existing ranking metrics.

On the prepared Windows workspace, run the guided walkthrough and interactive
shell with:

```powershell
cd 'D:\GAPTRACER MAIN\gaptrace'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\try.ps1
```

See [the terminal guide](docs/TERMINAL.md) for setup, visible run logs and
importing your own papers. Add `-NoShell` for a walkthrough that returns to
PowerShell.

Core correctness work and its validation are recorded in
[CORE_FIXES.md](docs/CORE_FIXES.md). Answers now undergo local sentence-level
source verification before being returned. Unsupported sentences are removed,
and a wholly unsupported answer becomes "Not found in these papers."
Validation decisions and added inference usage are recorded in each run's trail.

The terminal is the primary interface while the core is developed. After
installing the project, start with a sample corpus and open a persistent session:

```powershell
python -m gaptrace.cli demo --data data/demo
python -m gaptrace.cli shell --data data/demo
```

Inside the session, use the same commands as the standalone CLI:

```text
status
search "replay buffer"
gaps
trail <gap-id>
note <note-id>
passage <passage-id>
fact --method replay
ask "What limitations do these papers report?" --local-only
use "data/another-corpus"
help
exit
```

Search, gap discovery, evidence inspection, and status need no language-model
generation. The configured hybrid retrieval uses Ollama's local MiniLM embedding
model when opening a corpus, so keep Ollama running for these commands too.
`ask` and `ingest` use Ollama directly for the configured local model. Each `ask` is a fresh evidence
query; the shell retains the active corpus, not conversation history. Commands
reload corpus data so newly imported notes are available immediately. Quote
multiword questions and paths containing spaces. EOF or Ctrl+C at the prompt
closes the session; command errors leave it open.

To import real sectioned paper text, select a new output directory:

```powershell
python -m gaptrace.cli ingest papers.json --cutoff 2026-11-16 --data data/my-topic --local-only
python -m gaptrace.cli shell --data data/my-topic
```

Ingestion prints extraction, merge, verification, and save progress to stderr;
stdout remains JSON for scripting. Imports publish only after completion and
do not overwrite existing directories. Builds run in the foreground. Standalone
`ingest` does not support resume; topic `research` runs checkpoint completed work
and accept `--resume DIRECTORY`. `status --json` exposes corpus counts and import
statistics. Gap cards are candidates requiring closure review.

Run `gaptrace` (or `python -m gaptrace.cli`) without arguments to open the branded
interactive terminal. Long operations show elapsed time, actual tool/model
activity and measured per-request estimates when matching calls are available.
See [the terminal guide](docs/TERMINAL.md) for estimate limits and log behavior.

## Individual terminal commands

    pip install -e ".[dev,serve]"
    pytest -q
    gaptrace demo                   # writes a small sample corpus to ./data
    gaptrace rebuild
    gaptrace gaps
    gaptrace trail <gap id>
    gaptrace search "replay buffer"
    gaptrace fact --method replay --dataset cifar

`gaptrace ask "..."` needs Ollama with the local model. Hosted providers require `pip install litellm`.
`gaptrace mcp` runs the MCP server; `uvicorn --factory gaptrace.api:create_app` runs the API.

## Measure the local model

Start Ollama and make `nvidia-smi` available on `PATH`. Run these commands
from the repository root in PowerShell. The script writes `docs/hardware.md`.

```powershell
python -m pip install -e .
python -m pip install litellm==1.109.1 tokenizers==0.23.3
ollama pull qwen3.5:4b
curl.exe --fail --location https://huggingface.co/Qwen/Qwen3.5-4B/resolve/main/tokenizer.json --output "$env:TEMP\qwen35-tokenizer.json"
python scripts/measure.py --tokenizer "$env:TEMP\qwen35-tokenizer.json"
```

Qwen3.5 4B is the selected local language model. See
[GPU profiles](docs/gpu-profiles.md) for measurements and instructions for
running a separate Ollama instance. The earlier 2B comparison is historical;
that model has been retired.

## Import sectioned papers

Use `gaptrace ingest papers.json --cutoff YYYY-MM-DD --data my-corpus` to
build a new workspace from dated paper text. The input has a `papers` list
with an `id`, `title`, `published` date and `sections` for each paper. Each
section has a `heading` and `text`. `gaptrace corpus --data my-corpus` shows
the paper and passage IDs. See the [evaluation guide](evaluation/README.md)
for a real-paper example.

## Layout

    config/settings.yaml   budget, limits, tool caps, and an off switch per technique
    prompts/               every prompt the system sends, as text files
    src/gaptrace/
      schema.py            the fixed data model (notes, links, facts, gap cards, model outputs)
      bundle.py            note bundle (markdown + YAML) and passage store: the sources of truth
      facts.py             exact numbers in SQLite
      graph.py             note graph (verified links only for answers)
      search.py            BM25 + vectors merged by reciprocal rank fusion
      context.py           the context manager: budget, stubs, state file, trail
      tools.py             the tools shared by agent, MCP server, API and command line
      gaps.py              the four gap patterns and the code-only ranking
      gateway.py           one door for model calls: token gate, cache, schema check, routing
      pipeline.py          extract -> merge -> verify
      research.py          topic -> online abstracts -> evidence map -> gap review -> proposed experiments
      closure.py           closure check and doability
      agent.py             the step loop
      cli.py, mcp_server.py, api.py
    tests/                 run against a hand-made sample corpus with fake model backends
    evaluation/            research track only; nothing here is imported by the product
    docs/AGENT_BRIEF.md    what is done and what is left, phase by phase

See DEPENDENCIES.md for every dependency and its licence.
