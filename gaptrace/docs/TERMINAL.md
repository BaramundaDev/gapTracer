# Try GapTrace in PowerShell

## Interactive terminal

Open the GapTracer welcome screen and persistent prompt with no subcommand:

```powershell
cd 'D:\GAPTRACER MAIN\gaptrace'
.\.research-venv\Scripts\gaptrace.exe
```

`python -m gaptrace.cli` does the same thing. Use `gaptrace shell --data PATH`
to open a specific corpus. Opening the shell does not create sample data or
download papers. Run `research` to build a corpus, or `demo` for invented sample
papers. Run `help` to see all commands. Each `ask` is a fresh evidence query.

Research, import, question answering, and gap review display timestamped activity
and a live elapsed-time footer on interactive terminals. Activity describes real
operations: retrieval, model requests, tools, cache reuse, schema repairs and
source checks. It does not display private model reasoning. A first request has
no time estimate. Later requests show an approximate remaining time based on
the mean duration of completed calls using the same model and output schema
in that operation. If a request exceeds that average, remaining time returns
to unknown. These estimates concern the current request; the total research
duration remains unknown because searches, retries and tool loops vary.

Redirected logs use stable lines without animation. All activity goes to stderr;
`--json` keeps stdout as machine-readable JSON. Press Ctrl+C to interrupt a
command; inside the shell, the prompt remains available. Research retains its
existing checkpoint/resume behavior. `NO_COLOR=1` disables terminal colors.

## Research workflow

To start with your own research topic on this prepared workspace:

```powershell
cd 'D:\GAPTRACER MAIN\gaptrace'
.\.research-venv\Scripts\python.exe -m gaptrace.cli research
```

Choose a suggested topic number or enter your own topic. To run directly:

```powershell
.\.research-venv\Scripts\python.exe -m gaptrace.cli research "continual learning" --papers 30 --k 3
```

The command searches OpenAlex online, reads up to 30 abstracts with local Qwen,
extracts and verifies research notes, reviews the top three gap candidates for
closure, and proposes experiments for supported gaps left open by that search.
Progress appears during each stage. Reports and downloaded source abstracts
are saved under a new `data/research/` directory; `corpus/` can be opened with
`shell --data PATH_TO_CORPUS`. Inside an existing shell, `research` switches
automatically to the resulting corpus.

Options: `--papers 1..100`, `--k 1..10`, `--provider openalex|arxiv`,
`--as-of YYYY-MM-DD`, `--data NEW_DIRECTORY`, and `--json` (requires a topic
argument). OpenAlex optionally uses `OPENALEX_API_KEY` for a larger API budget;
see its [authentication guide](https://help.openalex.org/api/authentication/).
This initial collection workflow uses abstracts, not full PDFs. Missing
details remain unknown, and an open candidate is not proof of novelty.
Inference may take several minutes per paper on the 4 GB GPU; begin with
`--papers 2 --k 1` for a small run. For full sectioned paper text, use `ingest`.

If an inference request times out, resume the saved directory:

```powershell
.\.research-venv\Scripts\python.exe -m gaptrace.cli research --resume "data/research/YOUR_RUN_DIRECTORY"
```

This uses the saved topic, publication cutoff and downloaded papers. Completed
extractions and successful model responses are written after each request, so
future interruptions do not lose completed passages. Existing completed corpora
are reused. A run made before checkpoint support cannot recover its unsaved
extracted notes; it will read those passages once again. Resume supports one
process at a time per directory. The local adapter uses Ollama directly, with
180 seconds of socket inactivity allowed and a 600-second overall deadline per
attempt. A timeout, disconnection or crashed inference request retries once. If
it still fails, the research stage automatically resumes up to two more times,
waiting 2 and 4 seconds between resumptions. Successful passages and verified
responses are reused. Set `--auto-retries 0` to stop after request-level retry,
or choose up to 5 automatic stage resumptions. Permanent configuration and
file-permission errors stop without retrying. Exhausted recovery saves an
incomplete status and prints the resume command. GPU placement remains under
Ollama's control.

Each successful passage produces a durable checkpoint and updates the saved
completed-passage count. Damaged derived checkpoints and response caches are
renamed for inspection and rebuilt. Missing or malformed `papers.json` can be
restored from the saved source metadata; valid edited source text is never
silently replaced. Gap reviews are checkpointed individually. A process lock
prevents simultaneous writers and is released automatically if the process
exits. Ctrl+C records an interrupted state without automatically restarting.
If the app or computer exits, restart with `--resume`; recovery does not launch
a background service or change system startup settings.

To try the sample walkthrough instead:

From any PowerShell terminal on this machine:

```powershell
cd 'D:\GAPTRACER MAIN\gaptrace'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\try.ps1
```

The launcher uses the project Python environment without activation. It creates
`data/terminal-demo` on first use and reuses it thereafter. The sample contains
three invented papers, five source passages and four kinds of gap candidate.
It displays corpus counts, retrieval results, exact measurements, ranked gaps,
one evidence trail and a source passage. Then it asks Qwen a question, prints
answer/citation JSON and source-verification events, and opens the interactive
shell. The first model request can take longer while Ollama loads Qwen.

Inside the shell:

```text
status
search "replay buffer"
fact --method replay
gaps
ask "What limitations do these papers report?" --local-only
ask "Do these papers establish that replay helps language models?" --local-only
help
exit
```

Use IDs printed by search/gaps with `note ID` or `trail ID`. `passage a-1`
shows the original text. Each question creates a run directory under the
corpus's `runs` folder. Its `trail.jsonl` records model steps, tool use and
answer validation. Unsupported answers may correctly abstain. Each question
starts a fresh query; this is not a conversational memory session. Repeated
requests may reuse the gateway cache, which is recorded in the trail.

To watch events appear while a question is running, open a second PowerShell
terminal after starting the question and run:

```powershell
cd 'D:\GAPTRACER MAIN\gaptrace'
$latestRun = Get-ChildItem data/terminal-demo/runs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$trailFile = Join-Path $latestRun.FullName 'trail.jsonl'
while (-not (Test-Path -LiteralPath $trailFile)) { Start-Sleep -Seconds 1 }
Get-Content -LiteralPath $trailFile -Wait
```

Events appear after each completed model step or check; this is not a streamed
model response. Press Ctrl+C to stop watching. The first model call may take
time before producing its first event.

To run the walkthrough and return to PowerShell, add `-NoShell`. To skip the
answer call, add `-NoAsk`. Hybrid retrieval still uses Ollama's MiniLM model.
Choose a fresh demo location with `-Data data/another-demo`; use this launcher
with the sample corpus, since its walkthrough includes sample passage IDs.

## Runtime setup and recovery

The runtime needs Ollama listening on `http://127.0.0.1:11434`, Qwen3.5 4B and
MiniLM. Check the server/models with `ollama list`. If the server is stopped,
start the Ollama desktop app or run `ollama serve` in a separate terminal.
If a model is missing:

```powershell
ollama pull qwen3.5:4b
ollama pull all-minilm:l6-v2
```

If rebuilding the Python environment is necessary (Python 3.11 or newer):

```powershell
python -m venv .research-venv
.\.research-venv\Scripts\python.exe -m pip install -e '.[models]'
```

The bundled Qwen tokenizer is selected by `config/settings.yaml`. Cloud
inference is disabled. Source text, bundles, indexes, caches and run logs stay
under the selected data directory.

## Import your papers

Prepare JSON containing `papers`, each with `id`, `title`, `published`
(`YYYY-MM-DD`) and `sections` containing `heading` and `text`. See
`evaluation/examples/transformer-v1.json` for the smallest real-paper example
(a single excerpt, not a full paper). Import into a new output directory:

```powershell
.\.research-venv\Scripts\python.exe -m gaptrace.cli ingest .\evaluation\examples\transformer-v1.json --cutoff 2017-06-12 --data data/my-transformer --local-only
.\.research-venv\Scripts\python.exe -m gaptrace.cli shell --data data/my-transformer
```

Then try `ask "What research goal did the Transformer authors identify for
generation?" --local-only` as one line. Ingestion prints progress and does not
overwrite existing corpora. A tiny excerpt may produce no gap candidates.
Automatic literature collection and resumable imports remain future work.

The independent research benchmark remains incomplete. Terminal functionality
and software tests do not establish research accuracy or novelty of gap cards.

## Verified on this workspace

The prepared environment imports this checkout, includes LiteLLM, and passes
`pip check`. The complete offline audit passed all 230 tests; see
`data/evals/terminal-ready-001/summary.md`. The PowerShell walkthrough completed
with an answer of 71.2% for Paper A and 65.0% for Paper B, supported by passages
`a-1` and `b-1`. Its first successful local request took about 45 seconds,
including startup; cached requests can be faster. Shell status/help/exit were
also exercised. The real-paper example was imported separately into
`data/terminal-transformer`, preserving existing corpora.

To explore that prepared real-paper excerpt directly:

```powershell
.\.research-venv\Scripts\python.exe -m gaptrace.cli shell --data data/terminal-transformer
```
