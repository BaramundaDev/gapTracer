# Bounded development improvement program

Improve the frozen development objective without raising hallucination rate
or reducing evidence quality. This program governs the proposer; it never
authorizes changing the scorer or benchmark labels. See AUTORESEARCH.md.

Run `python -m evaluation.autoresearch <plan.json> --tokenizer <tokenizer.json> --out data/<new-run>`.

## Fixed boundaries

Read development inputs only. Held-out plans are rejected. Benchmark mode
requires at least three repeats. Preserve model, schema, the 8,000-token
budget, source bundle, corpus, questions, dates, thresholds and scoring.
The local proposer has no shell tool.

One proposal changes one target in a private copied workspace:

- Context engine: the experiment's context-policy text.
- Live engine: `prompts/step_agent.txt` OR one allowlisted retrieval/gap/step
  setting or named feature switch. No arbitrary file path or command.

## Loop

1. Read the incumbent's development note and recent trial decisions.
2. Propose one change with a falsifiable hypothesis citing an actual episode
   and step, beginning for example `E1 step 1 shows ...`.
3. Freeze the proposal and create an isolated candidate copy. The controller
   does not edit production prompts or settings.
4. Run the fixed adapter under its wall-clock limit. Preserve failures and
   partial artifacts; never convert missing grades into zeros.
5. Keep only a defined objective improvement exceeding both recorded repeat
   spreads and the minimum gain, with defined nonregressing hallucination
   and evidence guardrails. Otherwise retain the incumbent. Topic SE is
   diagnostic. Ties are rejected.
6. Record proposal, measurements, errors and decision in the per-round
   artifacts, state.json and results.tsv. Stop at the round/time limit.

The default temporal objective is hit_rate. The context-only example uses
answer_accuracy explicitly; that cannot establish better gap discovery.
Do not change the objective or thresholds after inspecting scores.

There is no Git repository in this workspace. Immutable snapshots and hashes
implement keep/revert; best.json points to the selected workspace. Cached
replays or a retained candidate do not establish scientific novelty or
independent improvement. Preserve negative results, and evaluate a final
configuration on untouched held-out topics outside this tuning loop.
