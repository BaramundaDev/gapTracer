"""Measure local model inference and write the Phase 0 hardware report."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import date
from importlib.metadata import version
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Literal

from pydantic import BaseModel

from gaptrace import tokens
from gaptrace.context import ContextBudgetExceeded
from gaptrace.gateway import Gateway, ModelOutputInvalid, litellm_backend
from gaptrace.settings import load_settings

OUTPUT_LIMIT = 250
PROMPT_SIZES = (1000, 4000, 8000)
GPU_SAMPLE_SECONDS = 0.1
SAMPLE_OUTPUT = " ".join(map(str, range(1, 85)))  # 246 Qwen tokens as compact JSON.


class MeasurementOutput(BaseModel, extra="forbid"):
    summary: Literal[SAMPLE_OUTPUT]


def measurement_prompt(target: int, label: str) -> str:
    prefix = (
        f"Run {label}. Return a compact JSON object with one key, summary. "
        "The value is a string with the numbers 1 through 84, separated by single spaces. Ignore this padding:\n\n"
    )
    low, high = 0, target
    while low <= high:
        middle = (low + high) // 2
        prompt = prefix + " x" * middle
        size = tokens.count(prompt)
        if size == target:
            return prompt
        if size < target:
            low = middle + 1
        else:
            high = middle - 1
    raise ValueError(f"Could not construct a {target:,}-token prompt with this tokenizer")


def gpu_query(index: int, fields: str) -> str:
    return subprocess.check_output(
        ["nvidia-smi", "-i", str(index), f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True, timeout=5,
    ).strip()


def measure_case(
    gateway: Gateway, response_log: list[dict], gpu: int, size: int,
) -> tuple[int, int, float, float, int, str]:
    prompt = measurement_prompt(size, str(size))
    first_response = len(response_log)
    samples = [int(gpu_query(gpu, "memory.used"))]
    stop = Event()
    errors = []

    def poll_gpu() -> None:
        try:
            while not stop.wait(GPU_SAMPLE_SECONDS):
                samples.append(int(gpu_query(gpu, "memory.used")))
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            errors.append(exc)

    sampler = Thread(target=poll_gpu)
    sampler.start()
    start = perf_counter()
    failure = None
    try:
        gateway.call("local", prompt, MeasurementOutput)
    except (ContextBudgetExceeded, ModelOutputInvalid) as exc:
        if len(response_log) == first_response:
            raise
        failure = exc
    finally:
        seconds = perf_counter() - start
        stop.set()
        sampler.join()
    if errors:
        raise RuntimeError("Could not sample GPU memory") from errors[0]
    samples.append(int(gpu_query(gpu, "memory.used")))
    replies = response_log[first_response:]
    if not replies:
        raise RuntimeError("LiteLLM did not expose Ollama timing data")
    last = replies[-1]
    decode_rate = last["eval_count"] * 1e9 / last["eval_duration"]
    if isinstance(failure, ContextBudgetExceeded):
        status = "Invalid JSON, retry blocked by budget"
    elif failure is not None:
        status = f"Invalid JSON after {len(replies)} attempts"
    else:
        status = "Yes" if len(replies) == 1 else f"Yes, after {len(replies)} attempts"
    return last["prompt_eval_count"], last["eval_count"], seconds, decode_rate, max(samples), status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path, help="Tokenizer JSON for the configured model")
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--ollama-url", help="Ollama server URL, for example http://127.0.0.1:11435")
    parser.add_argument("--gpu", type=int, default=0, help="nvidia-smi GPU index")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs/hardware.md")
    args = parser.parse_args()

    from tokenizers import Tokenizer

    settings = load_settings(args.settings)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokens.set_counter(lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids))
    gpu_name, memory, driver = gpu_query(args.gpu, "name,memory.total,driver_version").split(", ")
    ollama_env = dict(os.environ, OLLAMA_HOST=args.ollama_url) if args.ollama_url else None
    ollama_version = subprocess.check_output(["ollama", "--version"], env=ollama_env, text=True, timeout=10).strip()
    model = settings.model.local
    replies: list[dict] = []

    def capture_ollama_response(event: dict) -> None:
        if event.get("log_event_type") == "post_api_call":
            response = json.loads(event["original_response"])
            if "eval_duration" in response:
                replies.append(response)

    backend = litellm_backend(model, settings.model.local_num_ctx, api_base=args.ollama_url,
                              logger_fn=capture_ollama_response)

    def capped_backend(prompt: str, schema: dict, max_tokens: int) -> str:
        return backend(prompt, schema, min(OUTPUT_LIMIT, max_tokens))

    gateway = Gateway(settings, {"local": capped_backend})
    budget = settings.budget
    largest = budget.total - budget.output - budget.margin
    lines = [
        "# GapTrace Phase 0 hardware measurement", "",
        f"Measured {date.today():%Y-%m-%d} on {platform.system()} {platform.release()}.", "",
        f"- GPU: {gpu_name}, {memory} MiB, NVIDIA driver {driver}",
        f"- Model: `{model}`, configured context {settings.model.local_num_ctx:,} tokens",
        f"- Runtime: {ollama_version}, LiteLLM {version('litellm')}, tokenizers {version('tokenizers')}",
        f"- Output cap: {min(OUTPUT_LIMIT, budget.output)} tokens per attempt", "",
        "| Input target | Ollama input | Output | Wall time | Decode tokens/s | Peak GPU | Valid JSON |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for size in sorted({*PROMPT_SIZES, largest}):
        print(f"Measuring {size:,}-token input", flush=True)
        try:
            input_count, output, seconds, decode_rate, peak, json_status = measure_case(
                gateway, replies, args.gpu, size,
            )
        except ContextBudgetExceeded as exc:
            if size <= largest:
                raise
            lines.append(f"| {size:,} | — | — | — | — | — | Blocked by gateway |")
            print(exc, flush=True)
            continue
        lines.append(f"| {size:,} | {input_count:,} | {output} | {seconds:.2f} s | "
                     f"{decode_rate:.1f} | {peak:,} MiB | {json_status} |")
        print(lines[-1], flush=True)
    runners = subprocess.check_output(["ollama", "ps"], env=ollama_env, text=True, timeout=10).strip()
    print(runners)
    lines += [
        "", "## Method", "",
        f"The gateway reserves {budget.output:,} output tokens and {budget.margin:,} margin tokens "
        f"within the {budget.total:,}-token policy. The {largest:,}-token input exercises the configured ceiling.",
        "Input targets count user text with the model tokenizer. Ollama input counts include chat framing. "
        "GPU figures are sampled device-wide peaks, including other GPU activity. "
        "Ollama token counts and decode rates describe the final attempt. Wall time includes prompt processing, "
        "retries, and any model load. The model may stay loaded between rows. "
        "The fixed-value JSON schema keeps output length comparable across inputs.",
        "", "## Loaded runner", "", "```text", runners, "```",
    ]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
