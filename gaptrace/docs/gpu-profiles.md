# Local GPU profiles

Current selection: Qwen3.5 4B only for language-model tasks. The 2B model
was removed on November 16, 2026 by user request. Its measurements below
are retained as historical results.

Measured on November 16, 2026 on an RTX 3050 Laptop GPU with 4 GiB of
VRAM. The comparison used Ollama 0.40.16, one inference slot, an 8,192-token
runtime context, and a 250-token output cap. The test used a fixed JSON
schema and recorded decode throughput from Ollama. Each row is a single
run at that prompt size.

| Ollama profile | Model placement | Decode tokens/s at 1k / 4k / 5.3k | Peak GPU memory | Schema validation |
| --- | --- | --- | ---: | --- |
| Qwen3.5 4B, original server | 52% CPU / 48% GPU | 17.8 / 15.4 / 14.4 | 2,547 MiB | Passed all three on the first attempt |
| Qwen3.5 4B, Flash Attention and `q8_0` KV cache | 50% CPU / 50% GPU | 18.7 / 18.2 / 17.7 | 2,569 MiB | Passed all three on the first attempt |
| Qwen3.5 2B Q4_K_M, Flash Attention and `q8_0` KV cache | 100% GPU | 100.0 / 96.1 / 96.9 | 2,355 MiB | Passed 1k on retry. Output was truncated at 4k and 5.3k. |

Decode rates describe the last attempt, including attempts with truncated
output. The 2B model reached the 250-token cap before closing the JSON object
at 4k and 5.3k. The retry for 5.3k would have exceeded the prompt budget and was
blocked. The benchmark measures hardware placement and structured output on
one synthetic task. Corpus extraction accuracy and CPU utilization remain
unmeasured. The product still defaults to the 4B model.

## Reproduce the isolated profile

After installing the measurement dependencies listed in the README, run a
separate Ollama server on loopback from PowerShell. [Ollama GPU and cache settings](https://docs.ollama.com/faq#how-can-i-set-the-quantization-type-for-the-kv-cache)

```powershell
$env:OLLAMA_HOST = '127.0.0.1:11435'
$env:OLLAMA_FLASH_ATTENTION = '1'
$env:OLLAMA_KV_CACHE_TYPE = 'q8_0'
$env:OLLAMA_NUM_PARALLEL = '1'
ollama serve
```

With that server running, run the measurement from the repository root in
another PowerShell session. Use `--settings` with a copy of `config/settings.yaml`
whose `model.local` is `ollama_chat/qwen3.5:4b`. Use the selected 4B model
for subsequent measurements.

```powershell
python scripts/measure.py --ollama-url http://127.0.0.1:11435 --tokenizer "$env:TEMP\qwen35-tokenizer.json" --out "$env:TEMP\gaptrace-gpu-run.md"
```
