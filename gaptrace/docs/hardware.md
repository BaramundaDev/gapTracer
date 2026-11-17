# GapTrace Phase 0 hardware measurement

Measured 2026-11-16 on Windows 11.

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, NVIDIA driver 610.60
- Model: `ollama_chat/qwen3.5:4b`, configured context 8,192 tokens
- Runtime: ollama version is 0.40.16, LiteLLM 1.109.1, tokenizers 0.23.3
- Output cap: 250 tokens per attempt

| Input target | Ollama input | Output | Wall time | Decode tokens/s | Peak GPU | Valid JSON |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1,000 | 1,012 | 247 | 23.00 s | 17.8 | 2,547 MiB | Yes |
| 4,000 | 4,012 | 247 | 21.76 s | 15.4 | 2,547 MiB | Yes |
| 5,300 | 5,312 | 247 | 24.15 s | 14.4 | 2,547 MiB | Yes |
| 8,000 | — | — | — | — | — | Blocked by gateway |

## Method

The gateway reserves 1,200 output tokens and 1,500 margin tokens within the 8,000-token policy. The 5,300-token input exercises the configured ceiling.
Input targets count user text with the model tokenizer. Ollama input counts include chat framing. GPU figures are sampled device-wide peaks, including other GPU activity. Decode rate uses Ollama's generation token count and duration. Wall time includes prompt processing and any model load. Each row is one request sequence, and the model may stay loaded between rows. The fixed-value JSON schema keeps output length comparable across inputs.

## Loaded runner

```text
NAME          ID              SIZE      PROCESSOR          CONTEXT    RUNNER      UNTIL              
qwen3.5:4b    d8b0f5e9760c    3.9 GB    52%/48% CPU/GPU    8192       llamacpp    4 minutes from now
```
