# Dependencies and licences

## Research integration checked on 16 November 2026

The autoresearch controller reuses `Episode`, `build_note`, `parse_proposal`,
`Score`, and `diff_se` from the official
[Context Language Models repository](https://github.com/facebookresearch/context-language-models),
commit `483198229ab7b90c0eed4651306ab3b3505aaf50`. Six unmodified upstream
ICL files are relocated together in `evaluation/vendor/context_language_models/`.
Original copyright headers, `LICENSE`, `NOTICE`, and pinned provenance are
retained. The inspected license is **CC BY-NC 4.0**. This is a noncommercial
research integration, not an MIT/Apache product dependency. Do not include
these files under a blanket product license. Harbor, RL and serving stacks
are not installed. No upstream training code or weights from
[Karpathy autoresearch](https://github.com/karpathy/autoresearch) are copied;
its inspected MIT-licensed repository provides the experiment-loop pattern.

An isolated `.research-venv` uses system site packages plus these installations.
Licenses below were checked in the installed distribution metadata. Original
license files remain in those distributions. Only the tokenizer is used by
the adapter; its installer brought the other packages. No mandatory product
dependency was added. Preserve model/tokenizer hashes with results.

| Package | Installed version | Distribution license |
| --- | --- | --- |
| tokenizers | 0.23.3 | Apache-2.0 classifier |
| huggingface-hub | 2.4.0 | Apache-2.0 |
| filelock | 4.2.10 | MIT |
| fsspec | 2026.9.0 | BSD-3-Clause |
| hf-xet | 1.8.0 | Apache-2.0 |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |

Licences were read from the PyPI and npm registries on 14 Nov 2026 (latest
release shown). GROBID and the Context Language Models repository were read
from their LICENSE files. Re-check before a release; a licence can change
between versions.

## Used by the starter code

| Package | Latest | Licence | Used for |
| --- | --- | --- | --- |
| pydantic | 2.14.3 | MIT | Schemas and model-output validation |
| pyyaml | 6.0.3 | MIT | Note front matter, settings |
| networkx | 3.7 | BSD-3-Clause | Note graph |
| rank-bm25 | 0.2.2 | Apache-2.0 | Keyword search |
| numpy | 2.5.4 | BSD-3-Clause and others | Vector search, rank fusion |
| typer | 0.29.1 | MIT | Command line |
| mcp | 2.3.0 | MIT | MCP server (v2 API: MCPServer) |
| fastapi | 0.163.0 | MIT | HTTP API |
| uvicorn | 0.54.0 | BSD-3-Clause | API server |
| pytest | 9.2.0 | MIT | Tests |
| litellm | 1.109.1 | [MIT (SDK)](https://github.com/BerriAI/litellm/blob/main/LICENSE) | Model gateway backend and hardware measurement |
| tokenizers | 0.23.3 | [Apache-2.0](https://github.com/huggingface/tokenizers/blob/main/LICENSE) | Exact prompt counts in hardware, grounding probes and retrieval baseline runs |

## To add in later phases

| Package | Latest | Licence | Use for |
| --- | --- | --- | --- |
| ollama | 0.6.3 | MIT | Local model client, if LiteLLM is not used |
| sentence-transformers | 6.3.0 | Apache-2.0 | Embeddings |
| arxiv | 4.1.0 | MIT | arXiv search and download |
| pyalex | 0.21 | MIT | OpenAlex search |
| semanticscholar | 0.12.0 | MIT | Semantic Scholar search |
| docling | 2.150.1 | MIT | PDF to sectioned text |
| GROBID (Java service) | n/a | Apache-2.0 | Alternative PDF parser |
| llama-index-core | 0.14.26 | MIT | Text splitters, if needed |
| langchain-text-splitters | 1.1.3 | MIT | Text splitters, if needed |
| instructor | 1.17.0 | MIT | Structured output helper, if needed |
| bm25s | 0.3.16 | MIT | Faster BM25 for large corpora |
| chromadb | 1.5.9 | Apache-2.0 | Only if notes exceed what numpy handles |
| faiss-cpu | 1.16.0 | MIT | Same |

## Reference projects (read for patterns; copy only with attribution)

| Package | Latest | Licence |
| --- | --- | --- |
| lightrag-hku | 1.5.7 | MIT |
| nano-graphrag | 0.0.8.2 | MIT |
| graphrag | 3.3.0 | MIT |
| paper-qa | 2026.8.12 | Apache-2.0 |

## Front end (npm)

| Package | Latest | Licence |
| --- | --- | --- |
| next | 16.4.6 | MIT |
| react | 19.3.1 | MIT |
| tailwindcss | 4.3.5 | MIT |
| shadcn | 4.23.0 | MIT |
| @xyflow/react (React Flow) | 12.12.1 | MIT |
| @assistant-ui/react | 0.15.36 | MIT |
| @tanstack/react-table | 9.3.1 | MIT |
| cytoscape | 3.35.0 | MIT |
| @21st-dev/magic | 0.2.3 | ISC |

Components installed from 21st.dev are community-published: check each one.

## Evaluation model

The optional AWS connector build contains only standard-library application
code; it uses the AWS Lambda Python runtime's boto3 (Apache-2.0). Local
CloudFormation validation was run with cfn-lint 1.60.1 (MIT-0), installed as a
development tool. Neither cfn-lint nor a model runtime is included in the Lambda
zip. Local MCP continues to use the existing MIT-licensed mcp SDK.

| Model | Licence | Used for |
| --- | --- | --- |
| `all-minilm:l6-v2` via Ollama | [Apache-2.0](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Semantic ranking of later papers in the evaluation runner |

## Do not use without asking

| Project | Licence | Why |
| --- | --- | --- |
| PyMuPDF 2.1 | AGPL-3.0 or commercial | Copyleft; would bind a distributed product |
| facebookresearch/context-language-models | CC BY-NC 4.0 | Non-commercial; cite the paper, do not copy code |
| marker-pdf 2.2.0 | Apache-2.0 (package) | Fine as code, but check the licence of its model weights first |
