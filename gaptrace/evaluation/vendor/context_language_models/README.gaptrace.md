# Context Language Models research subset

Source: https://github.com/facebookresearch/context-language-models

Commit: `483198229ab7b90c0eed4651306ab3b3505aaf50`.

The six files in `upstream.json` are unmodified copies of `clm/clm_icl/`
files, relocated together as a Python package. `LICENSE` and `NOTICE` are
from the repository root. Copyright belongs to Meta Platforms, Inc. and
affiliates and named upstream contributors. The license is CC BY-NC 4.0;
this directory is for noncommercial evaluation. This README and provenance
JSON are local additions, not upstream files.

GapTrace imports `Episode`, `build_note`, `parse_proposal`, `Score` and
`diff_se`. The preserved `Evolution` class is a reference; our controller
adds evidence/hallucination guardrails and bounded private copies. The
upstream direct LiteLLM proposer is not executed: local research inference
passes through the existing product Gateway with the exact model tokenizer.
The unrestricted CLM agent, Harbor tasks, RL and suffix cache reuse are not
executed here. The full upstream NOTICE is preserved even for components
outside this subset.

Citation: Rulin Shao et al. Context Language Models, arXiv:2609.37725 (2026).
https://arxiv.org/abs/2609.37725
