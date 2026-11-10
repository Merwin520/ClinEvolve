# Notice

This repository is a derivative of HyperAgents (Zhang et al., 2026, arXiv:2603.19461), Copyright (c) Meta
Platforms, Inc. and affiliates, licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License (CC BY-NC-SA 4.0). The derivative work is distributed under the same license (LICENSE.md).

Files derived from HyperAgents, modified for ClinEvolve:
- `meta_agent.py`: generic repairer with objective, feedback and edit-scope arguments
- `agent/base_agent.py`, `agent/llm.py`, `agent/llm_withtools.py`, `agent/llm_withtools_meta.py`: local vLLM endpoints,
  tolerant tool-call parsing, and tool-loop limits
- `agent/tools/__init__.py`, `agent/tools/bash.py`, `agent/tools/edit.py`
- `utils/common.py`, `utils/thread_logger.py`

New in ClinEvolve: `task_agent.py`, the clinical tools in `agent/tools/get_*.py`, `agent/clinical_context.py`,
`agent/paths.py` and the `clinevolve/` package.

Several internal schema and version strings (for example `exp2_candidate_registry_v1`) keep their original names so
that released candidate records and evolved programs remain loadable.

Third-party data and benchmarks are not included. MIMIC-IV and MIMIC-CDM are subject to the PhysioNet credentialed
data-use agreement.
