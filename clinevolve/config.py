"""ClinEvolve configuration.

All method parameters live in one YAML file (see ``configs/clinevolve.yaml``). ``load_config`` deep-merges a user file over
the defaults below, which are the values used in the paper.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Clinical failure endpoints (binary, per patient trajectory; 1 = failure).
PD = "diagnostic_process_failure"
PL = "lab_interpretation_failure"
PT = "treatment_guideline_failure"
FD = "final_diagnosis_failure"
IT = "instruction_tool_failure"
WORKFLOW = [PD, PL, PT]
SHORT = {PD: "PD", PL: "PL", PT: "PT", FD: "FD", IT: "IT"}
TARGETS = {"PD": PD, "PL": PL, "PT": PT}

DEFAULTS: Dict[str, Any] = {
    "paths": {
        # MIMIC-CDM CSV directory (PhysioNet credentialed access; not distributed with this code).
        "mimic_cdm_dir": "data/mimic-cdm",
        # Case manifest: one row per patient case with columns case_id, subject_id, outer_fold, role
        # (role in {repair, validation, locked_test}); outer_fold is a partition id (use 1 for a single split).
        "cases": "data/cases.csv",
        "work_dir": "runs",
    },
    "models": {
        # Task agent model (the program under repair). Temperature, max_tokens and max_tool_calls are fixed inside
        # task_agent.py and are protected by the frozen-invariant check.
        "task": {"name": "openai/qwen3.5-9b", "servers": ["http://127.0.0.1:8801/v1"]},
        # Meta agent (repairer) model.
        "meta": {"name": "openai/qwen3.5-27b", "servers": ["http://127.0.0.1:8500/v1"],
                 "temperature": 0.7, "max_tokens": 2048},
        "api_key": "dummy",
    },
    "evaluation": {
        "shards_per_evaluation": 16,      # concurrent episode workers per program evaluation
        "evaluations_per_server": 8,      # concurrent program evaluations per task server
        "generations_per_server": 24,     # concurrent meta-agent sessions per meta server
        "feedback_max_examples": 20,
    },
    "sandbox": {
        # bubblewrap sandbox for every meta-agent session: the code base is read-only, the candidate workspace is the
        # only writable program copy, and patient data and every evaluation output are hidden.
        "enabled": True,
        "extra_hidden_paths": [],
        "extra_readonly_paths": [],
    },
    # Edit scope of a task-repair session.
    "scope": {
        "allowed_paths": ["task_agent.py"],
        "forbidden_paths": ["agent", "clinevolve", "configs", "meta_agent.py", "outputs", "runs", "data"],
    },
    # Safety gates G1-G6 (validation cohort; candidate vs reference).
    "gates": {
        "target_minimum_improvement_pp": 1.0,
        "retention_margin_pp": 2.0,
        "nontarget_workflow_margin_pp": 3.0,
        "final_diagnosis_margin_pp": 1.0,
        "tool_safety_margin_pp": 0.6,
        "resource_inflation_margin": {"mean_tool_calls_ratio": 1.25, "mean_total_tokens_ratio": 1.5},
        "new_regression_margin_pp": 5.0,
    },
    # Single-target repair from a fresh agent: k candidates, gate, select (CI@k).
    "repair": {"candidates": 10, "budgets_k": [5, 10], "generation_waves": [5, 10]},
    # Sequential repair PD -> PL -> PT with an archive.
    "sequential": {
        "stage_order": [PD, PL, PT],
        "candidates_per_stage": 4,
        "validation_candidates_per_stage": 2,
        "archive_capacity": 2,
        "use_gates": True,
    },
    # Tree-search single-target repair (Experiment 2 configuration).
    "tree_search": {
        "candidate_budget": 8,
        "policy": {"alpha": 0.6, "initial_root_expansions": 1, "tau_min": 1.0, "tau_max": 50.0,
                   "max_evaluations_per_node": 1, "random_seed": 20260908},
        "search_budget_units": 16,
        "validation_candidates": 2,
    },
    # Repairer self-improvement: two tournament rounds of self-edited meta_agent.py variants scored by expected safe gain.
    "self_improvement": {
        "rounds": 2,
        "variants_per_round": 3,
        "candidates_per_target": 4,
        "targets": [PD, PT],
    },
    # ClinEvolve-Lineage search over task-agent versions.
    "lineage": {"policy": "lineage", "target": PT, "steps": 12, "children_per_step": 4, "ucb_c_pp": 2.0,
                "seed": 20261018},
}


def _merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _abs(p: str) -> str:
    q = Path(os.path.expanduser(p))
    return str(q if q.is_absolute() else (REPO_ROOT / q).resolve())


def load_config(path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = DEFAULTS
    if path:
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    for k in ("mimic_cdm_dir", "cases", "work_dir"):
        cfg["paths"][k] = _abs(cfg["paths"][k])
    os.environ["MIMIC_CDM_DATA_DIR"] = cfg["paths"]["mimic_cdm_dir"]
    os.environ.setdefault("VLLM_API_KEY", cfg["models"].get("api_key", "dummy"))
    return cfg
