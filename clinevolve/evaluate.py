#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


# =============================================================================
# Project root
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =============================================================================
# Frozen Exp1 runtime/evaluators
# =============================================================================

from agent.clinical_context import (
    clear_current_hadm_id,
    set_current_hadm_id,
)

from clinevolve.endpoints.trajectory_metrics import (
    extract_trajectory_metrics,
)

from clinevolve.endpoints.diagnosis_evaluator import (
    DiagnosisGroundTruth,
    evaluate_trajectory as evaluate_diagnosis_trajectory,
)

from clinevolve.endpoints.instruction_tool_evaluator import (
    evaluate_instruction_tool,
)

from clinevolve.endpoints.clinical_failure_evaluator import (
    ClinicalGroundTruth,
    evaluate_clinical_failures,
)

from clinevolve.search.controller import (
    SearchEvaluationResult,
)
from agent.paths import mimic_cdm_dir


# =============================================================================
# Constants
# =============================================================================

EVALUATOR_VERSION = "exp2_candidate_evaluator_v1"

MIMIC_ROOT = mimic_cdm_dir()

DEFAULT_HPI_PATH = (
    MIMIC_ROOT
    / "history_of_present_illness.csv"
)

VALID_ROLES = {
    "repair",
    "validation",
    "locked_test",
}

FAILURE_DOMAINS = (
    "diagnostic_process_failure",
    "lab_interpretation_failure",
    "treatment_guideline_failure",
    "final_diagnosis_failure",
    "instruction_tool_failure",
)

TOOL_NAMES = (
    "get_physical_exam",
    "get_lab_results",
    "get_imaging_results",
)

DEFAULT_MAX_TOOL_CALLS = 40


# =============================================================================
# Helpers
# =============================================================================

def now_iso() -> str:

    return (
        datetime.now()
        .astimezone()
        .isoformat()
    )


def sha256_file(
    path: str | Path,
) -> str:

    path = Path(
        path
    )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def safe_json_dump(
    obj: Any,
    path: str | Path,
) -> None:

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        f.write(
            "\n"
        )

    tmp.replace(
        path
    )


def load_json(
    path: str | Path,
) -> Dict[str, Any]:

    path = Path(
        path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        payload = json.load(
            f
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return payload


def load_table(
    path: str | Path,
) -> pd.DataFrame:

    path = Path(
        path
    )

    suffix = (
        path.suffix.lower()
    )

    if suffix == ".csv":
        return pd.read_csv(
            path
        )

    if suffix in {
        ".parquet",
        ".pq",
    }:
        return pd.read_parquet(
            path
        )

    if suffix in {
        ".jsonl",
        ".ndjson",
    }:
        return pd.read_json(
            path,
            lines=True,
        )

    raise ValueError(
        "Unsupported table format: "
        f"{path.suffix}"
    )


def normalize_case_id(
    value: Any,
) -> str:

    text = str(
        value
    ).strip()

    if text.endswith(
        ".0"
    ):
        text = text[:-2]

    return text


def finite_or_none(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    try:
        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        result
    ):
        return None

    return result


# =============================================================================
# LLM usage
# =============================================================================

def aggregate_llm_usage(
    msg_history: Sequence[Any],
) -> Dict[str, Optional[int]]:
    """
    Same semantics as Exp1 runner:
    aggregate per-assistant-message llm_info usage.
    """

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    found_usage = False

    for message in (
        msg_history
    ):

        if not isinstance(
            message,
            dict,
        ):
            continue

        info = message.get(
            "llm_info"
        )

        if not isinstance(
            info,
            dict,
        ):
            continue

        prompt = info.get(
            "prompt_tokens"
        )

        completion = info.get(
            "completion_tokens"
        )

        total = info.get(
            "total_tokens"
        )

        if (
            prompt
            is not None
        ):

            prompt_tokens += int(
                prompt
            )

            found_usage = True

        if (
            completion
            is not None
        ):

            completion_tokens += int(
                completion
            )

            found_usage = True

        if (
            total
            is not None
        ):

            total_tokens += int(
                total
            )

            found_usage = True

        elif (
            prompt
            is not None
            or completion
            is not None
        ):

            total_tokens += (
                int(
                    prompt
                    or 0
                )
                + int(
                    completion
                    or 0
                )
            )

    if not found_usage:

        return {
            "prompt_tokens":
                None,

            "completion_tokens":
                None,

            "total_tokens":
                None,
        }

    return {
        "prompt_tokens":
            prompt_tokens,

        "completion_tokens":
            completion_tokens,

        "total_tokens":
            total_tokens,
    }


# =============================================================================
# Candidate TaskAgent loading
# =============================================================================

def load_candidate_task_agent(
    candidate_repo: str | Path,
):
    """
    Load TaskAgent from candidate workspace without replacing the root
    repository's task_agent.py.

    Exp2 candidate modification scope should normally make the candidate
    task_agent.py the modified implementation while the frozen tools and
    evaluators remain external.
    """

    candidate_repo = Path(
        candidate_repo
    ).resolve()

    task_agent_path = (
        candidate_repo
        / "task_agent.py"
    )

    if not (
        task_agent_path.exists()
    ):
        raise FileNotFoundError(
            "Candidate task_agent.py not found: "
            f"{task_agent_path}"
        )

    module_name = (
        "exp2_candidate_task_agent_"
        + sha256_file(
            task_agent_path
        )[:16]
    )

    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            task_agent_path,
        )
    )

    if (
        spec is None
        or spec.loader
        is None
    ):
        raise ImportError(
            "Could not create import spec for "
            f"{task_agent_path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    TaskAgent = getattr(
        module,
        "TaskAgent",
        None,
    )

    if TaskAgent is None:
        raise AttributeError(
            "Candidate task_agent.py does not "
            "define TaskAgent"
        )

    return (
        TaskAgent,
        task_agent_path,
    )


# =============================================================================
# Cohort / fold input
# =============================================================================

def load_hpi_lookup(
    hpi_path: str | Path,
) -> Dict[str, str]:

    hpi_path = Path(
        hpi_path
    )

    if not hpi_path.exists():
        raise FileNotFoundError(
            f"HPI file not found: {hpi_path}"
        )

    hpi_df = pd.read_csv(
        hpi_path,
        usecols=[
            "hadm_id",
            "hpi",
        ],
    )

    hpi_df[
        "hadm_id"
    ] = (
        hpi_df[
            "hadm_id"
        ]
        .map(
            normalize_case_id
        )
    )

    hpi_df[
        "hpi"
    ] = (
        hpi_df[
            "hpi"
        ]
        .fillna("")
        .astype(str)
    )

    hpi_df = (
        hpi_df
        .drop_duplicates(
            subset=[
                "hadm_id",
            ],
            keep="first",
        )
    )

    return dict(
        zip(
            hpi_df[
                "hadm_id"
            ],
            hpi_df[
                "hpi"
            ],
        )
    )


def select_manifest_cases(
    *,
    fold_manifest_path: str | Path,
    outer_fold: int,
    role: str,
) -> pd.DataFrame:
    """
    Select exactly one frozen role from one outer fold.

    No other role is returned.
    """

    if (
        not isinstance(outer_fold, int)
        or outer_fold < 1
    ):
        raise ValueError(
            "outer_fold (partition id) must be a positive integer"
        )

    if (
        role
        not in VALID_ROLES
    ):
        raise ValueError(
            f"Invalid role: {role}"
        )

    manifest = load_table(
        fold_manifest_path
    )

    required = {
        "outer_fold",
        "case_id",
        "patient_id",
        "disease",
        "role",
    }

    missing = (
        required
        - set(
            manifest.columns
        )
    )

    if missing:
        raise ValueError(
            "Fold manifest missing columns: "
            f"{sorted(missing)}"
        )

    selected = (
        manifest[
            (
                manifest[
                    "outer_fold"
                ]
                == outer_fold
            )
            &
            (
                manifest[
                    "role"
                ].astype(str)
                == role
            )
        ]
        [
            [
                "case_id",
                "patient_id",
                "disease",
                "role",
            ]
        ]
        .copy()
    )

    if selected.empty:
        raise RuntimeError(
            "No cases selected for "
            f"fold={outer_fold}, role={role}"
        )

    selected[
        "case_id"
    ] = (
        selected[
            "case_id"
        ].map(
            normalize_case_id
        )
    )

    selected[
        "patient_id"
    ] = (
        selected[
            "patient_id"
        ].map(
            normalize_case_id
        )
    )

    if (
        selected[
            "case_id"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Duplicate case_id within selected fold role"
        )

    return (
        selected
        .sort_values(
            "case_id"
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# Five-domain evaluator
# =============================================================================

def build_failure_mapping(
    *,
    trajectory: Mapping[str, Any],
    diagnosis_ground_truth: DiagnosisGroundTruth,
    clinical_ground_truth: ClinicalGroundTruth,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> Dict[str, Any]:
    """
    Reuse the exact frozen Exp1 evaluator modules.

    This function intentionally does not redesign any failure rule.
    """

    trajectory = dict(
        trajectory
    )

    trajectory_metrics = (
        extract_trajectory_metrics(
            trajectory.get(
                "message_history",
                [],
            ),
            trajectory.get(
                "prediction"
            ),
            trajectory.get(
                "latency_seconds"
            ),
        )
    )

    clinical_eval = (
        evaluate_clinical_failures(
            trajectory=(
                trajectory
            ),
            ground_truth=(
                clinical_ground_truth
            ),
        )
    )

    diagnostic_process_failure = (
        clinical_eval.get(
            "diagnostic_process_failure"
        )
    )

    lab_interpretation_failure = (
        clinical_eval.get(
            "lab_interpretation_failure"
        )
    )

    treatment_guideline_failure = (
        clinical_eval.get(
            "treatment_guideline_failure"
        )
    )

    if (
        trajectory.get(
            "status"
        )
        == "success"
    ):

        diagnosis_eval = (
            evaluate_diagnosis_trajectory(
                trajectory=(
                    trajectory
                ),
                ground_truth=(
                    diagnosis_ground_truth
                ),
            )
        )

    else:

        diagnosis_eval = {
            "diagnosis_correct":
                None,

            "final_diagnosis_failure":
                None,

            "evaluation_status":
                "trajectory_not_successful",

            "predicted_diagnosis_raw":
                "",

            "predicted_disease":
                None,

            "ground_truth_disease":
                trajectory.get(
                    "disease"
                ),

            "ground_truth_source":
                None,
        }

    instruction_eval = (
        evaluate_instruction_tool(
            trajectory=(
                trajectory
            ),
            max_tool_calls=(
                max_tool_calls
            ),
        )
    )

    final_diagnosis_failure = (
        diagnosis_eval.get(
            "final_diagnosis_failure"
        )
    )

    instruction_tool_failure = (
        instruction_eval.get(
            "instruction_tool_failure"
        )
    )

    five_values = [
        diagnostic_process_failure,
        lab_interpretation_failure,
        treatment_guideline_failure,
        final_diagnosis_failure,
        instruction_tool_failure,
    ]

    available_values = [
        value
        for value
        in five_values
        if value is not None
    ]

    complete_five_domain_mapping = (
        len(
            available_values
        )
        == 5
    )

    if (
        complete_five_domain_mapping
    ):

        error_burden_5 = int(
            sum(
                int(
                    value
                )
                for value
                in five_values
            )
        )

        any_failure = int(
            error_burden_5
            > 0
        )

    else:

        error_burden_5 = None
        any_failure = None

    diagnostic_details = (
        clinical_eval.get(
            "diagnostic_process_details",
            {},
        )
    )

    lab_details = (
        clinical_eval.get(
            "lab_interpretation_details",
            {},
        )
    )

    treatment_details = (
        clinical_eval.get(
            "treatment_guideline_details",
            {},
        )
    )

    n_invalid_calls = (
        int(
            instruction_eval.get(
                "n_unknown_tool_calls",
                0,
            )
            or 0
        )
        + int(
            instruction_eval.get(
                "n_invalid_tool_inputs",
                0,
            )
            or 0
        )
        + int(
            instruction_eval.get(
                "n_malformed_tool_calls",
                0,
            )
            or 0
        )
    )

    return {
        # identity
        "run_id":
            trajectory.get(
                "run_id"
            ),

        "case_id":
            str(
                trajectory.get(
                    "case_id",
                    "",
                )
            ),

        "disease":
            trajectory.get(
                "disease"
            ),

        "agent_id":
            trajectory.get(
                "agent_id"
            ),

        "base_model":
            trajectory.get(
                "base_model"
            ),

        "presentation_type":
            trajectory.get(
                "presentation_type",
                "main",
            ),

        "trajectory_status":
            trajectory.get(
                "status"
            ),

        # final diagnosis
        "diagnosis_correct":
            diagnosis_eval.get(
                "diagnosis_correct"
            ),

        # five failure domains
        "diagnostic_process_failure":
            diagnostic_process_failure,

        "lab_interpretation_failure":
            lab_interpretation_failure,

        "treatment_guideline_failure":
            treatment_guideline_failure,

        "final_diagnosis_failure":
            final_diagnosis_failure,

        "instruction_tool_failure":
            instruction_tool_failure,

        # applicability/status
        "diagnostic_process_applicable":
            clinical_eval.get(
                "diagnostic_process_applicable"
            ),

        "lab_interpretation_applicable":
            clinical_eval.get(
                "lab_interpretation_applicable"
            ),

        "treatment_guideline_applicable":
            clinical_eval.get(
                "treatment_guideline_applicable"
            ),

        "diagnostic_process_status":
            clinical_eval.get(
                "diagnostic_process_status"
            ),

        "lab_interpretation_status":
            clinical_eval.get(
                "lab_interpretation_status"
            ),

        "treatment_guideline_status":
            clinical_eval.get(
                "treatment_guideline_status"
            ),

        "diagnosis_evaluation_status":
            diagnosis_eval.get(
                "evaluation_status"
            ),

        # burden
        "complete_five_domain_mapping":
            complete_five_domain_mapping,

        "error_burden_5":
            error_burden_5,

        "any_failure":
            any_failure,

        "n_available_failure_domains":
            len(
                available_values
            ),

        # diagnosis audit
        "predicted_diagnosis_raw":
            diagnosis_eval.get(
                "predicted_diagnosis_raw"
            ),

        "predicted_disease":
            diagnosis_eval.get(
                "predicted_disease"
            ),

        "ground_truth_disease":
            diagnosis_eval.get(
                "ground_truth_disease"
            ),

        "ground_truth_source":
            diagnosis_eval.get(
                "ground_truth_source"
            ),

        # diagnostic-process audit
        "diagnostic_process_components":
            diagnostic_details.get(
                "components",
                {},
            ),

        "required_lab_categories":
            diagnostic_details.get(
                "required_lab_categories",
                [],
            ),

        "covered_lab_categories":
            diagnostic_details.get(
                "covered_lab_categories",
                [],
            ),

        "missing_lab_categories":
            diagnostic_details.get(
                "missing_lab_categories",
                [],
            ),

        "valid_imaging_calls":
            diagnostic_details.get(
                "valid_imaging_calls",
                [],
            ),

        # lab audit
        "n_evaluable_labs":
            lab_details.get(
                "n_evaluable_labs",
                0,
            ),

        "n_lab_interpretation_claims":
            lab_details.get(
                "n_interpretation_claims",
                0,
            ),

        "n_incorrect_lab_interpretations":
            lab_details.get(
                "n_incorrect_claims",
                0,
            ),

        "incorrect_lab_interpretations":
            lab_details.get(
                "incorrect_claims",
                [],
            ),

        # treatment audit
        "treatment_required":
            treatment_details.get(
                "treatment_required",
                {},
            ),

        "treatment_requested":
            treatment_details.get(
                "treatment_requested",
                {},
            ),

        "missing_required_treatments":
            treatment_details.get(
                "missing_required_treatments",
                [],
            ),

        "biliary_pancreatitis":
            treatment_details.get(
                "biliary_pancreatitis"
            ),

        # instruction/tool audit
        "instruction_tool_failure_reasons":
            instruction_eval.get(
                "failure_reasons",
                [],
            ),

        "n_invalid_calls":
            n_invalid_calls,

        "n_unknown_tool_calls":
            instruction_eval.get(
                "n_unknown_tool_calls",
                0,
            ),

        "n_invalid_tool_inputs":
            instruction_eval.get(
                "n_invalid_tool_inputs",
                0,
            ),

        "n_malformed_tool_calls":
            instruction_eval.get(
                "n_malformed_tool_calls",
                0,
            ),

        "n_tool_execution_errors":
            instruction_eval.get(
                "n_tool_execution_errors",
                0,
            ),

        "final_response_valid":
            instruction_eval.get(
                "final_response_valid"
            ),

        "final_response_errors":
            instruction_eval.get(
                "final_response_errors",
                [],
            ),

        # workflow
        "n_messages":
            trajectory_metrics.get(
                "n_messages"
            ),

        "n_rounds":
            trajectory_metrics.get(
                "n_rounds"
            ),

        "n_tool_calls":
            trajectory_metrics.get(
                "n_tool_calls"
            ),

        "n_history_calls":
            trajectory_metrics.get(
                "n_history_calls",
                0,
            ),

        "n_exam_calls":
            trajectory_metrics.get(
                "n_exam_calls",
                0,
            ),

        "n_lab_calls":
            trajectory_metrics.get(
                "n_lab_calls",
                0,
            ),

        "n_imaging_calls":
            trajectory_metrics.get(
                "n_imaging_calls",
                0,
            ),

        "n_repeated_calls":
            trajectory_metrics.get(
                "n_repeated_calls",
                0,
            ),

        "unique_tools_used":
            trajectory_metrics.get(
                "unique_tools_used"
            ),

        "time_to_diagnosis_round":
            trajectory_metrics.get(
                "time_to_diagnosis_round"
            ),

        "stop_reason":
            trajectory_metrics.get(
                "stop_reason"
            ),

        # compute
        "latency_seconds":
            trajectory_metrics.get(
                "latency_seconds"
            ),

        "prompt_tokens":
            trajectory_metrics.get(
                "prompt_tokens"
            ),

        "completion_tokens":
            trajectory_metrics.get(
                "completion_tokens"
            ),

        "total_tokens":
            trajectory_metrics.get(
                "total_tokens"
            ),

        "tool_sequence":
            trajectory_metrics.get(
                "tool_sequence",
                [],
            ),

        "clinical_failure_details": {
            "diagnostic_process":
                diagnostic_details,

            "lab_interpretation":
                lab_details,

            "treatment_guideline":
                treatment_details,
        },

        "instruction_tool_details": {
            "unknown_tool_calls":
                instruction_eval.get(
                    "unknown_tool_calls",
                    [],
                ),

            "invalid_tool_inputs":
                instruction_eval.get(
                    "invalid_tool_inputs",
                    [],
                ),

            "malformed_tool_calls":
                instruction_eval.get(
                    "malformed_tool_calls",
                    [],
                ),

            "tool_execution_errors":
                instruction_eval.get(
                    "tool_execution_errors",
                    [],
                ),
        },
    }


# =============================================================================
# Single case
# =============================================================================

def run_single_case(
    *,
    TaskAgent,
    case_id: str,
    patient_id: str,
    hpi: str,
    disease: str,
    role: str,
    outer_fold: int,
    candidate_id: str,
    model: str,
    output_root: Path,
    temperature: float,
    max_tokens: int,
    candidate_task_agent_sha256: str,
    diagnosis_ground_truth: DiagnosisGroundTruth,
    clinical_ground_truth: ClinicalGroundTruth,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> Dict[str, Any]:

    case_id = normalize_case_id(
        case_id
    )

    run_id = (
        f"exp2_"
        f"f{outer_fold}_"
        f"{role}_"
        f"{candidate_id}_"
        f"{case_id}"
    )

    trajectory_path = (
        output_root
        / "trajectories"
        / f"{case_id}.json"
    )

    failure_path = (
        output_root
        / "failure_mapping"
        / f"{case_id}.json"
    )

    chat_history_path = (
        output_root
        / "logs"
        / f"{case_id}.md"
    )

    started_at = (
        now_iso()
    )

    start_time = (
        time.perf_counter()
    )

    set_current_hadm_id(
        case_id
    )

    try:

        # Fresh TaskAgent for every patient.
        agent = TaskAgent(
            model=(
                model
            ),

            chat_history_file=str(
                chat_history_path
            ),

            temperature=(
                temperature
            ),

            max_tokens=(
                max_tokens
            ),
        )

        # Never expose case_id/patient_id/disease/role to TaskAgent.
        inputs = {
            "domain":
                "mimic_cdm",

            "hpi":
                str(
                    hpi
                ),
        }

        prediction, msg_history = (
            agent.forward(
                inputs
            )
        )

        latency = (
            time.perf_counter()
            - start_time
        )

        metrics = (
            extract_trajectory_metrics(
                msg_history,
                prediction,
                latency,
            )
        )

        usage = (
            aggregate_llm_usage(
                msg_history
            )
        )

        metrics.update(
            usage
        )

        trajectory = {
            "run_id":
                run_id,

            "case_id":
                case_id,

            "patient_id":
                patient_id,

            "disease":
                disease,

            "agent_id":
                candidate_id,

            "base_model":
                model,

            "presentation_type":
                "main",

            "status":
                "success",

            "started_at":
                started_at,

            "finished_at":
                now_iso(),

            "latency_seconds":
                latency,

            "config": {
                "temperature":
                    temperature,

                "max_tokens":
                    max_tokens,

                "max_tool_calls":
                    max_tool_calls,

                "tools":
                    list(
                        TOOL_NAMES
                    ),

                "outer_fold":
                    outer_fold,

                "role":
                    role,

                "candidate_task_agent_sha256":
                    candidate_task_agent_sha256,
            },

            "prediction":
                prediction,

            "metrics":
                metrics,

            "usage":
                usage,

            "message_history":
                msg_history,

            "evaluator":
                None,

            "failures":
                None,
        }

        safe_json_dump(
            trajectory,
            trajectory_path,
        )

        failure_mapping = (
            build_failure_mapping(
                trajectory=(
                    trajectory
                ),

                diagnosis_ground_truth=(
                    diagnosis_ground_truth
                ),

                clinical_ground_truth=(
                    clinical_ground_truth
                ),

                max_tool_calls=(
                    max_tool_calls
                ),
            )
        )

        failure_mapping[
            "patient_id"
        ] = patient_id

        failure_mapping[
            "outer_fold"
        ] = outer_fold

        failure_mapping[
            "role"
        ] = role

        failure_mapping[
            "candidate_task_agent_sha256"
        ] = (
            candidate_task_agent_sha256
        )

        safe_json_dump(
            failure_mapping,
            failure_path,
        )

        return {
            **failure_mapping,

            "status":
                "success",

            "error":
                None,

            "trajectory_path":
                str(
                    trajectory_path.resolve()
                ),

            "failure_mapping_path":
                str(
                    failure_path.resolve()
                ),
        }

    except Exception as exc:

        latency = (
            time.perf_counter()
            - start_time
        )

        error = (
            f"{type(exc).__name__}: {exc}"
        )

        # ---------------------------------------------------------------------
        # IMPORTANT:
        #
        # A successfully completed TaskAgent inference may already have been
        # persisted before a downstream evaluator fails. Never destroy that
        # successful trajectory by replacing it with an inference-error record.
        #
        # Leaving failure_mapping absent allows the resume path to rebuild the
        # evaluator result locally on the next run.
        # ---------------------------------------------------------------------

        existing_success_trajectory = None

        if trajectory_path.exists():

            try:

                existing = load_json(
                    trajectory_path
                )

                if (
                    existing.get(
                        "status"
                    )
                    == "success"
                    and existing.get(
                        "agent_id"
                    )
                    == candidate_id
                    and existing.get(
                        "config",
                        {},
                    ).get(
                        "candidate_task_agent_sha256"
                    )
                    == candidate_task_agent_sha256
                ):

                    existing_success_trajectory = (
                        existing
                    )

            except Exception:
                existing_success_trajectory = None

        if existing_success_trajectory is not None:

            return {
                "run_id":
                    run_id,

                "case_id":
                    case_id,

                "patient_id":
                    patient_id,

                "disease":
                    disease,

                "diagnostic_process_failure":
                    None,

                "lab_interpretation_failure":
                    None,

                "treatment_guideline_failure":
                    None,

                "final_diagnosis_failure":
                    None,

                "instruction_tool_failure":
                    None,

                "status":
                    "evaluator_error",

                "error":
                    error,

                "trajectory_path":
                    str(
                        trajectory_path.resolve()
                    ),

                "failure_mapping_path":
                    None,
            }

        trajectory = {
            "run_id":
                run_id,

            "case_id":
                case_id,

            "patient_id":
                patient_id,

            "disease":
                disease,

            "agent_id":
                candidate_id,

            "base_model":
                model,

            "presentation_type":
                "main",

            "status":
                "error",

            "error":
                error,

            "traceback":
                traceback.format_exc(),

            "started_at":
                started_at,

            "finished_at":
                now_iso(),

            "latency_seconds":
                latency,

            "config": {
                "temperature":
                    temperature,

                "max_tokens":
                    max_tokens,

                "max_tool_calls":
                    max_tool_calls,

                "tools":
                    list(
                        TOOL_NAMES
                    ),

                "outer_fold":
                    outer_fold,

                "role":
                    role,

                "candidate_task_agent_sha256":
                    candidate_task_agent_sha256,
            },

            "prediction":
                None,

            "metrics": {
                "latency_seconds":
                    latency,
            },

            "message_history":
                [],
        }

        safe_json_dump(
            trajectory,
            trajectory_path,
        )

        # Preserve NA rather than silently converting failed trajectory to
        # clinical success/failure.
        row = {
            "run_id":
                run_id,

            "case_id":
                case_id,

            "patient_id":
                patient_id,

            "disease":
                disease,

            "agent_id":
                candidate_id,

            "base_model":
                model,

            "outer_fold":
                outer_fold,

            "role":
                role,

            "trajectory_status":
                "error",

            "status":
                "error",

            "error":
                error,

            "diagnostic_process_failure":
                None,

            "lab_interpretation_failure":
                None,

            "treatment_guideline_failure":
                None,

            "final_diagnosis_failure":
                None,

            "instruction_tool_failure":
                None,

            "n_tool_calls":
                None,

            "n_invalid_calls":
                None,

            "n_tool_execution_errors":
                None,

            "n_repeated_calls":
                None,

            "latency_seconds":
                latency,

            "prompt_tokens":
                None,

            "completion_tokens":
                None,

            "total_tokens":
                None,

            "candidate_task_agent_sha256":
                candidate_task_agent_sha256,

            "trajectory_path":
                str(
                    trajectory_path.resolve()
                ),

            "failure_mapping_path":
                None,
        }

        return row

    finally:

        clear_current_hadm_id()


# =============================================================================
# Aggregation
# =============================================================================

def _endpoint_from_dataframe(
    df: pd.DataFrame,
    domain: str,
) -> Dict[str, Any]:
    """
    Preserve conditional NA:
        n_evaluable = non-NA domain rows
        n_failures  = failures among evaluable rows
    """

    series = pd.to_numeric(
        df[
            domain
        ],
        errors="coerce",
    )

    evaluable = (
        series.notna()
    )

    n_total = int(
        len(
            df
        )
    )

    n_evaluable = int(
        evaluable.sum()
    )

    n_failures = int(
        (
            series[
                evaluable
            ]
            == 1
        ).sum()
    )

    return {
        "n_total":
            n_total,

        "n_evaluable":
            n_evaluable,

        "n_failures":
            n_failures,

        "failure_rate":
            (
                n_failures
                / n_evaluable
                if n_evaluable
                > 0
                else None
            ),
    }


def _mean_numeric(
    df: pd.DataFrame,
    column: str,
) -> Optional[float]:

    if (
        column
        not in df.columns
    ):
        return None

    values = pd.to_numeric(
        df[
            column
        ],
        errors="coerce",
    )

    if not (
        values.notna().any()
    ):
        return None

    return float(
        values.mean()
    )


def build_aggregate_evidence(
    *,
    evaluations: pd.DataFrame,
    candidate_id: str,
    target_id: str,
    outer_fold: int,
    role: str,
) -> Dict[str, Any]:

    failure_endpoints = {
        domain:
            _endpoint_from_dataframe(
                evaluations,
                domain,
            )
        for domain
        in FAILURE_DOMAINS
    }

    n_tool_calls = pd.to_numeric(
        evaluations.get(
            "n_tool_calls",
            pd.Series(
                dtype=float
            ),
        ),
        errors="coerce",
    )

    n_invalid_calls = pd.to_numeric(
        evaluations.get(
            "n_invalid_calls",
            pd.Series(
                dtype=float
            ),
        ),
        errors="coerce",
    )

    n_tool_execution_errors = pd.to_numeric(
        evaluations.get(
            "n_tool_execution_errors",
            pd.Series(
                dtype=float
            ),
        ),
        errors="coerce",
    )

    n_repeated_calls = pd.to_numeric(
        evaluations.get(
            "n_repeated_calls",
            pd.Series(
                dtype=float
            ),
        ),
        errors="coerce",
    )

    total_tool_calls = float(
        n_tool_calls.fillna(
            0
        ).sum()
    )

    total_invalid_calls = float(
        n_invalid_calls.fillna(
            0
        ).sum()
    )

    total_execution_errors = float(
        n_tool_execution_errors.fillna(
            0
        ).sum()
    )

    total_repeated_calls = float(
        n_repeated_calls.fillna(
            0
        ).sum()
    )

    tool_metrics = {
        "mean_tool_calls":
            _mean_numeric(
                evaluations,
                "n_tool_calls",
            ),

        "invalid_tool_call_rate":
            (
                total_invalid_calls
                / total_tool_calls
                if total_tool_calls
                > 0
                else 0.0
            ),

        "tool_execution_error_rate":
            (
                total_execution_errors
                / total_tool_calls
                if total_tool_calls
                > 0
                else 0.0
            ),

        "repeated_tool_call_rate":
            (
                total_repeated_calls
                / total_tool_calls
                if total_tool_calls
                > 0
                else 0.0
            ),
    }

    resource_metrics = {
        "mean_total_tokens":
            _mean_numeric(
                evaluations,
                "total_tokens",
            ),

        "mean_prompt_tokens":
            _mean_numeric(
                evaluations,
                "prompt_tokens",
            ),

        "mean_completion_tokens":
            _mean_numeric(
                evaluations,
                "completion_tokens",
            ),

        "mean_latency_sec":
            _mean_numeric(
                evaluations,
                "latency_seconds",
            ),
    }

    n_cases = int(
        len(
            evaluations
        )
    )

    n_successful_trajectories = int(
        (
            evaluations[
                "status"
            ]
            == "success"
        ).sum()
    )

    return {
        "schema_version":
            EVALUATOR_VERSION,

        "candidate_id":
            candidate_id,

        "target_id":
            target_id,

        "outer_fold":
            outer_fold,

        "role":
            role,

        "n_cases":
            n_cases,

        "n_successful_trajectories":
            n_successful_trajectories,

        "n_failed_trajectories":
            (
                n_cases
                - n_successful_trajectories
            ),

        # Directly compatible with experiments.exp2.utility.py
        "metrics": {
            "failure_endpoints":
                failure_endpoints,

            "tool_metrics":
                tool_metrics,

            "resource_metrics":
                resource_metrics,
        },
    }


# =============================================================================
# Full candidate evaluation
# =============================================================================

def evaluate_candidate(
    *,
    candidate_repo: str | Path,
    candidate_id: str,
    model: str,
    target_id: str,
    outer_fold: int,
    role: str,
    fold_manifest_path: str | Path,
    output_root: str | Path,
    hpi_path: str | Path = DEFAULT_HPI_PATH,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    allow_locked_test: bool = False,
    resume: bool = True,
    search_success: Optional[int] = None,
) -> SearchEvaluationResult:
    """
    Evaluate one candidate on exactly one frozen role.

    search_success is deliberately NOT inferred here.

    The Exp2 search runner must define/freeze what binary search feedback means
    and pass 0/1/None explicitly.
    """

    if (
        target_id
        not in {
            "diagnostic_process_failure",
            "treatment_guideline_failure",
        }
    ):
        raise ValueError(
            f"Unsupported target_id: {target_id}"
        )

    if (
        role
        not in VALID_ROLES
    ):
        raise ValueError(
            f"Invalid role: {role}"
        )

    if (
        role
        == "locked_test"
        and not allow_locked_test
    ):
        raise RuntimeError(
            "locked_test evaluation is blocked by default. "
            "Use the dedicated locked-test stage and explicitly "
            "set allow_locked_test=True."
        )

    if (
        search_success
        not in {
            None,
            0,
            1,
        }
    ):
        raise ValueError(
            "search_success must be 0, 1, or None"
        )

    candidate_repo = Path(
        candidate_repo
    ).resolve()

    output_root = Path(
        output_root
    ).resolve()

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        TaskAgent,
        task_agent_path,
    ) = load_candidate_task_agent(
        candidate_repo
    )

    task_agent_hash = (
        sha256_file(
            task_agent_path
        )
    )

    selected_cases = (
        select_manifest_cases(
            fold_manifest_path=(
                fold_manifest_path
            ),

            outer_fold=(
                outer_fold
            ),

            role=(
                role
            ),
        )
    )

    hpi_lookup = (
        load_hpi_lookup(
            hpi_path
        )
    )

    missing_hpi = [
        case_id
        for case_id
        in selected_cases[
            "case_id"
        ].tolist()
        if (
            case_id
            not in hpi_lookup
        )
    ]

    if missing_hpi:

        raise RuntimeError(
            "Missing HPI for selected cases. "
            f"Examples: {missing_hpi[:20]}"
        )

    diagnosis_ground_truth = (
        DiagnosisGroundTruth()
    )

    clinical_ground_truth = (
        ClinicalGroundTruth()
    )

    rows: List[
        Dict[str, Any]
    ] = []

    case_output_root = (
        output_root
        / (
            f"fold{outer_fold}"
        )
        / target_id
        / role
        / candidate_id
    )

    case_output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for (
        index,
        row,
    ) in enumerate(
        selected_cases.itertuples(
            index=False
        ),
        start=1,
    ):

        case_id = normalize_case_id(
            row.case_id
        )

        trajectory_path = (
            case_output_root
            / "trajectories"
            / f"{case_id}.json"
        )

        failure_path = (
            case_output_root
            / "failure_mapping"
            / f"{case_id}.json"
        )

        # ---------------------------------------------------------------------
        # Resume only when BOTH trajectory and evaluated failure mapping exist.
        # ---------------------------------------------------------------------

        if (
            resume
            and trajectory_path.exists()
            and failure_path.exists()
        ):

            try:

                trajectory = load_json(
                    trajectory_path
                )

                failure_mapping = load_json(
                    failure_path
                )

                if (
                    trajectory.get(
                        "agent_id"
                    )
                    == candidate_id
                    and trajectory.get(
                        "config",
                        {},
                    ).get(
                        "candidate_task_agent_sha256"
                    )
                    == task_agent_hash
                ):

                    rows.append(
                        {
                            **failure_mapping,

                            "status":
                                trajectory.get(
                                    "status"
                                ),

                            "error":
                                trajectory.get(
                                    "error"
                                ),

                            "trajectory_path":
                                str(
                                    trajectory_path.resolve()
                                ),

                            "failure_mapping_path":
                                str(
                                    failure_path.resolve()
                                ),
                        }
                    )

                    print(
                        f"[{index}/{len(selected_cases)}] "
                        f"{case_id}: resume"
                    )

                    continue

            except Exception:
                pass

        # ---------------------------------------------------------------------
        # Recovery path:
        #
        # If inference completed successfully and trajectory was persisted, but
        # failure mapping is missing (for example evaluator crashed), rebuild
        # the evaluator output locally WITHOUT calling TaskAgent again.
        # ---------------------------------------------------------------------

        if (
            resume
            and trajectory_path.exists()
            and not failure_path.exists()
        ):

            try:

                trajectory = load_json(
                    trajectory_path
                )

                if (
                    trajectory.get(
                        "agent_id"
                    )
                    == candidate_id
                    and trajectory.get(
                        "status"
                    )
                    == "success"
                    and trajectory.get(
                        "config",
                        {},
                    ).get(
                        "candidate_task_agent_sha256"
                    )
                    == task_agent_hash
                ):

                    failure_mapping = (
                        build_failure_mapping(
                            trajectory=(
                                trajectory
                            ),

                            diagnosis_ground_truth=(
                                diagnosis_ground_truth
                            ),

                            clinical_ground_truth=(
                                clinical_ground_truth
                            ),

                            max_tool_calls=(
                                max_tool_calls
                            ),
                        )
                    )

                    failure_mapping[
                        "patient_id"
                    ] = normalize_case_id(
                        row.patient_id
                    )

                    failure_mapping[
                        "outer_fold"
                    ] = outer_fold

                    failure_mapping[
                        "role"
                    ] = role

                    failure_mapping[
                        "candidate_task_agent_sha256"
                    ] = task_agent_hash

                    safe_json_dump(
                        failure_mapping,
                        failure_path,
                    )

                    rows.append(
                        {
                            **failure_mapping,

                            "status":
                                trajectory.get(
                                    "status"
                                ),

                            "error":
                                trajectory.get(
                                    "error"
                                ),

                            "trajectory_path":
                                str(
                                    trajectory_path.resolve()
                                ),

                            "failure_mapping_path":
                                str(
                                    failure_path.resolve()
                                ),
                        }
                    )

                    print(
                        f"[{index}/{len(selected_cases)}] "
                        f"{case_id}: recover-evaluator"
                    )

                    continue

            except Exception as exc:

                print(
                    f"[{index}/{len(selected_cases)}] "
                    f"{case_id}: evaluator recovery failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        total_cases = len(selected_cases)
        progress_width = 30
        progress_done = int(
            progress_width * index / total_cases
        )
        progress_bar = (
            "#" * progress_done
            + "-" * (
                progress_width - progress_done
            )
        )
        progress_pct = (
            100.0 * index / total_cases
        )

        print(
            f"\r[{progress_bar}] "
            f"{index}/{total_cases} "
            f"({progress_pct:5.1f}%) "
            f"case={case_id}",
            end="",
            flush=True,
        )

        result = run_single_case(
            TaskAgent=(
                TaskAgent
            ),

            case_id=(
                case_id
            ),

            patient_id=(
                normalize_case_id(
                    row.patient_id
                )
            ),

            hpi=(
                hpi_lookup[
                    case_id
                ]
            ),

            disease=(
                str(
                    row.disease
                )
            ),

            role=(
                role
            ),

            outer_fold=(
                outer_fold
            ),

            candidate_id=(
                candidate_id
            ),

            model=(
                model
            ),

            output_root=(
                case_output_root
            ),

            temperature=(
                temperature
            ),

            max_tokens=(
                max_tokens
            ),

            candidate_task_agent_sha256=(
                task_agent_hash
            ),

            diagnosis_ground_truth=(
                diagnosis_ground_truth
            ),

            clinical_ground_truth=(
                clinical_ground_truth
            ),

            max_tool_calls=(
                max_tool_calls
            ),
        )

        rows.append(
            result
        )

    evaluations = pd.DataFrame(
        rows
    )

    # -------------------------------------------------------------------------
    # Verify exact selected-case coverage.
    # -------------------------------------------------------------------------

    expected_case_ids = set(
        selected_cases[
            "case_id"
        ].astype(str)
    )

    actual_case_ids = set(
        evaluations[
            "case_id"
        ].astype(str)
    )

    if (
        actual_case_ids
        != expected_case_ids
    ):

        raise RuntimeError(
            "Evaluation coverage does not match "
            "frozen fold manifest role"
        )

    if (
        evaluations[
            "case_id"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Duplicate case evaluation detected"
        )

    # -------------------------------------------------------------------------
    # Persist per-case evaluation table.
    # -------------------------------------------------------------------------

    evaluation_path = (
        case_output_root
        / "evaluations.parquet"
    )

    # ---------------------------------------------------------------------
    # Parquet-safe normalization.
    #
    # PyArrow cannot serialize an empty struct such as {}:
    #     ArrowNotImplementedError:
    #     Cannot write struct type 'parameters' with no child field
    #
    # Evaluation artifacts may legitimately contain dict/list values,
    # including empty dictionaries. Preserve their full content by
    # serializing nested Python containers to canonical JSON strings before
    # writing the aggregate DataFrame to Parquet.
    # ---------------------------------------------------------------------
    import json as _json

    def _parquet_safe_value(value):
        if isinstance(value, (dict, list, tuple)):
            return _json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        return value

    for _column in evaluations.columns:
        if evaluations[_column].dtype == "object":
            evaluations[_column] = evaluations[_column].map(
                _parquet_safe_value
            )

    evaluations.to_parquet(
        evaluation_path,
        index=False,
    )

    # Feedback builder may want CSV too; parquet remains source of truth.
    evaluation_csv_path = (
        case_output_root
        / "evaluations.csv"
    )

    evaluations.to_csv(
        evaluation_csv_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Aggregate validation/search evidence.
    # -------------------------------------------------------------------------

    aggregate = (
        build_aggregate_evidence(
            evaluations=(
                evaluations
            ),

            candidate_id=(
                candidate_id
            ),

            target_id=(
                target_id
            ),

            outer_fold=(
                outer_fold
            ),

            role=(
                role
            ),
        )
    )

    aggregate[
        "provenance"
    ] = {
        "candidate_repo":
            str(
                candidate_repo
            ),

        "candidate_task_agent_path":
            str(
                task_agent_path.resolve()
            ),

        "candidate_task_agent_sha256":
            task_agent_hash,

        "fold_manifest_path":
            str(
                Path(
                    fold_manifest_path
                ).resolve()
            ),

        "fold_manifest_sha256":
            sha256_file(
                fold_manifest_path
            ),

        "hpi_path":
            str(
                Path(
                    hpi_path
                ).resolve()
            ),

        "hpi_sha256":
            sha256_file(
                hpi_path
            ),

        "temperature":
            temperature,

        "max_tokens":
            max_tokens,

        "max_tool_calls":
            max_tool_calls,

        "evaluator_version":
            EVALUATOR_VERSION,
    }

    aggregate_path = (
        case_output_root
        / "aggregate_evidence.json"
    )

    safe_json_dump(
        aggregate,
        aggregate_path,
    )

    # -------------------------------------------------------------------------
    # SearchEvaluationResult
    #
    # No experiment-specific success rule is invented here.
    # -------------------------------------------------------------------------

    return SearchEvaluationResult(
        success=True,

        evaluation_id=(
            f"exp2_f{outer_fold}_"
            f"{target_id}_"
            f"{role}_"
            f"{candidate_id}"
        ),

        search_success=(
            search_success
        ),

        metrics={
            **aggregate[
                "metrics"
            ],

            "n_cases":
                aggregate[
                    "n_cases"
                ],

            "n_successful_trajectories":
                aggregate[
                    "n_successful_trajectories"
                ],

            "n_failed_trajectories":
                aggregate[
                    "n_failed_trajectories"
                ],
        },

        resource_metrics={
            **aggregate[
                "metrics"
            ][
                "resource_metrics"
            ]
        },

        artifact_paths={
            "evaluation_parquet":
                str(
                    evaluation_path.resolve()
                ),

            "evaluation_csv":
                str(
                    evaluation_csv_path.resolve()
                ),

            "aggregate_evidence":
                str(
                    aggregate_path.resolve()
                ),

            "trajectory_dir":
                str(
                    (
                        case_output_root
                        / "trajectories"
                    ).resolve()
                ),

            "failure_mapping_dir":
                str(
                    (
                        case_output_root
                        / "failure_mapping"
                    ).resolve()
                ),
        },

        metadata={
            "candidate_id":
                candidate_id,

            "outer_fold":
                outer_fold,

            "target_id":
                target_id,

            "role":
                role,

            "candidate_task_agent_sha256":
                task_agent_hash,
        },
    )


# =============================================================================
# CandidateEvidence conversion
# =============================================================================

def aggregate_json_to_candidate_evidence(
    aggregate_path: str | Path,
):
    """
    Convert aggregate_evidence.json into generic search.utility.CandidateEvidence.
    """

    from clinevolve.search.utility import (
        CandidateEvidence,
    )

    payload = load_json(
        aggregate_path
    )

    candidate_id = str(
        payload[
            "candidate_id"
        ]
    )

    metrics = dict(
        payload[
            "metrics"
        ]
    )

    return CandidateEvidence(
        candidate_id=(
            candidate_id
        ),

        metrics=(
            metrics
        ),

        metadata={
            "target_id":
                payload.get(
                    "target_id"
                ),

            "outer_fold":
                payload.get(
                    "outer_fold"
                ),

            "role":
                payload.get(
                    "role"
                ),

            "aggregate_path":
                str(
                    Path(
                        aggregate_path
                    ).resolve()
                ),
        },
    )


# =============================================================================
# CLI
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one Exp2 candidate using the frozen Exp1 "
            "Clinical TaskAgent trajectory/evaluator protocol."
        )
    )

    parser.add_argument(
        "--candidate-repo",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--candidate-id",
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--target-id",
        required=True,
        choices=[
            "diagnostic_process_failure",
            "treatment_guideline_failure",
        ],
    )

    parser.add_argument(
        "--outer-fold",
        required=True,
        type=int,
        choices=[
            1,
            2,
            3,
            4,
            5,
        ],
    )

    parser.add_argument(
        "--role",
        required=True,
        choices=[
            "repair",
            "validation",
            "locked_test",
        ],
    )

    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=Path(
            "outputs/exp2/frozen/fold_manifest.csv"
        ),
    )

    parser.add_argument(
        "--hpi-path",
        type=Path,
        default=(
            DEFAULT_HPI_PATH
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/exp2/evaluations"
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--allow-locked-test",
        action="store_true",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
    )

    args = parser.parse_args()

    result = evaluate_candidate(
        candidate_repo=(
            args.candidate_repo
        ),

        candidate_id=(
            args.candidate_id
        ),

        model=(
            args.model
        ),

        target_id=(
            args.target_id
        ),

        outer_fold=(
            args.outer_fold
        ),

        role=(
            args.role
        ),

        fold_manifest_path=(
            args.fold_manifest
        ),

        output_root=(
            args.output_root
        ),

        hpi_path=(
            args.hpi_path
        ),

        temperature=(
            args.temperature
        ),

        max_tokens=(
            args.max_tokens
        ),

        max_tool_calls=(
            args.max_tool_calls
        ),

        allow_locked_test=(
            args.allow_locked_test
        ),

        resume=(
            not args.no_resume
        ),

        search_success=None,
    )

    print(
        json.dumps(
            {
                "success":
                    result.success,

                "evaluation_id":
                    result.evaluation_id,

                "search_success":
                    result.search_success,

                "metrics":
                    result.metrics,

                "resource_metrics":
                    result.resource_metrics,

                "artifact_paths":
                    result.artifact_paths,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    if not result.success:
        raise SystemExit(
            1
        )

    print(
        "evaluate_candidate.py: PASS"
    )


if __name__ == "__main__":
    main()
