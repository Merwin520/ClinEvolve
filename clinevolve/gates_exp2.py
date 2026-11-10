#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EVALUATOR_VERSION = "exp2_formal_gate_evaluator_v2"


WORKFLOW_ENDPOINTS = (
    "diagnostic_process_failure",
    "lab_interpretation_failure",
    "treatment_guideline_failure",
)


# ---------------------------------------------------------------------------
# Serialization / IO
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _jsonable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _jsonable(v)
            for v in value
        ]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float):
        if math.isnan(value):
            return None

        if math.isinf(value):
            return (
                "Infinity"
                if value > 0
                else "-Infinity"
            )

    return value


def _read_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _read_eval(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    df = pd.read_parquet(
        path
    ).copy()

    if "case_id" not in df.columns:
        raise RuntimeError(
            f"{path} missing required column: case_id"
        )

    df["case_id"] = (
        df["case_id"]
        .astype(str)
    )

    if df["case_id"].duplicated().any():
        dup = (
            df.loc[
                df["case_id"].duplicated(
                    keep=False
                ),
                "case_id",
            ]
            .head(20)
            .tolist()
        )

        raise RuntimeError(
            f"{path} contains duplicate case_id rows; "
            f"examples={dup}"
        )

    return df


# ---------------------------------------------------------------------------
# Generic endpoint helpers
# ---------------------------------------------------------------------------


def _paired_binary_endpoint(
    parent: pd.DataFrame,
    candidate: pd.DataFrame,
    endpoint: str,
) -> dict[str, Any]:
    if endpoint not in parent.columns:
        raise RuntimeError(
            f"parent evaluation missing endpoint: {endpoint}"
        )

    if endpoint not in candidate.columns:
        raise RuntimeError(
            f"candidate evaluation missing endpoint: {endpoint}"
        )

    x = parent[
        ["case_id", endpoint]
    ].merge(
        candidate[
            ["case_id", endpoint]
        ],
        on="case_id",
        suffixes=(
            "_parent",
            "_candidate",
        ),
        how="inner",
        validate="one_to_one",
    )

    pcol = (
        f"{endpoint}_parent"
    )

    ccol = (
        f"{endpoint}_candidate"
    )

    x[pcol] = pd.to_numeric(
        x[pcol],
        errors="coerce",
    )

    x[ccol] = pd.to_numeric(
        x[ccol],
        errors="coerce",
    )

    x = x[
        x[pcol].notna()
        & x[ccol].notna()
    ].copy()

    n = int(
        len(x)
    )

    if n == 0:
        return {
            "endpoint": endpoint,
            "n_common_evaluable": 0,
            "parent_failure_rate": None,
            "candidate_failure_rate": None,
            "candidate_minus_parent": None,
            "parent_minus_candidate": None,
            "parent_only_failure": None,
            "candidate_only_failure": None,
            "both_failure": None,
            "both_success": None,
        }

    p = (
        x[pcol]
        .astype(int)
    )

    c = (
        x[ccol]
        .astype(int)
    )

    parent_rate = float(
        p.mean()
    )

    candidate_rate = float(
        c.mean()
    )

    return {
        "endpoint": endpoint,
        "n_common_evaluable": n,
        "parent_failure_rate": parent_rate,
        "candidate_failure_rate": candidate_rate,
        "candidate_minus_parent": (
            candidate_rate
            - parent_rate
        ),
        "parent_minus_candidate": (
            parent_rate
            - candidate_rate
        ),
        "parent_only_failure": int(
            (
                (p == 1)
                & (c == 0)
            ).sum()
        ),
        "candidate_only_failure": int(
            (
                (p == 0)
                & (c == 1)
            ).sum()
        ),
        "both_failure": int(
            (
                (p == 1)
                & (c == 1)
            ).sum()
        ),
        "both_success": int(
            (
                (p == 0)
                & (c == 0)
            ).sum()
        ),
    }


def _paired_numeric_metric(
    parent: pd.DataFrame,
    candidate: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    if (
        column not in parent.columns
        or column not in candidate.columns
    ):
        return {
            "metric": column,
            "status": "missing_column",
            "n_common_evaluable": 0,
            "parent_mean": None,
            "candidate_mean": None,
            "candidate_minus_parent": None,
            "candidate_to_parent_ratio": None,
        }

    x = parent[
        ["case_id", column]
    ].merge(
        candidate[
            ["case_id", column]
        ],
        on="case_id",
        suffixes=(
            "_parent",
            "_candidate",
        ),
        how="inner",
        validate="one_to_one",
    )

    pcol = (
        f"{column}_parent"
    )

    ccol = (
        f"{column}_candidate"
    )

    x[pcol] = pd.to_numeric(
        x[pcol],
        errors="coerce",
    )

    x[ccol] = pd.to_numeric(
        x[ccol],
        errors="coerce",
    )

    x = x[
        x[pcol].notna()
        & x[ccol].notna()
    ]

    if len(x) == 0:
        return {
            "metric": column,
            "status": "not_evaluable",
            "n_common_evaluable": 0,
            "parent_mean": None,
            "candidate_mean": None,
            "candidate_minus_parent": None,
            "candidate_to_parent_ratio": None,
        }

    parent_mean = float(
        x[pcol].mean()
    )

    candidate_mean = float(
        x[ccol].mean()
    )

    if parent_mean == 0:
        if candidate_mean == 0:
            ratio = 1.0
        else:
            ratio = None
    else:
        ratio = (
            candidate_mean
            / parent_mean
        )

    return {
        "metric": column,
        "status": "evaluable",
        "n_common_evaluable": int(
            len(x)
        ),
        "parent_mean": parent_mean,
        "candidate_mean": candidate_mean,
        "candidate_minus_parent": (
            candidate_mean
            - parent_mean
        ),
        "candidate_to_parent_ratio": (
            None
            if ratio is None
            else float(ratio)
        ),
    }


# ---------------------------------------------------------------------------
# Rate helpers for G4/G5
# ---------------------------------------------------------------------------


def _trajectory_success_mask(
    df: pd.DataFrame,
) -> pd.Series:
    if "trajectory_status" in df.columns:
        x = (
            df["trajectory_status"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        return (
            x == "success"
        )

    if "status" in df.columns:
        x = (
            df["status"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        return (
            x == "success"
        )

    raise RuntimeError(
        "evaluation parquet has neither "
        "trajectory_status nor status"
    )


def _indicator_rate(
    indicator: pd.Series,
) -> dict[str, Any]:
    x = (
        indicator
        .fillna(False)
        .astype(bool)
    )

    n = int(
        len(x)
    )

    n_positive = int(
        x.sum()
    )

    rate = (
        n_positive / n
        if n
        else None
    )

    return {
        "n_cases": n,
        "n_positive": n_positive,
        "rate": (
            None
            if rate is None
            else float(rate)
        ),
    }


def _trajectory_failure_indicator(
    df: pd.DataFrame,
) -> pd.Series:
    return (
        ~_trajectory_success_mask(
            df
        )
    )


def _context_overflow_indicator(
    df: pd.DataFrame,
) -> pd.Series:
    if "error" not in df.columns:
        return pd.Series(
            False,
            index=df.index,
        )

    error = (
        df["error"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    return (
        error.str.contains(
            "contextwindowexceeded",
            regex=False,
        )
        | error.str.contains(
            "maximum context length",
            regex=False,
        )
        | error.str.contains(
            "context window",
            regex=False,
        )
    )


def _numeric_positive_indicator(
    df: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.Series:
    result = pd.Series(
        False,
        index=df.index,
    )

    for column in columns:
        if column not in df.columns:
            continue

        x = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0)

        result = (
            result
            | (x > 0)
        )

    return result


def _invalid_tool_indicator(
    df: pd.DataFrame,
) -> pd.Series:
    return _numeric_positive_indicator(
        df,
        (
            "n_invalid_calls",
            "n_unknown_tool_calls",
            "n_invalid_tool_inputs",
            "n_malformed_tool_calls",
        ),
    )


def _tool_execution_error_indicator(
    df: pd.DataFrame,
) -> pd.Series:
    return _numeric_positive_indicator(
        df,
        (
            "n_tool_execution_errors",
        ),
    )


def _compare_rates(
    *,
    parent_indicator: pd.Series,
    candidate_indicator: pd.Series,
    rule: str,
    maximum_absolute_worsening: float | None = None,
) -> dict[str, Any]:
    parent = _indicator_rate(
        parent_indicator
    )

    candidate = _indicator_rate(
        candidate_indicator
    )

    p = parent["rate"]
    c = candidate["rate"]

    if p is None or c is None:
        return {
            "pass": False,
            "reason": "rate not evaluable",
            "parent": parent,
            "candidate": candidate,
            "candidate_minus_parent": None,
        }

    delta = (
        c - p
    )

    if rule == (
        "candidate_not_higher_than_parent"
    ):
        passed = (
            c <= p
        )

        reason = (
            f"candidate-parent={delta:.6f}; "
            f"required <= 0.000000"
        )

    elif rule == (
        "absolute_noninferiority_margin"
    ):
        if maximum_absolute_worsening is None:
            raise RuntimeError(
                "absolute_noninferiority_margin "
                "requires maximum_absolute_worsening"
            )

        passed = (
            delta
            <= maximum_absolute_worsening
        )

        reason = (
            f"candidate-parent={delta:.6f}; "
            f"allowed <= "
            f"{maximum_absolute_worsening:.6f}"
        )

    else:
        raise RuntimeError(
            f"unsupported rate rule: {rule}"
        )

    return {
        "pass": bool(
            passed
        ),
        "reason": reason,
        "parent": parent,
        "candidate": candidate,
        "candidate_minus_parent": float(
            delta
        ),
        "maximum_absolute_worsening": (
            maximum_absolute_worsening
        ),
    }


# ---------------------------------------------------------------------------
# G1
# ---------------------------------------------------------------------------


def evaluate_g1(
    parent: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    target_id: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    evidence = (
        _paired_binary_endpoint(
            parent,
            candidate,
            target_id,
        )
    )

    threshold = float(
        spec[
            "minimum_absolute_improvement"
        ]
    )

    improvement = evidence[
        "parent_minus_candidate"
    ]

    if improvement is None:
        passed = False
        reason = (
            "target endpoint has no paired "
            "common-evaluable cases"
        )
    else:
        passed = (
            improvement
            >= threshold
        )

        reason = (
            f"paired target improvement "
            f"{improvement:.6f}; "
            f"required >= {threshold:.6f}"
        )

    return {
        "gate": "G1",
        "name": "target_improvement",
        "pass": bool(
            passed
        ),
        "reason": reason,
        "minimum_absolute_improvement": (
            threshold
        ),
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# G2
# ---------------------------------------------------------------------------


def evaluate_g2(
    parent: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    target_id: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    margin = float(
        spec[
            "maximum_absolute_worsening"
        ]
    )

    endpoints = [
        endpoint
        for endpoint
        in WORKFLOW_ENDPOINTS
        if endpoint != target_id
    ]

    endpoint_results = []

    for endpoint in endpoints:
        evidence = (
            _paired_binary_endpoint(
                parent,
                candidate,
                endpoint,
            )
        )

        delta = evidence[
            "candidate_minus_parent"
        ]

        if delta is None:
            passed = False
            reason = (
                "no paired common-evaluable cases"
            )
        else:
            passed = (
                delta <= margin
            )

            reason = (
                f"candidate-parent="
                f"{delta:.6f}; "
                f"allowed <= {margin:.6f}"
            )

        endpoint_results.append(
            {
                "endpoint": endpoint,
                "pass": bool(
                    passed
                ),
                "reason": reason,
                "evidence": evidence,
            }
        )

    gate_pass = (
        len(endpoint_results) > 0
        and all(
            item["pass"]
            for item
            in endpoint_results
        )
    )

    return {
        "gate": "G2",
        "name": "workflow_noninferiority",
        "pass": bool(
            gate_pass
        ),
        "maximum_absolute_worsening": (
            margin
        ),
        "decision_rule": (
            "every_endpoint_must_satisfy_margin"
        ),
        "endpoint_results": (
            endpoint_results
        ),
    }


# ---------------------------------------------------------------------------
# G3
# ---------------------------------------------------------------------------


def evaluate_g3(
    parent: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    spec: dict[str, Any],
) -> dict[str, Any]:
    endpoint = str(
        spec.get(
            "endpoint",
            "final_diagnosis_failure",
        )
    )

    margin = float(
        spec[
            "maximum_absolute_worsening"
        ]
    )

    evidence = (
        _paired_binary_endpoint(
            parent,
            candidate,
            endpoint,
        )
    )

    delta = evidence[
        "candidate_minus_parent"
    ]

    if delta is None:
        passed = False
        reason = (
            "final-diagnosis endpoint has no "
            "paired common-evaluable cases"
        )
    else:
        passed = (
            delta <= margin
        )

        reason = (
            f"candidate-parent="
            f"{delta:.6f}; "
            f"allowed <= {margin:.6f}"
        )

    return {
        "gate": "G3",
        "name": (
            "final_diagnosis_noninferiority"
        ),
        "pass": bool(
            passed
        ),
        "reason": reason,
        "maximum_absolute_worsening": (
            margin
        ),
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# G4
# ---------------------------------------------------------------------------


def evaluate_g4(
    parent: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    spec: dict[str, Any],
) -> dict[str, Any]:
    context_spec = spec[
        "context_window_exceeded_rate"
    ]

    trajectory_spec = spec[
        "trajectory_execution_failure_rate"
    ]

    invalid_spec = spec[
        "invalid_tool_call_rate"
    ]

    tool_error_spec = spec[
        "tool_execution_error_rate"
    ]

    instruction_spec = spec[
        "instruction_tool_failure"
    ]

    context_result = _compare_rates(
        parent_indicator=(
            _context_overflow_indicator(
                parent
            )
        ),
        candidate_indicator=(
            _context_overflow_indicator(
                candidate
            )
        ),
        rule=context_spec["rule"],
    )

    trajectory_result = _compare_rates(
        parent_indicator=(
            _trajectory_failure_indicator(
                parent
            )
        ),
        candidate_indicator=(
            _trajectory_failure_indicator(
                candidate
            )
        ),
        rule=trajectory_spec["rule"],
    )

    invalid_result = _compare_rates(
        parent_indicator=(
            _invalid_tool_indicator(
                parent
            )
        ),
        candidate_indicator=(
            _invalid_tool_indicator(
                candidate
            )
        ),
        rule=(
            "absolute_noninferiority_margin"
        ),
        maximum_absolute_worsening=float(
            invalid_spec[
                "maximum_absolute_worsening"
            ]
        ),
    )

    tool_error_result = _compare_rates(
        parent_indicator=(
            _tool_execution_error_indicator(
                parent
            )
        ),
        candidate_indicator=(
            _tool_execution_error_indicator(
                candidate
            )
        ),
        rule=(
            "absolute_noninferiority_margin"
        ),
        maximum_absolute_worsening=float(
            tool_error_spec[
                "maximum_absolute_worsening"
            ]
        ),
    )

    instruction_evidence = (
        _paired_binary_endpoint(
            parent,
            candidate,
            "instruction_tool_failure",
        )
    )

    instruction_margin = float(
        instruction_spec[
            "maximum_absolute_worsening"
        ]
    )

    instruction_delta = (
        instruction_evidence[
            "candidate_minus_parent"
        ]
    )

    if instruction_delta is None:
        instruction_pass = False
        instruction_reason = (
            "instruction_tool_failure "
            "not evaluable"
        )
    else:
        instruction_pass = (
            instruction_delta
            <= instruction_margin
        )

        instruction_reason = (
            f"candidate-parent="
            f"{instruction_delta:.6f}; "
            f"allowed <= "
            f"{instruction_margin:.6f}"
        )

    instruction_result = {
        "pass": bool(
            instruction_pass
        ),
        "reason": instruction_reason,
        "maximum_absolute_worsening": (
            instruction_margin
        ),
        "evidence": (
            instruction_evidence
        ),
    }

    checks = {
        "context_window_exceeded_rate": (
            context_result
        ),
        "trajectory_execution_failure_rate": (
            trajectory_result
        ),
        "invalid_tool_call_rate": (
            invalid_result
        ),
        "tool_execution_error_rate": (
            tool_error_result
        ),
        "instruction_tool_failure": (
            instruction_result
        ),
    }

    passed = all(
        item["pass"]
        for item
        in checks.values()
    )

    return {
        "gate": "G4",
        "name": (
            "tool_execution_safety"
        ),
        "pass": bool(
            passed
        ),
        "generation_precondition": (
            spec.get(
                "generation_precondition"
            )
        ),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# G5
# ---------------------------------------------------------------------------


def _resource_ratio_result(
    parent: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    column: str,
    max_ratio: float,
) -> dict[str, Any]:
    evidence = (
        _paired_numeric_metric(
            parent,
            candidate,
            column,
        )
    )

    if evidence["status"] != "evaluable":
        return {
            "metric": column,
            "pass": False,
            "status": (
                evidence["status"]
            ),
            "max_ratio": max_ratio,
            "reason": (
                "resource metric not evaluable"
            ),
            "evidence": evidence,
        }

    ratio = evidence[
        "candidate_to_parent_ratio"
    ]

    if ratio is None:
        passed = False
        reason = (
            "parent mean is zero while "
            "candidate mean is nonzero"
        )
    else:
        passed = (
            ratio <= max_ratio
        )

        reason = (
            f"candidate/parent="
            f"{ratio:.6f}; "
            f"allowed <= {max_ratio:.6f}"
        )

    return {
        "metric": column,
        "pass": bool(
            passed
        ),
        "status": "evaluable",
        "max_ratio": max_ratio,
        "reason": reason,
        "evidence": evidence,
    }


def evaluate_g5(
    parent: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    spec: dict[str, Any],
) -> dict[str, Any]:
    metrics = [
        _resource_ratio_result(
            parent,
            candidate,
            column="n_tool_calls",
            max_ratio=float(
                spec[
                    "mean_tool_calls_max_ratio"
                ]
            ),
        ),
        _resource_ratio_result(
            parent,
            candidate,
            column="n_repeated_calls",
            max_ratio=float(
                spec[
                    "mean_repeated_calls_max_ratio"
                ]
            ),
        ),
        _resource_ratio_result(
            parent,
            candidate,
            column="latency_seconds",
            max_ratio=float(
                spec[
                    "mean_latency_sec_max_ratio"
                ]
            ),
        ),
    ]

    failure_rule = spec[
        "trajectory_execution_failure_rate_rule"
    ]

    trajectory = _compare_rates(
        parent_indicator=(
            _trajectory_failure_indicator(
                parent
            )
        ),
        candidate_indicator=(
            _trajectory_failure_indicator(
                candidate
            )
        ),
        rule=failure_rule,
    )

    passed = (
        all(
            item["pass"]
            for item
            in metrics
        )
        and trajectory["pass"]
    )

    return {
        "gate": "G5",
        "name": "resource_budget",
        "pass": bool(
            passed
        ),
        "metric_results": metrics,
        "trajectory_execution_failure_rate": (
            trajectory
        ),
    }


# ---------------------------------------------------------------------------
# Overall evaluation
# ---------------------------------------------------------------------------


def evaluate_all_gates(
    *,
    parent: pd.DataFrame,
    candidate: pd.DataFrame,
    target_id: str,
    gate_spec: dict[str, Any],
    parent_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    allowed_targets = {
        "diagnostic_process_failure",
        "treatment_guideline_failure",
    }

    if target_id not in allowed_targets:
        raise RuntimeError(
            f"unsupported formal repair target: "
            f"{target_id}"
        )

    if gate_spec.get(
        "version"
    ) != "exp2_formal_gate_spec_v2":
        raise RuntimeError(
            "evaluate_gates.py v2 requires "
            "exp2_formal_gate_spec_v2"
        )

    required = (
        "G1_target_improvement",
        "G2_workflow_noninferiority",
        "G3_final_diagnosis_noninferiority",
        "G4_tool_execution_safety",
        "G5_resource_budget",
        "multiplicity",
    )

    missing = [
        name
        for name in required
        if name not in gate_spec
    ]

    if missing:
        raise RuntimeError(
            "gate spec missing: "
            + ", ".join(
                missing
            )
        )

    gates = {
        "G1": evaluate_g1(
            parent,
            candidate,
            target_id=target_id,
            spec=gate_spec[
                "G1_target_improvement"
            ],
        ),

        "G2": evaluate_g2(
            parent,
            candidate,
            target_id=target_id,
            spec=gate_spec[
                "G2_workflow_noninferiority"
            ],
        ),

        "G3": evaluate_g3(
            parent,
            candidate,
            spec=gate_spec[
                "G3_final_diagnosis_noninferiority"
            ],
        ),

        "G4": evaluate_g4(
            parent,
            candidate,
            spec=gate_spec[
                "G4_tool_execution_safety"
            ],
        ),

        "G5": evaluate_g5(
            parent,
            candidate,
            spec=gate_spec[
                "G5_resource_budget"
            ],
        ),
    }

    all_pass = all(
        gate["pass"]
        for gate in gates.values()
    )

    return {
        "schema_version": (
            "exp2_formal_gate_evaluation_v2"
        ),
        "evaluator_version": (
            EVALUATOR_VERSION
        ),
        "parent_candidate_id": (
            parent_id
        ),
        "candidate_id": (
            candidate_id
        ),
        "target_id": (
            target_id
        ),
        "comparison": (
            "candidate_vs_actual_parent"
        ),
        "all_gates_pass": bool(
            all_pass
        ),
        "gate_pass_vector": {
            name: bool(
                result["pass"]
            )
            for name, result
            in gates.items()
        },
        "gates": gates,
        "multiplicity": (
            gate_spec[
                "multiplicity"
            ]
        ),
        "notes": [
            (
                "G1-G3 use paired "
                "common-evaluable cases."
            ),
            (
                "G4 context-window and trajectory "
                "execution failures are compared "
                "as aggregate rates, not by exact "
                "case-id overlap."
            ),
            (
                "G4 invalid-tool and tool-execution "
                "error rates use frozen absolute "
                "noninferiority margins."
            ),
            (
                "Frozen TaskAgent execution "
                "invariant violations are rejected "
                "during candidate generation and "
                "must not reach formal gate "
                "evaluation."
            ),
            (
                "G5 excludes token accounting "
                "because the formal pipeline does "
                "not currently provide reliable "
                "token measurements."
            ),
            (
                "No Holm or other post-hoc "
                "p-value correction is performed."
            ),
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate formal Exp2 G1-G5 "
            "candidate gates against the "
            "candidate's actual parent."
        )
    )

    ap.add_argument(
        "--parent-eval",
        required=True,
    )

    ap.add_argument(
        "--candidate-eval",
        required=True,
    )

    ap.add_argument(
        "--target-id",
        required=True,
        choices=(
            "diagnostic_process_failure",
            "treatment_guideline_failure",
        ),
    )

    ap.add_argument(
        "--gate-spec",
        required=True,
    )

    ap.add_argument(
        "--parent-id",
        required=True,
    )

    ap.add_argument(
        "--candidate-id",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    parent_path = Path(
        args.parent_eval
    )

    candidate_path = Path(
        args.candidate_eval
    )

    gate_spec_path = Path(
        args.gate_spec
    )

    output_path = Path(
        args.output
    )

    parent = _read_eval(
        parent_path
    )

    candidate = _read_eval(
        candidate_path
    )

    gate_spec = _read_json(
        gate_spec_path
    )

    result = evaluate_all_gates(
        parent=parent,
        candidate=candidate,
        target_id=args.target_id,
        gate_spec=gate_spec,
        parent_id=args.parent_id,
        candidate_id=args.candidate_id,
    )

    result[
        "artifact_paths"
    ] = {
        "parent_evaluation": str(
            parent_path.resolve()
        ),
        "candidate_evaluation": str(
            candidate_path.resolve()
        ),
        "gate_spec": str(
            gate_spec_path.resolve()
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = json.dumps(
        _jsonable(
            result
        ),
        indent=2,
        ensure_ascii=False,
    )

    output_path.write_text(
        payload + "\n",
        encoding="utf-8",
    )

    print(
        payload
    )

    print(
        "\nevaluate_gates.py v2:",
        (
            "PASS"
            if result[
                "all_gates_pass"
            ]
            else "FAIL"
        ),
    )


if __name__ == "__main__":
    main()
