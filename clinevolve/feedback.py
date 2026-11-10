#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd


# =============================================================================
# Constants
# =============================================================================

FEEDBACK_SCHEMA_VERSION = "exp2_repair_feedback_v1"

VALID_REPAIR_TARGETS = {
    "diagnostic_process_failure",
    "lab_interpretation_failure",
    "treatment_guideline_failure",
}

VALID_FAILURE_DOMAINS = {
    "diagnostic_process_failure",
    "lab_interpretation_failure",
    "treatment_guideline_failure",
    "final_diagnosis_failure",
    "instruction_tool_failure",
}

ALLOWED_ROLE = "repair"


# =============================================================================
# Helpers
# =============================================================================

def _json_safe(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            bool,
            int,
            float,
        ),
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            _json_safe(v)
            for v in value
        ]

    if hasattr(
        value,
        "item",
    ):
        try:
            return _json_safe(
                value.item()
            )
        except Exception:
            pass

    return str(value)


def sha256_file(
    path: Path,
) -> str:

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


def load_table(
    path: Path,
) -> pd.DataFrame:

    suffix = path.suffix.lower()

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
        f"Unsupported table format: {path.suffix}"
    )


# =============================================================================
# Feedback records
# =============================================================================

@dataclass
class RepairCaseFeedback:

    case_id: str

    patient_id: str

    disease: str

    target_id: str

    target_failure: bool

    failure_domains: Dict[
        str,
        Optional[bool],
    ]

    observed_problem: Optional[str] = None

    evaluator_feedback: Optional[str] = None

    diagnostic_summary: Optional[str] = None

    treatment_plan: Optional[str] = None

    tool_trace_summary: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return _json_safe(
            asdict(self)
        )


@dataclass
class RepairFeedbackBundle:

    schema_version: str

    outer_fold: int

    target_id: str

    source_agent_id: str

    role: str

    n_cases: int

    n_target_failures: int

    target_failure_rate: float

    disease_counts: Dict[
        str,
        int,
    ]

    disease_target_failures: Dict[
        str,
        int,
    ]

    failure_domain_counts: Dict[
        str,
        int,
    ]

    cases: List[
        RepairCaseFeedback,
    ]

    text_feedback: str

    provenance: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return _json_safe(
            asdict(self)
        )


# =============================================================================
# Column resolution
# =============================================================================

def _require_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    *,
    label: str,
) -> None:

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label} missing required columns: "
            f"{missing}"
        )


def _coerce_optional_bool(
    value: Any,
) -> Optional[bool]:

    if pd.isna(
        value
    ):
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        if value == 1:
            return True
        if value == 0:
            return False

    text = str(
        value
    ).strip().lower()

    if text in {
        "1",
        "true",
        "yes",
        "y",
        "fail",
        "failure",
        "failed",
    }:
        return True

    if text in {
        "0",
        "false",
        "no",
        "n",
        "pass",
        "passed",
        "success",
    }:
        return False

    if text in {
        "",
        "na",
        "n/a",
        "none",
        "null",
    }:
        return None

    raise ValueError(
        f"Cannot interpret boolean value: {value!r}"
    )


# =============================================================================
# Manifest filtering
# =============================================================================

def get_repair_case_ids(
    manifest: pd.DataFrame,
    *,
    outer_fold: int,
) -> pd.DataFrame:

    _require_columns(
        manifest,
        [
            "outer_fold",
            "case_id",
            "patient_id",
            "disease",
            "role",
        ],
        label="fold manifest",
    )

    fold_df = (
        manifest[
            manifest[
                "outer_fold"
            ]
            == outer_fold
        ]
        .copy()
    )

    if fold_df.empty:
        raise ValueError(
            f"No rows found for outer_fold={outer_fold}"
        )

    # Hard leakage guard.
    unexpected_roles = (
        set(
            fold_df[
                "role"
            ].dropna().astype(str)
        )
        - {
            "repair",
            "validation",
            "locked_test",
        }
    )

    if unexpected_roles:
        raise ValueError(
            "Unknown roles in fold manifest: "
            f"{sorted(unexpected_roles)}"
        )

    repair_df = (
        fold_df[
            fold_df[
                "role"
            ]
            == ALLOWED_ROLE
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

    if repair_df.empty:
        raise ValueError(
            f"No repair cases found for outer_fold={outer_fold}"
        )

    if (
        repair_df[
            "case_id"
        ].duplicated().any()
    ):
        raise ValueError(
            "Duplicate repair case_id values detected"
        )

    return repair_df


# =============================================================================
# Join repair cases to evaluation outputs
# =============================================================================

def build_case_feedback_records(
    *,
    repair_cases: pd.DataFrame,
    evaluations: pd.DataFrame,
    target_id: str,
    agent_id: str,
) -> List[
    RepairCaseFeedback
]:

    if (
        target_id
        not in VALID_REPAIR_TARGETS
    ):
        raise ValueError(
            f"Unsupported target_id: {target_id}"
        )

    _require_columns(
        evaluations,
        [
            "case_id",
        ],
        label="evaluation table",
    )

    eval_df = (
        evaluations.copy()
    )

    eval_df[
        "case_id"
    ] = (
        eval_df[
            "case_id"
        ]
        .astype(str)
    )

    repair_cases = (
        repair_cases.copy()
    )

    repair_cases[
        "case_id"
    ] = (
        repair_cases[
            "case_id"
        ]
        .astype(str)
    )

    # Optional safety check if role is present in evaluation table.
    if (
        "role"
        in eval_df.columns
    ):

        bad_role_rows = (
            eval_df[
                eval_df[
                    "role"
                ].astype(str)
                != ALLOWED_ROLE
            ]
        )

        if not bad_role_rows.empty:
            raise RuntimeError(
                "Evaluation table contains non-repair rows. "
                "Refusing to build repair feedback."
            )

    # Optional agent check.
    if (
        "agent_id"
        in eval_df.columns
    ):

        eval_df = (
            eval_df[
                eval_df[
                    "agent_id"
                ].astype(str)
                == str(
                    agent_id
                )
            ]
            .copy()
        )

    merged = repair_cases.merge(
        eval_df,
        on="case_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "_manifest",
            "_eval",
        ),
    )

    # ---------------------------------------------------------------------
    # Canonicalize manifest identity columns after merge.
    #
    # Both repair_cases and eval_df may contain patient_id / disease / role,
    # in which case pandas suffixes them. The fold manifest is authoritative
    # for split identity and disease labels.
    # ---------------------------------------------------------------------

    for canonical in (
        "patient_id",
        "disease",
        "role",
    ):
        if canonical in merged.columns:
            continue

        manifest_col = (
            f"{canonical}_manifest"
        )

        eval_col = (
            f"{canonical}_eval"
        )

        if manifest_col in merged.columns:
            merged[
                canonical
            ] = merged[
                manifest_col
            ]

        elif eval_col in merged.columns:
            merged[
                canonical
            ] = merged[
                eval_col
            ]

        else:
            raise KeyError(
                f"Could not recover canonical column "
                f"{canonical!r}; available columns="
                f"{list(merged.columns)}"
            )

    # ---------------------------------------------------------------------
    # Every repair case should have an evaluation result.
    #
    # After merge, evaluation columns that overlap manifest columns are
    # suffixed with "_eval". Use their actual merged names rather than the
    # original eval_df column names.
    # ---------------------------------------------------------------------

    eval_merged_columns = []

    for col in eval_df.columns:

        if col == "case_id":
            continue

        eval_suffixed = (
            f"{col}_eval"
        )

        if eval_suffixed in merged.columns:
            eval_merged_columns.append(
                eval_suffixed
            )

        elif col in merged.columns:
            eval_merged_columns.append(
                col
            )

        else:
            raise KeyError(
                f"Evaluation column {col!r} missing after merge; "
                f"available columns={list(merged.columns)}"
            )

    missing_eval = (
        merged[
            eval_merged_columns
        ]
        .isna()
        .all(
            axis=1
        )
    )

    if (
        len(
            eval_df.columns
        )
        > 1
        and missing_eval.any()
    ):

        bad_cases = (
            merged.loc[
                missing_eval,
                "case_id",
            ]
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            "Missing evaluation results for repair cases. "
            f"Examples: {bad_cases[:20]}"
        )

    records: List[
        RepairCaseFeedback
    ] = []

    for row in (
        merged.itertuples(
            index=False
        )
    ):

        row_dict = (
            row._asdict()
        )

        failure_domains: Dict[
            str,
            Optional[bool],
        ] = {}

        for domain in sorted(
            VALID_FAILURE_DOMAINS
        ):

            value = (
                row_dict.get(
                    domain
                )
            )

            if value is None:
                failure_domains[
                    domain
                ] = None
            else:
                failure_domains[
                    domain
                ] = _coerce_optional_bool(
                    value
                )

        target_failure = (
            failure_domains.get(
                target_id
            )
        )

        if target_failure is None:
            # Unevaluable target outcome (for example context-window overflow).
            # Preserve it as NA in evaluation/audit artifacts, but do not feed
            # it to MetaAgent as either a target failure or a target success.
            continue

        patient_id = (
            row_dict.get(
                "patient_id"
            )
            or row_dict.get(
                "patient_id_manifest"
            )
        )

        disease = (
            row_dict.get(
                "disease"
            )
            or row_dict.get(
                "disease_manifest"
            )
        )

        record = RepairCaseFeedback(
            case_id=str(
                row_dict[
                    "case_id"
                ]
            ),

            patient_id=str(
                patient_id
            ),

            disease=str(
                disease
            ),

            target_id=(
                target_id
            ),

            target_failure=bool(
                target_failure
            ),

            failure_domains=(
                failure_domains
            ),

            observed_problem=(
                row_dict.get(
                    "observed_problem"
                )
                or row_dict.get(
                    "failure_reason"
                )
                or row_dict.get(
                    "error_reason"
                )
            ),

            evaluator_feedback=(
                row_dict.get(
                    "evaluator_feedback"
                )
                or row_dict.get(
                    "feedback"
                )
            ),

            diagnostic_summary=(
                row_dict.get(
                    "diagnostic_summary"
                )
            ),

            treatment_plan=(
                row_dict.get(
                    "treatment_plan"
                )
            ),

            tool_trace_summary=(
                row_dict.get(
                    "tool_trace_summary"
                )
                or row_dict.get(
                    "tool_summary"
                )
            ),

            metadata={
                "agent_id":
                    agent_id,
            },
        )

        records.append(
            record
        )

    return records


# =============================================================================
# Aggregation
# =============================================================================

def build_text_feedback(
    *,
    outer_fold: int,
    target_id: str,
    source_agent_id: str,
    records: Sequence[
        RepairCaseFeedback
    ],
    max_examples: int,
) -> str:

    failures = [
        record
        for record in records
        if record.target_failure
    ]

    disease_counts = Counter(
        record.disease
        for record in records
    )

    disease_failures = Counter(
        record.disease
        for record in failures
    )

    other_failure_counts = Counter()

    for record in records:

        for (
            domain,
            failed,
        ) in (
            record.failure_domains.items()
        ):

            if (
                failed is True
                and domain
                != target_id
            ):
                other_failure_counts[
                    domain
                ] += 1

    lines: List[str] = []

    lines.append(
        "Repair-feedback context for one search partition."
    )

    lines.append(
        f"Outer fold: {outer_fold}"
    )

    lines.append(
        f"Repair target: {target_id}"
    )

    lines.append(
        f"Source agent: {source_agent_id}"
    )

    lines.append(
        "Only repair-role cases are included. "
        "Candidate-validation and locked-test cases are excluded."
    )

    lines.append(
        f"Repair cases evaluated: {len(records)}"
    )

    lines.append(
        f"Target failures observed: {len(failures)}"
    )

    target_rate = (
        len(
            failures
        )
        / len(
            records
        )
        if records
        else 0.0
    )

    lines.append(
        f"Target failure rate on repair set: {target_rate:.6f}"
    )

    lines.append(
        "Disease distribution: "
        + json.dumps(
            dict(
                sorted(
                    disease_counts.items()
                )
            ),
            ensure_ascii=False,
        )
    )

    lines.append(
        "Target failures by disease: "
        + json.dumps(
            dict(
                sorted(
                    disease_failures.items()
                )
            ),
            ensure_ascii=False,
        )
    )

    if other_failure_counts:

        lines.append(
            "Other observed failure-domain counts: "
            + json.dumps(
                dict(
                    sorted(
                        other_failure_counts.items()
                    )
                ),
                ensure_ascii=False,
            )
        )

    lines.append(
        ""
    )

    lines.append(
        "Representative target-failure cases:"
    )

    selected_examples = (
        failures[
            :max_examples
        ]
    )

    if not selected_examples:

        lines.append(
            "- No target-failure cases observed in the repair set."
        )

    for (
        index,
        record,
    ) in enumerate(
        selected_examples,
        start=1,
    ):

        lines.append(
            f"{index}. case_id={record.case_id}; "
            f"disease={record.disease}"
        )

        if (
            record.observed_problem
        ):
            lines.append(
                "   observed_problem: "
                f"{record.observed_problem}"
            )

        if (
            record.evaluator_feedback
        ):
            lines.append(
                "   evaluator_feedback: "
                f"{record.evaluator_feedback}"
            )

        if (
            record.diagnostic_summary
        ):
            lines.append(
                "   diagnostic_summary: "
                f"{record.diagnostic_summary}"
            )

        if (
            record.treatment_plan
        ):
            lines.append(
                "   treatment_plan: "
                f"{record.treatment_plan}"
            )

        if (
            record.tool_trace_summary
        ):
            lines.append(
                "   tool_trace_summary: "
                f"{record.tool_trace_summary}"
            )

        other_failures = [
            domain
            for domain, failed
            in record.failure_domains.items()
            if (
                failed is True
                and domain
                != target_id
            )
        ]

        if other_failures:

            lines.append(
                "   co-occurring_failures: "
                + ", ".join(
                    sorted(
                        other_failures
                    )
                )
            )

    lines.append(
        ""
    )

    lines.append(
        "Use this repair-set evidence to propose one coherent improvement "
        "to the agent. Do not assume access to candidate-validation or "
        "locked-test outcomes."
    )

    return "\n".join(
        lines
    )


def build_feedback_bundle(
    *,
    outer_fold: int,
    target_id: str,
    source_agent_id: str,
    records: Sequence[
        RepairCaseFeedback
    ],
    max_examples: int = 20,
    provenance: Optional[
        Mapping[str, Any]
    ] = None,
) -> RepairFeedbackBundle:

    if (
        target_id
        not in VALID_REPAIR_TARGETS
    ):
        raise ValueError(
            f"Unsupported target_id: {target_id}"
        )

    if (
        max_examples
        <= 0
    ):
        raise ValueError(
            "max_examples must be > 0"
        )

    if not records:
        raise ValueError(
            "No repair records supplied"
        )

    target_failures = [
        record
        for record in records
        if record.target_failure
    ]

    disease_counts = Counter(
        record.disease
        for record in records
    )

    disease_target_failures = Counter(
        record.disease
        for record in target_failures
    )

    failure_domain_counts: Dict[
        str,
        int,
    ] = {}

    for domain in sorted(
        VALID_FAILURE_DOMAINS
    ):

        failure_domain_counts[
            domain
        ] = sum(
            1
            for record in records
            if (
                record.failure_domains.get(
                    domain
                )
                is True
            )
        )

    text_feedback = (
        build_text_feedback(
            outer_fold=(
                outer_fold
            ),
            target_id=(
                target_id
            ),
            source_agent_id=(
                source_agent_id
            ),
            records=(
                records
            ),
            max_examples=(
                max_examples
            ),
        )
    )

    return RepairFeedbackBundle(
        schema_version=(
            FEEDBACK_SCHEMA_VERSION
        ),

        outer_fold=(
            outer_fold
        ),

        target_id=(
            target_id
        ),

        source_agent_id=(
            source_agent_id
        ),

        role=(
            ALLOWED_ROLE
        ),

        n_cases=(
            len(
                records
            )
        ),

        n_target_failures=(
            len(
                target_failures
            )
        ),

        target_failure_rate=(
            len(
                target_failures
            )
            / len(
                records
            )
        ),

        disease_counts=dict(
            sorted(
                disease_counts.items()
            )
        ),

        disease_target_failures=dict(
            sorted(
                disease_target_failures.items()
            )
        ),

        failure_domain_counts=(
            failure_domain_counts
        ),

        cases=list(
            records
        ),

        text_feedback=(
            text_feedback
        ),

        provenance=(
            dict(
                provenance
                or {}
            )
        ),
    )


# =============================================================================
# Save
# =============================================================================

def save_feedback_bundle(
    bundle: RepairFeedbackBundle,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            bundle.to_dict(),
            f,
            indent=2,
            ensure_ascii=False,
        )

        f.write(
            "\n"
        )


# =============================================================================
# High-level builder
# =============================================================================

def build_from_files(
    *,
    fold_manifest_path: Path,
    evaluation_path: Path,
    output_path: Path,
    outer_fold: int,
    target_id: str,
    agent_id: str,
    max_examples: int = 20,
) -> RepairFeedbackBundle:

    manifest = load_table(
        fold_manifest_path
    )

    evaluations = load_table(
        evaluation_path
    )

    repair_cases = (
        get_repair_case_ids(
            manifest,
            outer_fold=(
                outer_fold
            ),
        )
    )

    # Hard filter evaluations to repair case IDs only.
    repair_case_ids = set(
        repair_cases[
            "case_id"
        ]
        .astype(str)
        .tolist()
    )

    evaluations[
        "case_id"
    ] = (
        evaluations[
            "case_id"
        ]
        .astype(str)
    )

    evaluation_case_ids = set(
        evaluations[
            "case_id"
        ].tolist()
    )

    non_repair_eval_ids = (
        evaluation_case_ids
        - repair_case_ids
    )

    if non_repair_eval_ids:

        raise RuntimeError(
            "Evaluation input contains case IDs outside "
            "the repair role for this fold. "
            "Refusing to build feedback. "
            f"Examples: {sorted(non_repair_eval_ids)[:20]}"
        )

    records = (
        build_case_feedback_records(
            repair_cases=(
                repair_cases
            ),
            evaluations=(
                evaluations
            ),
            target_id=(
                target_id
            ),
            agent_id=(
                agent_id
            ),
        )
    )

    bundle = (
        build_feedback_bundle(
            outer_fold=(
                outer_fold
            ),
            target_id=(
                target_id
            ),
            source_agent_id=(
                agent_id
            ),
            records=(
                records
            ),
            max_examples=(
                max_examples
            ),
            provenance={
                "fold_manifest_path":
                    str(
                        fold_manifest_path.resolve()
                    ),

                "fold_manifest_sha256":
                    sha256_file(
                        fold_manifest_path
                    ),

                "evaluation_path":
                    str(
                        evaluation_path.resolve()
                    ),

                "evaluation_sha256":
                    sha256_file(
                        evaluation_path
                    ),
            },
        )
    )

    save_feedback_bundle(
        bundle,
        output_path,
    )

    return bundle


# =============================================================================
# CLI
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build Exp2 repair-only feedback for MetaAgent. "
            "Validation and locked-test cases are forbidden."
        )
    )

    parser.add_argument(
        "--fold-manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--evaluation",
        required=True,
        type=Path,
        help=(
            "Per-case evaluation table containing only repair-role cases "
            "for the requested outer fold."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
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
        "--target-id",
        required=True,
        choices=sorted(
            VALID_REPAIR_TARGETS
        ),
    )

    parser.add_argument(
        "--agent-id",
        required=True,
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    bundle = build_from_files(
        fold_manifest_path=(
            args.fold_manifest
        ),
        evaluation_path=(
            args.evaluation
        ),
        output_path=(
            args.output
        ),
        outer_fold=(
            args.outer_fold
        ),
        target_id=(
            args.target_id
        ),
        agent_id=(
            args.agent_id
        ),
        max_examples=(
            args.max_examples
        ),
    )

    print(
        json.dumps(
            {
                "outer_fold":
                    bundle.outer_fold,

                "target_id":
                    bundle.target_id,

                "agent_id":
                    bundle.source_agent_id,

                "n_cases":
                    bundle.n_cases,

                "n_target_failures":
                    bundle.n_target_failures,

                "target_failure_rate":
                    bundle.target_failure_rate,

                "output":
                    str(
                        args.output.resolve()
                    ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "feedback_builder.py: PASS"
    )


if __name__ == "__main__":
    main()
