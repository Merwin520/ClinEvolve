#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd


SEARCH_FEEDBACK_SCHEMA_VERSION = "exp2_search_feedback_v2"

SEARCH_FEEDBACK_RULE_VERSION = (
    "repair_target_failure_rate_paired_common_evaluable_strict_improvement_v2"
)

VALID_TARGETS = {
    "diagnostic_process_failure",
    "treatment_guideline_failure",
}

ALLOWED_ROLE = "repair"


def load_json(
    path: str | Path,
) -> Dict[str, Any]:

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        payload = json.load(f)

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return payload


def safe_json_dump(
    payload: Mapping[str, Any],
    path: str | Path,
) -> None:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")


def _validate_evidence_identity(
    *,
    parent_evidence: Mapping[str, Any],
    candidate_evidence: Mapping[str, Any],
    target_id: str,
) -> tuple[str, str]:

    if target_id not in VALID_TARGETS:
        raise ValueError(
            f"Unsupported target_id: {target_id}"
        )

    if parent_evidence.get("role") != ALLOWED_ROLE:
        raise RuntimeError(
            "Parent search feedback evidence "
            "must come from repair role"
        )

    if candidate_evidence.get("role") != ALLOWED_ROLE:
        raise RuntimeError(
            "Candidate search feedback evidence "
            "must come from repair role"
        )

    if (
        parent_evidence.get("target_id") != target_id
        or candidate_evidence.get("target_id") != target_id
    ):
        raise RuntimeError(
            "Evidence target_id does not match "
            "requested repair target"
        )

    parent_candidate_id = str(
        parent_evidence.get(
            "candidate_id",
            "UNKNOWN_PARENT",
        )
    )

    candidate_id = str(
        candidate_evidence.get(
            "candidate_id",
            "UNKNOWN_CANDIDATE",
        )
    )

    return (
        parent_candidate_id,
        candidate_id,
    )


@dataclass
class SearchFeedbackResult:

    schema_version: str
    rule_version: str
    target_id: str
    role: str
    parent_candidate_id: str
    candidate_id: str

    parent_target_failure_rate: Optional[float]
    candidate_target_failure_rate: Optional[float]
    absolute_improvement: Optional[float]
    search_success: Optional[int]
    reason: str

    parent_n_evaluable: Optional[int]
    candidate_n_evaluable: Optional[int]
    parent_n_failures: Optional[int]
    candidate_n_failures: Optional[int]

    comparison_population: str = (
        "paired_common_evaluable_cases"
    )

    n_parent_rows: Optional[int] = None
    n_candidate_rows: Optional[int] = None
    n_matched_cases: Optional[int] = None
    n_common_evaluable: Optional[int] = None

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(self)


def compute_search_feedback(
    *,
    parent_evidence: Mapping[str, Any],
    candidate_evidence: Mapping[str, Any],
    target_id: str,
) -> SearchFeedbackResult:
    """
    Aggregate-only compatibility interface.

    Exp2 tree search should use compute_from_files(), which performs the
    required paired common-evaluable case comparison from evaluations.parquet.
    """

    raise RuntimeError(
        "Aggregate-only search feedback is disabled for Exp2 v2. "
        "Use compute_from_files() so parent and candidate are compared "
        "on paired common-evaluable case_ids."
    )


def _load_target_table(
    *,
    evaluation_path: Path,
    target_id: str,
    side: str,
) -> pd.DataFrame:

    if not evaluation_path.is_file():
        raise FileNotFoundError(
            f"{side} evaluation parquet missing: "
            f"{evaluation_path}"
        )

    df = pd.read_parquet(
        evaluation_path,
        columns=[
            "case_id",
            target_id,
        ],
    )

    if df["case_id"].isna().any():
        raise RuntimeError(
            f"{side} evaluation contains null case_id"
        )

    case_id = (
        df["case_id"]
        .astype(str)
    )

    if case_id.duplicated().any():
        dup = (
            case_id[
                case_id.duplicated(
                    keep=False
                )
            ]
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            f"{side} evaluation contains duplicate case_id: "
            f"{dup}"
        )

    df = df.copy()

    df["case_id"] = case_id

    df = df.rename(
        columns={
            target_id:
                f"{side}_failure"
        }
    )

    return df


def compute_from_files(
    *,
    parent_aggregate_path: str | Path,
    candidate_aggregate_path: str | Path,
    target_id: str,
    output_path: Optional[str | Path] = None,
) -> SearchFeedbackResult:
    """
    Exp2 repair-search feedback on paired common-evaluable cases.

    Population:
        parent case_id ∩ candidate case_id
        restricted to rows where BOTH target endpoints are non-null.

    Search success:
        candidate paired failure rate
            < parent paired failure rate
                -> 1
        otherwise -> 0

    This is scheduling feedback only, not G1-G5.
    """

    parent_aggregate_path = Path(
        parent_aggregate_path
    )

    candidate_aggregate_path = Path(
        candidate_aggregate_path
    )

    parent_evidence = load_json(
        parent_aggregate_path
    )

    candidate_evidence = load_json(
        candidate_aggregate_path
    )

    (
        parent_candidate_id,
        candidate_id,
    ) = _validate_evidence_identity(
        parent_evidence=(
            parent_evidence
        ),
        candidate_evidence=(
            candidate_evidence
        ),
        target_id=(
            target_id
        ),
    )

    parent_evaluation_path = (
        parent_aggregate_path.parent
        / "evaluations.parquet"
    )

    candidate_evaluation_path = (
        candidate_aggregate_path.parent
        / "evaluations.parquet"
    )

    parent_df = _load_target_table(
        evaluation_path=(
            parent_evaluation_path
        ),
        target_id=(
            target_id
        ),
        side="parent",
    )

    candidate_df = _load_target_table(
        evaluation_path=(
            candidate_evaluation_path
        ),
        target_id=(
            target_id
        ),
        side="candidate",
    )

    n_parent_rows = len(
        parent_df
    )

    n_candidate_rows = len(
        candidate_df
    )

    paired = parent_df.merge(
        candidate_df,
        on="case_id",
        how="inner",
        validate="one_to_one",
    )

    n_matched_cases = len(
        paired
    )

    common = paired[
        paired["parent_failure"].notna()
        & paired["candidate_failure"].notna()
    ].copy()

    n_common_evaluable = len(
        common
    )

    if n_common_evaluable == 0:

        result = SearchFeedbackResult(
            schema_version=(
                SEARCH_FEEDBACK_SCHEMA_VERSION
            ),
            rule_version=(
                SEARCH_FEEDBACK_RULE_VERSION
            ),
            target_id=(
                target_id
            ),
            role=(
                ALLOWED_ROLE
            ),
            parent_candidate_id=(
                parent_candidate_id
            ),
            candidate_id=(
                candidate_id
            ),
            parent_target_failure_rate=None,
            candidate_target_failure_rate=None,
            absolute_improvement=None,
            search_success=None,
            reason=(
                "no paired common-evaluable cases "
                "for repair target"
            ),
            parent_n_evaluable=0,
            candidate_n_evaluable=0,
            parent_n_failures=None,
            candidate_n_failures=None,
            n_parent_rows=(
                n_parent_rows
            ),
            n_candidate_rows=(
                n_candidate_rows
            ),
            n_matched_cases=(
                n_matched_cases
            ),
            n_common_evaluable=0,
        )

    else:

        parent_values = pd.to_numeric(
            common["parent_failure"],
            errors="raise",
        )

        candidate_values = pd.to_numeric(
            common["candidate_failure"],
            errors="raise",
        )

        valid_values = {
            0.0,
            1.0,
        }

        if not set(
            parent_values.unique()
        ).issubset(
            valid_values
        ):
            raise RuntimeError(
                "Parent target failure values "
                "must be binary 0/1"
            )

        if not set(
            candidate_values.unique()
        ).issubset(
            valid_values
        ):
            raise RuntimeError(
                "Candidate target failure values "
                "must be binary 0/1"
            )

        parent_n_failures = int(
            parent_values.sum()
        )

        candidate_n_failures = int(
            candidate_values.sum()
        )

        parent_rate = (
            parent_n_failures
            / n_common_evaluable
        )

        candidate_rate = (
            candidate_n_failures
            / n_common_evaluable
        )

        absolute_improvement = (
            parent_rate
            - candidate_rate
        )

        search_success = (
            1
            if candidate_rate
            < parent_rate
            else 0
        )

        reason = (
            "candidate strictly improved target "
            "failure rate on paired common-evaluable "
            "repair cases"
            if search_success == 1
            else
            "candidate did not strictly improve target "
            "failure rate on paired common-evaluable "
            "repair cases"
        )

        result = SearchFeedbackResult(
            schema_version=(
                SEARCH_FEEDBACK_SCHEMA_VERSION
            ),
            rule_version=(
                SEARCH_FEEDBACK_RULE_VERSION
            ),
            target_id=(
                target_id
            ),
            role=(
                ALLOWED_ROLE
            ),
            parent_candidate_id=(
                parent_candidate_id
            ),
            candidate_id=(
                candidate_id
            ),
            parent_target_failure_rate=(
                parent_rate
            ),
            candidate_target_failure_rate=(
                candidate_rate
            ),
            absolute_improvement=(
                absolute_improvement
            ),
            search_success=(
                search_success
            ),
            reason=(
                reason
            ),

            # These counts now deliberately refer to the SAME paired
            # common-evaluable comparison population.
            parent_n_evaluable=(
                n_common_evaluable
            ),
            candidate_n_evaluable=(
                n_common_evaluable
            ),
            parent_n_failures=(
                parent_n_failures
            ),
            candidate_n_failures=(
                candidate_n_failures
            ),

            n_parent_rows=(
                n_parent_rows
            ),
            n_candidate_rows=(
                n_candidate_rows
            ),
            n_matched_cases=(
                n_matched_cases
            ),
            n_common_evaluable=(
                n_common_evaluable
            ),
        )

    if output_path is not None:
        safe_json_dump(
            result.to_dict(),
            output_path,
        )

    return result


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--parent-aggregate",
        required=True,
    )

    parser.add_argument(
        "--candidate-aggregate",
        required=True,
    )

    parser.add_argument(
        "--target-id",
        required=True,
        choices=sorted(
            VALID_TARGETS
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    result = compute_from_files(
        parent_aggregate_path=(
            args.parent_aggregate
        ),
        candidate_aggregate_path=(
            args.candidate_aggregate
        ),
        target_id=(
            args.target_id
        ),
        output_path=(
            args.output
        ),
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
