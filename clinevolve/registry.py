#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


# =============================================================================
# Constants
# =============================================================================

REGISTRY_SCHEMA_VERSION = "exp2_candidate_registry_v1"

VALID_TARGETS = {
    "diagnostic_process_failure",
    "lab_interpretation_failure",
    "treatment_guideline_failure",
}

VALID_GENERATION_STATUS = {
    "pending",
    "generating",
    "generated",
    "generation_failed",
    "rejected_scope_violation",
}

VALID_VALIDATION_STATUS = {
    "not_run",
    "running",
    "eligible",
    "rejected",
    "validation_failed",
}

VALID_LOCKED_TEST_STATUS = {
    "not_run",
    "running",
    "completed",
    "failed",
}

VALID_SELECTION_STATUS = {
    "not_considered",
    "candidate",
    "selected",
    "not_selected",
}


# =============================================================================
# Helpers
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


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
            str(key): _json_safe(item)
            for key, item
            in value.items()
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
            _json_safe(item)
            for item in value
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
    path: str | Path,
) -> str:

    path = Path(path)

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


def sha256_text(
    text: str,
) -> str:

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def canonical_json(
    payload: Mapping[
        str,
        Any,
    ],
) -> str:

    return json.dumps(
        _json_safe(
            payload
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_partition_key(
    partition: Mapping[
        str,
        Any,
    ],
) -> str:

    return canonical_json(
        partition
    )


def atomic_write_json(
    path: Path,
    payload: Mapping[
        str,
        Any,
    ],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = json.dumps(
        _json_safe(
            payload
        ),
        indent=2,
        ensure_ascii=False,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(
            path.parent
        ),
        prefix=(
            path.name
            + "."
        ),
        suffix=".tmp",
        delete=False,
    ) as tmp:

        tmp.write(
            text
            + "\n"
        )

        temp_name = (
            tmp.name
        )

    os.replace(
        temp_name,
        path,
    )


# =============================================================================
# Candidate record
# =============================================================================

@dataclass
class CandidateRecord:
    """
    One formal Exp2 candidate audit record.

    This is NOT the generic search-tree node.

    Search lineage/state is maintained by search.archive.AgentArchive.
    This record stores Exp2-specific provenance and experiment status.
    """

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------

    candidate_id: str

    node_id: str

    outer_fold: int

    target_id: str

    partition: Dict[
        str,
        Any,
    ]

    parent_candidate_id: Optional[str]

    parent_node_id: Optional[str]

    generation: int

    is_a0: bool = False

    # -------------------------------------------------------------------------
    # Workspace / patch provenance
    # -------------------------------------------------------------------------

    repo_path: Optional[str] = None

    source_repo_path: Optional[str] = None

    base_commit: Optional[str] = None

    patch_path: Optional[str] = None

    patch_sha256: Optional[str] = None

    modified_files: List[
        str
    ] = field(
        default_factory=list
    )

    lines_added: Optional[int] = None

    lines_deleted: Optional[int] = None

    # -------------------------------------------------------------------------
    # Generation provenance
    # -------------------------------------------------------------------------

    generation_status: str = "pending"

    generation_model: Optional[str] = None

    generation_temperature: Optional[
        float
    ] = None

    generation_max_tokens: Optional[
        int
    ] = None

    generation_tokens: Optional[
        int
    ] = None

    generation_calls: Optional[
        int
    ] = None

    generation_seconds: Optional[
        float
    ] = None

    generation_error: Optional[
        str
    ] = None

    feedback_path: Optional[
        str
    ] = None

    feedback_sha256: Optional[
        str
    ] = None

    # -------------------------------------------------------------------------
    # Scope audit
    # -------------------------------------------------------------------------

    allowed_modification_paths: List[
        str
    ] = field(
        default_factory=list
    )

    forbidden_modification_paths: List[
        str
    ] = field(
        default_factory=list
    )

    scope_check_passed: Optional[
        bool
    ] = None

    scope_violations: List[
        str
    ] = field(
        default_factory=list
    )

    # -------------------------------------------------------------------------
    # Repair evaluation
    # -------------------------------------------------------------------------

    repair_evaluation_path: Optional[
        str
    ] = None

    repair_evaluation_sha256: Optional[
        str
    ] = None

    repair_search_success: Optional[
        int
    ] = None

    # -------------------------------------------------------------------------
    # Candidate validation
    # -------------------------------------------------------------------------

    validation_status: str = "not_run"

    validation_evidence_path: Optional[
        str
    ] = None

    validation_evidence_sha256: Optional[
        str
    ] = None

    validation_gate_result_path: Optional[
        str
    ] = None

    validation_gate_result_sha256: Optional[
        str
    ] = None

    validation_eligible: Optional[
        bool
    ] = None

    utility_primary_score: Optional[
        float
    ] = None

    # -------------------------------------------------------------------------
    # Candidate selection
    # -------------------------------------------------------------------------

    selection_status: str = "not_considered"

    selection_rank: Optional[
        int
    ] = None

    # -------------------------------------------------------------------------
    # Locked test
    # -------------------------------------------------------------------------

    locked_test_status: str = "not_run"

    locked_test_result_path: Optional[
        str
    ] = None

    locked_test_result_sha256: Optional[
        str
    ] = None

    # -------------------------------------------------------------------------
    # Misc audit
    # -------------------------------------------------------------------------

    metadata: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    created_at_utc: str = field(
        default_factory=utc_now_iso
    )

    updated_at_utc: str = field(
        default_factory=utc_now_iso
    )

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(
        self,
    ) -> None:

        if not str(
            self.candidate_id
        ).strip():
            raise ValueError(
                "candidate_id must be non-empty"
            )

        if not str(
            self.node_id
        ).strip():
            raise ValueError(
                "node_id must be non-empty"
            )

        if (
            not isinstance(self.outer_fold, int)
            or self.outer_fold < 1
        ):
            raise ValueError(
                "outer_fold (partition id) must be a positive integer"
            )

        if (
            self.target_id
            not in VALID_TARGETS
        ):
            raise ValueError(
                "Unsupported target_id: "
                f"{self.target_id}"
            )

        if not isinstance(
            self.partition,
            dict,
        ):
            raise TypeError(
                "partition must be a dict"
            )

        if (
            self.generation
            < 0
        ):
            raise ValueError(
                "generation must be >= 0"
            )

        if self.is_a0:

            if (
                self.generation
                != 0
            ):
                raise ValueError(
                    "A0 must have generation=0"
                )

            if (
                self.parent_candidate_id
                is not None
                or self.parent_node_id
                is not None
            ):
                raise ValueError(
                    "A0 cannot have a parent"
                )

        else:

            if (
                self.generation
                <= 0
            ):
                raise ValueError(
                    "Non-A0 candidate must have "
                    "generation > 0"
                )

            if (
                self.parent_candidate_id
                is None
            ):
                raise ValueError(
                    "Non-A0 candidate requires "
                    "parent_candidate_id"
                )

            if (
                self.parent_node_id
                is None
            ):
                raise ValueError(
                    "Non-A0 candidate requires "
                    "parent_node_id"
                )

        if (
            self.generation_status
            not in VALID_GENERATION_STATUS
        ):
            raise ValueError(
                "Unsupported generation_status: "
                f"{self.generation_status}"
            )

        if (
            self.validation_status
            not in VALID_VALIDATION_STATUS
        ):
            raise ValueError(
                "Unsupported validation_status: "
                f"{self.validation_status}"
            )

        if (
            self.locked_test_status
            not in VALID_LOCKED_TEST_STATUS
        ):
            raise ValueError(
                "Unsupported locked_test_status: "
                f"{self.locked_test_status}"
            )

        if (
            self.selection_status
            not in VALID_SELECTION_STATUS
        ):
            raise ValueError(
                "Unsupported selection_status: "
                f"{self.selection_status}"
            )

        if (
            self.repair_search_success
            not in (
                None,
                0,
                1,
            )
        ):
            raise ValueError(
                "repair_search_success must be "
                "0, 1, or None"
            )

        if (
            self.selection_rank
            is not None
            and self.selection_rank
            <= 0
        ):
            raise ValueError(
                "selection_rank must be > 0"
            )

        if (
            self.modified_files
            and self.scope_check_passed
            is None
        ):
            # Not fatal during intermediate generation,
            # but generated candidates should eventually receive a scope audit.
            pass

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        self.validate()

        return {
            "schema_version":
                REGISTRY_SCHEMA_VERSION,

            **_json_safe(
                asdict(
                    self
                )
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[
            str,
            Any,
        ],
    ) -> "CandidateRecord":

        raw = dict(
            payload
        )

        schema_version = raw.pop(
            "schema_version",
            REGISTRY_SCHEMA_VERSION,
        )

        if (
            schema_version
            != REGISTRY_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported candidate record schema: "
                f"{schema_version}"
            )

        record = cls(
            **raw
        )

        record.validate()

        return record


# =============================================================================
# Registry
# =============================================================================

class CandidateRegistry:
    """
    Persistent Exp2 candidate registry.

    JSON is the source of truth.

    Optional CSV export is intended only for human-readable audit/reporting.
    """

    def __init__(
        self,
        registry_path: str | Path,
        *,
        autosave: bool = True,
    ) -> None:

        self.registry_path = Path(
            registry_path
        )

        self.autosave = bool(
            autosave
        )

        self.records: Dict[
            str,
            CandidateRecord,
        ] = {}

        if (
            self.registry_path.exists()
        ):
            self.load()

    # =========================================================================
    # Persistence
    # =========================================================================

    def __len__(
        self,
    ) -> int:

        return len(
            self.records
        )

    def __contains__(
        self,
        candidate_id: str,
    ) -> bool:

        return (
            candidate_id
            in self.records
        )

    def to_dict(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        self.validate_registry()

        return {
            "schema_version":
                REGISTRY_SCHEMA_VERSION,

            "updated_at_utc":
                utc_now_iso(),

            "n_candidates":
                len(
                    self.records
                ),

            "candidates": [
                self.records[
                    candidate_id
                ].to_dict()
                for candidate_id
                in sorted(
                    self.records
                )
            ],
        }

    def save(
        self,
    ) -> None:

        self.validate_registry()

        atomic_write_json(
            self.registry_path,
            self.to_dict(),
        )

    def load(
        self,
    ) -> None:

        with self.registry_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            payload = json.load(
                f
            )

        schema_version = (
            payload.get(
                "schema_version"
            )
        )

        if (
            schema_version
            != REGISTRY_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported registry schema: "
                f"{schema_version}"
            )

        loaded: Dict[
            str,
            CandidateRecord,
        ] = {}

        for raw in (
            payload.get(
                "candidates",
                []
            )
        ):

            record = (
                CandidateRecord.from_dict(
                    raw
                )
            )

            if (
                record.candidate_id
                in loaded
            ):
                raise ValueError(
                    "Duplicate candidate_id: "
                    f"{record.candidate_id}"
                )

            loaded[
                record.candidate_id
            ] = record

        self.records = (
            loaded
        )

        self.validate_registry()

    def _save_if_needed(
        self,
    ) -> None:

        if self.autosave:
            self.save()

    # =========================================================================
    # CRUD
    # =========================================================================

    def add(
        self,
        record: CandidateRecord,
    ) -> None:

        record.validate()

        if (
            record.candidate_id
            in self.records
        ):
            raise ValueError(
                "candidate_id already exists: "
                f"{record.candidate_id}"
            )

        if any(
            existing.node_id
            == record.node_id
            for existing
            in self.records.values()
        ):
            raise ValueError(
                "node_id already exists: "
                f"{record.node_id}"
            )

        if (
            not record.is_a0
            and record.parent_candidate_id
            not in self.records
        ):
            raise ValueError(
                "Parent candidate not registered: "
                f"{record.parent_candidate_id}"
            )

        if (
            not record.is_a0
        ):

            parent = self.records[
                record.parent_candidate_id
            ]

            if (
                record.parent_node_id
                != parent.node_id
            ):
                raise ValueError(
                    "parent_node_id does not match "
                    "parent candidate record"
                )

            if (
                record.generation
                != parent.generation
                + 1
            ):
                raise ValueError(
                    "Candidate generation must equal "
                    "parent generation + 1"
                )

            if (
                canonical_partition_key(
                    record.partition
                )
                != canonical_partition_key(
                    parent.partition
                )
            ):
                raise ValueError(
                    "Parent and child candidates must "
                    "belong to same partition"
                )

        self.records[
            record.candidate_id
        ] = record

        self._save_if_needed()

    def get(
        self,
        candidate_id: str,
    ) -> CandidateRecord:

        if (
            candidate_id
            not in self.records
        ):
            raise KeyError(
                f"Unknown candidate_id: "
                f"{candidate_id}"
            )

        return self.records[
            candidate_id
        ]

    def update(
        self,
        candidate_id: str,
        **updates: Any,
    ) -> CandidateRecord:

        record = self.get(
            candidate_id
        )

        immutable_fields = {
            "candidate_id",
            "node_id",
            "outer_fold",
            "target_id",
            "partition",
            "parent_candidate_id",
            "parent_node_id",
            "generation",
            "is_a0",
            "created_at_utc",
        }

        illegal = (
            immutable_fields
            & set(
                updates
            )
        )

        if illegal:
            raise ValueError(
                "Cannot update immutable fields: "
                f"{sorted(illegal)}"
            )

        for (
            key,
            value,
        ) in updates.items():

            if not hasattr(
                record,
                key,
            ):
                raise AttributeError(
                    "CandidateRecord has no field: "
                    f"{key}"
                )

            setattr(
                record,
                key,
                value,
            )

        record.updated_at_utc = (
            utc_now_iso()
        )

        record.validate()

        self._save_if_needed()

        return record

    # =========================================================================
    # Queries
    # =========================================================================

    def all_records(
        self,
    ) -> List[
        CandidateRecord
    ]:

        return sorted(
            self.records.values(),
            key=lambda record: (
                record.outer_fold,
                record.target_id,
                record.generation,
                record.candidate_id,
            ),
        )

    def filter(
        self,
        *,
        outer_fold: Optional[
            int
        ] = None,
        target_id: Optional[
            str
        ] = None,
        generation_status: Optional[
            str
        ] = None,
        validation_status: Optional[
            str
        ] = None,
        selection_status: Optional[
            str
        ] = None,
        is_a0: Optional[
            bool
        ] = None,
    ) -> List[
        CandidateRecord
    ]:

        result = list(
            self.records.values()
        )

        if (
            outer_fold
            is not None
        ):
            result = [
                record
                for record in result
                if (
                    record.outer_fold
                    == outer_fold
                )
            ]

        if (
            target_id
            is not None
        ):
            result = [
                record
                for record in result
                if (
                    record.target_id
                    == target_id
                )
            ]

        if (
            generation_status
            is not None
        ):
            result = [
                record
                for record in result
                if (
                    record.generation_status
                    == generation_status
                )
            ]

        if (
            validation_status
            is not None
        ):
            result = [
                record
                for record in result
                if (
                    record.validation_status
                    == validation_status
                )
            ]

        if (
            selection_status
            is not None
        ):
            result = [
                record
                for record in result
                if (
                    record.selection_status
                    == selection_status
                )
            ]

        if (
            is_a0
            is not None
        ):
            result = [
                record
                for record in result
                if (
                    record.is_a0
                    == is_a0
                )
            ]

        return sorted(
            result,
            key=lambda record: (
                record.outer_fold,
                record.target_id,
                record.generation,
                record.candidate_id,
            ),
        )

    def partition_records(
        self,
        *,
        outer_fold: int,
        target_id: str,
    ) -> List[
        CandidateRecord
    ]:

        return self.filter(
            outer_fold=(
                outer_fold
            ),
            target_id=(
                target_id
            ),
        )

    def get_a0(
        self,
        *,
        outer_fold: int,
        target_id: str,
    ) -> CandidateRecord:

        matches = [
            record
            for record
            in self.partition_records(
                outer_fold=(
                    outer_fold
                ),
                target_id=(
                    target_id
                ),
            )
            if record.is_a0
        ]

        if (
            len(
                matches
            )
            != 1
        ):
            raise RuntimeError(
                "Expected exactly one A0 for "
                f"fold={outer_fold}, target={target_id}; "
                f"found {len(matches)}"
            )

        return matches[
            0
        ]

    def selected_candidate(
        self,
        *,
        outer_fold: int,
        target_id: str,
    ) -> Optional[
        CandidateRecord
    ]:

        matches = [
            record
            for record
            in self.partition_records(
                outer_fold=(
                    outer_fold
                ),
                target_id=(
                    target_id
                ),
            )
            if (
                record.selection_status
                == "selected"
            )
        ]

        if (
            len(
                matches
            )
            > 1
        ):
            raise RuntimeError(
                "Multiple selected candidates in "
                "one fold × target partition"
            )

        return (
            matches[0]
            if matches
            else None
        )

    # =========================================================================
    # Status helpers
    # =========================================================================

    def mark_generation_started(
        self,
        candidate_id: str,
    ) -> None:

        self.update(
            candidate_id,
            generation_status=(
                "generating"
            ),
            generation_error=None,
        )

    def mark_generation_failed(
        self,
        candidate_id: str,
        *,
        error: str,
    ) -> None:

        self.update(
            candidate_id,
            generation_status=(
                "generation_failed"
            ),
            generation_error=(
                str(
                    error
                )
            ),
        )

    def mark_scope_violation(
        self,
        candidate_id: str,
        *,
        violations: Iterable[
            str
        ],
    ) -> None:

        violations = sorted(
            {
                str(item)
                for item in violations
            }
        )

        self.update(
            candidate_id,
            generation_status=(
                "rejected_scope_violation"
            ),
            scope_check_passed=False,
            scope_violations=(
                violations
            ),
        )

    def mark_generated(
        self,
        candidate_id: str,
        *,
        repo_path: str | Path,
        patch_path: Optional[
            str | Path
        ],
        modified_files: Iterable[
            str
        ],
        lines_added: Optional[
            int
        ] = None,
        lines_deleted: Optional[
            int
        ] = None,
        generation_tokens: Optional[
            int
        ] = None,
        generation_calls: Optional[
            int
        ] = None,
        generation_seconds: Optional[
            float
        ] = None,
    ) -> None:

        patch_sha256 = None

        if (
            patch_path
            is not None
        ):

            patch_path = Path(
                patch_path
            )

            if (
                not patch_path.exists()
            ):
                raise FileNotFoundError(
                    f"Patch does not exist: "
                    f"{patch_path}"
                )

            patch_sha256 = (
                sha256_file(
                    patch_path
                )
            )

        self.update(
            candidate_id,

            generation_status=(
                "generated"
            ),

            repo_path=(
                str(
                    Path(
                        repo_path
                    ).resolve()
                )
            ),

            patch_path=(
                str(
                    Path(
                        patch_path
                    ).resolve()
                )
                if patch_path
                is not None
                else None
            ),

            patch_sha256=(
                patch_sha256
            ),

            modified_files=sorted(
                {
                    str(item)
                    for item
                    in modified_files
                }
            ),

            lines_added=(
                lines_added
            ),

            lines_deleted=(
                lines_deleted
            ),

            generation_tokens=(
                generation_tokens
            ),

            generation_calls=(
                generation_calls
            ),

            generation_seconds=(
                generation_seconds
            ),

            scope_check_passed=True,

            scope_violations=[],

            generation_error=None,
        )

    def attach_feedback(
        self,
        candidate_id: str,
        *,
        feedback_path: str | Path,
    ) -> None:

        path = Path(
            feedback_path
        ).resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Feedback file does not exist: "
                f"{path}"
            )

        self.update(
            candidate_id,

            feedback_path=(
                str(
                    path
                )
            ),

            feedback_sha256=(
                sha256_file(
                    path
                )
            ),
        )

    def attach_repair_evaluation(
        self,
        candidate_id: str,
        *,
        evaluation_path: str | Path,
        search_success: Optional[
            int
        ],
    ) -> None:

        if (
            search_success
            not in (
                None,
                0,
                1,
            )
        ):
            raise ValueError(
                "search_success must be "
                "0, 1, or None"
            )

        path = Path(
            evaluation_path
        ).resolve()

        if not path.exists():
            raise FileNotFoundError(
                "Repair evaluation file "
                f"does not exist: {path}"
            )

        self.update(
            candidate_id,

            repair_evaluation_path=(
                str(
                    path
                )
            ),

            repair_evaluation_sha256=(
                sha256_file(
                    path
                )
            ),

            repair_search_success=(
                search_success
            ),
        )

    def mark_validation_started(
        self,
        candidate_id: str,
    ) -> None:

        self.update(
            candidate_id,
            validation_status=(
                "running"
            ),
        )

    def mark_validation_failed(
        self,
        candidate_id: str,
        *,
        error: str,
    ) -> None:

        record = self.get(
            candidate_id
        )

        metadata = dict(
            record.metadata
        )

        metadata.setdefault(
            "validation_errors",
            [],
        ).append(
            {
                "error":
                    str(
                        error
                    ),

                "recorded_at_utc":
                    utc_now_iso(),
            }
        )

        self.update(
            candidate_id,
            validation_status=(
                "validation_failed"
            ),
            metadata=(
                metadata
            ),
        )

    def attach_validation_result(
        self,
        candidate_id: str,
        *,
        evidence_path: str | Path,
        gate_result_path: str | Path,
        eligible: bool,
        primary_score: Optional[
            float
        ],
    ) -> None:

        evidence_path = Path(
            evidence_path
        ).resolve()

        gate_result_path = Path(
            gate_result_path
        ).resolve()

        if not evidence_path.exists():
            raise FileNotFoundError(
                "Validation evidence does not exist: "
                f"{evidence_path}"
            )

        if not gate_result_path.exists():
            raise FileNotFoundError(
                "Validation gate result does not exist: "
                f"{gate_result_path}"
            )

        self.update(
            candidate_id,

            validation_status=(
                "eligible"
                if eligible
                else "rejected"
            ),

            validation_evidence_path=(
                str(
                    evidence_path
                )
            ),

            validation_evidence_sha256=(
                sha256_file(
                    evidence_path
                )
            ),

            validation_gate_result_path=(
                str(
                    gate_result_path
                )
            ),

            validation_gate_result_sha256=(
                sha256_file(
                    gate_result_path
                )
            ),

            validation_eligible=(
                bool(
                    eligible
                )
            ),

            utility_primary_score=(
                primary_score
            ),

            selection_status=(
                "candidate"
                if eligible
                else "not_considered"
            ),
        )

    def mark_selected(
        self,
        candidate_id: str,
        *,
        rank: int = 1,
    ) -> None:

        selected = self.get(
            candidate_id
        )

        if (
            selected.validation_eligible
            is not True
        ):
            raise RuntimeError(
                "Cannot select a candidate that "
                "did not pass validation"
            )

        for record in (
            self.records.values()
        ):

            if (
                record.outer_fold
                != selected.outer_fold
                or record.target_id
                != selected.target_id
            ):
                continue

            if (
                record.candidate_id
                == candidate_id
            ):
                record.selection_status = (
                    "selected"
                )
                record.selection_rank = (
                    rank
                )

            elif (
                record.validation_eligible
                is True
            ):
                record.selection_status = (
                    "not_selected"
                )

                if (
                    record.selection_rank
                    == rank
                ):
                    record.selection_rank = None

            record.updated_at_utc = (
                utc_now_iso()
            )

        self._save_if_needed()

    def mark_locked_test_started(
        self,
        candidate_id: str,
    ) -> None:

        record = self.get(
            candidate_id
        )

        if (
            record.selection_status
            != "selected"
            and not record.is_a0
        ):
            raise RuntimeError(
                "Locked test may only be run for "
                "A0 or the selected repaired candidate"
            )

        self.update(
            candidate_id,
            locked_test_status=(
                "running"
            ),
        )

    def attach_locked_test_result(
        self,
        candidate_id: str,
        *,
        result_path: str | Path,
    ) -> None:

        record = self.get(
            candidate_id
        )

        if (
            record.selection_status
            != "selected"
            and not record.is_a0
        ):
            raise RuntimeError(
                "Locked test may only be attached "
                "to A0 or selected candidate"
            )

        path = Path(
            result_path
        ).resolve()

        if not path.exists():
            raise FileNotFoundError(
                "Locked-test result does not exist: "
                f"{path}"
            )

        self.update(
            candidate_id,

            locked_test_status=(
                "completed"
            ),

            locked_test_result_path=(
                str(
                    path
                )
            ),

            locked_test_result_sha256=(
                sha256_file(
                    path
                )
            ),
        )

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_registry(
        self,
    ) -> None:

        node_ids = set()

        partition_a0_counts: Dict[
            str,
            int,
        ] = {}

        partition_selected_counts: Dict[
            str,
            int,
        ] = {}

        for record in (
            self.records.values()
        ):

            record.validate()

            if (
                record.node_id
                in node_ids
            ):
                raise ValueError(
                    "Duplicate node_id: "
                    f"{record.node_id}"
                )

            node_ids.add(
                record.node_id
            )

            partition_key = (
                canonical_partition_key(
                    record.partition
                )
            )

            if record.is_a0:

                partition_a0_counts[
                    partition_key
                ] = (
                    partition_a0_counts.get(
                        partition_key,
                        0,
                    )
                    + 1
                )

            if (
                record.selection_status
                == "selected"
            ):

                partition_selected_counts[
                    partition_key
                ] = (
                    partition_selected_counts.get(
                        partition_key,
                        0,
                    )
                    + 1
                )

            if (
                not record.is_a0
            ):

                if (
                    record.parent_candidate_id
                    not in self.records
                ):
                    raise ValueError(
                        "Missing parent candidate for "
                        f"{record.candidate_id}"
                    )

                parent = self.records[
                    record.parent_candidate_id
                ]

                if (
                    parent.node_id
                    != record.parent_node_id
                ):
                    raise ValueError(
                        "Parent node mismatch for "
                        f"{record.candidate_id}"
                    )

                if (
                    parent.generation
                    + 1
                    != record.generation
                ):
                    raise ValueError(
                        "Generation mismatch for "
                        f"{record.candidate_id}"
                    )

                if (
                    canonical_partition_key(
                        parent.partition
                    )
                    != partition_key
                ):
                    raise ValueError(
                        "Cross-partition lineage for "
                        f"{record.candidate_id}"
                    )

        for (
            partition_key,
            count,
        ) in (
            partition_a0_counts.items()
        ):

            if count > 1:
                raise ValueError(
                    "Multiple A0 records in partition: "
                    f"{partition_key}"
                )

        for (
            partition_key,
            count,
        ) in (
            partition_selected_counts.items()
        ):

            if count > 1:
                raise ValueError(
                    "Multiple selected candidates in "
                    f"partition: {partition_key}"
                )

    # =========================================================================
    # CSV audit export
    # =========================================================================

    def export_csv(
        self,
        path: str | Path,
    ) -> None:

        import csv

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        columns = [
            "candidate_id",
            "node_id",
            "outer_fold",
            "target_id",
            "parent_candidate_id",
            "parent_node_id",
            "generation",
            "is_a0",
            "generation_status",
            "repo_path",
            "patch_path",
            "patch_sha256",
            "modified_files",
            "lines_added",
            "lines_deleted",
            "generation_model",
            "generation_tokens",
            "generation_calls",
            "generation_seconds",
            "scope_check_passed",
            "scope_violations",
            "repair_search_success",
            "validation_status",
            "validation_eligible",
            "utility_primary_score",
            "selection_status",
            "selection_rank",
            "locked_test_status",
            "created_at_utc",
            "updated_at_utc",
        ]

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=(
                    columns
                ),
            )

            writer.writeheader()

            for record in (
                self.all_records()
            ):

                raw = record.to_dict()

                row = {
                    column:
                        raw.get(
                            column
                        )
                    for column
                    in columns
                }

                row[
                    "modified_files"
                ] = json.dumps(
                    record.modified_files,
                    ensure_ascii=False,
                )

                row[
                    "scope_violations"
                ] = json.dumps(
                    record.scope_violations,
                    ensure_ascii=False,
                )

                writer.writerow(
                    row
                )

    # =========================================================================
    # Summary
    # =========================================================================

    def summary(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        return {
            "schema_version":
                REGISTRY_SCHEMA_VERSION,

            "registry_path":
                str(
                    self.registry_path
                ),

            "n_candidates":
                len(
                    self.records
                ),

            "generation_status_counts":
                self._count_field(
                    "generation_status"
                ),

            "validation_status_counts":
                self._count_field(
                    "validation_status"
                ),

            "selection_status_counts":
                self._count_field(
                    "selection_status"
                ),

            "locked_test_status_counts":
                self._count_field(
                    "locked_test_status"
                ),
        }

    def _count_field(
        self,
        field_name: str,
    ) -> Dict[
        str,
        int,
    ]:

        counts: Dict[
            str,
            int,
        ] = {}

        for record in (
            self.records.values()
        ):

            value = str(
                getattr(
                    record,
                    field_name,
                )
            )

            counts[
                value
            ] = (
                counts.get(
                    value,
                    0,
                )
                + 1
            )

        return dict(
            sorted(
                counts.items()
            )
        )


# =============================================================================
# Constructors
# =============================================================================

def make_exp2_partition(
    *,
    outer_fold: int,
    target_id: str,
) -> Dict[
    str,
    Any,
]:

    if (
        not isinstance(outer_fold, int)
        or outer_fold < 1
    ):
        raise ValueError(
            "outer_fold (partition id) must be a positive integer"
        )

    if (
        target_id
        not in VALID_TARGETS
    ):
        raise ValueError(
            f"Unsupported target_id: {target_id}"
        )

    return {
        "experiment":
            "exp2",

        "outer_fold":
            outer_fold,

        "repair_target":
            target_id,
    }


def make_a0_record(
    *,
    outer_fold: int,
    target_id: str,
    node_id: str,
    repo_path: str | Path,
    base_commit: Optional[
        str
    ],
    agent_model: str,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> CandidateRecord:

    partition = (
        make_exp2_partition(
            outer_fold=(
                outer_fold
            ),
            target_id=(
                target_id
            ),
        )
    )

    return CandidateRecord(
        candidate_id="A0",

        node_id=(
            node_id
        ),

        outer_fold=(
            outer_fold
        ),

        target_id=(
            target_id
        ),

        partition=(
            partition
        ),

        parent_candidate_id=None,

        parent_node_id=None,

        generation=0,

        is_a0=True,

        repo_path=str(
            Path(
                repo_path
            ).resolve()
        ),

        base_commit=(
            base_commit
        ),

        generation_status=(
            "generated"
        ),

        generation_model=(
            agent_model
        ),

        scope_check_passed=True,

        validation_status=(
            "not_run"
        ),

        selection_status=(
            "not_considered"
        ),

        locked_test_status=(
            "not_run"
        ),

        metadata=dict(
            metadata
            or {}
        ),
    )


def make_candidate_record(
    *,
    candidate_id: str,
    node_id: str,
    parent: CandidateRecord,
    source_repo_path: str | Path,
    generation_model: Optional[
        str
    ],
    generation_temperature: Optional[
        float
    ] = None,
    generation_max_tokens: Optional[
        int
    ] = None,
    feedback_path: Optional[
        str | Path
    ] = None,
    allowed_modification_paths: Optional[
        Iterable[str]
    ] = None,
    forbidden_modification_paths: Optional[
        Iterable[str]
    ] = None,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> CandidateRecord:

    if parent.is_a0:
        parent_generation = 0
    else:
        parent_generation = (
            parent.generation
        )

    feedback_sha256 = None
    feedback_path_value = None

    if (
        feedback_path
        is not None
    ):

        feedback_path = Path(
            feedback_path
        ).resolve()

        if not feedback_path.exists():
            raise FileNotFoundError(
                "Feedback file does not exist: "
                f"{feedback_path}"
            )

        feedback_path_value = str(
            feedback_path
        )

        feedback_sha256 = (
            sha256_file(
                feedback_path
            )
        )

    return CandidateRecord(
        candidate_id=(
            candidate_id
        ),

        node_id=(
            node_id
        ),

        outer_fold=(
            parent.outer_fold
        ),

        target_id=(
            parent.target_id
        ),

        partition=dict(
            parent.partition
        ),

        parent_candidate_id=(
            parent.candidate_id
        ),

        parent_node_id=(
            parent.node_id
        ),

        generation=(
            parent_generation
            + 1
        ),

        is_a0=False,

        source_repo_path=str(
            Path(
                source_repo_path
            ).resolve()
        ),

        generation_status=(
            "pending"
        ),

        generation_model=(
            generation_model
        ),

        generation_temperature=(
            generation_temperature
        ),

        generation_max_tokens=(
            generation_max_tokens
        ),

        feedback_path=(
            feedback_path_value
        ),

        feedback_sha256=(
            feedback_sha256
        ),

        allowed_modification_paths=sorted(
            {
                str(item)
                for item
                in (
                    allowed_modification_paths
                    or []
                )
            }
        ),

        forbidden_modification_paths=sorted(
            {
                str(item)
                for item
                in (
                    forbidden_modification_paths
                    or []
                )
            }
        ),

        metadata=dict(
            metadata
            or {}
        ),
    )


# =============================================================================
# Minimal self-test
# =============================================================================

def _self_test() -> None:

    import tempfile

    with tempfile.TemporaryDirectory() as tempdir:

        root = Path(
            tempdir
        )

        registry_path = (
            root
            / "candidate_registry.json"
        )

        registry = CandidateRegistry(
            registry_path
        )

        a0_repo = (
            root
            / "A0"
        )

        a0_repo.mkdir()

        a0 = make_a0_record(
            outer_fold=1,

            target_id=(
                "diagnostic_process_failure"
            ),

            node_id=(
                "exp2_f1_pd_A0"
            ),

            repo_path=(
                a0_repo
            ),

            base_commit=(
                "deadbeef"
            ),

            agent_model=(
                "test-model"
            ),
        )

        registry.add(
            a0
        )

        feedback_path = (
            root
            / "feedback.json"
        )

        feedback_path.write_text(
            '{"feedback":"test"}\n',
            encoding="utf-8",
        )

        candidate = (
            make_candidate_record(
                candidate_id=(
                    "c001"
                ),

                node_id=(
                    "exp2_f1_pd_c001"
                ),

                parent=(
                    a0
                ),

                source_repo_path=(
                    a0_repo
                ),

                generation_model=(
                    "meta-test-model"
                ),

                feedback_path=(
                    feedback_path
                ),

                allowed_modification_paths=[
                    "task_agent.py",
                ],

                forbidden_modification_paths=[
                    "search/",
                    "experiments/exp2/frozen/",
                ],
            )
        )

        registry.add(
            candidate
        )

        registry.mark_generation_started(
            "c001"
        )

        candidate_repo = (
            root
            / "c001"
        )

        candidate_repo.mkdir()

        patch_path = (
            root
            / "c001.patch"
        )

        patch_path.write_text(
            "fake patch\n",
            encoding="utf-8",
        )

        registry.mark_generated(
            "c001",

            repo_path=(
                candidate_repo
            ),

            patch_path=(
                patch_path
            ),

            modified_files=[
                "task_agent.py",
            ],

            lines_added=5,

            lines_deleted=1,

            generation_tokens=1234,

            generation_calls=1,

            generation_seconds=3.5,
        )

        assert (
            registry.get(
                "c001"
            ).generation_status
            == "generated"
        )

        assert (
            registry.get_a0(
                outer_fold=1,
                target_id=(
                    "diagnostic_process_failure"
                ),
            ).candidate_id
            == "A0"
        )

        registry.export_csv(
            root
            / "candidate_registry.csv"
        )

        registry.save()

        reloaded = CandidateRegistry(
            registry_path
        )

        assert (
            len(
                reloaded
            )
            == 2
        )

        reloaded.validate_registry()

        print(
            json.dumps(
                reloaded.summary(),
                indent=2,
                ensure_ascii=False,
            )
        )

        print(
            "candidate_registry.py self-test: PASS"
        )


if __name__ == "__main__":
    _self_test()
