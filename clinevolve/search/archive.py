#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set


# =============================================================================
# Constants
# =============================================================================

ARCHIVE_SCHEMA_VERSION = "hyperagent_archive_v2"

VALID_STATUSES = {
    "root",
    "generated",
    "pending_evaluation",
    "evaluating",
    "evaluated",
    "rejected",
    "selected",
    "failed_generation",
    "failed_evaluation",
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
        ),
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if (
            math.isnan(value)
            or math.isinf(value)
        ):
            return None

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


def _atomic_write_text(
    path: Path,
    text: str,
) -> None:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:

        tmp.write(text)

        temp_name = tmp.name

    os.replace(
        temp_name,
        path,
    )


def canonical_partition(
    partition: Optional[
        Mapping[str, Any]
    ],
) -> Dict[str, Any]:
    """
    Normalize a generic search partition.

    Archive does not interpret partition fields.

    Examples:
        {"fold_id": 1, "target_id": "x"}
        {"task_group": "reasoning", "seed": 3}
        {}
    """

    if partition is None:
        return {}

    if not isinstance(
        partition,
        Mapping,
    ):
        raise TypeError(
            "partition must be a mapping or None"
        )

    return _json_safe(
        dict(partition)
    )


def partition_key(
    partition: Optional[
        Mapping[str, Any]
    ],
) -> str:
    """
    Stable canonical string for a partition.
    """

    normalized = canonical_partition(
        partition
    )

    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def partition_id(
    partition: Optional[
        Mapping[str, Any]
    ],
) -> str:
    """
    Short stable hash for a partition.
    """

    key = partition_key(
        partition
    )

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).hexdigest()

    return digest[:16]


# =============================================================================
# Archive node
# =============================================================================

@dataclass
class ArchiveNode:
    """
    One node/version in a generic HyperAgent search tree.

    ArchiveNode stores:
        - identity
        - lineage
        - repository/patch metadata
        - generation metadata
        - generic search statistics
        - generic metrics
        - generic metadata

    It does NOT define:
        - experiment objective
        - evaluation rules
        - utility
        - dataset semantics
        - target semantics
    """

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------

    node_id: str

    parent_id: Optional[str]

    generation: int

    candidate_id: str

    partition: Dict[str, Any] = field(
        default_factory=dict
    )

    status: str = "generated"

    # -------------------------------------------------------------------------
    # Agent / repository identity
    # -------------------------------------------------------------------------

    agent_version: Optional[str] = None

    repo_path: Optional[str] = None

    base_commit: Optional[str] = None

    patch_path: Optional[str] = None

    patch_hash: Optional[str] = None

    modified_files: List[str] = field(
        default_factory=list
    )

    lines_added: Optional[int] = None

    lines_deleted: Optional[int] = None

    # -------------------------------------------------------------------------
    # Generation metadata
    # -------------------------------------------------------------------------

    generation_model: Optional[str] = None

    generation_temperature: Optional[float] = None

    generation_max_tokens: Optional[int] = None

    generation_tokens: Optional[int] = None

    generation_calls: Optional[int] = None

    generation_seconds: Optional[float] = None

    iterations_left_at_generation: Optional[int] = None

    generation_error: Optional[str] = None

    # -------------------------------------------------------------------------
    # Search statistics
    #
    # These are generic outer-loop bookkeeping fields.
    # Search policy decides how to interpret them.
    # -------------------------------------------------------------------------

    n_children: int = 0

    n_successful_children: int = 0

    n_search_evaluations: int = 0

    node_successes: int = 0

    node_failures: int = 0

    clade_successes: int = 0

    clade_failures: int = 0

    # -------------------------------------------------------------------------
    # Generic evaluation/search metrics
    # -------------------------------------------------------------------------

    metrics: Dict[str, Any] = field(
        default_factory=dict
    )

    resource_metrics: Dict[str, Any] = field(
        default_factory=dict
    )

    # -------------------------------------------------------------------------
    # Generic final-selection state
    # -------------------------------------------------------------------------

    selected_final: bool = False

    rejection_reason: Optional[str] = None

    utility_result: Optional[
        Dict[str, Any]
    ] = None

    # -------------------------------------------------------------------------
    # Arbitrary audit metadata
    # -------------------------------------------------------------------------

    metadata: Dict[str, Any] = field(
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
            self.node_id
        ).strip():
            raise ValueError(
                "node_id must be non-empty"
            )

        if (
            self.parent_id
            is not None
            and not str(
                self.parent_id
            ).strip()
        ):
            raise ValueError(
                "parent_id must be None "
                "or non-empty"
            )

        if (
            self.parent_id
            == self.node_id
        ):
            raise ValueError(
                "node cannot be its own parent"
            )

        if (
            self.generation
            < 0
        ):
            raise ValueError(
                "generation must be >= 0"
            )

        if not str(
            self.candidate_id
        ).strip():
            raise ValueError(
                "candidate_id must be non-empty"
            )

        if not isinstance(
            self.partition,
            dict,
        ):
            raise TypeError(
                "partition must be a dict"
            )

        self.partition = canonical_partition(
            self.partition
        )

        if (
            self.status
            not in VALID_STATUSES
        ):
            raise ValueError(
                f"Unsupported status: "
                f"{self.status}"
            )

        integer_fields = {
            "n_children":
                self.n_children,

            "n_successful_children":
                self.n_successful_children,

            "n_search_evaluations":
                self.n_search_evaluations,

            "node_successes":
                self.node_successes,

            "node_failures":
                self.node_failures,

            "clade_successes":
                self.clade_successes,

            "clade_failures":
                self.clade_failures,
        }

        for (
            name,
            value,
        ) in (
            integer_fields.items()
        ):

            if (
                value
                < 0
            ):
                raise ValueError(
                    f"{name} must be >= 0"
                )

        if not isinstance(
            self.metrics,
            dict,
        ):
            raise TypeError(
                "metrics must be a dict"
            )

        if not isinstance(
            self.resource_metrics,
            dict,
        ):
            raise TypeError(
                "resource_metrics must be a dict"
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "metadata must be a dict"
            )

    # =========================================================================
    # Partition helpers
    # =========================================================================

    @property
    def partition_key(
        self,
    ) -> str:

        return partition_key(
            self.partition
        )

    @property
    def partition_id(
        self,
    ) -> str:

        return partition_id(
            self.partition
        )

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        self.validate()

        payload = asdict(
            self
        )

        payload[
            "schema_version"
        ] = (
            ARCHIVE_SCHEMA_VERSION
        )

        return _json_safe(
            payload
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[
            str,
            Any,
        ],
    ) -> "ArchiveNode":

        raw = dict(
            payload
        )

        schema_version = raw.pop(
            "schema_version",
            ARCHIVE_SCHEMA_VERSION,
        )

        if (
            schema_version
            != ARCHIVE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported archive node schema: "
                f"{schema_version}"
            )

        node = cls(
            **raw
        )

        node.validate()

        return node

    # =========================================================================
    # Search-stat helpers
    # =========================================================================

    def record_search_outcome(
        self,
        *,
        success: Optional[int],
        metrics: Optional[
            Mapping[str, Any]
        ] = None,
        resource_metrics: Optional[
            Mapping[str, Any]
        ] = None,
        evaluation_id: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> None:
        """
        Record one generic search-feedback evaluation.

        success:
            1    -> node_successes += 1
            0    -> node_failures += 1
            None -> evaluation counted, but no binary outcome added

        This binary value is for search policy bookkeeping only.
        It is NOT final experiment utility.
        """

        if (
            success
            not in (
                None,
                0,
                1,
            )
        ):
            raise ValueError(
                "success must be 0, 1, or None"
            )

        self.n_search_evaluations += 1

        if success == 1:
            self.node_successes += 1

        elif success == 0:
            self.node_failures += 1

        if metrics:

            self.metrics.update(
                _json_safe(
                    dict(metrics)
                )
            )

        if resource_metrics:

            self.resource_metrics.update(
                _json_safe(
                    dict(
                        resource_metrics
                    )
                )
            )

        self.metadata.setdefault(
            "search_evaluations",
            [],
        ).append(
            {
                "evaluation_id":
                    evaluation_id,

                "success":
                    success,

                "metrics":
                    _json_safe(
                        dict(
                            metrics
                            or {}
                        )
                    ),

                "resource_metrics":
                    _json_safe(
                        dict(
                            resource_metrics
                            or {}
                        )
                    ),

                "metadata":
                    _json_safe(
                        dict(
                            metadata
                            or {}
                        )
                    ),

                "recorded_at_utc":
                    utc_now_iso(),
            }
        )

        self.updated_at_utc = (
            utc_now_iso()
        )

        self.validate()

    # =========================================================================
    # Generic selection helper
    # =========================================================================

    def set_utility_result(
        self,
        result: Mapping[
            str,
            Any
        ],
    ) -> None:

        self.utility_result = (
            _json_safe(
                dict(result)
            )
        )

        self.updated_at_utc = (
            utc_now_iso()
        )


# =============================================================================
# Archive
# =============================================================================

class AgentArchive:
    """
    Persistent generic search-tree archive.

    The archive knows tree structure and node state only.

    It does not know what an experiment's partition fields mean.
    """

    def __init__(
        self,
        archive_path: str | Path,
        *,
        autosave: bool = True,
    ) -> None:

        self.archive_path = Path(
            archive_path
        )

        self.autosave = (
            autosave
        )

        self.nodes: Dict[
            str,
            ArchiveNode,
        ] = {}

        if (
            self.archive_path.exists()
        ):
            self.load()

    # =========================================================================
    # Persistence
    # =========================================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "schema_version":
                ARCHIVE_SCHEMA_VERSION,

            "updated_at_utc":
                utc_now_iso(),

            "n_nodes":
                len(
                    self.nodes
                ),

            "nodes": [
                self.nodes[
                    node_id
                ].to_dict()
                for node_id
                in sorted(
                    self.nodes
                )
            ],
        }

    def save(
        self,
    ) -> None:

        self.validate_archive()

        text = json.dumps(
            self.to_dict(),
            indent=2,
            ensure_ascii=False,
        )

        _atomic_write_text(
            self.archive_path,
            text + "\n",
        )

    def load(
        self,
    ) -> None:

        with self.archive_path.open(
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
            != ARCHIVE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported archive schema: "
                f"{schema_version}"
            )

        loaded: Dict[
            str,
            ArchiveNode,
        ] = {}

        for raw in (
            payload.get(
                "nodes",
                []
            )
        ):

            node = (
                ArchiveNode.from_dict(
                    raw
                )
            )

            if (
                node.node_id
                in loaded
            ):
                raise ValueError(
                    "Duplicate node_id: "
                    f"{node.node_id}"
                )

            loaded[
                node.node_id
            ] = node

        self.nodes = (
            loaded
        )

        self.validate_archive()

        self.recompute_child_counts(
            save=False
        )

    def _save_if_needed(
        self,
    ) -> None:

        if self.autosave:
            self.save()

    # =========================================================================
    # Basic operations
    # =========================================================================

    def __len__(
        self,
    ) -> int:

        return len(
            self.nodes
        )

    def __contains__(
        self,
        node_id: str,
    ) -> bool:

        return (
            node_id
            in self.nodes
        )

    def get(
        self,
        node_id: str,
    ) -> ArchiveNode:

        if (
            node_id
            not in self.nodes
        ):
            raise KeyError(
                f"Unknown node_id: "
                f"{node_id}"
            )

        return self.nodes[
            node_id
        ]

    def add_node(
        self,
        node: ArchiveNode,
    ) -> None:

        node.validate()

        if (
            node.node_id
            in self.nodes
        ):
            raise ValueError(
                "node_id already exists: "
                f"{node.node_id}"
            )

        if (
            node.parent_id
            is None
        ):

            if (
                node.generation
                != 0
            ):
                raise ValueError(
                    "Root node must have "
                    "generation=0"
                )

        else:

            if (
                node.parent_id
                not in self.nodes
            ):
                raise ValueError(
                    "Parent does not exist: "
                    f"{node.parent_id}"
                )

            parent = self.nodes[
                node.parent_id
            ]

            if (
                node.generation
                != parent.generation
                + 1
            ):
                raise ValueError(
                    "Child generation must equal "
                    "parent.generation + 1"
                )

            if (
                node.partition_key
                != parent.partition_key
            ):
                raise ValueError(
                    "Parent and child must belong "
                    "to the same search partition"
                )

        self.nodes[
            node.node_id
        ] = node

        self.recompute_child_counts(
            save=False
        )

        self._save_if_needed()

    def update_node(
        self,
        node_id: str,
        **updates: Any,
    ) -> ArchiveNode:

        node = self.get(
            node_id
        )

        protected = {
            "node_id",
            "parent_id",
            "generation",
            "candidate_id",
            "partition",
            "created_at_utc",
        }

        illegal = (
            protected
            & set(
                updates
            )
        )

        if illegal:

            raise ValueError(
                "Immutable node fields cannot "
                "be updated: "
                f"{sorted(illegal)}"
            )

        for (
            key,
            value,
        ) in (
            updates.items()
        ):

            if not hasattr(
                node,
                key,
            ):
                raise AttributeError(
                    "ArchiveNode has no field: "
                    f"{key}"
                )

            setattr(
                node,
                key,
                value,
            )

        node.updated_at_utc = (
            utc_now_iso()
        )

        node.validate()

        self._save_if_needed()

        return node

    def set_status(
        self,
        node_id: str,
        status: str,
    ) -> None:

        if (
            status
            not in VALID_STATUSES
        ):
            raise ValueError(
                f"Unsupported status: "
                f"{status}"
            )

        node = self.get(
            node_id
        )

        node.status = status

        node.updated_at_utc = (
            utc_now_iso()
        )

        self._save_if_needed()

    # =========================================================================
    # Partition queries
    # =========================================================================

    def partition_nodes(
        self,
        partition: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> List[
        ArchiveNode
    ]:

        wanted = partition_key(
            partition
        )

        return sorted(
            [
                node
                for node
                in self.nodes.values()
                if (
                    node.partition_key
                    == wanted
                )
            ],
            key=lambda node: (
                node.generation,
                node.node_id,
            ),
        )

    def partitions(
        self,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Return unique partition dictionaries.
        """

        by_key: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for node in (
            self.nodes.values()
        ):

            by_key.setdefault(
                node.partition_key,
                node.partition,
            )

        return [
            by_key[key]
            for key
            in sorted(
                by_key
            )
        ]

    # =========================================================================
    # Tree queries
    # =========================================================================

    def roots(
        self,
        *,
        partition: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> List[
        ArchiveNode
    ]:

        nodes = (
            self.nodes.values()
            if partition
            is None
            else self.partition_nodes(
                partition
            )
        )

        return sorted(
            [
                node
                for node in nodes
                if (
                    node.parent_id
                    is None
                )
            ],
            key=lambda node: (
                node.partition_key,
                node.node_id,
            ),
        )

    def children(
        self,
        node_id: str,
    ) -> List[
        ArchiveNode
    ]:

        self.get(
            node_id
        )

        return sorted(
            [
                node
                for node
                in self.nodes.values()
                if (
                    node.parent_id
                    == node_id
                )
            ],
            key=lambda node: (
                node.generation,
                node.node_id,
            ),
        )

    def ancestors(
        self,
        node_id: str,
        *,
        include_self: bool = False,
    ) -> List[
        ArchiveNode
    ]:

        node = self.get(
            node_id
        )

        result: List[
            ArchiveNode
        ] = []

        if include_self:
            result.append(
                node
            )

        seen: Set[str] = set()

        current = node

        while (
            current.parent_id
            is not None
        ):

            parent_id = (
                current.parent_id
            )

            if (
                parent_id
                in seen
            ):
                raise RuntimeError(
                    "Cycle detected in archive"
                )

            seen.add(
                parent_id
            )

            parent = self.get(
                parent_id
            )

            result.append(
                parent
            )

            current = (
                parent
            )

        return result

    def descendants(
        self,
        node_id: str,
        *,
        include_self: bool = False,
    ) -> List[
        ArchiveNode
    ]:

        self.get(
            node_id
        )

        result: List[
            ArchiveNode
        ] = []

        queue = [
            node_id
        ]

        visited: Set[
            str
        ] = set()

        while queue:

            current_id = (
                queue.pop(0)
            )

            if (
                current_id
                in visited
            ):
                raise RuntimeError(
                    "Cycle detected in archive"
                )

            visited.add(
                current_id
            )

            if (
                include_self
                or current_id
                != node_id
            ):
                result.append(
                    self.get(
                        current_id
                    )
                )

            for child in (
                self.children(
                    current_id
                )
            ):

                queue.append(
                    child.node_id
                )

        return sorted(
            result,
            key=lambda node: (
                node.generation,
                node.node_id,
            ),
        )

    def clade(
        self,
        node_id: str,
    ) -> List[
        ArchiveNode
    ]:

        return self.descendants(
            node_id,
            include_self=True,
        )

    def lineage(
        self,
        node_id: str,
    ) -> List[
        ArchiveNode
    ]:
        """
        Root -> ... -> node
        """

        node = self.get(
            node_id
        )

        result = self.ancestors(
            node_id,
            include_self=False,
        )

        result.reverse()

        result.append(
            node
        )

        return result

    # =========================================================================
    # Filtering
    # =========================================================================

    def filter_nodes(
        self,
        *,
        partition: Optional[
            Mapping[str, Any]
        ] = None,
        status: Optional[str] = None,
        generation: Optional[int] = None,
        selected_final: Optional[bool] = None,
    ) -> List[
        ArchiveNode
    ]:

        if partition is None:

            result = list(
                self.nodes.values()
            )

        else:

            result = (
                self.partition_nodes(
                    partition
                )
            )

        if status is not None:

            result = [
                node
                for node in result
                if (
                    node.status
                    == status
                )
            ]

        if generation is not None:

            result = [
                node
                for node in result
                if (
                    node.generation
                    == generation
                )
            ]

        if (
            selected_final
            is not None
        ):

            result = [
                node
                for node in result
                if (
                    node.selected_final
                    == selected_final
                )
            ]

        return sorted(
            result,
            key=lambda node: (
                node.partition_key,
                node.generation,
                node.node_id,
            ),
        )

    # =========================================================================
    # Search statistics
    # =========================================================================

    def recompute_child_counts(
        self,
        *,
        save: bool = True,
    ) -> None:

        for node in (
            self.nodes.values()
        ):

            children = (
                self.children(
                    node.node_id
                )
            )

            node.n_children = (
                len(
                    children
                )
            )

            node.n_successful_children = sum(
                1
                for child in children
                if (
                    child.selected_final
                    or child.status
                    == "selected"
                    or child.node_successes
                    > child.node_failures
                )
            )

            node.updated_at_utc = (
                utc_now_iso()
            )

        if save:
            self._save_if_needed()

    def recompute_clade_statistics(
        self,
        *,
        save: bool = True,
    ) -> None:
        """
        Aggregate generic search successes/failures across each clade.

        This does not define what success means.
        """

        for node in (
            self.nodes.values()
        ):

            clade_nodes = (
                self.clade(
                    node.node_id
                )
            )

            node.clade_successes = sum(
                int(
                    item.node_successes
                )
                for item
                in clade_nodes
            )

            node.clade_failures = sum(
                int(
                    item.node_failures
                )
                for item
                in clade_nodes
            )

            node.updated_at_utc = (
                utc_now_iso()
            )

        if save:
            self._save_if_needed()

    def clade_empirical_rate(
        self,
        node_id: str,
    ) -> Optional[
        float
    ]:

        node = self.get(
            node_id
        )

        denominator = (
            node.clade_successes
            + node.clade_failures
        )

        if denominator <= 0:
            return None

        return (
            node.clade_successes
            / denominator
        )

    # =========================================================================
    # Final selection
    # =========================================================================

    def mark_selected_final(
        self,
        node_id: str,
    ) -> None:
        """
        Mark one selected-final candidate within its own partition.

        At most one node may be selected_final per partition.
        """

        node = self.get(
            node_id
        )

        for other in (
            self.nodes.values()
        ):

            if (
                other.partition_key
                != node.partition_key
            ):
                continue

            other.selected_final = (
                False
            )

            if (
                other.status
                == "selected"
            ):

                other.status = (
                    "evaluated"
                )

        node.selected_final = True
        node.status = "selected"

        node.updated_at_utc = (
            utc_now_iso()
        )

        self._save_if_needed()

    def selected_final(
        self,
        *,
        partition: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> Optional[
        ArchiveNode
    ]:

        wanted_key = (
            partition_key(
                partition
            )
        )

        matches = [
            node
            for node
            in self.nodes.values()
            if (
                node.partition_key
                == wanted_key
                and node.selected_final
            )
        ]

        if (
            len(
                matches
            )
            > 1
        ):

            raise RuntimeError(
                "Multiple selected_final nodes "
                "in the same partition"
            )

        return (
            matches[0]
            if matches
            else None
        )

    # =========================================================================
    # Archive validation
    # =========================================================================

    def validate_archive(
        self,
    ) -> None:

        # Individual nodes
        for node in (
            self.nodes.values()
        ):
            node.validate()

        # Parent edges
        for node in (
            self.nodes.values()
        ):

            if (
                node.parent_id
                is None
            ):
                continue

            if (
                node.parent_id
                not in self.nodes
            ):
                raise ValueError(
                    "Missing parent for "
                    f"{node.node_id}: "
                    f"{node.parent_id}"
                )

            parent = (
                self.nodes[
                    node.parent_id
                ]
            )

            if (
                node.generation
                != parent.generation
                + 1
            ):
                raise ValueError(
                    "Generation mismatch: "
                    f"{node.node_id}"
                )

            if (
                node.partition_key
                != parent.partition_key
            ):
                raise ValueError(
                    "Cross-partition parent/child "
                    f"edge detected: {node.node_id}"
                )

        # Cycle detection
        for node_id in (
            self.nodes
        ):

            seen: Set[str] = set()

            current = self.nodes[
                node_id
            ]

            while (
                current.parent_id
                is not None
            ):

                parent_id = (
                    current.parent_id
                )

                if (
                    parent_id
                    in seen
                ):
                    raise ValueError(
                        "Cycle detected involving "
                        f"{node_id}"
                    )

                seen.add(
                    parent_id
                )

                current = (
                    self.nodes[
                        parent_id
                    ]
                )

        # One selected_final per partition
        selected_partitions: Set[
            str
        ] = set()

        for node in (
            self.nodes.values()
        ):

            if not node.selected_final:
                continue

            key = (
                node.partition_key
            )

            if (
                key
                in selected_partitions
            ):
                raise ValueError(
                    "Multiple selected_final nodes "
                    "for partition "
                    f"{key}"
                )

            selected_partitions.add(
                key
            )

    # =========================================================================
    # Reporting
    # =========================================================================

    def summary(
        self,
    ) -> Dict[
        str,
        Any
    ]:

        status_counts: Dict[
            str,
            int,
        ] = {}

        partition_counts: Dict[
            str,
            int,
        ] = {}

        for node in (
            self.nodes.values()
        ):

            status_counts[
                node.status
            ] = (
                status_counts.get(
                    node.status,
                    0,
                )
                + 1
            )

            partition_counts[
                node.partition_id
            ] = (
                partition_counts.get(
                    node.partition_id,
                    0,
                )
                + 1
            )

        return {
            "schema_version":
                ARCHIVE_SCHEMA_VERSION,

            "archive_path":
                str(
                    self.archive_path
                ),

            "n_nodes":
                len(
                    self.nodes
                ),

            "n_partitions":
                len(
                    self.partitions()
                ),

            "n_roots":
                len(
                    self.roots()
                ),

            "status_counts":
                status_counts,

            "partition_counts":
                partition_counts,
        }

    def print_summary(
        self,
    ) -> None:

        print(
            json.dumps(
                self.summary(),
                indent=2,
                ensure_ascii=False,
            )
        )


# =============================================================================
# Convenience constructors
# =============================================================================

def make_root_node(
    *,
    node_id: str,
    candidate_id: str = "A0",
    partition: Optional[
        Mapping[str, Any]
    ] = None,
    agent_version: str = "A0",
    repo_path: Optional[str] = None,
    base_commit: Optional[str] = None,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> ArchiveNode:

    return ArchiveNode(
        node_id=(
            node_id
        ),

        parent_id=None,

        generation=0,

        candidate_id=(
            candidate_id
        ),

        partition=(
            canonical_partition(
                partition
            )
        ),

        status="root",

        agent_version=(
            agent_version
        ),

        repo_path=(
            repo_path
        ),

        base_commit=(
            base_commit
        ),

        metadata=(
            dict(
                metadata
            )
            if metadata
            else {}
        ),
    )


def make_child_node(
    *,
    node_id: str,
    parent: ArchiveNode,
    candidate_id: str,
    agent_version: Optional[str] = None,
    repo_path: Optional[str] = None,
    patch_path: Optional[str] = None,
    patch_hash: Optional[str] = None,
    modified_files: Optional[
        Iterable[str]
    ] = None,
    generation_model: Optional[
        str
    ] = None,
    generation_temperature: Optional[
        float
    ] = None,
    generation_max_tokens: Optional[
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
    iterations_left_at_generation: Optional[
        int
    ] = None,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> ArchiveNode:

    return ArchiveNode(
        node_id=(
            node_id
        ),

        parent_id=(
            parent.node_id
        ),

        generation=(
            parent.generation
            + 1
        ),

        candidate_id=(
            candidate_id
        ),

        partition=(
            dict(
                parent.partition
            )
        ),

        status="generated",

        agent_version=(
            agent_version
        ),

        repo_path=(
            repo_path
        ),

        base_commit=(
            parent.base_commit
        ),

        patch_path=(
            patch_path
        ),

        patch_hash=(
            patch_hash
        ),

        modified_files=(
            list(
                modified_files
            )
            if modified_files
            else []
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

        generation_tokens=(
            generation_tokens
        ),

        generation_calls=(
            generation_calls
        ),

        generation_seconds=(
            generation_seconds
        ),

        iterations_left_at_generation=(
            iterations_left_at_generation
        ),

        metadata=(
            dict(
                metadata
            )
            if metadata
            else {}
        ),
    )


# =============================================================================
# Minimal self-test
# =============================================================================

def _self_test() -> None:

    with tempfile.TemporaryDirectory() as tempdir:

        archive_path = (
            Path(
                tempdir
            )
            / "archive.json"
        )

        archive = AgentArchive(
            archive_path
        )

        partition = {
            "experiment":
                "self_test",

            "track":
                "track_a",
        }

        root = make_root_node(
            node_id="track_a_A0",
            candidate_id="A0",
            partition=(
                partition
            ),
            repo_path="/tmp/A0",
            base_commit="deadbeef",
        )

        archive.add_node(
            root
        )

        child1 = make_child_node(
            node_id="track_a_c001",
            parent=(
                root
            ),
            candidate_id="c001",
            repo_path="/tmp/c001",
            patch_hash="hash001",
            modified_files=[
                "task_agent.py"
            ],
            generation_model="fake-model",
        )

        archive.add_node(
            child1
        )

        child2 = make_child_node(
            node_id="track_a_c002",
            parent=(
                root
            ),
            candidate_id="c002",
            repo_path="/tmp/c002",
            patch_hash="hash002",
        )

        archive.add_node(
            child2
        )

        child1.record_search_outcome(
            success=1,
            evaluation_id="eval001",
            metrics={
                "score":
                    0.8,
            },
            resource_metrics={
                "tokens":
                    1000,
            },
        )

        child2.record_search_outcome(
            success=0,
            evaluation_id="eval002",
            metrics={
                "score":
                    0.4,
            },
        )

        archive.recompute_child_counts(
            save=False
        )

        archive.recompute_clade_statistics(
            save=False
        )

        assert (
            root.n_children
            == 2
        )

        assert (
            root.clade_successes
            == 1
        )

        assert (
            root.clade_failures
            == 1
        )

        assert (
            len(
                archive.children(
                    root.node_id
                )
            )
            == 2
        )

        assert (
            len(
                archive.clade(
                    root.node_id
                )
            )
            == 3
        )

        assert (
            len(
                archive.lineage(
                    child1.node_id
                )
            )
            == 2
        )

        assert (
            archive.partition_nodes(
                partition
            )[
                0
            ].partition
            == partition
        )

        archive.mark_selected_final(
            child1.node_id
        )

        selected = (
            archive.selected_final(
                partition=partition
            )
        )

        assert (
            selected
            is not None
        )

        assert (
            selected.node_id
            == child1.node_id
        )

        archive.save()

        reloaded = AgentArchive(
            archive_path
        )

        assert (
            len(
                reloaded
            )
            == 3
        )

        assert (
            reloaded.selected_final(
                partition=partition
            ).node_id
            == child1.node_id
        )

        reloaded.validate_archive()

        print(
            json.dumps(
                reloaded.summary(),
                indent=2,
                ensure_ascii=False,
            )
        )

        print(
            "archive.py self-test: PASS"
        )


if __name__ == "__main__":
    _self_test()
