#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from clinevolve.search.archive import (
    AgentArchive,
    ArchiveNode,
    canonical_partition,
)


# =============================================================================
# Constants
# =============================================================================

POLICY_SCHEMA_VERSION = "hyperagent_policy_v2"

ACTION_EXPAND = "expand"
ACTION_EVALUATE = "evaluate"
ACTION_STOP = "stop"

VALID_ACTIONS = {
    ACTION_EXPAND,
    ACTION_EVALUATE,
    ACTION_STOP,
}


DEFAULT_EXPANDABLE_STATUSES = {
    "root",
    "generated",
    "pending_evaluation",
    "evaluating",
    "evaluated",
}

DEFAULT_EVALUATABLE_STATUSES = {
    "root",
    "generated",
    "pending_evaluation",
    "evaluating",
    "evaluated",
}


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
        ),
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            return None

        return value

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key):
                _json_safe(item)
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

    return str(
        value
    )


def _stable_sorted_nodes(
    nodes: Iterable[
        ArchiveNode
    ],
) -> List[
    ArchiveNode
]:

    return sorted(
        list(
            nodes
        ),
        key=lambda node: (
            node.partition_key,
            node.generation,
            node.node_id,
        ),
    )


# =============================================================================
# Search budget
# =============================================================================

@dataclass
class SearchBudget:
    """
    Generic search-budget bookkeeping.

    The policy does not define what one budget unit means.

    The experiment/controller decides that.
    """

    total_budget: int

    used_budget: int = 0

    completed_evaluations: int = 0

    completed_expansions: int = 0

    def validate(
        self,
    ) -> None:

        if (
            self.total_budget
            <= 0
        ):
            raise ValueError(
                "total_budget must be > 0"
            )

        if (
            self.used_budget
            < 0
        ):
            raise ValueError(
                "used_budget must be >= 0"
            )

        if (
            self.used_budget
            > self.total_budget
        ):
            raise ValueError(
                "used_budget cannot exceed "
                "total_budget"
            )

        if (
            self.completed_evaluations
            < 0
        ):
            raise ValueError(
                "completed_evaluations "
                "must be >= 0"
            )

        if (
            self.completed_expansions
            < 0
        ):
            raise ValueError(
                "completed_expansions "
                "must be >= 0"
            )

    @property
    def remaining_budget(
        self,
    ) -> int:

        return max(
            0,
            self.total_budget
            - self.used_budget,
        )

    @property
    def exhausted(
        self,
    ) -> bool:

        return (
            self.remaining_budget
            <= 0
        )

    @property
    def fraction_used(
        self,
    ) -> float:

        return (
            self.used_budget
            / self.total_budget
        )

    def consume(
        self,
        units: int = 1,
        *,
        action: Optional[str] = None,
    ) -> None:

        if (
            units
            <= 0
        ):
            raise ValueError(
                "units must be > 0"
            )

        if (
            self.used_budget
            + units
            > self.total_budget
        ):
            raise RuntimeError(
                "Search budget exceeded"
            )

        self.used_budget += (
            units
        )

        if (
            action
            == ACTION_EXPAND
        ):
            self.completed_expansions += 1

        elif (
            action
            == ACTION_EVALUATE
        ):
            self.completed_evaluations += 1

        self.validate()


# =============================================================================
# Policy configuration
# =============================================================================

@dataclass(frozen=True)
class SearchPolicyConfig:
    """
    Generic HGM-inspired search policy configuration.

    alpha:
        Adaptive widening exponent.

        Expansion is preferred when:

            completed_evaluations ** alpha
            >= number_of_nodes_in_partition

    initial_root_expansions:
        Number of direct root expansions before normal adaptive scheduling.

    tau_min / tau_max:
        Thompson posterior concentration schedule bounds.

    max_children_per_node:
        Optional expansion cap per node.

    max_evaluations_per_node:
        Optional search-evaluation cap per node.

    min_evaluations_before_expansion:
        Optional requirement before a non-root node may itself expand.
    """

    policy_version: str

    alpha: float = 0.6

    initial_root_expansions: int = 5

    tau_min: float = 1.0

    tau_max: float = 50.0

    max_children_per_node: Optional[
        int
    ] = None

    max_evaluations_per_node: Optional[
        int
    ] = None

    min_evaluations_before_expansion: int = 0

    expandable_statuses: Tuple[
        str,
        ...
    ] = tuple(
        sorted(
            DEFAULT_EXPANDABLE_STATUSES
        )
    )

    evaluatable_statuses: Tuple[
        str,
        ...
    ] = tuple(
        sorted(
            DEFAULT_EVALUATABLE_STATUSES
        )
    )

    random_seed: int = 20260908

    def validate(
        self,
    ) -> None:

        if not str(
            self.policy_version
        ).strip():
            raise ValueError(
                "policy_version must be "
                "non-empty"
            )

        if not (
            0.0
            < self.alpha
            <= 1.0
        ):
            raise ValueError(
                "alpha must be in (0,1]"
            )

        if (
            self.initial_root_expansions
            < 0
        ):
            raise ValueError(
                "initial_root_expansions "
                "must be >= 0"
            )

        if (
            self.tau_min
            <= 0
        ):
            raise ValueError(
                "tau_min must be > 0"
            )

        if (
            self.tau_max
            < self.tau_min
        ):
            raise ValueError(
                "tau_max must be >= tau_min"
            )

        if (
            self.max_children_per_node
            is not None
            and self.max_children_per_node
            <= 0
        ):
            raise ValueError(
                "max_children_per_node "
                "must be > 0 or None"
            )

        if (
            self.max_evaluations_per_node
            is not None
            and self.max_evaluations_per_node
            <= 0
        ):
            raise ValueError(
                "max_evaluations_per_node "
                "must be > 0 or None"
            )

        if (
            self.min_evaluations_before_expansion
            < 0
        ):
            raise ValueError(
                "min_evaluations_before_expansion "
                "must be >= 0"
            )

    def to_dict(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        self.validate()

        return {
            "schema_version":
                POLICY_SCHEMA_VERSION,

            **_json_safe(
                asdict(
                    self
                )
            ),
        }


# =============================================================================
# Policy decision
# =============================================================================

@dataclass
class PolicyDecision:
    """
    One outer-loop scheduling decision.
    """

    action: str

    node_id: Optional[str]

    reason: str

    tau: float

    sampled_scores: Dict[
        str,
        float,
    ] = field(
        default_factory=dict
    )

    metadata: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:

        if (
            self.action
            not in VALID_ACTIONS
        ):
            raise ValueError(
                f"Unknown action: "
                f"{self.action}"
            )

        if (
            self.action
            in {
                ACTION_EXPAND,
                ACTION_EVALUATE,
            }
            and not self.node_id
        ):
            raise ValueError(
                f"{self.action} "
                "requires node_id"
            )

    def to_dict(
        self,
    ) -> Dict[
        str,
        Any,
    ]:

        return _json_safe(
            asdict(
                self
            )
        )


# =============================================================================
# Generic search policy
# =============================================================================

class SearchPolicy:
    """
    Generic HGM-inspired outer-loop search policy.

    Responsibilities:
        - decide EXPAND vs EVALUATE
        - select expansion parent
        - select node for further evaluation
        - schedule exploration/exploitation

    It does NOT define:
        - experiment semantics
        - utility
        - success criterion
        - evaluator
        - dataset
        - candidate-generation mechanism

    The only grouping concept it understands is a generic `partition`.
    """

    def __init__(
        self,
        config: SearchPolicyConfig,
    ) -> None:

        config.validate()

        self.config = (
            config
        )

        self.rng = (
            np.random.default_rng(
                config.random_seed
            )
        )

    # =========================================================================
    # Tau schedule
    # =========================================================================

    def compute_tau(
        self,
        budget: SearchBudget,
    ) -> float:
        """
        Increase posterior concentration as remaining budget decreases.

        raw_tau =
            total_budget / remaining_budget

        Then clamp to [tau_min, tau_max].
        """

        budget.validate()

        remaining = max(
            1,
            budget.remaining_budget,
        )

        raw_tau = (
            budget.total_budget
            / remaining
        )

        return float(
            min(
                self.config.tau_max,
                max(
                    self.config.tau_min,
                    raw_tau,
                ),
            )
        )

    # =========================================================================
    # Node eligibility
    # =========================================================================

    def is_expandable(
        self,
        node: ArchiveNode,
    ) -> bool:

        if (
            node.status
            not in self.config.expandable_statuses
        ):
            return False

        if (
            node.selected_final
        ):
            return False

        if (
            self.config.max_children_per_node
            is not None
            and node.n_children
            >= self.config.max_children_per_node
        ):
            return False

        # Root is allowed to bootstrap immediately.
        if (
            node.generation
            == 0
        ):
            return True

        if (
            node.n_search_evaluations
            < self.config.min_evaluations_before_expansion
        ):
            return False

        return True

    def is_evaluatable(
        self,
        node: ArchiveNode,
    ) -> bool:

        if (
            node.status
            not in self.config.evaluatable_statuses
        ):
            return False

        if (
            node.selected_final
        ):
            return False

        if (
            self.config.max_evaluations_per_node
            is not None
            and node.n_search_evaluations
            >= self.config.max_evaluations_per_node
        ):
            return False

        return True

    def expandable_nodes(
        self,
        archive: AgentArchive,
        *,
        partition: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> List[
        ArchiveNode
    ]:

        normalized_partition = (
            canonical_partition(
                partition
            )
        )

        nodes = (
            archive.filter_nodes(
                partition=normalized_partition
            )
        )

        return _stable_sorted_nodes(
            node
            for node in nodes
            if self.is_expandable(
                node
            )
        )

    def evaluatable_nodes(
        self,
        archive: AgentArchive,
        *,
        partition: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> List[
        ArchiveNode
    ]:

        normalized_partition = (
            canonical_partition(
                partition
            )
        )

        nodes = (
            archive.filter_nodes(
                partition=normalized_partition
            )
        )

        return _stable_sorted_nodes(
            node
            for node in nodes
            if self.is_evaluatable(
                node
            )
        )

    # =========================================================================
    # Thompson sampling
    # =========================================================================

    def _beta_sample(
        self,
        *,
        successes: int,
        failures: int,
        tau: float,
    ) -> float:

        if (
            successes
            < 0
        ):
            raise ValueError(
                "successes must be >= 0"
            )

        if (
            failures
            < 0
        ):
            raise ValueError(
                "failures must be >= 0"
            )

        if (
            tau
            <= 0
        ):
            raise ValueError(
                "tau must be > 0"
            )

        alpha = (
            tau
            * (
                1.0
                + successes
            )
        )

        beta = (
            tau
            * (
                1.0
                + failures
            )
        )

        return float(
            self.rng.beta(
                alpha,
                beta,
            )
        )

    # =========================================================================
    # Expansion policy
    # =========================================================================

    def select_parent(
        self,
        archive: AgentArchive,
        *,
        partition: Optional[
            Mapping[str, Any]
        ] = None,
        budget: SearchBudget,
    ) -> PolicyDecision:
        """
        Select expansion parent using clade-level Thompson sampling.
        """

        normalized_partition = (
            canonical_partition(
                partition
            )
        )

        tau = self.compute_tau(
            budget
        )

        candidates = (
            self.expandable_nodes(
                archive,
                partition=normalized_partition,
            )
        )

        if not candidates:

            return PolicyDecision(
                action=ACTION_STOP,
                node_id=None,
                reason=(
                    "no expandable nodes"
                ),
                tau=tau,
                metadata={
                    "partition":
                        normalized_partition
                },
            )

        sampled_scores: Dict[
            str,
            float,
        ] = {}

        for node in candidates:

            sampled_scores[
                node.node_id
            ] = (
                self._beta_sample(
                    successes=(
                        node.clade_successes
                    ),
                    failures=(
                        node.clade_failures
                    ),
                    tau=tau,
                )
            )

        best_node = max(
            candidates,
            key=lambda node: (
                sampled_scores[
                    node.node_id
                ],

                # Prefer nodes with fewer children on exact score ties.
                -node.n_children,

                # Prefer shallower node on exact tie.
                -node.generation,

                node.node_id,
            ),
        )

        return PolicyDecision(
            action=ACTION_EXPAND,
            node_id=(
                best_node.node_id
            ),
            reason=(
                "selected parent using "
                "clade Thompson sampling"
            ),
            tau=tau,
            sampled_scores=(
                sampled_scores
            ),
            metadata={
                "partition":
                    normalized_partition,

                "n_candidates":
                    len(
                        candidates
                    ),

                "selected_generation":
                    best_node.generation,

                "selected_n_children":
                    best_node.n_children,

                "selected_clade_successes":
                    best_node.clade_successes,

                "selected_clade_failures":
                    best_node.clade_failures,
            },
        )

    # =========================================================================
    # Evaluation policy
    # =========================================================================

    def select_for_evaluation(
        self,
        archive: AgentArchive,
        *,
        partition: Optional[
            Mapping[str, Any]
        ] = None,
        budget: SearchBudget,
    ) -> PolicyDecision:
        """
        Select node for additional search evaluation using node-level
        Thompson sampling.
        """

        normalized_partition = (
            canonical_partition(
                partition
            )
        )

        tau = self.compute_tau(
            budget
        )

        candidates = (
            self.evaluatable_nodes(
                archive,
                partition=normalized_partition,
            )
        )

        if not candidates:

            return PolicyDecision(
                action=ACTION_STOP,
                node_id=None,
                reason=(
                    "no evaluatable nodes"
                ),
                tau=tau,
                metadata={
                    "partition":
                        normalized_partition
                },
            )

        sampled_scores: Dict[
            str,
            float,
        ] = {}

        for node in candidates:

            sampled_scores[
                node.node_id
            ] = (
                self._beta_sample(
                    successes=(
                        node.node_successes
                    ),
                    failures=(
                        node.node_failures
                    ),
                    tau=tau,
                )
            )

        best_node = max(
            candidates,
            key=lambda node: (
                sampled_scores[
                    node.node_id
                ],

                # Prefer less-evaluated node on exact score ties.
                -node.n_search_evaluations,

                -node.generation,

                node.node_id,
            ),
        )

        return PolicyDecision(
            action=ACTION_EVALUATE,
            node_id=(
                best_node.node_id
            ),
            reason=(
                "selected candidate using "
                "node-level Thompson sampling"
            ),
            tau=tau,
            sampled_scores=(
                sampled_scores
            ),
            metadata={
                "partition":
                    normalized_partition,

                "n_candidates":
                    len(
                        candidates
                    ),

                "selected_generation":
                    best_node.generation,

                "selected_node_successes":
                    best_node.node_successes,

                "selected_node_failures":
                    best_node.node_failures,

                "selected_n_search_evaluations":
                    best_node.n_search_evaluations,
            },
        )

    # =========================================================================
    # Initial expansion
    # =========================================================================

    def should_force_initial_root_expansion(
        self,
        archive: AgentArchive,
        *,
        partition: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> bool:

        if (
            self.config.initial_root_expansions
            <= 0
        ):
            return False

        normalized_partition = (
            canonical_partition(
                partition
            )
        )

        roots = archive.roots(
            partition=normalized_partition
        )

        if (
            len(
                roots
            )
            != 1
        ):
            raise RuntimeError(
                "Expected exactly one root "
                "per search partition"
            )

        root = roots[
            0
        ]

        return (
            root.n_children
            < self.config.initial_root_expansions
            and self.is_expandable(
                root
            )
        )

    # =========================================================================
    # Widening
    # =========================================================================

    def widening_prefers_expansion(
        self,
        archive: AgentArchive,
        *,
        partition: Optional[
            Mapping[str, Any]
        ] = None,
        budget: SearchBudget,
    ) -> bool:
        """
        Adaptive widening:

            completed_evaluations ** alpha
            >= number_of_nodes_in_partition

        True:
            EXPAND

        False:
            EVALUATE
        """

        normalized_partition = (
            canonical_partition(
                partition
            )
        )

        nodes = archive.filter_nodes(
            partition=normalized_partition
        )

        n_nodes = len(
            nodes
        )

        if (
            n_nodes
            <= 0
        ):
            raise RuntimeError(
                "Search partition contains no nodes"
            )

        n_eval = (
            budget.completed_evaluations
        )

        widening_value = (
            float(
                n_eval
            )
            ** self.config.alpha
        )

        return (
            widening_value
            >= n_nodes
        )

    # =========================================================================
    # Full scheduling policy
    # =========================================================================

    def decide_action(
        self,
        archive: AgentArchive,
        *,
        partition: Optional[
            Mapping[str, Any]
        ] = None,
        budget: SearchBudget,
    ) -> PolicyDecision:
        """
        Decide the next outer-loop action.

        Priority:
            1. stop if budget exhausted
            2. perform configured initial root expansion
            3. if only one action type is possible, use it
            4. otherwise apply adaptive widening
        """

        budget.validate()

        normalized_partition = (
            canonical_partition(
                partition
            )
        )

        tau = self.compute_tau(
            budget
        )

        if (
            budget.exhausted
        ):

            return PolicyDecision(
                action=ACTION_STOP,
                node_id=None,
                reason=(
                    "search budget exhausted"
                ),
                tau=tau,
                metadata={
                    "partition":
                        normalized_partition
                },
            )

        partition_nodes = (
            archive.filter_nodes(
                partition=normalized_partition
            )
        )

        if not partition_nodes:

            return PolicyDecision(
                action=ACTION_STOP,
                node_id=None,
                reason=(
                    "search partition contains "
                    "no nodes"
                ),
                tau=tau,
                metadata={
                    "partition":
                        normalized_partition
                },
            )

        expandable = (
            self.expandable_nodes(
                archive,
                partition=normalized_partition,
            )
        )

        evaluatable = (
            self.evaluatable_nodes(
                archive,
                partition=normalized_partition,
            )
        )

        if (
            not expandable
            and not evaluatable
        ):

            return PolicyDecision(
                action=ACTION_STOP,
                node_id=None,
                reason=(
                    "no expandable or evaluatable "
                    "nodes"
                ),
                tau=tau,
                metadata={
                    "partition":
                        normalized_partition
                },
            )

        # ---------------------------------------------------------------------
        # Root bootstrap
        # ---------------------------------------------------------------------

        if (
            expandable
            and self.should_force_initial_root_expansion(
                archive,
                partition=normalized_partition,
            )
        ):

            roots = archive.roots(
                partition=normalized_partition
            )

            root = roots[
                0
            ]

            return PolicyDecision(
                action=ACTION_EXPAND,
                node_id=(
                    root.node_id
                ),
                reason=(
                    "initial root expansion "
                    "bootstrap"
                ),
                tau=tau,
                metadata={
                    "partition":
                        normalized_partition,

                    "root_n_children":
                        root.n_children,

                    "required_initial_root_expansions":
                        self.config.initial_root_expansions,
                },
            )

        # ---------------------------------------------------------------------
        # Only one available action
        # ---------------------------------------------------------------------

        if (
            expandable
            and not evaluatable
        ):

            return self.select_parent(
                archive,
                partition=normalized_partition,
                budget=budget,
            )

        if (
            evaluatable
            and not expandable
        ):

            return self.select_for_evaluation(
                archive,
                partition=normalized_partition,
                budget=budget,
            )

        # ---------------------------------------------------------------------
        # Adaptive widening
        # ---------------------------------------------------------------------

        if (
            self.widening_prefers_expansion(
                archive,
                partition=normalized_partition,
                budget=budget,
            )
        ):

            return self.select_parent(
                archive,
                partition=normalized_partition,
                budget=budget,
            )

        return self.select_for_evaluation(
            archive,
            partition=normalized_partition,
            budget=budget,
        )

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def node_expansion_scores(
        self,
        archive: AgentArchive,
        *,
        partition: Optional[
            Mapping[str, Any]
        ] = None,
        budget: SearchBudget,
    ) -> List[
        Dict[str, Any]
    ]:

        normalized_partition = (
            canonical_partition(
                partition
            )
        )

        tau = self.compute_tau(
            budget
        )

        rows: List[
            Dict[str, Any]
        ] = []

        for node in (
            self.expandable_nodes(
                archive,
                partition=normalized_partition,
            )
        ):

            score = self._beta_sample(
                successes=(
                    node.clade_successes
                ),
                failures=(
                    node.clade_failures
                ),
                tau=tau,
            )

            rows.append(
                {
                    "node_id":
                        node.node_id,

                    "partition":
                        normalized_partition,

                    "generation":
                        node.generation,

                    "n_children":
                        node.n_children,

                    "clade_successes":
                        node.clade_successes,

                    "clade_failures":
                        node.clade_failures,

                    "tau":
                        tau,

                    "sampled_score":
                        score,
                }
            )

        return rows

    def node_evaluation_scores(
        self,
        archive: AgentArchive,
        *,
        partition: Optional[
            Mapping[str, Any]
        ] = None,
        budget: SearchBudget,
    ) -> List[
        Dict[str, Any]
    ]:

        normalized_partition = (
            canonical_partition(
                partition
            )
        )

        tau = self.compute_tau(
            budget
        )

        rows: List[
            Dict[str, Any]
        ] = []

        for node in (
            self.evaluatable_nodes(
                archive,
                partition=normalized_partition,
            )
        ):

            score = self._beta_sample(
                successes=(
                    node.node_successes
                ),
                failures=(
                    node.node_failures
                ),
                tau=tau,
            )

            rows.append(
                {
                    "node_id":
                        node.node_id,

                    "partition":
                        normalized_partition,

                    "generation":
                        node.generation,

                    "n_search_evaluations":
                        node.n_search_evaluations,

                    "node_successes":
                        node.node_successes,

                    "node_failures":
                        node.node_failures,

                    "tau":
                        tau,

                    "sampled_score":
                        score,
                }
            )

        return rows


# =============================================================================
# Config serialization
# =============================================================================

def save_policy_config(
    config: SearchPolicyConfig,
    path: str,
) -> None:

    config.validate()

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config.to_dict(),
            f,
            indent=2,
            ensure_ascii=False,
        )

        f.write(
            "\n"
        )


# =============================================================================
# Backward-compatible alias
# =============================================================================

ClinicalSearchPolicy = SearchPolicy


# =============================================================================
# Minimal self-test
# =============================================================================

def _self_test() -> None:

    import tempfile
    from pathlib import Path

    from clinevolve.search.archive import (
        AgentArchive,
        make_child_node,
        make_root_node,
    )

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
            partition=partition,
            repo_path="/tmp/A0",
            base_commit="deadbeef",
        )

        archive.add_node(
            root
        )

        policy = SearchPolicy(
            SearchPolicyConfig(
                policy_version=(
                    "policy_self_test_v2"
                ),
                alpha=0.6,
                initial_root_expansions=2,
                max_children_per_node=3,
                max_evaluations_per_node=5,
                random_seed=12345,
            )
        )

        budget = SearchBudget(
            total_budget=20
        )

        # ---------------------------------------------------------------------
        # Initial root expansion #1
        # ---------------------------------------------------------------------

        decision = (
            policy.decide_action(
                archive,
                partition=partition,
                budget=budget,
            )
        )

        assert (
            decision.action
            == ACTION_EXPAND
        )

        assert (
            decision.node_id
            == root.node_id
        )

        child1 = make_child_node(
            node_id="track_a_c001",
            parent=root,
            candidate_id="c001",
        )

        archive.add_node(
            child1
        )

        budget.consume(
            1,
            action=ACTION_EXPAND,
        )

        # ---------------------------------------------------------------------
        # Initial root expansion #2
        # ---------------------------------------------------------------------

        decision = (
            policy.decide_action(
                archive,
                partition=partition,
                budget=budget,
            )
        )

        assert (
            decision.action
            == ACTION_EXPAND
        )

        assert (
            decision.node_id
            == root.node_id
        )

        child2 = make_child_node(
            node_id="track_a_c002",
            parent=root,
            candidate_id="c002",
        )

        archive.add_node(
            child2
        )

        budget.consume(
            1,
            action=ACTION_EXPAND,
        )

        # ---------------------------------------------------------------------
        # Synthetic search evidence
        # ---------------------------------------------------------------------

        for _ in range(4):
            child1.record_search_outcome(
                success=1
            )

        child1.record_search_outcome(
            success=0
        )

        child2.record_search_outcome(
            success=1
        )

        for _ in range(4):
            child2.record_search_outcome(
                success=0
            )

        root.record_search_outcome(
            success=1
        )

        root.record_search_outcome(
            success=0
        )

        archive.recompute_child_counts(
            save=False
        )

        archive.recompute_clade_statistics(
            save=False
        )

        archive.save()

        # ---------------------------------------------------------------------
        # Expansion policy
        # ---------------------------------------------------------------------

        parent_decision = (
            policy.select_parent(
                archive,
                partition=partition,
                budget=budget,
            )
        )

        assert (
            parent_decision.action
            == ACTION_EXPAND
        )

        assert (
            parent_decision.node_id
            in archive.nodes
        )

        # ---------------------------------------------------------------------
        # Evaluation policy
        # ---------------------------------------------------------------------

        eval_decision = (
            policy.select_for_evaluation(
                archive,
                partition=partition,
                budget=budget,
            )
        )

        assert (
            eval_decision.action
            == ACTION_EVALUATE
        )

        assert (
            eval_decision.node_id
            in archive.nodes
        )

        # ---------------------------------------------------------------------
        # Partition isolation
        # ---------------------------------------------------------------------

        other_partition = {
            "experiment":
                "self_test",

            "track":
                "track_b",
        }

        other_root = make_root_node(
            node_id="track_b_A0",
            candidate_id="A0",
            partition=other_partition,
        )

        archive.add_node(
            other_root
        )

        track_a_nodes = (
            policy.expandable_nodes(
                archive,
                partition=partition,
            )
        )

        assert all(
            node.partition
            == partition
            for node in track_a_nodes
        )

        # ---------------------------------------------------------------------
        # Tau increases later in search
        # ---------------------------------------------------------------------

        tau_early = (
            policy.compute_tau(
                budget
            )
        )

        for _ in range(10):

            if budget.exhausted:
                break

            budget.consume(
                1,
                action=ACTION_EVALUATE,
            )

        tau_late = (
            policy.compute_tau(
                budget
            )
        )

        assert (
            tau_late
            >= tau_early
        )

        # ---------------------------------------------------------------------
        # Stop when exhausted
        # ---------------------------------------------------------------------

        while (
            not budget.exhausted
        ):

            budget.consume(
                1
            )

        stop_decision = (
            policy.decide_action(
                archive,
                partition=partition,
                budget=budget,
            )
        )

        assert (
            stop_decision.action
            == ACTION_STOP
        )

        print(
            "policy.py self-test: PASS"
        )


if __name__ == "__main__":
    _self_test()
