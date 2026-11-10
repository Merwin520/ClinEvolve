#!/usr/bin/env python3

from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
)

from clinevolve.search.archive import (
    AgentArchive,
    ArchiveNode,
    canonical_partition,
    make_child_node,
    make_root_node,
)

from clinevolve.search.policy import (
    ACTION_EVALUATE,
    ACTION_EXPAND,
    ACTION_STOP,
    PolicyDecision,
    SearchBudget,
    SearchPolicy,
)

from clinevolve.search.utility import (
    CandidateEvidence,
    RankingConfig,
    UtilityEvaluator,
    UtilityResult,
    rank_candidates,
    select_best_candidate,
)


# =============================================================================
# Constants
# =============================================================================

CONTROLLER_SCHEMA_VERSION = "hyperagent_controller_v3"


# =============================================================================
# Callback result contracts
# =============================================================================

@dataclass
class ChildGenerationResult:
    """
    Generic result returned by generate_child_fn.

    The controller does not care whether generation is performed by:
        - MetaAgent
        - prompt mutation
        - code mutation
        - another optimizer
        - manual candidate construction
    """

    success: bool

    candidate_id: str

    node_id: Optional[str] = None

    agent_version: Optional[str] = None

    repo_path: Optional[str] = None

    patch_path: Optional[str] = None

    patch_hash: Optional[str] = None

    modified_files: List[str] = field(
        default_factory=list
    )

    lines_added: Optional[int] = None

    lines_deleted: Optional[int] = None

    generation_model: Optional[str] = None

    generation_temperature: Optional[float] = None

    generation_max_tokens: Optional[int] = None

    generation_tokens: Optional[int] = None

    generation_calls: Optional[int] = None

    generation_seconds: Optional[float] = None

    generation_error: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:

        if not str(
            self.candidate_id
        ).strip():
            raise ValueError(
                "candidate_id must be non-empty"
            )

        if (
            not self.success
            and not self.generation_error
        ):
            # Failure without error text is allowed, but still make it explicit.
            self.generation_error = (
                "candidate generation failed"
            )


@dataclass
class SearchEvaluationResult:
    """
    Generic result returned by evaluate_node_fn.

    `success`:
        Whether the evaluation operation itself completed successfully.

    `search_success`:
        Optional binary search-feedback outcome used by policy bookkeeping.

        1    = positive search outcome
        0    = negative search outcome
        None = no binary outcome defined

    IMPORTANT:
        search_success is NOT final experiment utility.

        Final candidate qualification/ranking is performed separately by
        UtilityEvaluator.
    """

    success: bool

    evaluation_id: Optional[str] = None

    search_success: Optional[int] = None

    metrics: Dict[str, Any] = field(
        default_factory=dict
    )

    resource_metrics: Dict[str, Any] = field(
        default_factory=dict
    )

    artifact_paths: Dict[str, str] = field(
        default_factory=dict
    )

    evaluation_error: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(
        self,
    ) -> None:

        if (
            self.search_success
            not in (
                None,
                0,
                1,
            )
        ):
            raise ValueError(
                "search_success must be 0, 1, or None"
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
            self.artifact_paths,
            dict,
        ):
            raise TypeError(
                "artifact_paths must be a dict"
            )


# =============================================================================
# Controller configuration
# =============================================================================

@dataclass(frozen=True)
class SearchControllerConfig:
    """
    Generic outer-loop controller configuration.

    Search-core does not define what one budget unit means.
    The experiment-specific runner must freeze that meaning.
    """

    controller_version: str

    search_budget_units: int

    expansion_cost_units: int = 1

    evaluation_cost_units: int = 1

    autosave_each_step: bool = True

    stop_when_no_action: bool = True

    def validate(
        self,
    ) -> None:

        if not str(
            self.controller_version
        ).strip():
            raise ValueError(
                "controller_version must be non-empty"
            )

        if (
            self.search_budget_units
            <= 0
        ):
            raise ValueError(
                "search_budget_units must be > 0"
            )

        if (
            self.expansion_cost_units
            <= 0
        ):
            raise ValueError(
                "expansion_cost_units must be > 0"
            )

        if (
            self.evaluation_cost_units
            <= 0
        ):
            raise ValueError(
                "evaluation_cost_units must be > 0"
            )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        self.validate()

        return {
            "schema_version":
                CONTROLLER_SCHEMA_VERSION,

            **asdict(
                self
            ),
        }


# =============================================================================
# Search event
# =============================================================================

@dataclass
class SearchEvent:
    """
    One auditable outer-loop event.
    """

    step: int

    action: str

    partition: Dict[str, Any]

    node_id: Optional[str]

    reason: str

    budget_used: int

    budget_remaining: int

    success: bool

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(
            self
        )


# =============================================================================
# Search Controller
# =============================================================================

class SearchController:
    """
    Generic HyperAgent outer-loop controller.

    Responsibilities
    ----------------
    1. Maintain one search partition.
    2. Maintain search budget.
    3. Ask SearchPolicy for EXPAND / EVALUATE / STOP.
    4. Invoke candidate-generation callback.
    5. Invoke search-evaluation callback.
    6. Update AgentArchive.
    7. Update node/clade search statistics.
    8. Persist event logs.
    9. Invoke experiment-specific UtilityEvaluator.
    10. Mark final selected candidate.

    It does NOT define
    ------------------
    - dataset
    - experiment
    - task
    - objective
    - evaluator semantics
    - utility semantics
    - candidate-generation mechanism
    - safety gates
    - statistical thresholds
    """

    def __init__(
        self,
        *,
        archive: AgentArchive,
        policy: SearchPolicy,
        controller_config: SearchControllerConfig,
        partition: Optional[
            Mapping[str, Any]
        ],
        generate_child_fn: Callable[
            [
                ArchiveNode,
                int,
                int,
            ],
            ChildGenerationResult,
        ],
        evaluate_node_fn: Callable[
            [
                ArchiveNode,
                int,
                int,
            ],
            SearchEvaluationResult,
        ],
        event_log_path: Optional[
            str | Path
        ] = None,
    ) -> None:

        controller_config.validate()

        self.archive = archive

        self.policy = policy

        self.controller_config = (
            controller_config
        )

        self.partition = (
            canonical_partition(
                partition
            )
        )

        self.generate_child_fn = (
            generate_child_fn
        )

        self.evaluate_node_fn = (
            evaluate_node_fn
        )

        self.budget = SearchBudget(
            total_budget=(
                controller_config.search_budget_units
            )
        )

        self.step = 0

        self.events: List[
            SearchEvent
        ] = []

        self.event_log_path = (
            Path(
                event_log_path
            )
            if event_log_path
            is not None
            else None
        )

        if (
            self.event_log_path
            is not None
        ):
            self.event_log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

    # =========================================================================
    # Event logging
    # =========================================================================

    def _append_event(
        self,
        event: SearchEvent,
    ) -> None:

        self.events.append(
            event
        )

        if (
            self.event_log_path
            is None
        ):
            return

        with self.event_log_path.open(
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    event.to_dict(),
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    # =========================================================================
    # Budget
    # =========================================================================

    def _action_cost(
        self,
        action: str,
    ) -> int:

        if (
            action
            == ACTION_EXPAND
        ):
            return (
                self.controller_config.expansion_cost_units
            )

        if (
            action
            == ACTION_EVALUATE
        ):
            return (
                self.controller_config.evaluation_cost_units
            )

        return 0

    def _can_afford(
        self,
        action: str,
    ) -> bool:

        return (
            self.budget.remaining_budget
            >= self._action_cost(
                action
            )
        )

    def _budget_stop_event(
        self,
        *,
        requested_action: str,
    ) -> SearchEvent:

        event = SearchEvent(
            step=(
                self.step
            ),

            action=(
                ACTION_STOP
            ),

            partition=(
                dict(
                    self.partition
                )
            ),

            node_id=None,

            reason=(
                "insufficient remaining budget "
                f"for requested action: {requested_action}"
            ),

            budget_used=(
                self.budget.used_budget
            ),

            budget_remaining=(
                self.budget.remaining_budget
            ),

            success=True,

            metadata={
                "requested_action":
                    requested_action,

                "required_cost":
                    self._action_cost(
                        requested_action
                    ),
            },
        )

        self._append_event(
            event
        )

        return event

    # =========================================================================
    # Root management
    # =========================================================================

    def ensure_root(
        self,
        *,
        root_node_id: str,
        candidate_id: str = "A0",
        repo_path: Optional[str] = None,
        base_commit: Optional[str] = None,
        agent_version: str = "A0",
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> ArchiveNode:
        """
        Ensure exactly one root exists in this partition.

        If root already exists, return it.

        If no root exists, create it.
        """

        roots = self.archive.roots(
            partition=(
                self.partition
            )
        )

        if (
            len(
                roots
            )
            > 1
        ):
            raise RuntimeError(
                "Multiple roots found in "
                "the same search partition"
            )

        if (
            len(
                roots
            )
            == 1
        ):
            return roots[
                0
            ]

        root = make_root_node(
            node_id=(
                root_node_id
            ),

            candidate_id=(
                candidate_id
            ),

            partition=(
                self.partition
            ),

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

        self.archive.add_node(
            root
        )

        return root

    # =========================================================================
    # Expansion
    # =========================================================================

    def _expand(
        self,
        decision: PolicyDecision,
    ) -> SearchEvent:

        if (
            not self._can_afford(
                ACTION_EXPAND
            )
        ):
            return self._budget_stop_event(
                requested_action=(
                    ACTION_EXPAND
                )
            )

        if (
            decision.node_id
            is None
        ):
            raise RuntimeError(
                "EXPAND decision missing node_id"
            )

        parent = self.archive.get(
            decision.node_id
        )

        if (
            parent.partition_key
            != make_partition_key(
                self.partition
            )
        ):
            raise RuntimeError(
                "Policy selected a parent outside "
                "the controller partition"
            )

        remaining_before = (
            self.budget.remaining_budget
        )

        try:

            result = (
                self.generate_child_fn(
                    parent,
                    self.step,
                    remaining_before,
                )
            )

            if not isinstance(
                result,
                ChildGenerationResult,
            ):
                raise TypeError(
                    "generate_child_fn must return "
                    "ChildGenerationResult"
                )

            result.validate()

        except Exception as exc:

            result = ChildGenerationResult(
                success=False,

                candidate_id=(
                    f"generation_error_step_{self.step}"
                ),

                generation_error=(
                    f"{type(exc).__name__}: {exc}"
                ),

                metadata={
                    "traceback":
                        traceback.format_exc()
                },
            )

        # Every attempted expansion consumes budget.
        self.budget.consume(
            self.controller_config.expansion_cost_units,
            action=(
                ACTION_EXPAND
            ),
        )

        # ---------------------------------------------------------------------
        # Failed generation
        # ---------------------------------------------------------------------

        if not result.success:

            parent.metadata.setdefault(
                "generation_failures",
                [],
            ).append(
                {
                    "step":
                        self.step,

                    "candidate_id":
                        result.candidate_id,

                    "generation_error":
                        result.generation_error,

                    "metadata":
                        result.metadata,
                }
            )

            if (
                self.controller_config.autosave_each_step
            ):
                self.archive.save()

            event = SearchEvent(
                step=(
                    self.step
                ),

                action=(
                    ACTION_EXPAND
                ),

                partition=(
                    dict(
                        self.partition
                    )
                ),

                node_id=(
                    parent.node_id
                ),

                reason=(
                    "candidate generation failed"
                ),

                budget_used=(
                    self.budget.used_budget
                ),

                budget_remaining=(
                    self.budget.remaining_budget
                ),

                success=False,

                metadata={
                    "policy":
                        decision.to_dict(),

                    "candidate_id":
                        result.candidate_id,

                    "generation_error":
                        result.generation_error,

                    **result.metadata,
                },
            )

            self._append_event(
                event
            )

            return event

        # ---------------------------------------------------------------------
        # Successful generation
        # ---------------------------------------------------------------------

        child_node_id = (
            result.node_id
            or (
                f"{parent.node_id}__"
                f"{result.candidate_id}"
            )
        )

        child = make_child_node(
            node_id=(
                child_node_id
            ),

            parent=(
                parent
            ),

            candidate_id=(
                result.candidate_id
            ),

            agent_version=(
                result.agent_version
            ),

            repo_path=(
                result.repo_path
            ),

            patch_path=(
                result.patch_path
            ),

            patch_hash=(
                result.patch_hash
            ),

            modified_files=(
                result.modified_files
            ),

            generation_model=(
                result.generation_model
            ),

            generation_temperature=(
                result.generation_temperature
            ),

            generation_max_tokens=(
                result.generation_max_tokens
            ),

            generation_tokens=(
                result.generation_tokens
            ),

            generation_calls=(
                result.generation_calls
            ),

            generation_seconds=(
                result.generation_seconds
            ),

            iterations_left_at_generation=(
                self.budget.remaining_budget
            ),

            metadata={
                **result.metadata,

                "controller_step":
                    self.step,

                "parent_node_id":
                    parent.node_id,
            },
        )

        child.lines_added = (
            result.lines_added
        )

        child.lines_deleted = (
            result.lines_deleted
        )

        self.archive.add_node(
            child
        )

        self.archive.set_status(
            child.node_id,
            "pending_evaluation",
        )

        self.archive.recompute_child_counts(
            save=False
        )

        if (
            self.controller_config.autosave_each_step
        ):
            self.archive.save()

        event = SearchEvent(
            step=(
                self.step
            ),

            action=(
                ACTION_EXPAND
            ),

            partition=(
                dict(
                    self.partition
                )
            ),

            node_id=(
                child.node_id
            ),

            reason=(
                "candidate generated successfully"
            ),

            budget_used=(
                self.budget.used_budget
            ),

            budget_remaining=(
                self.budget.remaining_budget
            ),

            success=True,

            metadata={
                "policy":
                    decision.to_dict(),

                "parent_node_id":
                    parent.node_id,

                "candidate_id":
                    child.candidate_id,

                "patch_path":
                    child.patch_path,

                "patch_hash":
                    child.patch_hash,

                "modified_files":
                    child.modified_files,
            },
        )

        self._append_event(
            event
        )

        return event

    # =========================================================================
    # Search evaluation
    # =========================================================================

    def _evaluate(
        self,
        decision: PolicyDecision,
    ) -> SearchEvent:

        if (
            not self._can_afford(
                ACTION_EVALUATE
            )
        ):
            return self._budget_stop_event(
                requested_action=(
                    ACTION_EVALUATE
                )
            )

        if (
            decision.node_id
            is None
        ):
            raise RuntimeError(
                "EVALUATE decision missing node_id"
            )

        node = self.archive.get(
            decision.node_id
        )

        if (
            node.partition_key
            != make_partition_key(
                self.partition
            )
        ):
            raise RuntimeError(
                "Policy selected an evaluation node "
                "outside the controller partition"
            )

        self.archive.set_status(
            node.node_id,
            "evaluating",
        )

        remaining_before = (
            self.budget.remaining_budget
        )

        try:

            result = (
                self.evaluate_node_fn(
                    node,
                    self.step,
                    remaining_before,
                )
            )

            if not isinstance(
                result,
                SearchEvaluationResult,
            ):
                raise TypeError(
                    "evaluate_node_fn must return "
                    "SearchEvaluationResult"
                )

            result.validate()

        except Exception as exc:

            result = SearchEvaluationResult(
                success=False,

                evaluation_id=None,

                search_success=None,

                evaluation_error=(
                    f"{type(exc).__name__}: {exc}"
                ),

                metadata={
                    "traceback":
                        traceback.format_exc()
                },
            )

        # Every attempted evaluation consumes budget.
        self.budget.consume(
            self.controller_config.evaluation_cost_units,
            action=(
                ACTION_EVALUATE
            ),
        )

        # ---------------------------------------------------------------------
        # Failed evaluation
        # ---------------------------------------------------------------------

        if not result.success:

            self.archive.set_status(
                node.node_id,
                "failed_evaluation",
            )

            node.metadata.setdefault(
                "evaluation_failures",
                [],
            ).append(
                {
                    "step":
                        self.step,

                    "evaluation_id":
                        result.evaluation_id,

                    "evaluation_error":
                        result.evaluation_error,

                    "metadata":
                        result.metadata,
                }
            )

            if (
                self.controller_config.autosave_each_step
            ):
                self.archive.save()

            event = SearchEvent(
                step=(
                    self.step
                ),

                action=(
                    ACTION_EVALUATE
                ),

                partition=(
                    dict(
                        self.partition
                    )
                ),

                node_id=(
                    node.node_id
                ),

                reason=(
                    "candidate search evaluation failed"
                ),

                budget_used=(
                    self.budget.used_budget
                ),

                budget_remaining=(
                    self.budget.remaining_budget
                ),

                success=False,

                metadata={
                    "policy":
                        decision.to_dict(),

                    "evaluation_id":
                        result.evaluation_id,

                    "evaluation_error":
                        result.evaluation_error,

                    **result.metadata,
                },
            )

            self._append_event(
                event
            )

            return event

        # ---------------------------------------------------------------------
        # Successful evaluation
        # ---------------------------------------------------------------------

        node.record_search_outcome(
            success=(
                result.search_success
            ),

            metrics=(
                result.metrics
            ),

            resource_metrics=(
                result.resource_metrics
            ),

            evaluation_id=(
                result.evaluation_id
            ),

            metadata={
                "artifact_paths":
                    result.artifact_paths,

                **result.metadata,
            },
        )

        self.archive.set_status(
            node.node_id,
            "evaluated",
        )

        # Recompute HGM-style tree statistics after every evaluation.
        self.archive.recompute_child_counts(
            save=False
        )

        self.archive.recompute_clade_statistics(
            save=False
        )

        if (
            self.controller_config.autosave_each_step
        ):
            self.archive.save()

        event = SearchEvent(
            step=(
                self.step
            ),

            action=(
                ACTION_EVALUATE
            ),

            partition=(
                dict(
                    self.partition
                )
            ),

            node_id=(
                node.node_id
            ),

            reason=(
                "candidate search evaluation completed"
            ),

            budget_used=(
                self.budget.used_budget
            ),

            budget_remaining=(
                self.budget.remaining_budget
            ),

            success=True,

            metadata={
                "policy":
                    decision.to_dict(),

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

                "n_search_evaluations":
                    node.n_search_evaluations,

                "node_successes":
                    node.node_successes,

                "node_failures":
                    node.node_failures,

                "clade_successes":
                    node.clade_successes,

                "clade_failures":
                    node.clade_failures,
            },
        )

        self._append_event(
            event
        )

        return event

    # =========================================================================
    # One step
    # =========================================================================

    def run_step(
        self,
    ) -> SearchEvent:

        decision = (
            self.policy.decide_action(
                self.archive,
                partition=(
                    self.partition
                ),
                budget=(
                    self.budget
                ),
            )
        )

        if (
            decision.action
            == ACTION_STOP
        ):

            event = SearchEvent(
                step=(
                    self.step
                ),

                action=(
                    ACTION_STOP
                ),

                partition=(
                    dict(
                        self.partition
                    )
                ),

                node_id=None,

                reason=(
                    decision.reason
                ),

                budget_used=(
                    self.budget.used_budget
                ),

                budget_remaining=(
                    self.budget.remaining_budget
                ),

                success=True,

                metadata={
                    "policy":
                        decision.to_dict()
                },
            )

            self._append_event(
                event
            )

            return event

        if (
            decision.action
            == ACTION_EXPAND
        ):
            return self._expand(
                decision
            )

        if (
            decision.action
            == ACTION_EVALUATE
        ):
            return self._evaluate(
                decision
            )

        raise RuntimeError(
            "Unhandled policy action: "
            f"{decision.action}"
        )

    # =========================================================================
    # Full search
    # =========================================================================

    def run(
        self,
        *,
        max_steps: Optional[int] = None,
    ) -> List[
        SearchEvent
    ]:
        """
        Run until:
            - search budget is exhausted
            - policy returns STOP
            - optional max_steps is reached

        max_steps is intended mainly for testing/debugging.
        """

        executed = 0

        while True:

            if (
                self.budget.exhausted
            ):
                break

            if (
                max_steps
                is not None
                and executed
                >= max_steps
            ):
                break

            self.step += 1

            event = (
                self.run_step()
            )

            executed += 1

            if (
                event.action
                == ACTION_STOP
                and self.controller_config.stop_when_no_action
            ):
                break

        self.archive.recompute_child_counts(
            save=False
        )

        self.archive.recompute_clade_statistics(
            save=False
        )

        self.archive.save()

        return list(
            self.events
        )

    # =========================================================================
    # Generic experiment utility
    # =========================================================================

    def evaluate_candidates(
        self,
        *,
        evaluator: UtilityEvaluator,
        candidate_evidence: Mapping[
            str,
            CandidateEvidence,
        ],
        baseline_evidence: Optional[
            CandidateEvidence
        ] = None,
        context: Optional[
            Mapping[str, Any]
        ] = None,
        ranking_config: Optional[
            RankingConfig
        ] = None,
        mark_selected: bool = True,
    ) -> List[
        UtilityResult
    ]:
        """
        Apply an experiment-specific UtilityEvaluator to candidates.

        candidate_evidence mapping:
            archive_node_id -> CandidateEvidence

        The controller never interprets metrics or utility semantics.
        """

        if (
            ranking_config
            is None
        ):
            ranking_config = (
                RankingConfig()
            )

        results: List[
            UtilityResult
        ] = []

        candidate_to_node: Dict[
            str,
            str,
        ] = {}

        for (
            node_id,
            evidence,
        ) in (
            candidate_evidence.items()
        ):

            node = self.archive.get(
                node_id
            )

            if (
                node.partition_key
                != make_partition_key(
                    self.partition
                )
            ):
                raise ValueError(
                    f"{node_id}: node does not "
                    "belong to controller partition"
                )

            evidence.validate()

            if (
                evidence.candidate_id
                in candidate_to_node
            ):
                raise ValueError(
                    "Duplicate candidate_id across "
                    "candidate evidence: "
                    f"{evidence.candidate_id}"
                )

            result = evaluator.evaluate(
                baseline_evidence,
                evidence,
                context=context,
            )

            result.validate()

            if (
                result.candidate_id
                != evidence.candidate_id
            ):
                raise ValueError(
                    "UtilityEvaluator returned a "
                    "different candidate_id"
                )

            candidate_to_node[
                result.candidate_id
            ] = node_id

            node.set_utility_result(
                result.to_dict()
            )

            if (
                result.eligible
            ):

                node.rejection_reason = None

                if (
                    node.status
                    not in {
                        "selected",
                        "root",
                    }
                ):
                    node.status = (
                        "evaluated"
                    )

            else:

                node.status = (
                    "rejected"
                )

                node.rejection_reason = (
                    "; ".join(
                        result.rejection_reasons
                    )
                    if result.rejection_reasons
                    else
                    "utility evaluator rejected candidate"
                )

            results.append(
                result
            )

        ranked = rank_candidates(
            results,
            config=(
                ranking_config
            ),
        )

        # ---------------------------------------------------------------------
        # Optional final selection
        # ---------------------------------------------------------------------

        if (
            mark_selected
        ):

            best = (
                select_best_candidate(
                    ranked,
                    config=(
                        ranking_config
                    ),
                )
            )

            if (
                best
                is not None
            ):

                selected_node_id = (
                    candidate_to_node.get(
                        best.candidate_id
                    )
                )

                if (
                    selected_node_id
                    is None
                ):
                    raise RuntimeError(
                        "Could not map selected "
                        "candidate back to archive node"
                    )

                self.archive.mark_selected_final(
                    selected_node_id
                )

        self.archive.save()

        return ranked

    # =========================================================================
    # Summary
    # =========================================================================

    def summary(
        self,
    ) -> Dict[str, Any]:

        selected = (
            self.archive.selected_final(
                partition=(
                    self.partition
                )
            )
        )

        partition_nodes = (
            self.archive.partition_nodes(
                self.partition
            )
        )

        return {
            "schema_version":
                CONTROLLER_SCHEMA_VERSION,

            "partition":
                dict(
                    self.partition
                ),

            "step":
                self.step,

            "budget": {
                "total":
                    self.budget.total_budget,

                "used":
                    self.budget.used_budget,

                "remaining":
                    self.budget.remaining_budget,

                "completed_expansions":
                    self.budget.completed_expansions,

                "completed_evaluations":
                    self.budget.completed_evaluations,
            },

            "partition_n_nodes":
                len(
                    partition_nodes
                ),

            "selected_final_node_id":
                (
                    selected.node_id
                    if selected
                    is not None
                    else None
                ),

            "n_events":
                len(
                    self.events
                ),
        }


# =============================================================================
# Small helper
# =============================================================================

def make_partition_key(
    partition: Mapping[
        str,
        Any
    ],
) -> str:
    """
    Local helper used only to avoid duplicating partition-comparison logic.

    ArchiveNode.partition_key uses the same canonical representation.
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


# =============================================================================
# Minimal self-test
# =============================================================================

def _self_test() -> None:

    import tempfile

    from clinevolve.search.policy import (
        SearchPolicyConfig,
    )

    from clinevolve.search.utility import (
        UtilityEvaluator,
    )

    # -------------------------------------------------------------------------
    # Example utility evaluator
    # -------------------------------------------------------------------------

    class ExampleEvaluator(
        UtilityEvaluator
    ):

        def evaluate(
            self,
            baseline,
            candidate,
            *,
            context=None,
        ) -> UtilityResult:

            candidate_score = float(
                candidate.metrics[
                    "score"
                ]
            )

            baseline_score = (
                float(
                    baseline.metrics[
                        "score"
                    ]
                )
                if baseline
                is not None
                else None
            )

            eligible = (
                baseline_score
                is None
                or candidate_score
                >= baseline_score
            )

            return UtilityResult(
                candidate_id=(
                    candidate.candidate_id
                ),

                eligible=(
                    eligible
                ),

                primary_score=(
                    candidate_score
                ),

                secondary_scores={
                    "robustness":
                        float(
                            candidate.metrics.get(
                                "robustness",
                                0.0,
                            )
                        )
                },

                secondary_order=[
                    "robustness"
                ],

                complexity=(
                    float(
                        candidate.metrics.get(
                            "patch_size",
                            0.0,
                        )
                    )
                ),

                resource_cost=(
                    float(
                        candidate.metrics.get(
                            "resource_cost",
                            0.0,
                        )
                    )
                ),

                rejection_reasons=(
                    []
                    if eligible
                    else [
                        "below baseline"
                    ]
                ),
            )

    # -------------------------------------------------------------------------
    # Temporary search
    # -------------------------------------------------------------------------

    with tempfile.TemporaryDirectory() as tempdir:

        root_dir = Path(
            tempdir
        )

        archive = AgentArchive(
            root_dir
            / "archive.json"
        )

        partition = {
            "experiment":
                "self_test",

            "track":
                "track_a",
        }

        policy = SearchPolicy(
            SearchPolicyConfig(
                policy_version=(
                    "controller_self_test_policy_v3"
                ),

                alpha=0.6,

                initial_root_expansions=2,

                max_children_per_node=3,

                max_evaluations_per_node=3,

                random_seed=123,
            )
        )

        controller_config = (
            SearchControllerConfig(
                controller_version=(
                    "controller_self_test_v3"
                ),

                search_budget_units=8,
            )
        )

        counter = {
            "candidate":
                0,

            "evaluation":
                0,
        }

        # ---------------------------------------------------------------------
        # Fake generator
        # ---------------------------------------------------------------------

        def fake_generate_child(
            parent: ArchiveNode,
            step: int,
            remaining_budget: int,
        ) -> ChildGenerationResult:

            counter[
                "candidate"
            ] += 1

            candidate_id = (
                f"c{counter['candidate']:03d}"
            )

            return ChildGenerationResult(
                success=True,

                candidate_id=(
                    candidate_id
                ),

                node_id=(
                    f"track_a_{candidate_id}"
                ),

                agent_version=(
                    candidate_id
                ),

                repo_path=(
                    f"/tmp/{candidate_id}"
                ),

                patch_hash=(
                    f"hash_{candidate_id}"
                ),

                modified_files=[
                    "task_agent.py"
                ],

                lines_added=5,

                lines_deleted=1,

                generation_model=(
                    "fake-generator"
                ),

                generation_tokens=100,
            )

        # ---------------------------------------------------------------------
        # Fake search evaluator
        # ---------------------------------------------------------------------

        def fake_evaluate_node(
            node: ArchiveNode,
            step: int,
            remaining_budget: int,
        ) -> SearchEvaluationResult:

            counter[
                "evaluation"
            ] += 1

            outcome = (
                1
                if (
                    counter[
                        "evaluation"
                    ]
                    % 3
                    != 0
                )
                else 0
            )

            return SearchEvaluationResult(
                success=True,

                evaluation_id=(
                    f"eval_"
                    f"{counter['evaluation']:03d}"
                ),

                search_success=(
                    outcome
                ),

                metrics={
                    "raw_score":
                        float(
                            outcome
                        )
                },

                resource_metrics={
                    "tokens":
                        1000
                },
            )

        # ---------------------------------------------------------------------
        # Controller
        # ---------------------------------------------------------------------

        controller = SearchController(
            archive=(
                archive
            ),

            policy=(
                policy
            ),

            controller_config=(
                controller_config
            ),

            partition=(
                partition
            ),

            generate_child_fn=(
                fake_generate_child
            ),

            evaluate_node_fn=(
                fake_evaluate_node
            ),

            event_log_path=(
                root_dir
                / "events.jsonl"
            ),
        )

        root = controller.ensure_root(
            root_node_id=(
                "track_a_A0"
            ),

            candidate_id=(
                "A0"
            ),

            repo_path=(
                "/tmp/A0"
            ),

            base_commit=(
                "deadbeef"
            ),
        )

        assert (
            root.partition
            == partition
        )

        events = controller.run()

        assert (
            len(
                events
            )
            > 0
        )

        assert (
            len(
                archive.partition_nodes(
                    partition
                )
            )
            >= 2
        )

        assert (
            controller.budget.used_budget
            <= controller.budget.total_budget
        )

        archive.validate_archive()

        # ---------------------------------------------------------------------
        # Generic final utility evaluation
        # ---------------------------------------------------------------------

        candidate_nodes = [
            node
            for node
            in archive.partition_nodes(
                partition
            )
            if (
                node.generation
                > 0
            )
        ]

        baseline_evidence = (
            CandidateEvidence(
                candidate_id="A0",

                metrics={
                    "score":
                        0.50
                },
            )
        )

        evidence_map: Dict[
            str,
            CandidateEvidence,
        ] = {}

        for (
            index,
            node,
        ) in enumerate(
            candidate_nodes,
            start=1,
        ):

            evidence_map[
                node.node_id
            ] = CandidateEvidence(
                candidate_id=(
                    node.candidate_id
                ),

                metrics={
                    "score":
                        0.50
                        + 0.05
                        * index,

                    "robustness":
                        0.80
                        + 0.01
                        * index,

                    "patch_size":
                        float(
                            index
                        ),

                    "resource_cost":
                        1.0
                        + 0.1
                        * index,
                },
            )

        if (
            evidence_map
        ):

            ranked = (
                controller.evaluate_candidates(
                    evaluator=(
                        ExampleEvaluator()
                    ),

                    baseline_evidence=(
                        baseline_evidence
                    ),

                    candidate_evidence=(
                        evidence_map
                    ),

                    ranking_config=(
                        RankingConfig(
                            secondary_order=(
                                "robustness",
                            )
                        )
                    ),

                    mark_selected=True,
                )
            )

            assert (
                len(
                    ranked
                )
                == len(
                    evidence_map
                )
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

        archive.validate_archive()

        print(
            json.dumps(
                controller.summary(),
                indent=2,
                ensure_ascii=False,
            )
        )

        print(
            "controller.py self-test: PASS"
        )


if __name__ == "__main__":
    _self_test()
