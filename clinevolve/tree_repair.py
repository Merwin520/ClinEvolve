#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from meta_agent import MetaAgent

from agent.paths import mimic_cdm_dir

from clinevolve.search.archive import (
    AgentArchive,
    make_root_node,
)

from clinevolve.search.policy import (
    SearchPolicy,
    SearchPolicyConfig,
)

from clinevolve.search.controller import (
    SearchController,
    SearchControllerConfig,
    SearchEvaluationResult,
)

from clinevolve.registry import (
    CandidateRegistry,
    make_a0_record,
)

from clinevolve.evaluate import (
    evaluate_candidate,
)

from clinevolve.feedback import (
    build_from_files as build_feedback_from_files,
)

from clinevolve.generate import (
    generate_candidate,
)

from clinevolve.search_feedback import (
    compute_from_files as compute_search_feedback_from_files,
)


# =============================================================================
# Constants
# =============================================================================

RUNNER_VERSION = "exp2_run_repair_search_tree_v5_meta_sampling"

VALID_TARGETS = {
    "diagnostic_process_failure",
    "treatment_guideline_failure",
}

REQUIRED_PARENT_RULE = "tree_search"


# =============================================================================
# Basic helpers
# =============================================================================

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


def load_run_manifest(
    path: str | Path,
) -> Dict[str, Any]:

    manifest = load_json(
        path
    )

    if (
        manifest.get(
            "selection_status"
        )
        != "frozen"
    ):
        raise RuntimeError(
            "Exp2 run manifest is not frozen"
        )

    parent_rule = (
        manifest.get(
            "search",
            {}
        ).get(
            "candidate_parent_rule"
        )
    )

    if (
        parent_rule
        != REQUIRED_PARENT_RULE
    ):
        raise RuntimeError(
            "Formal Exp2 repair search requires "
            "candidate_parent_rule='tree_search'. "
            f"Frozen manifest contains: {parent_rule!r}"
        )

    return manifest


def get_candidate_budget(
    manifest: Mapping[str, Any],
) -> int:

    budget = int(
        manifest[
            "search"
        ][
            "candidate_budget"
        ]
    )

    if budget <= 0:
        raise ValueError(
            "candidate_budget must be > 0"
        )

    return budget


def get_a0_model(
    manifest: Mapping[str, Any],
) -> str:

    model = (
        manifest.get(
            "a0",
            {}
        ).get(
            "model"
        )
    )

    if not model:
        raise ValueError(
            "Frozen run manifest missing A0 model"
        )

    return str(model)


def get_policy_config(
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:

    payload = (
        manifest.get(
            "search",
            {}
        )
        .get(
            "search_policy_config",
            {}
        )
        .get(
            "content"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Frozen run manifest missing "
            "SearchPolicyConfig content"
        )

    return dict(payload)


def get_controller_config(
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:

    payload = (
        manifest.get(
            "search",
            {}
        )
        .get(
            "controller_config",
            {}
        )
        .get(
            "content"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Frozen run manifest missing "
            "SearchControllerConfig content"
        )

    return dict(payload)


def make_partition(
    *,
    outer_fold: int,
    target_id: str,
) -> Dict[str, Any]:

    return {
        "experiment":
            "exp2",

        "outer_fold":
            int(outer_fold),

        "repair_target":
            str(target_id),
    }


def target_short_name(
    target_id: str,
) -> str:

    if (
        target_id
        == "diagnostic_process_failure"
    ):
        return "pd"

    if (
        target_id
        == "treatment_guideline_failure"
    ):
        return "pt"

    raise ValueError(
        f"Unsupported target_id: {target_id}"
    )


def candidate_id_for_index(
    *,
    outer_fold: int,
    target_id: str,
    index: int,
) -> str:

    return (
        f"f{outer_fold}_"
        f"{target_short_name(target_id)}_"
        f"c{index:04d}"
    )


def node_id_for_candidate(
    *,
    outer_fold: int,
    target_id: str,
    candidate_id: str,
) -> str:

    return (
        f"exp2_f{outer_fold}_"
        f"{target_short_name(target_id)}_"
        f"{candidate_id}"
    )


# =============================================================================
# Artifact paths
# =============================================================================

def candidate_evaluation_root(
    *,
    evaluation_root: Path,
    outer_fold: int,
    target_id: str,
    candidate_id: str,
) -> Path:

    return (
        evaluation_root
        / f"fold{outer_fold}"
        / target_id
        / "repair"
        / candidate_id
    )


def aggregate_path_for(
    *,
    evaluation_root: Path,
    outer_fold: int,
    target_id: str,
    candidate_id: str,
) -> Path:

    return (
        candidate_evaluation_root(
            evaluation_root=(
                evaluation_root
            ),
            outer_fold=(
                outer_fold
            ),
            target_id=(
                target_id
            ),
            candidate_id=(
                candidate_id
            ),
        )
        / "aggregate_evidence.json"
    )


def evaluation_table_path_for(
    *,
    evaluation_root: Path,
    outer_fold: int,
    target_id: str,
    candidate_id: str,
) -> Path:

    return (
        candidate_evaluation_root(
            evaluation_root=(
                evaluation_root
            ),
            outer_fold=(
                outer_fold
            ),
            target_id=(
                target_id
            ),
            candidate_id=(
                candidate_id
            ),
        )
        / "evaluations.parquet"
    )


def feedback_path_for(
    *,
    feedback_root: Path,
    outer_fold: int,
    target_id: str,
    candidate_id: str,
) -> Path:

    return (
        feedback_root
        / f"fold{outer_fold}"
        / target_id
        / f"{candidate_id}_feedback.json"
    )


def search_feedback_path_for(
    *,
    search_feedback_root: Path,
    outer_fold: int,
    target_id: str,
    candidate_id: str,
) -> Path:

    return (
        search_feedback_root
        / f"fold{outer_fold}"
        / target_id
        / f"{candidate_id}.json"
    )


# =============================================================================
# A0 setup
# =============================================================================

def ensure_a0_registry_record(
    *,
    registry: CandidateRegistry,
    outer_fold: int,
    target_id: str,
    a0_repo: Path,
    a0_model: str,
):

    try:

        return registry.get_a0(
            outer_fold=(
                outer_fold
            ),
            target_id=(
                target_id
            ),
        )

    except RuntimeError:
        pass

    node_id = (
        node_id_for_candidate(
            outer_fold=(
                outer_fold
            ),
            target_id=(
                target_id
            ),
            candidate_id="A0",
        )
    )

    record = make_a0_record(
        outer_fold=(
            outer_fold
        ),

        target_id=(
            target_id
        ),

        node_id=(
            node_id
        ),

        repo_path=(
            a0_repo
        ),

        base_commit=None,

        agent_model=(
            a0_model
        ),

        metadata={
            "runner_version":
                RUNNER_VERSION,

            "role":
                "root",
        },
    )

    registry.add(
        record
    )

    return record


def ensure_archive_root(
    *,
    archive: AgentArchive,
    a0_record,
) -> None:

    try:

        archive.get(
            a0_record.node_id
        )

        return

    except Exception:
        pass

    root = make_root_node(
        node_id=(
            a0_record.node_id
        ),

        candidate_id="A0",

        partition=dict(
            a0_record.partition
        ),

        repo_path=(
            a0_record.repo_path
        ),

        agent_version="A0",
    )

    archive.add_node(
        root
    )


# =============================================================================
# Repair evaluation
# =============================================================================

def ensure_repair_evaluation(
    *,
    registry: CandidateRegistry,
    candidate_id: str,
    candidate_repo: Path,
    model: str,
    target_id: str,
    outer_fold: int,
    fold_manifest_path: Path,
    evaluation_root: Path,
    hpi_path: Path,
) -> Path:

    aggregate_path = (
        aggregate_path_for(
            evaluation_root=(
                evaluation_root
            ),

            outer_fold=(
                outer_fold
            ),

            target_id=(
                target_id
            ),

            candidate_id=(
                candidate_id
            ),
        )
    )

    evaluation_path = (
        evaluation_table_path_for(
            evaluation_root=(
                evaluation_root
            ),

            outer_fold=(
                outer_fold
            ),

            target_id=(
                target_id
            ),

            candidate_id=(
                candidate_id
            ),
        )
    )

    if (
        aggregate_path.exists()
        and evaluation_path.exists()
    ):

        return aggregate_path

    result = evaluate_candidate(
        candidate_repo=(
            candidate_repo
        ),

        candidate_id=(
            candidate_id
        ),

        model=(
            model
        ),

        target_id=(
            target_id
        ),

        outer_fold=(
            outer_fold
        ),

        role="repair",

        fold_manifest_path=(
            fold_manifest_path
        ),

        output_root=(
            evaluation_root
        ),

        hpi_path=(
            hpi_path
        ),

        temperature=0.0,

        max_tokens=4096,

        max_tool_calls=40,

        search_success=None,
    )

    if not result.success:
        raise RuntimeError(
            "Repair evaluation failed for "
            f"{candidate_id}"
        )

    if not aggregate_path.exists():
        raise RuntimeError(
            "aggregate_evidence.json not produced for "
            f"{candidate_id}"
        )

    return aggregate_path


# =============================================================================
# Repair feedback
# =============================================================================

def ensure_feedback(
    *,
    candidate_id: str,
    target_id: str,
    outer_fold: int,
    fold_manifest_path: Path,
    evaluation_root: Path,
    feedback_root: Path,
) -> Path:

    evaluation_path = (
        evaluation_table_path_for(
            evaluation_root=(
                evaluation_root
            ),

            outer_fold=(
                outer_fold
            ),

            target_id=(
                target_id
            ),

            candidate_id=(
                candidate_id
            ),
        )
    )

    if not evaluation_path.exists():
        raise FileNotFoundError(
            "Cannot build repair feedback because "
            "evaluation file is missing:\n"
            f"{evaluation_path}"
        )

    output_path = (
        feedback_path_for(
            feedback_root=(
                feedback_root
            ),

            outer_fold=(
                outer_fold
            ),

            target_id=(
                target_id
            ),

            candidate_id=(
                candidate_id
            ),
        )
    )

    if output_path.exists():
        return output_path

    build_feedback_from_files(
        fold_manifest_path=(
            fold_manifest_path
        ),

        evaluation_path=(
            evaluation_path
        ),

        output_path=(
            output_path
        ),

        outer_fold=(
            outer_fold
        ),

        target_id=(
            target_id
        ),

        agent_id=(
            candidate_id
        ),

        max_examples=20,
    )

    return output_path


# =============================================================================
# Search feedback
# =============================================================================

def evaluate_child_against_parent(
    *,
    registry: CandidateRegistry,
    candidate_id: str,
    model: str,
    target_id: str,
    outer_fold: int,
    fold_manifest_path: Path,
    evaluation_root: Path,
    search_feedback_root: Path,
    hpi_path: Path,
) -> SearchEvaluationResult:
    """
    Evaluate child on repair set and compare it with its ACTUAL parent.

    This is the core tree-search semantics:

        child vs parent
            -> search_success

    not:
        child vs A0 always
    """

    candidate = registry.get(
        candidate_id
    )

    if (
        candidate.parent_candidate_id
        is None
    ):
        raise RuntimeError(
            "Generated child has no parent_candidate_id"
        )

    parent_id = (
        candidate.parent_candidate_id
    )

    parent = registry.get(
        parent_id
    )

    if (
        candidate.repo_path
        is None
    ):
        raise RuntimeError(
            f"{candidate_id} has no candidate workspace"
        )

    if (
        parent.repo_path
        is None
    ):
        raise RuntimeError(
            f"{parent_id} has no parent workspace"
        )

    # -------------------------------------------------------------------------
    # Parent must already have repair evidence.
    # -------------------------------------------------------------------------

    parent_aggregate = (
        ensure_repair_evaluation(
            registry=(
                registry
            ),

            candidate_id=(
                parent_id
            ),

            candidate_repo=(
                Path(
                    parent.repo_path
                )
            ),

            model=(
                model
            ),

            target_id=(
                target_id
            ),

            outer_fold=(
                outer_fold
            ),

            fold_manifest_path=(
                fold_manifest_path
            ),

            evaluation_root=(
                evaluation_root
            ),

            hpi_path=(
                hpi_path
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Evaluate child.
    # -------------------------------------------------------------------------

    candidate_eval = evaluate_candidate(
        candidate_repo=(
            candidate.repo_path
        ),

        candidate_id=(
            candidate_id
        ),

        model=(
            model
        ),

        target_id=(
            target_id
        ),

        outer_fold=(
            outer_fold
        ),

        role="repair",

        fold_manifest_path=(
            fold_manifest_path
        ),

        output_root=(
            evaluation_root
        ),

        hpi_path=(
            hpi_path
        ),

        temperature=0.0,

        max_tokens=4096,

        max_tool_calls=40,

        search_success=None,
    )

    if not candidate_eval.success:
        return candidate_eval

    candidate_aggregate = (
        aggregate_path_for(
            evaluation_root=(
                evaluation_root
            ),

            outer_fold=(
                outer_fold
            ),

            target_id=(
                target_id
            ),

            candidate_id=(
                candidate_id
            ),
        )
    )

    if not candidate_aggregate.exists():
        raise RuntimeError(
            "Candidate aggregate evidence missing: "
            f"{candidate_aggregate}"
        )

    # -------------------------------------------------------------------------
    # Child vs actual parent.
    # -------------------------------------------------------------------------

    feedback_output = (
        search_feedback_path_for(
            search_feedback_root=(
                search_feedback_root
            ),

            outer_fold=(
                outer_fold
            ),

            target_id=(
                target_id
            ),

            candidate_id=(
                candidate_id
            ),
        )
    )

    search_feedback = (
        compute_search_feedback_from_files(
            parent_aggregate_path=(
                parent_aggregate
            ),

            candidate_aggregate_path=(
                candidate_aggregate
            ),

            target_id=(
                target_id
            ),

            output_path=(
                feedback_output
            ),
        )
    )

    evaluation_path = (
        evaluation_table_path_for(
            evaluation_root=(
                evaluation_root
            ),

            outer_fold=(
                outer_fold
            ),

            target_id=(
                target_id
            ),

            candidate_id=(
                candidate_id
            ),
        )
    )

    registry.attach_repair_evaluation(
        candidate_id,

        evaluation_path=(
            evaluation_path
        ),

        search_success=(
            search_feedback.search_success
        ),
    )

    return SearchEvaluationResult(
        success=True,

        evaluation_id=(
            candidate_eval.evaluation_id
        ),

        search_success=(
            search_feedback.search_success
        ),

        metrics={
            **dict(
                candidate_eval.metrics
            ),

            "target_improvement_vs_parent":
                search_feedback.absolute_improvement,

            "parent_target_failure_rate":
                search_feedback.parent_target_failure_rate,

            "candidate_target_failure_rate":
                search_feedback.candidate_target_failure_rate,
        },

        resource_metrics=dict(
            candidate_eval.resource_metrics
        ),

        artifact_paths={
            **dict(
                getattr(
                    candidate_eval,
                    "artifact_paths",
                    {},
                )
                or {}
            ),

            "search_feedback":
                str(
                    feedback_output.resolve()
                ),
        },

        metadata={
            **dict(
                getattr(
                    candidate_eval,
                    "metadata",
                    {},
                )
                or {}
            ),

            "parent_candidate_id":
                parent_id,

            "parent_node_id":
                candidate.parent_node_id,

            "search_feedback_rule":
                search_feedback.rule_version,
        },
    )


# =============================================================================
# Search callbacks
# =============================================================================

class Exp2RepairSearchCallbacks:
    """
    Stateful callbacks connecting generic SearchController to Exp2.

    SearchController owns:
        - selection / expansion policy
        - archive tree
        - node/clade bookkeeping

    This class owns:
        - candidate IDs
        - repair feedback
        - MetaAgent generation
        - repair evaluation
        - parent-relative binary feedback
    """

    def __init__(
        self,
        *,
        registry: CandidateRegistry,
        run_manifest_path: Path,
        meta_agent: MetaAgent,
        model: str,
        target_id: str,
        outer_fold: int,
        candidate_budget: int,
        fold_manifest_path: Path,
        evaluation_root: Path,
        feedback_root: Path,
        search_feedback_root: Path,
        workspace_root: Path,
        patch_root: Path,
        hpi_path: Path,
    ) -> None:

        self.registry = (
            registry
        )

        self.run_manifest_path = (
            run_manifest_path
        )

        self.meta_agent = (
            meta_agent
        )

        self.model = (
            model
        )

        self.target_id = (
            target_id
        )

        self.outer_fold = (
            outer_fold
        )

        self.candidate_budget = (
            candidate_budget
        )

        self.fold_manifest_path = (
            fold_manifest_path
        )

        self.evaluation_root = (
            evaluation_root
        )

        self.feedback_root = (
            feedback_root
        )

        self.search_feedback_root = (
            search_feedback_root
        )

        self.workspace_root = (
            workspace_root
        )

        self.patch_root = (
            patch_root
        )

        self.hpi_path = (
            hpi_path
        )

        existing = (
            registry.filter(
                outer_fold=(
                    outer_fold
                ),

                target_id=(
                    target_id
                ),

                is_a0=False,
            )
        )

        self.next_candidate_index = (
            len(
                existing
            )
            + 1
        )

        # Bound later by the runner after SearchPolicy is created.
        # Exp2 uses this only to disable further EXPAND actions when
        # the experiment-level candidate generation budget is exhausted.
        self.search_policy = None

    # -------------------------------------------------------------------------
    # EXPAND
    # -------------------------------------------------------------------------

    def generation_budget_exhausted(
        self,
    ) -> bool:

        return (
            self.next_candidate_index
            > self.candidate_budget
        )

    def disable_further_expansion(
        self,
    ) -> None:

        if self.search_policy is None:
            return

        from dataclasses import replace

        self.search_policy.config = replace(
            self.search_policy.config,
            expandable_statuses=tuple(),
        )

    def generate_child(
        self,
        parent_archive_node,
        *args,
        **kwargs,
    ):

        if (
            self.next_candidate_index
            > self.candidate_budget
        ):
            raise RuntimeError(
                "Candidate generation budget exhausted"
            )

        parent_candidate_id = (
            parent_archive_node.candidate_id
        )

        parent = self.registry.get(
            parent_candidate_id
        )

        # ---------------------------------------------------------------------
        # The chosen HGM parent must already have repair evidence.
        # ---------------------------------------------------------------------

        ensure_repair_evaluation(
            registry=(
                self.registry
            ),

            candidate_id=(
                parent_candidate_id
            ),

            candidate_repo=(
                Path(
                    parent.repo_path
                )
            ),

            model=(
                self.model
            ),

            target_id=(
                self.target_id
            ),

            outer_fold=(
                self.outer_fold
            ),

            fold_manifest_path=(
                self.fold_manifest_path
            ),

            evaluation_root=(
                self.evaluation_root
            ),

            hpi_path=(
                self.hpi_path
            ),
        )

        # ---------------------------------------------------------------------
        # MetaAgent gets feedback from ACTUAL selected parent.
        # ---------------------------------------------------------------------

        parent_feedback_path = (
            ensure_feedback(
                candidate_id=(
                    parent_candidate_id
                ),

                target_id=(
                    self.target_id
                ),

                outer_fold=(
                    self.outer_fold
                ),

                fold_manifest_path=(
                    self.fold_manifest_path
                ),

                evaluation_root=(
                    self.evaluation_root
                ),

                feedback_root=(
                    self.feedback_root
                ),
            )
        )

        candidate_id = (
            candidate_id_for_index(
                outer_fold=(
                    self.outer_fold
                ),

                target_id=(
                    self.target_id
                ),

                index=(
                    self.next_candidate_index
                ),
            )
        )

        node_id = (
            node_id_for_candidate(
                outer_fold=(
                    self.outer_fold
                ),

                target_id=(
                    self.target_id
                ),

                candidate_id=(
                    candidate_id
                ),
            )
        )

        current_index = (
            self.next_candidate_index
        )

        self.next_candidate_index += 1

        if self.generation_budget_exhausted():
            self.disable_further_expansion()

        return generate_candidate(
            registry=(
                self.registry
            ),

            run_manifest_path=(
                self.run_manifest_path
            ),

            parent_candidate_id=(
                parent_candidate_id
            ),

            candidate_id=(
                candidate_id
            ),

            node_id=(
                node_id
            ),

            feedback_path=(
                parent_feedback_path
            ),

            workspace_root=(
                self.workspace_root
            ),

            patch_root=(
                self.patch_root
            ),

            meta_agent=(
                self.meta_agent
            ),

            iterations_left=(
                self.candidate_budget
                - current_index
            ),

            extra_context={
                "experiment":
                    "exp2",

                "outer_fold":
                    self.outer_fold,

                "repair_target":
                    self.target_id,

                "search_mode":
                    "tree_search",

                "selected_parent_candidate_id":
                    parent_candidate_id,

                "selected_parent_node_id":
                    parent_archive_node.node_id,

                "candidate_index":
                    current_index,

                "candidate_budget":
                    self.candidate_budget,
            },
        )

    # -------------------------------------------------------------------------
    # EVALUATE
    # -------------------------------------------------------------------------

    def evaluate_node(
        self,
        archive_node,
        *args,
        **kwargs,
    ) -> SearchEvaluationResult:

        candidate_id = (
            archive_node.candidate_id
        )

        if (
            candidate_id
            == "A0"
        ):

            # A0 is baseline/root, not a parent-relative mutation.
            return SearchEvaluationResult(
                success=True,

                evaluation_id=(
                    f"exp2_f{self.outer_fold}_"
                    f"{self.target_id}_"
                    "repair_A0"
                ),

                search_success=None,

                metrics={},

                resource_metrics={},

                metadata={
                    "candidate_id":
                        "A0",

                    "role":
                        "root_baseline",
                },
            )

        return evaluate_child_against_parent(
            registry=(
                self.registry
            ),

            candidate_id=(
                candidate_id
            ),

            model=(
                self.model
            ),

            target_id=(
                self.target_id
            ),

            outer_fold=(
                self.outer_fold
            ),

            fold_manifest_path=(
                self.fold_manifest_path
            ),

            evaluation_root=(
                self.evaluation_root
            ),

            search_feedback_root=(
                self.search_feedback_root
            ),

            hpi_path=(
                self.hpi_path
            ),
        )


# =============================================================================
# Main formal search
# =============================================================================

def run_repair_search(
    *,
    outer_fold: int,
    target_id: str,
    run_manifest_path: Path,
    a0_repo: Path,
    meta_model: str,
    fold_manifest_path: Path,
    hpi_path: Path,
    registry_path: Path,
    archive_path: Path,
    evaluation_root: Path,
    feedback_root: Path,
    search_feedback_root: Path,
    workspace_root: Path,
    patch_root: Path,
    meta_chat_history: Path,
) -> Dict[str, Any]:

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

    run_manifest_path = (
        run_manifest_path.resolve()
    )

    a0_repo = (
        a0_repo.resolve()
    )

    fold_manifest_path = (
        fold_manifest_path.resolve()
    )

    hpi_path = (
        hpi_path.resolve()
    )

    manifest = (
        load_run_manifest(
            run_manifest_path
        )
    )

    candidate_budget = (
        get_candidate_budget(
            manifest
        )
    )

    a0_model = (
        get_a0_model(
            manifest
        )
    )

    if not a0_repo.is_dir():

        raise FileNotFoundError(
            f"A0 repo does not exist: {a0_repo}"
        )

    if not fold_manifest_path.exists():

        raise FileNotFoundError(
            "Fold manifest does not exist: "
            f"{fold_manifest_path}"
        )

    # -------------------------------------------------------------------------
    # Persistent Exp2 audit registry
    # -------------------------------------------------------------------------

    registry = CandidateRegistry(
        registry_path
    )

    # ---------------------------------------------------------------------
    # Resume recovery for interrupted candidate generation.
    #
    # A persisted "generating" record at fresh runner startup means the
    # previous process did not reach a terminal generation state. Finalize
    # it as generation_failed for auditability. It still counts as a
    # consumed candidate-generation attempt in this pilot.
    # ---------------------------------------------------------------------

    stale_generating = registry.filter(
        outer_fold=(
            outer_fold
        ),
        target_id=(
            target_id
        ),
        generation_status=(
            "generating"
        ),
        is_a0=False,
    )

    for record in stale_generating:
        registry.mark_generation_failed(
            record.candidate_id,
            error=(
                "stale generating state recovered on runner startup; "
                "previous generation did not reach a terminal state"
            ),
        )

    a0_record = (
        ensure_a0_registry_record(
            registry=(
                registry
            ),

            outer_fold=(
                outer_fold
            ),

            target_id=(
                target_id
            ),

            a0_repo=(
                a0_repo
            ),

            a0_model=(
                a0_model
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Persistent generic search archive
    # -------------------------------------------------------------------------

    archive = AgentArchive(
        archive_path
    )

    ensure_archive_root(
        archive=(
            archive
        ),

        a0_record=(
            a0_record
        ),
    )

    # -------------------------------------------------------------------------
    # A0 repair evidence must exist before search.
    # -------------------------------------------------------------------------

    ensure_repair_evaluation(
        registry=(
            registry
        ),

        candidate_id="A0",

        candidate_repo=(
            a0_repo
        ),

        model=(
            a0_model
        ),

        target_id=(
            target_id
        ),

        outer_fold=(
            outer_fold
        ),

        fold_manifest_path=(
            fold_manifest_path
        ),

        evaluation_root=(
            evaluation_root
        ),

        hpi_path=(
            hpi_path
        ),
    )

    ensure_feedback(
        candidate_id="A0",

        target_id=(
            target_id
        ),

        outer_fold=(
            outer_fold
        ),

        fold_manifest_path=(
            fold_manifest_path
        ),

        evaluation_root=(
            evaluation_root
        ),

        feedback_root=(
            feedback_root
        ),
    )

    # -------------------------------------------------------------------------
    # MetaAgent
    # -------------------------------------------------------------------------

    meta_chat_history.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    meta_agent = MetaAgent(
        model=(
            meta_model
        ),

        chat_history_file=str(
            meta_chat_history
        ),

        # Exp2 v5: MetaAgent samples (default 0.7) so that repeated expansions explore
        # different repairs instead of replaying one deterministic failure (v3 finding).
        # TaskAgent temperature stays frozen at 0.0.
        temperature=float(
            os.environ.get("EXP2_META_TEMPERATURE", "0.7")
        ),

        max_tokens=4096,
    )

    # Exp2-only MetaAgent context headroom.
    # TaskAgent remains frozen at max_tokens=4096.
    meta_agent.max_tokens = 2048

    # -------------------------------------------------------------------------
    # HGM policy
    # -------------------------------------------------------------------------

    policy_config_dict = (
        get_policy_config(
            manifest
        )
    )

    # ---------------------------------------------------------------------
    # Exp2 resume / exhausted-generation-budget guard.
    #
    # Candidate generation budget is tracked by the experiment-level
    # CandidateRegistry, independently of SearchController's unit budget.
    # On resume, all candidate slots may already be occupied while a
    # generated child is still only partially evaluated. In that state,
    # allowing EXPAND causes the policy to repeatedly choose a parent for
    # expansion even though generate_child() can only raise
    # "Candidate generation budget exhausted".
    #
    # Therefore:
    #   - root/evaluated nodes are expansion candidates only while a
    #     generation slot remains;
    #   - generated/pending_evaluation/evaluating nodes remain evaluatable;
    #   - when the frozen candidate budget is exhausted, this resumed run
    #     becomes evaluation-only and finishes outstanding candidates.
    # ---------------------------------------------------------------------

    existing_non_a0_candidates = (
        registry.filter(
            outer_fold=(
                outer_fold
            ),
            target_id=(
                target_id
            ),
            is_a0=False,
        )
    )

    generation_budget_exhausted = (
        len(
            existing_non_a0_candidates
        )
        >= candidate_budget
    )

    # Keep Exp2 role separation explicit rather than relying on the generic
    # SearchPolicy defaults, which allow the same status in both action sets.
    policy_config_dict[
        "evaluatable_statuses"
    ] = (
        "generated",
        "pending_evaluation",
        "evaluating",
    )

    if generation_budget_exhausted:

        # No further candidate may legally be generated in this partition.
        # Existing unfinished candidates must be evaluated/resumed instead.
        policy_config_dict[
            "expandable_statuses"
        ] = tuple()

    else:

        policy_config_dict[
            "expandable_statuses"
        ] = (
            "root",
            "evaluated",
        )

    policy_config = (
        SearchPolicyConfig(
            **policy_config_dict
        )
    )

    policy = SearchPolicy(
        policy_config
    )

    # -------------------------------------------------------------------------
    # Controller
    # -------------------------------------------------------------------------

    controller_config_dict = (
        get_controller_config(
            manifest
        )
    )

    controller_config = (
        SearchControllerConfig(
            **controller_config_dict
        )
    )

    callbacks = (
        Exp2RepairSearchCallbacks(
            registry=(
                registry
            ),

            run_manifest_path=(
                run_manifest_path
            ),

            meta_agent=(
                meta_agent
            ),

            model=(
                a0_model
            ),

            target_id=(
                target_id
            ),

            outer_fold=(
                outer_fold
            ),

            candidate_budget=(
                candidate_budget
            ),

            fold_manifest_path=(
                fold_manifest_path
            ),

            evaluation_root=(
                evaluation_root
            ),

            feedback_root=(
                feedback_root
            ),

            search_feedback_root=(
                search_feedback_root
            ),

            workspace_root=(
                workspace_root
            ),

            patch_root=(
                patch_root
            ),

            hpi_path=(
                hpi_path
            ),
        )
    )

    # Bind Exp2 callbacks to the active SearchPolicy so the
    # candidate-budget guard can dynamically disable EXPAND.
    callbacks.search_policy = policy

    partition = (
        make_partition(
            outer_fold=(
                outer_fold
            ),

            target_id=(
                target_id
            ),
        )
    )

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
            callbacks.generate_child
        ),

        evaluate_node_fn=(
            callbacks.evaluate_node
        ),
    )

    # -------------------------------------------------------------------------
    # Formal tree search.
    # -------------------------------------------------------------------------

    search_result = (
        controller.run()
    )

    # -------------------------------------------------------------------------
    # Persist final audit state.
    # -------------------------------------------------------------------------

    archive.recompute_clade_statistics()

    archive.save()

    registry.export_csv(
        registry_path.with_suffix(
            ".csv"
        )
    )

    summary = {
        "runner_version":
            RUNNER_VERSION,

        "search_mode":
            "tree_search",

        "outer_fold":
            outer_fold,

        "target_id":
            target_id,

        "partition":
            partition,

        "a0_model":
            a0_model,

        "meta_model":
            meta_model,

        "candidate_budget":
            candidate_budget,

        "root_candidate_id":
            "A0",

        "root_node_id":
            a0_record.node_id,

        "registry_path":
            str(
                registry_path.resolve()
            ),

        "archive_path":
            str(
                archive_path.resolve()
            ),

        "controller_summary":
            controller.summary(),

        "search_result":
            search_result,

        "protocol_guards": {
            "repair_feedback_only":
                True,

            "child_compared_to_actual_parent":
                True,

            "descendant_generation_allowed":
                True,

            "validation_used_for_search":
                False,

            "locked_test_used_for_search":
                False,
        },
    }

    summary_path = (
        archive_path.parent
        / "repair_search_summary.json"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        f.write(
            "\n"
        )

    return summary


# =============================================================================
# CLI
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run formal Exp2 HGM-style repair tree search "
            "for one outer-fold × repair-target partition."
        )
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
            VALID_TARGETS
        ),
    )

    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=Path(
            "outputs/exp2/frozen/run_manifest.yaml"
        ),
    )

    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=Path(
            "outputs/exp2/frozen/fold_manifest.csv"
        ),
    )

    parser.add_argument(
        "--a0-repo",
        type=Path,
        default=Path("."),
        help=(
            "Frozen A0 repository/workspace."
        ),
    )

    parser.add_argument(
        "--meta-model",
        required=True,
    )

    parser.add_argument(
        "--hpi-path",
        type=Path,
        default=mimic_cdm_dir() / "history_of_present_illness.csv",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/exp2"
        ),
    )

    args = parser.parse_args()

    partition_root = (
        args.output_root
        / "search"
        / f"fold{args.outer_fold}"
        / args.target_id
    )

    registry_path = (
        args.output_root
        / "candidates"
        / f"fold{args.outer_fold}"
        / args.target_id
        / "candidate_registry.json"
    )

    archive_path = (
        partition_root
        / "archive.json"
    )

    result = run_repair_search(
        outer_fold=(
            args.outer_fold
        ),

        target_id=(
            args.target_id
        ),

        run_manifest_path=(
            args.run_manifest
        ),

        a0_repo=(
            args.a0_repo
        ),

        meta_model=(
            args.meta_model
        ),

        fold_manifest_path=(
            args.fold_manifest
        ),

        hpi_path=(
            args.hpi_path
        ),

        registry_path=(
            registry_path
        ),

        archive_path=(
            archive_path
        ),

        evaluation_root=(
            args.output_root
            / "evaluations"
        ),

        feedback_root=(
            args.output_root
            / "feedback"
        ),

        search_feedback_root=(
            args.output_root
            / "search_feedback"
        ),

        workspace_root=(
            args.output_root
            / "workspaces"
        ),

        patch_root=(
            args.output_root
            / "patches"
        ),

        meta_chat_history=(
            partition_root
            / "meta_chat_history.md"
        ),
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    print(
        "run_repair_search.py: PASS"
    )


if __name__ == "__main__":
    main()
