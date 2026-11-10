#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from meta_agent import MetaAgent

from clinevolve.search.controller import ChildGenerationResult

from clinevolve.registry import (
    CandidateRecord,
    CandidateRegistry,
    make_candidate_record,
)


# =============================================================================
# Constants
# =============================================================================

GENERATOR_VERSION = "exp2_generate_candidate_v3_workspace_safe_execution_guard"

VALID_PARENT_RULES = {
    "a0_only",
    "tree_search",
}

IGNORED_DIR_NAMES = {
    # Runtime/output trees must never enter candidate workspaces.
    # In particular, excluding outputs/ prevents recursive copying when
    # workspaces themselves live under repo/outputs/....
    "outputs",
    "runs",
    "data",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}

IGNORED_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
}


# =============================================================================
# Helpers
# =============================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

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


def load_json(
    path: Path,
) -> Dict[str, Any]:

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


def normalize_relpath(
    value: str | Path,
) -> str:
    """
    Normalize repository-relative paths.

    Examples:
        ./task_agent.py -> task_agent.py
        search/         -> search
    """

    text = str(
        value
    ).strip()

    text = text.replace(
        "\\",
        "/",
    )

    while text.startswith(
        "./"
    ):
        text = text[2:]

    text = text.rstrip(
        "/"
    )

    if (
        text.startswith("/")
        or text == ".."
        or text.startswith("../")
        or "/../" in text
    ):
        raise ValueError(
            f"Modification scope must be repo-relative: {value}"
        )

    return text


def path_matches_scope(
    relpath: str,
    scope: str,
) -> bool:

    relpath = normalize_relpath(
        relpath
    )

    scope = normalize_relpath(
        scope
    )

    if not scope:
        return True

    return (
        relpath == scope
        or relpath.startswith(
            scope + "/"
        )
    )


def is_ignored_path(
    relpath: str,
) -> bool:

    path = Path(
        relpath
    )

    if any(
        part in IGNORED_DIR_NAMES
        for part in path.parts
    ):
        return True

    if (
        path.suffix
        in IGNORED_FILE_SUFFIXES
    ):
        return True

    return False


# =============================================================================
# Run manifest
# =============================================================================

def load_run_manifest(
    path: Path,
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

    search = manifest.get(
        "search"
    )

    if not isinstance(
        search,
        Mapping,
    ):
        raise ValueError(
            "run manifest missing search section"
        )

    parent_rule = search.get(
        "candidate_parent_rule"
    )

    if (
        parent_rule
        not in VALID_PARENT_RULES
    ):
        raise ValueError(
            "Invalid candidate_parent_rule in run manifest: "
            f"{parent_rule}"
        )

    scope = manifest.get(
        "candidate_modification_scope"
    )

    if not isinstance(
        scope,
        Mapping,
    ):
        raise ValueError(
            "run manifest missing "
            "candidate_modification_scope"
        )

    allowed = scope.get(
        "allowed_paths"
    )

    if (
        not isinstance(
            allowed,
            list,
        )
        or not allowed
    ):
        raise ValueError(
            "Frozen allowed_paths are missing"
        )

    return manifest


def frozen_parent_rule(
    manifest: Mapping[str, Any],
) -> str:

    return str(
        manifest[
            "search"
        ][
            "candidate_parent_rule"
        ]
    )


def frozen_allowed_paths(
    manifest: Mapping[str, Any],
) -> List[str]:

    return sorted(
        {
            normalize_relpath(item)
            for item
            in manifest[
                "candidate_modification_scope"
            ][
                "allowed_paths"
            ]
        }
    )


def frozen_forbidden_paths(
    manifest: Mapping[str, Any],
) -> List[str]:

    raw = (
        manifest[
            "candidate_modification_scope"
        ].get(
            "forbidden_paths",
            [],
        )
    )

    return sorted(
        {
            normalize_relpath(item)
            for item in raw
        }
    )


# =============================================================================
# Workspace snapshot
# =============================================================================

def snapshot_workspace(
    repo_path: Path,
) -> Dict[
    str,
    str,
]:
    """
    relpath -> SHA256

    Runtime cache files are ignored.
    Source/config files are not ignored.
    """

    repo_path = (
        repo_path.resolve()
    )

    if not repo_path.is_dir():
        raise NotADirectoryError(
            f"Workspace does not exist: {repo_path}"
        )

    snapshot: Dict[
        str,
        str,
    ] = {}

    for path in sorted(
        repo_path.rglob("*")
    ):

        if not path.is_file():
            continue

        relpath = (
            path.relative_to(
                repo_path
            )
            .as_posix()
        )

        if is_ignored_path(
            relpath
        ):
            continue

        snapshot[
            relpath
        ] = sha256_file(
            path
        )

    return snapshot


def compare_snapshots(
    before: Mapping[
        str,
        str,
    ],
    after: Mapping[
        str,
        str,
    ],
) -> Tuple[
    List[str],
    List[str],
    List[str],
]:
    """
    Returns:
        modified
        added
        deleted
    """

    before_paths = set(
        before
    )

    after_paths = set(
        after
    )

    added = sorted(
        after_paths
        - before_paths
    )

    deleted = sorted(
        before_paths
        - after_paths
    )

    modified = sorted(
        path
        for path
        in (
            before_paths
            & after_paths
        )
        if (
            before[path]
            != after[path]
        )
    )

    return (
        modified,
        added,
        deleted,
    )


# =============================================================================
# Modification-scope audit
# =============================================================================

def check_modification_scope(
    changed_files: Sequence[str],
    *,
    allowed_paths: Sequence[str],
    forbidden_paths: Sequence[str],
) -> Tuple[
    bool,
    List[str],
]:

    violations: List[
        str
    ] = []

    for raw_path in (
        changed_files
    ):

        relpath = normalize_relpath(
            raw_path
        )

        forbidden_match = any(
            path_matches_scope(
                relpath,
                scope,
            )
            for scope
            in forbidden_paths
        )

        allowed_match = any(
            path_matches_scope(
                relpath,
                scope,
            )
            for scope
            in allowed_paths
        )

        if forbidden_match:

            violations.append(
                f"forbidden:{relpath}"
            )

        elif not allowed_match:

            violations.append(
                f"not_allowed:{relpath}"
            )

    return (
        len(
            violations
        )
        == 0,
        sorted(
            violations
        ),
    )


# =============================================================================
# Workspace copy
# =============================================================================

def copy_workspace(
    source: Path,
    destination: Path,
) -> None:
    """
    Copy a parent agent repository into an isolated candidate workspace.

    Safety invariant:
        Runtime/output trees are never copied. In particular, outputs/ is
        always excluded. This is essential because Exp2 workspaces normally
        live under the source repository's own outputs/ tree; copying outputs/
        would recursively copy the destination into itself and cause explosive
        disk growth.

    The same ignore policy is used by snapshot_workspace(), so excluded runtime
    artifacts cannot appear as candidate modifications.
    """

    source = source.resolve()
    destination = destination.resolve()

    if not source.is_dir():
        raise NotADirectoryError(
            f"Source workspace not found: {source}"
        )

    if destination.exists():
        raise FileExistsError(
            "Candidate workspace already exists: "
            f"{destination}"
        )

    try:
        destination_rel = destination.relative_to(source)
    except ValueError:
        destination_rel = None

    if destination_rel is not None:
        if not destination_rel.parts:
            raise RuntimeError(
                "Candidate workspace destination cannot equal source"
            )

        top_component = destination_rel.parts[0]
        if top_component not in IGNORED_DIR_NAMES:
            raise RuntimeError(
                "Candidate workspace is nested inside source under a "
                "non-ignored directory, which can cause recursive copying: "
                f"source={source}, destination={destination}, "
                f"top_component={top_component!r}"
            )

    def ignore(
        directory: str,
        names: List[str],
    ) -> Set[str]:

        ignored: Set[str] = set()

        for name in names:
            if name in IGNORED_DIR_NAMES:
                ignored.add(name)
            elif Path(name).suffix in IGNORED_FILE_SUFFIXES:
                ignored.add(name)

        return ignored

    shutil.copytree(
        source,
        destination,
        ignore=ignore,
        symlinks=True,
    )


# =============================================================================
# Patch generation
# =============================================================================

def create_patch(
    *,
    before_dir: Path,
    after_dir: Path,
    patch_path: Path,
    changed_files: Sequence[str],
) -> None:
    """
    Create an audit patch containing only files detected as changed by the
    filtered workspace snapshots.

    The A0 repository contains runtime outputs that are intentionally not
    copied into candidate workspaces. Diffing whole directories would therefore
    generate a huge patch showing outputs/ as deleted. We instead materialize
    tiny temporary before/after trees containing only changed files.
    """

    before_dir = before_dir.resolve()
    after_dir = after_dir.resolve()
    patch_path = patch_path.resolve()

    patch_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_changed = sorted({
        normalize_relpath(path)
        for path in changed_files
        if normalize_relpath(path)
    })

    if not normalized_changed:
        patch_path.write_bytes(b"")
        return

    with tempfile.TemporaryDirectory(
        prefix="exp2_patch_",
        dir=str(patch_path.parent),
    ) as tmp:

        tmp_root = Path(tmp)
        before_filtered = tmp_root / "before"
        after_filtered = tmp_root / "after"
        before_filtered.mkdir(parents=True, exist_ok=True)
        after_filtered.mkdir(parents=True, exist_ok=True)

        for relpath in normalized_changed:
            before_path = before_dir / relpath
            after_path = after_dir / relpath

            if before_path.exists() and before_path.is_file():
                dst = before_filtered / relpath
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(before_path, dst)

            if after_path.exists() and after_path.is_file():
                dst = after_filtered / relpath
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(after_path, dst)

        command = [
            "git",
            "diff",
            "--no-index",
            "--binary",
            "--",
            "before",
            "after",
        ]

        completed = subprocess.run(
            command,
            cwd=str(tmp_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if completed.returncode not in {0, 1}:
            raise RuntimeError(
                "git diff --no-index failed:\n"
                + completed.stderr.decode(
                    "utf-8",
                    errors="replace",
                )
            )

        patch_path.write_bytes(completed.stdout)


def patch_line_counts(
    patch_path: Path,
) -> Tuple[
    int,
    int,
]:

    added = 0
    deleted = 0

    text = patch_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    for line in (
        text.splitlines()
    ):

        if (
            line.startswith(
                "+++"
            )
            or line.startswith(
                "---"
            )
        ):
            continue

        if line.startswith("+"):
            added += 1

        elif line.startswith("-"):
            deleted += 1

    return (
        added,
        deleted,
    )


# =============================================================================
# Frozen A0 execution invariants
# =============================================================================

_FROZEN_TASK_AGENT_PATTERNS = {
    "tools_available": (
        '"get_physical_exam"',
        '"get_lab_results"',
        '"get_imaging_results"',
    ),
    "multiple_tool_calls": "multiple_tool_calls=False",
    "max_tool_calls": "max_tool_calls=40",
    "temperature": "temperature=self.temperature",
    "max_tokens": "max_tokens=self.max_tokens",
}


def validate_frozen_task_agent_invariants(
    *,
    parent_repo: Path,
    candidate_repo: Path,
) -> list[str]:
    """
    Protect Exp1-frozen execution conditions while still allowing
    MetaAgent to modify task_agent.py prompt / decision logic.
    """

    parent_path = parent_repo / "task_agent.py"
    candidate_path = candidate_repo / "task_agent.py"

    if not parent_path.is_file():
        raise RuntimeError(
            f"Frozen parent task_agent.py missing: {parent_path}"
        )

    if not candidate_path.is_file():
        return [
            "candidate deleted task_agent.py"
        ]

    parent_text = parent_path.read_text(
        encoding="utf-8"
    )

    candidate_text = candidate_path.read_text(
        encoding="utf-8"
    )

    violations: list[str] = []

    # Exact frozen call arguments.
    for name in (
        "multiple_tool_calls",
        "max_tool_calls",
        "temperature",
        "max_tokens",
    ):
        pattern = _FROZEN_TASK_AGENT_PATTERNS[name]

        if pattern not in parent_text:
            raise RuntimeError(
                f"Frozen invariant pattern missing from parent "
                f"task_agent.py: {pattern}"
            )

        if pattern not in candidate_text:
            violations.append(
                f"frozen execution invariant changed: {pattern}"
            )

    # Frozen tool set: all three required tools must remain present.
    for tool_name in _FROZEN_TASK_AGENT_PATTERNS[
        "tools_available"
    ]:
        if tool_name not in parent_text:
            raise RuntimeError(
                f"Frozen tool missing from parent task_agent.py: "
                f"{tool_name}"
            )

        if tool_name not in candidate_text:
            violations.append(
                f"frozen tool removed or renamed: {tool_name}"
            )

    return violations



# =============================================================================
# Frozen TaskAgent execution invariants
# =============================================================================

def validate_frozen_task_agent_execution(
    *,
    parent_repo: Path,
    candidate_repo: Path,
) -> list[str]:
    """
    Candidate may modify TaskAgent reasoning/prompt logic, but must not modify
    the frozen execution interface inherited from its actual parent.

    Frozen chat_with_agent kwargs:
        tools_available
        multiple_tool_calls
        max_tool_calls
        temperature
        max_tokens
    """

    import ast

    frozen_kwargs = (
        "tools_available",
        "multiple_tool_calls",
        "max_tool_calls",
        "temperature",
        "max_tokens",
    )

    def extract(path: Path) -> dict[str, str]:
        if not path.is_file():
            raise RuntimeError(
                f"task_agent.py missing: {path}"
            )

        tree = ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )

        matches = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            func = node.func

            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            else:
                continue

            if name != "chat_with_agent":
                continue

            kw = {
                item.arg: item.value
                for item in node.keywords
                if item.arg is not None
            }

            if all(
                key in kw
                for key in frozen_kwargs
            ):
                matches.append(kw)

        if len(matches) != 1:
            raise RuntimeError(
                "Expected exactly one TaskAgent chat_with_agent call "
                f"containing frozen execution kwargs in {path}; "
                f"found {len(matches)}"
            )

        kw = matches[0]

        return {
            key: ast.dump(
                kw[key],
                annotate_fields=True,
                include_attributes=False,
            )
            for key in frozen_kwargs
        }

    parent_values = extract(
        parent_repo / "task_agent.py"
    )

    candidate_values = extract(
        candidate_repo / "task_agent.py"
    )

    violations = []

    for key in frozen_kwargs:
        if candidate_values[key] != parent_values[key]:
            violations.append(
                f"frozen TaskAgent execution setting changed: {key}"
            )

    return violations


# =============================================================================
# MetaAgent invocation
# =============================================================================

def call_meta_agent(
    *,
    meta_agent: MetaAgent,
    repo_path: Path,
    objective: str,
    feedback: str,
    allowed_paths: Sequence[str],
    forbidden_paths: Sequence[str],
    iterations_left: Optional[int],
    extra_context: Optional[
        Mapping[str, Any]
    ] = None,
) -> Any:
    """
    Call the generic MetaAgent without assuming optional kwargs that its
    installed forward() implementation does not support.

    This matters because the original repository MetaAgent.forward only had:
        repo_path, eval_path, iterations_left

    while the generic version may additionally support:
        objective, feedback, allowed_paths, forbidden_paths, extra_context.

    Required safety:
        the generic Exp2 run should use a MetaAgent implementation supporting
        the experiment context arguments. If it does not, this function fails
        rather than silently dropping repair feedback.
    """

    signature = inspect.signature(
        meta_agent.forward
    )

    parameters = signature.parameters

    desired_kwargs: Dict[
        str,
        Any,
    ] = {
        "repo_path":
            str(
                repo_path.resolve()
            ),

        "iterations_left":
            iterations_left,

        "objective":
            objective,

        "feedback":
            feedback,

        "allowed_paths":
            list(
                allowed_paths
            ),

        "forbidden_paths":
            list(
                forbidden_paths
            ),

        "extra_context":
            dict(
                extra_context
                or {}
            ),
    }

    # Exp2 requires these arguments semantically.
    required_context_args = {
        "objective",
        "feedback",
        "allowed_paths",
        "forbidden_paths",
    }

    missing_required = [
        name
        for name
        in sorted(
            required_context_args
        )
        if name
        not in parameters
    ]

    has_var_kwargs = any(
        parameter.kind
        == inspect.Parameter.VAR_KEYWORD
        for parameter
        in parameters.values()
    )

    if (
        missing_required
        and not has_var_kwargs
    ):
        raise RuntimeError(
            "Current MetaAgent.forward is not the generic "
            "interface required by Exp2. Missing parameters: "
            f"{missing_required}. "
            "Do not silently run Exp2 without objective/feedback/scope."
        )

    call_kwargs = {}

    for (
        key,
        value,
    ) in desired_kwargs.items():

        if (
            key in parameters
            or has_var_kwargs
        ):
            call_kwargs[
                key
            ] = value

    return meta_agent.forward(
        **call_kwargs
    )


# =============================================================================
# Feedback
# =============================================================================

def load_feedback_text(
    feedback_path: Path,
) -> Tuple[
    str,
    str,
]:

    feedback_path = (
        feedback_path.resolve()
    )

    payload = load_json(
        feedback_path
    )

    role = payload.get(
        "role"
    )

    if (
        role
        != "repair"
    ):
        raise RuntimeError(
            "Feedback artifact is not repair-role only"
        )

    text_feedback = payload.get(
        "text_feedback"
    )

    if (
        not isinstance(
            text_feedback,
            str,
        )
        or not text_feedback.strip()
    ):
        raise ValueError(
            "Feedback JSON missing non-empty text_feedback"
        )

    return (
        text_feedback,
        sha256_file(
            feedback_path
        ),
    )


# =============================================================================
# Parent-rule enforcement
# =============================================================================

def validate_parent(
    *,
    parent: CandidateRecord,
    registry: CandidateRegistry,
    parent_rule: str,
) -> None:

    if (
        parent_rule
        == "a0_only"
    ):

        if not parent.is_a0:

            raise RuntimeError(
                "Frozen candidate_parent_rule=a0_only, "
                f"but requested parent {parent.candidate_id} "
                "is not A0. Refusing descendant generation."
            )

    elif (
        parent_rule
        == "tree_search"
    ):

        if (
            parent.generation_status
            != "generated"
        ):

            raise RuntimeError(
                "tree_search parent must be a successfully "
                "generated candidate"
            )

    else:

        raise ValueError(
            f"Unsupported parent rule: {parent_rule}"
        )

    # A0 consistency check.
    a0 = registry.get_a0(
        outer_fold=(
            parent.outer_fold
        ),
        target_id=(
            parent.target_id
        ),
    )

    if (
        parent_rule
        == "a0_only"
        and parent.candidate_id
        != a0.candidate_id
    ):
        raise RuntimeError(
            "a0_only parent does not match partition A0"
        )


# =============================================================================
# Candidate objective
# =============================================================================

def build_candidate_objective(
    *,
    target_id: str,
) -> str:

    if (
        target_id
        == "diagnostic_process_failure"
    ):

        return (
            "Improve the agent's diagnostic process on the provided "
            "repair-set evidence while preserving other workflow behavior, "
            "final diagnosis quality, tool safety, and resource discipline. "
            "Make one coherent modification within the allowed paths only."
        )

    if (
        target_id
        == "treatment_guideline_failure"
    ):

        return (
            "Improve the agent's treatment-guideline reasoning on the "
            "provided repair-set evidence while preserving other workflow "
            "behavior, final diagnosis quality, tool safety, and resource "
            "discipline. Make one coherent modification within the allowed "
            "paths only."
        )

    raise ValueError(
        f"Unsupported repair target: {target_id}"
    )


# =============================================================================
# Main candidate generation function
# =============================================================================


def _finalize_generation_failure(
    registry,
    candidate_id,
    *,
    error,
    generation_seconds=None,
    metadata_updates=None,
):
    """Persist terminal state for non-exception generation failures."""
    registry.mark_generation_failed(
        candidate_id,
        error=str(error),
    )

    current = registry.get(
        candidate_id
    )

    metadata = dict(
        current.metadata
    )

    if metadata_updates:
        metadata.update(
            dict(metadata_updates)
        )

    updates = {
        "metadata": metadata,
    }

    if generation_seconds is not None:
        updates[
            "generation_seconds"
        ] = generation_seconds

    registry.update(
        candidate_id,
        **updates,
    )


def generate_candidate(
    *,
    registry: CandidateRegistry,
    run_manifest_path: Path,
    parent_candidate_id: str,
    candidate_id: str,
    node_id: str,
    feedback_path: Path,
    workspace_root: Path,
    patch_root: Path,
    meta_agent: MetaAgent,
    iterations_left: Optional[int] = None,
    extra_context: Optional[
        Mapping[str, Any]
    ] = None,
    objective: Optional[str] = None,
) -> ChildGenerationResult:
    """
    Generate exactly one candidate.

    ``objective`` overrides the default single-target objective (used by sequential repair, whose stage objective
    also asks the repairer to preserve earlier repairs).

    This function is suitable for wrapping as SearchController's
    generate_child_fn.
    """

    manifest = load_run_manifest(
        run_manifest_path
    )

    parent_rule = (
        frozen_parent_rule(
            manifest
        )
    )

    allowed_paths = (
        frozen_allowed_paths(
            manifest
        )
    )

    forbidden_paths = (
        frozen_forbidden_paths(
            manifest
        )
    )

    parent = registry.get(
        parent_candidate_id
    )

    validate_parent(
        parent=parent,
        registry=registry,
        parent_rule=parent_rule,
    )

    if (
        parent.repo_path
        is None
    ):
        raise RuntimeError(
            f"Parent {parent.candidate_id} has no repo_path"
        )

    source_repo = Path(
        parent.repo_path
    ).resolve()

    if not source_repo.is_dir():
        raise RuntimeError(
            f"Parent workspace does not exist: {source_repo}"
        )

    # -------------------------------------------------------------------------
    # Repair feedback only
    # -------------------------------------------------------------------------

    (
        feedback_text,
        feedback_hash,
    ) = load_feedback_text(
        feedback_path
    )

    # -------------------------------------------------------------------------
    # Register candidate BEFORE generation begins.
    # -------------------------------------------------------------------------

    candidate_workspace = (
        workspace_root
        / (
            f"fold{parent.outer_fold}"
        )
        / parent.target_id
        / candidate_id
    ).resolve()

    patch_path = (
        patch_root
        / (
            f"fold{parent.outer_fold}"
        )
        / parent.target_id
        / f"{candidate_id}.patch"
    ).resolve()

    candidate = make_candidate_record(
        candidate_id=(
            candidate_id
        ),

        node_id=(
            node_id
        ),

        parent=(
            parent
        ),

        source_repo_path=(
            source_repo
        ),

        generation_model=(
            getattr(
                meta_agent,
                "model",
                None,
            )
        ),

        generation_temperature=(
            getattr(
                meta_agent,
                "temperature",
                None,
            )
        ),

        generation_max_tokens=(
            getattr(
                meta_agent,
                "max_tokens",
                None,
            )
        ),

        feedback_path=(
            feedback_path
        ),

        allowed_modification_paths=(
            allowed_paths
        ),

        forbidden_modification_paths=(
            forbidden_paths
        ),

        metadata={
            "generator_version":
                GENERATOR_VERSION,

            "run_manifest_path":
                str(
                    run_manifest_path.resolve()
                ),

            "run_manifest_sha256":
                sha256_file(
                    run_manifest_path
                ),

            "candidate_parent_rule":
                parent_rule,

            "feedback_sha256":
                feedback_hash,
        },
    )

    registry.add(
        candidate
    )

    registry.mark_generation_started(
        candidate_id
    )

    started = time.monotonic()

    try:

        # ---------------------------------------------------------------------
        # Copy parent into independent candidate workspace.
        # ---------------------------------------------------------------------

        candidate_workspace.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        copy_workspace(
            source_repo,
            candidate_workspace,
        )

        # Hard safety check: candidate workspaces must never contain runtime
        # outputs. Abort before MetaAgent runs if this invariant is violated.
        if (candidate_workspace / "outputs").exists():
            raise RuntimeError(
                "Unsafe candidate workspace: outputs/ was copied into "
                f"{candidate_workspace}. Refusing generation to prevent "
                "recursive workspace growth."
            )

        before_snapshot = snapshot_workspace(
            source_repo
        )

        # ---------------------------------------------------------------------
        # MetaAgent modification
        # ---------------------------------------------------------------------

        objective = objective or build_candidate_objective(
            target_id=(
                parent.target_id
            )
        )

        meta_result = call_meta_agent(
            meta_agent=(
                meta_agent
            ),

            repo_path=(
                candidate_workspace
            ),

            objective=(
                objective
            ),

            feedback=(
                feedback_text
            ),

            allowed_paths=(
                allowed_paths
            ),

            forbidden_paths=(
                forbidden_paths
            ),

            iterations_left=(
                iterations_left
            ),

            extra_context={
                "experiment":
                    "exp2",

                "outer_fold":
                    parent.outer_fold,

                "repair_target":
                    parent.target_id,

                "candidate_id":
                    candidate_id,

                "parent_candidate_id":
                    parent.candidate_id,

                "candidate_parent_rule":
                    parent_rule,

                **dict(
                    extra_context
                    or {}
                ),
            },
        )

        after_snapshot = snapshot_workspace(
            candidate_workspace
        )

        (
            modified,
            added,
            deleted,
        ) = compare_snapshots(
            before_snapshot,
            after_snapshot,
        )

        changed_files = sorted(
            set(
                modified
                + added
                + deleted
            )
        )

        if not changed_files:

            raise RuntimeError(
                "MetaAgent completed without modifying "
                "any repository file"
            )

        # ---------------------------------------------------------------------
        # Scope guard
        # ---------------------------------------------------------------------

        (
            scope_ok,
            scope_violations,
        ) = check_modification_scope(
            changed_files,
            allowed_paths=(
                allowed_paths
            ),
            forbidden_paths=(
                forbidden_paths
            ),
        )

        if not scope_ok:

            registry.mark_scope_violation(
                candidate_id,
                violations=(
                    scope_violations
                ),
            )

            elapsed = (
                time.monotonic()
                - started
            )

            return ChildGenerationResult(
                success=False,

                candidate_id=(
                    candidate_id
                ),

                node_id=(
                    node_id
                ),

                repo_path=(
                    str(
                        candidate_workspace
                    )
                ),

                modified_files=(
                    changed_files
                ),

                generation_model=(
                    getattr(
                        meta_agent,
                        "model",
                        None,
                    )
                ),

                generation_temperature=(
                    getattr(
                        meta_agent,
                        "temperature",
                        None,
                    )
                ),

                generation_max_tokens=(
                    getattr(
                        meta_agent,
                        "max_tokens",
                        None,
                    )
                ),

                generation_seconds=(
                    elapsed
                ),

                generation_error=(
                    "candidate violated frozen "
                    "modification scope: "
                    + "; ".join(
                        scope_violations
                    )
                ),

                metadata={
                    "scope_violations":
                        scope_violations,

                    "modified_files":
                        modified,

                    "added_files":
                        added,

                    "deleted_files":
                        deleted,
                },
            )

        # ---------------------------------------------------------------------
        # Frozen A0 execution invariant guard
        # ---------------------------------------------------------------------

        invariant_violations = (
            validate_frozen_task_agent_invariants(
                parent_repo=(
                    source_repo
                ),
                candidate_repo=(
                    candidate_workspace
                ),
            )
        )

        if invariant_violations:

            elapsed = (
                time.monotonic()
                - started
            )

            invariant_error = (
                "candidate violated frozen A0 execution "
                "invariants: "
                + "; ".join(
                    invariant_violations
                )
            )

            registry.mark_generation_failed(
                candidate_id,
                error=invariant_error,
            )

            current = registry.get(
                candidate_id
            )

            metadata = dict(
                current.metadata
            )

            metadata.update(
                {
                    "invariant_violations":
                        invariant_violations,

                    "modified_files":
                        modified,

                    "added_files":
                        added,

                    "deleted_files":
                        deleted,
                }
            )

            registry.update(
                candidate_id,
                generation_seconds=(
                    elapsed
                ),
                metadata=(
                    metadata
                ),
            )

            return ChildGenerationResult(
                success=False,

                candidate_id=(
                    candidate_id
                ),

                node_id=(
                    node_id
                ),

                repo_path=(
                    str(
                        candidate_workspace
                    )
                ),

                modified_files=(
                    changed_files
                ),

                generation_model=(
                    getattr(
                        meta_agent,
                        "model",
                        None,
                    )
                ),

                generation_temperature=(
                    getattr(
                        meta_agent,
                        "temperature",
                        None,
                    )
                ),

                generation_max_tokens=(
                    getattr(
                        meta_agent,
                        "max_tokens",
                        None,
                    )
                ),

                generation_seconds=(
                    elapsed
                ),

                generation_error=(
                    invariant_error
                ),

                metadata={
                    "invariant_violations":
                        invariant_violations,

                    "modified_files":
                        modified,

                    "added_files":
                        added,

                    "deleted_files":
                        deleted,
                },
            )


        # ---------------------------------------------------------------------
        # Frozen TaskAgent execution guard
        # ---------------------------------------------------------------------

        execution_violations = (
            validate_frozen_task_agent_execution(
                parent_repo=(
                    source_repo
                ),
                candidate_repo=(
                    candidate_workspace
                ),
            )
        )

        if execution_violations:

            registry.mark_scope_violation(
                candidate_id,
                violations=(
                    execution_violations
                ),
            )

            elapsed = (
                time.monotonic()
                - started
            )

            return ChildGenerationResult(
                success=False,

                candidate_id=(
                    candidate_id
                ),

                node_id=(
                    node_id
                ),

                repo_path=(
                    str(
                        candidate_workspace
                    )
                ),

                modified_files=(
                    changed_files
                ),

                generation_model=(
                    getattr(
                        meta_agent,
                        "model",
                        None,
                    )
                ),

                generation_temperature=(
                    getattr(
                        meta_agent,
                        "temperature",
                        None,
                    )
                ),

                generation_max_tokens=(
                    getattr(
                        meta_agent,
                        "max_tokens",
                        None,
                    )
                ),

                generation_seconds=(
                    elapsed
                ),

                generation_error=(
                    "candidate violated frozen TaskAgent "
                    "execution invariants: "
                    + "; ".join(
                        execution_violations
                    )
                ),

                metadata={
                    "execution_invariant_violations":
                        execution_violations,

                    "modified_files":
                        modified,

                    "added_files":
                        added,

                    "deleted_files":
                        deleted,
                },
            )

        # ---------------------------------------------------------------------
        # Patch
        # ---------------------------------------------------------------------

        patch_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        create_patch(
            before_dir=(
                source_repo
            ),
            after_dir=(
                candidate_workspace
            ),
            patch_path=(
                patch_path
            ),
            changed_files=(
                changed_files
            ),
        )

        patch_hash = sha256_file(
            patch_path
        )

        (
            lines_added,
            lines_deleted,
        ) = patch_line_counts(
            patch_path
        )

        elapsed = (
            time.monotonic()
            - started
        )

        # MetaAgent return value is intentionally treated as opaque.
        # We only record a JSON-safe repr in audit metadata.
        meta_result_repr = repr(
            meta_result
        )

        if (
            len(
                meta_result_repr
            )
            > 10000
        ):
            meta_result_repr = (
                meta_result_repr[
                    :10000
                ]
                + "...<truncated>"
            )

        # ---------------------------------------------------------------------
        # Registry
        # ---------------------------------------------------------------------

        registry.mark_generated(
            candidate_id,

            repo_path=(
                candidate_workspace
            ),

            patch_path=(
                patch_path
            ),

            modified_files=(
                changed_files
            ),

            lines_added=(
                lines_added
            ),

            lines_deleted=(
                lines_deleted
            ),

            generation_tokens=None,

            generation_calls=None,

            generation_seconds=(
                elapsed
            ),
        )

        current = registry.get(
            candidate_id
        )

        metadata = dict(
            current.metadata
        )

        metadata.update(
            {
                "modified_files":
                    modified,

                "added_files":
                    added,

                "deleted_files":
                    deleted,

                "meta_agent_return_repr":
                    meta_result_repr,
            }
        )

        registry.update(
            candidate_id,
            metadata=metadata,
        )

        return ChildGenerationResult(
            success=True,

            candidate_id=(
                candidate_id
            ),

            node_id=(
                node_id
            ),

            agent_version=(
                candidate_id
            ),

            repo_path=(
                str(
                    candidate_workspace
                )
            ),

            patch_path=(
                str(
                    patch_path
                )
            ),

            patch_hash=(
                patch_hash
            ),

            modified_files=(
                changed_files
            ),

            lines_added=(
                lines_added
            ),

            lines_deleted=(
                lines_deleted
            ),

            generation_model=(
                getattr(
                    meta_agent,
                    "model",
                    None,
                )
            ),

            generation_temperature=(
                getattr(
                    meta_agent,
                    "temperature",
                    None,
                )
            ),

            generation_max_tokens=(
                getattr(
                    meta_agent,
                    "max_tokens",
                    None,
                )
            ),

            generation_tokens=None,

            generation_calls=None,

            generation_seconds=(
                elapsed
            ),

            metadata={
                "generator_version":
                    GENERATOR_VERSION,

                "parent_candidate_id":
                    parent.candidate_id,

                "parent_rule":
                    parent_rule,

                "feedback_path":
                    str(
                        feedback_path.resolve()
                    ),

                "feedback_sha256":
                    feedback_hash,

                "modified_files":
                    modified,

                "added_files":
                    added,

                "deleted_files":
                    deleted,
            },
        )

    except Exception as exc:

        elapsed = (
            time.monotonic()
            - started
        )

        error = (
            f"{type(exc).__name__}: {exc}"
        )

        try:

            registry.mark_generation_failed(
                candidate_id,
                error=error,
            )

            current = registry.get(
                candidate_id
            )

            metadata = dict(
                current.metadata
            )

            metadata[
                "generation_traceback"
            ] = traceback.format_exc()

            registry.update(
                candidate_id,
                generation_seconds=(
                    elapsed
                ),
                metadata=(
                    metadata
                ),
            )

        except Exception:
            pass

        return ChildGenerationResult(
            success=False,

            candidate_id=(
                candidate_id
            ),

            node_id=(
                node_id
            ),

            generation_model=(
                getattr(
                    meta_agent,
                    "model",
                    None,
                )
            ),

            generation_temperature=(
                getattr(
                    meta_agent,
                    "temperature",
                    None,
                )
            ),

            generation_max_tokens=(
                getattr(
                    meta_agent,
                    "max_tokens",
                    None,
                )
            ),

            generation_seconds=(
                elapsed
            ),

            generation_error=(
                error
            ),

            metadata={
                "traceback":
                    traceback.format_exc(),

                "generator_version":
                    GENERATOR_VERSION,
            },
        )


# =============================================================================
# Controller callback factory
# =============================================================================

def make_generate_child_callback(
    *,
    registry: CandidateRegistry,
    run_manifest_path: Path,
    feedback_path: Path,
    workspace_root: Path,
    patch_root: Path,
    meta_agent: MetaAgent,
    candidate_id_prefix: str,
) -> Any:
    """
    Build a SearchController-compatible callback:

        fn(parent_archive_node, step, remaining_budget)
            -> ChildGenerationResult

    Important:
        CandidateRegistry and AgentArchive candidate IDs must stay aligned.
    """

    def callback(
        parent_archive_node,
        step: int,
        remaining_budget: int,
    ) -> ChildGenerationResult:

        parent_candidate_id = (
            parent_archive_node.candidate_id
        )

        candidate_id = (
            f"{candidate_id_prefix}"
            f"{step:04d}"
        )

        node_id = (
            f"{parent_archive_node.node_id}"
            f"__{candidate_id}"
        )

        return generate_candidate(
            registry=(
                registry
            ),

            run_manifest_path=(
                run_manifest_path
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
                feedback_path
            ),

            workspace_root=(
                workspace_root
            ),

            patch_root=(
                patch_root
            ),

            meta_agent=(
                meta_agent
            ),

            iterations_left=(
                remaining_budget
            ),

            extra_context={
                "controller_step":
                    step,
            },
        )

    return callback


# =============================================================================
# CLI
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate one Exp2 candidate in an isolated workspace."
        )
    )

    parser.add_argument(
        "--registry",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--run-manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--parent-candidate-id",
        required=True,
    )

    parser.add_argument(
        "--candidate-id",
        required=True,
    )

    parser.add_argument(
        "--node-id",
        required=True,
    )

    parser.add_argument(
        "--feedback",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(
            "outputs/exp2/workspaces"
        ),
    )

    parser.add_argument(
        "--patch-root",
        type=Path,
        default=Path(
            "outputs/exp2/patches"
        ),
    )

    parser.add_argument(
        "--meta-model",
        required=True,
        help=(
            "Model name passed to MetaAgent."
        ),
    )

    parser.add_argument(
        "--chat-history-file",
        type=Path,
        default=Path(
            "outputs/exp2/meta_chat_history.md"
        ),
    )

    parser.add_argument(
        "--iterations-left",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    registry = CandidateRegistry(
        args.registry
    )

    args.chat_history_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    meta_agent = MetaAgent(
        model=(
            args.meta_model
        ),
        chat_history_file=str(
            args.chat_history_file
        ),
        temperature=0.0,
        max_tokens=4096,
    )

    result = generate_candidate(
        registry=(
            registry
        ),

        run_manifest_path=(
            args.run_manifest
        ),

        parent_candidate_id=(
            args.parent_candidate_id
        ),

        candidate_id=(
            args.candidate_id
        ),

        node_id=(
            args.node_id
        ),

        feedback_path=(
            args.feedback
        ),

        workspace_root=(
            args.workspace_root
        ),

        patch_root=(
            args.patch_root
        ),

        meta_agent=(
            meta_agent
        ),

        iterations_left=(
            args.iterations_left
        ),
    )

    print(
        json.dumps(
            {
                "success":
                    result.success,

                "candidate_id":
                    result.candidate_id,

                "node_id":
                    result.node_id,

                "repo_path":
                    result.repo_path,

                "patch_path":
                    result.patch_path,

                "patch_hash":
                    result.patch_hash,

                "modified_files":
                    result.modified_files,

                "generation_error":
                    result.generation_error,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if not result.success:
        raise SystemExit(1)

    print(
        "generate_candidate.py: PASS"
    )


if __name__ == "__main__":
    main()
