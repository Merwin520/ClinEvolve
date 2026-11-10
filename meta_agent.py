# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

from agent.base_agent import AgentSystem
from agent.llm_withtools_meta import chat_with_agent


class MetaAgent(AgentSystem):
    """
    General-purpose self-modification agent.

    The MetaAgent receives:
        - a repository to modify,
        - optional evaluation / feedback context,
        - optional task objective,
        - optional modification constraints,
        - optional remaining search budget.

    It does NOT encode any experiment-specific objective, evaluator,
    target, dataset, or selection rule.

    Experiment-specific logic should be supplied by the caller.
    """

    def forward(
        self,
        repo_path: str,
        eval_path: Optional[str] = None,
        iterations_left: Optional[int] = None,
        *,
        objective: Optional[str] = None,
        feedback: Optional[str] = None,
        allowed_paths: Optional[Sequence[str]] = None,
        forbidden_paths: Optional[Sequence[str]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate one self-modification attempt.

        Args:
            repo_path:
                Path to the repository that the MetaAgent may inspect
                and modify.

            eval_path:
                Optional path containing previous evaluation results,
                logs, trajectories, or other development evidence.

                The MetaAgent may inspect this path if it is useful.

            iterations_left:
                Optional number of future self-modification iterations
                remaining in the current outer-loop search.

            objective:
                Optional high-level objective supplied by the caller.

                Example:
                    "Improve the agent's performance on the current task
                    while preserving existing capabilities."

                This field is deliberately generic.

            feedback:
                Optional textual feedback supplied by the caller.

                This can contain failure summaries, evaluator comments,
                development observations, or other experiment-specific
                information.

            allowed_paths:
                Optional list of repository paths the MetaAgent is
                permitted to modify.

                If supplied, the MetaAgent must restrict modifications
                to these paths.

            forbidden_paths:
                Optional list of repository paths that must not be
                modified.

            extra_context:
                Optional structured metadata supplied by the outer-loop
                controller.

                The MetaAgent should treat it as contextual information,
                not as authority to override modification constraints.

        Returns:
            list:
                Message history returned by chat_with_agent().
        """

        repo_path = str(
            repo_path
        ).strip()

        if not repo_path:
            raise ValueError(
                "repo_path must be non-empty"
            )

        instruction_parts = [
            (
                "You are a meta-level agent responsible for improving "
                "an agentic system by modifying its codebase."
            ),
            "",
            (
                f"The repository you may inspect is located at:\n"
                f"`{repo_path}`"
            ),
            "",
            (
                "Your task is to make one coherent self-modification "
                "that is expected to improve the system with respect "
                "to the objective and evidence supplied below."
            ),
            "",
            (
                "You may inspect the repository before deciding what "
                "to change."
            ),
            "",
            (
                "Do not make unrelated edits."
            ),
            (
                "Do not modify evaluation results, benchmark data, "
                "or external evidence merely to make performance "
                "appear better."
            ),
            (
                "Do not fabricate successful evaluations."
            ),
            (
                "Preserve compatibility with the existing codebase "
                "unless the supplied objective explicitly requires "
                "an interface change."
            ),
        ]

        # ------------------------------------------------------------------
        # Objective
        # ------------------------------------------------------------------

        if objective:

            instruction_parts.extend(
                [
                    "",
                    "====================",
                    "OBJECTIVE",
                    "====================",
                    str(
                        objective
                    ).strip(),
                ]
            )

        # ------------------------------------------------------------------
        # Evaluation / development evidence
        # ------------------------------------------------------------------

        if eval_path:

            instruction_parts.extend(
                [
                    "",
                    "====================",
                    "EVALUATION EVIDENCE",
                    "====================",
                    (
                        "Previous evaluation results or development "
                        "evidence are available at:"
                    ),
                    f"`{eval_path}`",
                    (
                        "Inspect this evidence when useful before "
                        "deciding what to modify."
                    ),
                ]
            )

        if feedback:

            instruction_parts.extend(
                [
                    "",
                    "====================",
                    "FEEDBACK",
                    "====================",
                    str(
                        feedback
                    ).strip(),
                ]
            )

        # ------------------------------------------------------------------
        # Remaining budget
        # ------------------------------------------------------------------

        if iterations_left is not None:

            if iterations_left < 0:
                raise ValueError(
                    "iterations_left must be >= 0 or None"
                )

            instruction_parts.extend(
                [
                    "",
                    "====================",
                    "SEARCH CONTEXT",
                    "====================",
                    (
                        "Remaining future self-modification "
                        f"iterations: {iterations_left}"
                    ),
                    (
                        "Prefer changes whose expected value is "
                        "appropriate for the remaining search budget."
                    ),
                ]
            )

        # ------------------------------------------------------------------
        # Modification constraints
        # ------------------------------------------------------------------

        if allowed_paths:

            cleaned_allowed = [
                str(
                    path
                ).strip()
                for path
                in allowed_paths
                if str(
                    path
                ).strip()
            ]

            if cleaned_allowed:

                instruction_parts.extend(
                    [
                        "",
                        "====================",
                        "ALLOWED MODIFICATION PATHS",
                        "====================",
                        (
                            "You may modify only the following "
                            "repository paths:"
                        ),
                    ]
                )

                instruction_parts.extend(
                    [
                        f"- {path}"
                        for path
                        in cleaned_allowed
                    ]
                )

        if forbidden_paths:

            cleaned_forbidden = [
                str(
                    path
                ).strip()
                for path
                in forbidden_paths
                if str(
                    path
                ).strip()
            ]

            if cleaned_forbidden:

                instruction_parts.extend(
                    [
                        "",
                        "====================",
                        "FORBIDDEN MODIFICATION PATHS",
                        "====================",
                        (
                            "Do not modify any of the following "
                            "repository paths:"
                        ),
                    ]
                )

                instruction_parts.extend(
                    [
                        f"- {path}"
                        for path
                        in cleaned_forbidden
                    ]
                )

        # ------------------------------------------------------------------
        # Generic structured context
        # ------------------------------------------------------------------

        if extra_context:

            instruction_parts.extend(
                [
                    "",
                    "====================",
                    "ADDITIONAL CONTEXT",
                    "====================",
                ]
            )

            for key, value in (
                extra_context.items()
            ):

                instruction_parts.append(
                    f"{key}: {value}"
                )

        # ------------------------------------------------------------------
        # Final instruction
        # ------------------------------------------------------------------

        instruction_parts.extend(
            [
                "",
                "====================",
                "MODIFICATION PROCEDURE",
                "====================",
                (
                    "1. Inspect the relevant implementation and "
                    "available evidence."
                ),
                (
                    "2. Identify a specific weakness or improvement "
                    "opportunity."
                ),
                (
                    "3. Make the smallest coherent code change that "
                    "addresses it."
                ),
                (
                    "4. Keep the repository executable and internally "
                    "consistent."
                ),
                (
                    "5. Do not modify files outside the permitted scope."
                ),
                (
                    "6. Do not change benchmark/evaluation artifacts "
                    "to improve reported scores."
                ),
                (
                    "7. Finish after implementing one coherent "
                    "candidate modification."
                ),
            ]
        )

        instruction = "\n".join(
            instruction_parts
        ).strip()

        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available=["bash", "edit"],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return new_msg_history