import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================================================
# Allowed Experiment-1 clinical tools
# =========================================================

CLINICAL_TOOLS = {
    "get_physical_exam",
    "get_lab_results",
    "get_imaging_results",
}


# =========================================================
# Helpers
# =========================================================

def _get_text(message: Dict[str, Any]) -> str:
    if "text" in message:
        value = message.get("text", "")
    else:
        value = message.get("content", "")

    if value is None:
        return ""

    return str(value)


def _get_role(message: Dict[str, Any]) -> str:
    return str(
        message.get("role", "")
    ).lower()


def _load_json(path: Path) -> Dict[str, Any]:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def _save_json(
    obj: Dict[str, Any],
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    temp_path.replace(path)


# =========================================================
# Tool-call parsing
# =========================================================

def _extract_json_blocks(text: str) -> List[str]:
    """
    Extract <json> ... </json> blocks.
    """
    return re.findall(
        r"<json>\s*(.*?)\s*</json>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )


def extract_assistant_tool_calls(
    msg_history: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """
    Parse assistant tool calls.

    Returns:
        parsed_calls,
        malformed_calls
    """
    calls = []
    malformed = []

    for message_index, message in enumerate(
        msg_history
    ):
        if _get_role(message) != "assistant":
            continue

        text = _get_text(message)

        if "tool_name" not in text:
            continue

        blocks = _extract_json_blocks(
            text
        )

        # Fallback for attempted calls without <json> wrapper.
        if not blocks:
            blocks = [text]

        parsed_any = False
        malformed_in_this_message = False

        for block in blocks:
            try:
                obj = json.loads(block)

            except json.JSONDecodeError:
                if "tool_name" in block:
                    malformed.append(
                        {
                            "message_index": message_index,
                            "reason": "malformed_tool_json",
                            "raw": block,
                        }
                    )
                    malformed_in_this_message = True

                continue

            if not isinstance(
                obj,
                dict,
            ):
                continue

            if "tool_name" not in obj:
                continue

            parsed_any = True

            tool_name = str(
                obj.get(
                    "tool_name",
                    "",
                )
            ).strip()

            tool_input = obj.get(
                "tool_input",
                None,
            )

            calls.append(
                {
                    "message_index": message_index,
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "known_tool": (
                        tool_name
                        in CLINICAL_TOOLS
                    ),
                    "tool_input_is_object": isinstance(
                        tool_input,
                        dict,
                    ),
                }
            )

        if (
            not parsed_any
            and "tool_name" in text
            and not malformed_in_this_message
        ):
            malformed.append(
                {
                    "message_index": message_index,
                    "reason": "unparseable_tool_call",
                    "raw": text,
                }
            )

    return (
        calls,
        malformed,
    )


# =========================================================
# Tool input schema validation
# =========================================================

def _validate_physical_exam_input(
    tool_input: Any,
) -> List[str]:
    errors = []

    if not isinstance(
        tool_input,
        dict,
    ):
        return [
            "tool_input_not_object"
        ]

    if tool_input:
        errors.append(
            "physical_exam_input_must_be_empty_object"
        )

    return errors


def _validate_lab_input(
    tool_input: Any,
) -> List[str]:
    errors = []

    if not isinstance(
        tool_input,
        dict,
    ):
        return [
            "tool_input_not_object"
        ]

    if "test_names" not in tool_input:
        errors.append(
            "missing_test_names"
        )
        return errors

    test_names = tool_input.get(
        "test_names"
    )

    if not isinstance(
        test_names,
        list,
    ):
        errors.append(
            "test_names_not_list"
        )
        return errors

    if len(test_names) == 0:
        errors.append(
            "test_names_empty"
        )
        return errors

    for index, value in enumerate(
        test_names
    ):
        if not isinstance(
            value,
            str,
        ):
            errors.append(
                f"test_name_not_string:{index}"
            )
            continue

        if not value.strip():
            errors.append(
                f"test_name_empty:{index}"
            )

    # No extra fields in formal tool schema.
    extra_fields = (
        set(tool_input.keys())
        - {"test_names"}
    )

    if extra_fields:
        errors.append(
            "unexpected_lab_fields:"
            + ",".join(
                sorted(extra_fields)
            )
        )

    return errors


def _validate_imaging_input(
    tool_input: Any,
) -> List[str]:
    errors = []

    if not isinstance(
        tool_input,
        dict,
    ):
        return [
            "tool_input_not_object"
        ]

    required_fields = {
        "modality",
        "region",
    }

    missing = (
        required_fields
        - set(tool_input.keys())
    )

    if missing:
        errors.append(
            "missing_imaging_fields:"
            + ",".join(
                sorted(missing)
            )
        )

    modality = tool_input.get(
        "modality"
    )

    region = tool_input.get(
        "region"
    )

    if "modality" in tool_input:
        if not isinstance(
            modality,
            str,
        ):
            errors.append(
                "modality_not_string"
            )
        elif not modality.strip():
            errors.append(
                "modality_empty"
            )

    if "region" in tool_input:
        if not isinstance(
            region,
            str,
        ):
            errors.append(
                "region_not_string"
            )
        elif not region.strip():
            errors.append(
                "region_empty"
            )

    extra_fields = (
        set(tool_input.keys())
        - required_fields
    )

    if extra_fields:
        errors.append(
            "unexpected_imaging_fields:"
            + ",".join(
                sorted(extra_fields)
            )
        )

    return errors


def validate_tool_input_schema(
    tool_name: str,
    tool_input: Any,
) -> List[str]:
    """
    Deterministically validate Experiment-1 tool arguments.
    """
    if tool_name == "get_physical_exam":
        return _validate_physical_exam_input(
            tool_input
        )

    if tool_name == "get_lab_results":
        return _validate_lab_input(
            tool_input
        )

    if tool_name == "get_imaging_results":
        return _validate_imaging_input(
            tool_input
        )

    return []


def validate_tool_calls(
    tool_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Return records for tool calls whose input schema is invalid.
    """
    invalid_calls = []

    for call in tool_calls:
        tool_name = call.get(
            "tool_name",
            "",
        )

        # Unknown tools are handled separately.
        if tool_name not in CLINICAL_TOOLS:
            continue

        schema_errors = (
            validate_tool_input_schema(
                tool_name=tool_name,
                tool_input=call.get(
                    "tool_input"
                ),
            )
        )

        if schema_errors:
            invalid_calls.append(
                {
                    **call,
                    "schema_errors":
                        schema_errors,
                }
            )

    return invalid_calls


# =========================================================
# Final response validation
# =========================================================

def validate_final_response(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate the structured final Clinical TaskAgent output.
    """
    prediction = trajectory.get(
        "prediction"
    )

    errors = []

    if not isinstance(
        prediction,
        dict,
    ):
        errors.append(
            "prediction_not_dict"
        )

        return {
            "valid": False,
            "errors": errors,
        }

    required_fields = {
        "diagnosis",
        "diagnostic_summary",
        "treatment_plan",
        "confidence",
        "stop_reason",
    }

    missing = (
        required_fields
        - set(prediction.keys())
    )

    if missing:
        errors.append(
            "missing_final_fields:"
            + ",".join(
                sorted(missing)
            )
        )

    diagnosis = prediction.get(
        "diagnosis"
    )

    if (
        diagnosis is None
        or not str(
            diagnosis
        ).strip()
        or str(
            diagnosis
        ).strip().lower()
        == "none"
    ):
        errors.append(
            "missing_or_invalid_diagnosis"
        )

    diagnostic_summary = (
        prediction.get(
            "diagnostic_summary"
        )
    )

    if (
        diagnostic_summary is None
        or not str(
            diagnostic_summary
        ).strip()
    ):
        errors.append(
            "missing_diagnostic_summary"
        )

    treatment_plan = prediction.get(
        "treatment_plan"
    )

    if treatment_plan is None:
        errors.append(
            "missing_treatment_plan"
        )

    confidence = str(
        prediction.get(
            "confidence",
            "",
        )
    ).strip().lower()

    if confidence not in {
        "low",
        "moderate",
        "high",
    }:
        errors.append(
            "invalid_confidence"
        )

    stop_reason = str(
        prediction.get(
            "stop_reason",
            "",
        )
    ).strip()

    if not stop_reason:
        errors.append(
            "missing_stop_reason"
        )

    return {
        "valid":
            len(errors) == 0,

        "errors":
            errors,
    }


# =========================================================
# Tool execution failure detection
# =========================================================

def detect_tool_execution_errors(
    msg_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Detect actual tool execution errors.

    Only inspect tool outputs returned by the harness.

    Do NOT scan arbitrary assistant reasoning, because phrases
    such as:

        "the electrolytes were not found in the results"

    are clinical observations, not tool execution failures.
    """

    error_patterns = [
        (
            "tool_not_found",
            r"\btool\s+['\"]?[\w.\-]+['\"]?\s+not\s+found\b",
        ),
        (
            "unknown_tool",
            r"\bunknown\s+tool\b",
        ),
        (
            "invalid_tool",
            r"\binvalid\s+tool\b",
        ),
        (
            "error_executing_tool",
            r"\berror\s+executing\s+tool\b",
        ),
        (
            "tool_execution_error",
            r"\btool\s+execution\s+error\b",
        ),
        (
            "traceback",
            r"traceback\s*\(most\s+recent\s+call\s+last\)",
        ),
        (
            "file_not_found_error",
            r"\bfilenotfounderror\b",
        ),
        (
            "type_error",
            r"\btypeerror\s*:",
        ),
        (
            "value_error",
            r"\bvalueerror\s*:",
        ),
        (
            "key_error",
            r"\bkeyerror\s*:",
        ),
    ]

    errors = []

    for message_index, message in enumerate(
        msg_history
    ):
        # Tool results in the current HyperAgent protocol are
        # returned by the harness as user-role messages.
        if _get_role(message) != "user":
            continue

        text = _get_text(
            message
        )

        blocks = _extract_json_blocks(
            text
        )

        if not blocks:
            continue

        for block in blocks:
            try:
                outer = json.loads(
                    block
                )
            except json.JSONDecodeError:
                continue

            if not isinstance(
                outer,
                dict,
            ):
                continue

            # This is not a tool-result block.
            if "tool_output" not in outer:
                continue

            tool_name = str(
                outer.get(
                    "tool_name",
                    "",
                )
            ).strip()

            raw_output = outer.get(
                "tool_output"
            )

            parsed_output = None

            if isinstance(
                raw_output,
                dict,
            ):
                parsed_output = raw_output

            elif isinstance(
                raw_output,
                str,
            ):
                stripped = raw_output.strip()

                try:
                    candidate = json.loads(
                        stripped
                    )

                    if isinstance(
                        candidate,
                        dict,
                    ):
                        parsed_output = candidate

                except json.JSONDecodeError:
                    parsed_output = None

            # -------------------------------------------------
            # Structured error status
            # -------------------------------------------------

            structured_status = None

            if isinstance(
                parsed_output,
                dict,
            ):
                structured_status = str(
                    parsed_output.get(
                        "status",
                        "",
                    )
                ).strip().lower()

            matched = []

            if structured_status in {
                "error",
                "failed",
                "failure",
            }:
                matched.append(
                    f"status={structured_status}"
                )

            # -------------------------------------------------
            # Explicit runtime-error text
            # -------------------------------------------------

            if isinstance(
                raw_output,
                str,
            ):
                output_text = raw_output

            else:
                try:
                    output_text = json.dumps(
                        raw_output,
                        ensure_ascii=False,
                        default=str,
                    )
                except Exception:
                    output_text = str(
                        raw_output
                    )

            lowered = output_text.lower()

            for label, pattern in error_patterns:
                if re.search(
                    pattern,
                    lowered,
                    flags=re.IGNORECASE,
                ):
                    matched.append(
                        label
                    )

            if not matched:
                continue

            errors.append(
                {
                    "message_index":
                        message_index,

                    "tool_name":
                        tool_name,

                    "patterns":
                        sorted(
                            set(
                                matched
                            )
                        ),

                    "text_excerpt":
                        output_text[:1000],
                }
            )

    return errors


# =========================================================
# Main evaluator
# =========================================================

def evaluate_instruction_tool(
    trajectory: Dict[str, Any],
    max_tool_calls: int = 40,
) -> Dict[str, Any]:
    """
    Deterministic Experiment-1 instruction/tool evaluator.

    instruction_tool_failure = 1 if any of the following occur:

    1. trajectory runtime status is not success
    2. malformed tool-call JSON
    3. unknown tool called
    4. tool_input is not a JSON object
    5. known tool has invalid argument schema
    6. tool execution produced an obvious runtime/interface error
    7. tool-call count exceeds configured maximum
    8. final structured prediction is invalid
    """
    status = trajectory.get(
        "status",
        "",
    )

    msg_history = trajectory.get(
        "message_history",
        [],
    )

    if not isinstance(
        msg_history,
        list,
    ):
        msg_history = []

    tool_calls, malformed_calls = (
        extract_assistant_tool_calls(
            msg_history
        )
    )

    unknown_tools = [
        call
        for call in tool_calls
        if not call[
            "known_tool"
        ]
    ]

    non_object_tool_inputs = [
        call
        for call in tool_calls
        if not call[
            "tool_input_is_object"
        ]
    ]

    schema_invalid_tool_inputs = (
        validate_tool_calls(
            tool_calls
        )
    )

    execution_errors = (
        detect_tool_execution_errors(
            msg_history
        )
    )

    final_validation = (
        validate_final_response(
            trajectory
        )
    )

    n_tool_calls = len(
        tool_calls
    )

    exceeded_max_tool_calls = (
        n_tool_calls
        > max_tool_calls
    )

    failure_reasons = []

    if status != "success":
        failure_reasons.append(
            "trajectory_status_not_success"
        )

    if malformed_calls:
        failure_reasons.append(
            "malformed_tool_call"
        )

    if unknown_tools:
        failure_reasons.append(
            "unknown_tool_call"
        )

    if non_object_tool_inputs:
        failure_reasons.append(
            "tool_input_not_object"
        )

    if schema_invalid_tool_inputs:
        failure_reasons.append(
            "invalid_tool_input_schema"
        )

    if execution_errors:
        failure_reasons.append(
            "tool_execution_error"
        )

    if exceeded_max_tool_calls:
        failure_reasons.append(
            "max_tool_calls_exceeded"
        )

    if not final_validation[
        "valid"
    ]:
        failure_reasons.append(
            "invalid_final_response"
        )

    instruction_tool_failure = int(
        len(failure_reasons) > 0
    )

    return {
        "run_id":
            trajectory.get(
                "run_id"
            ),

        "case_id":
            str(
                trajectory.get(
                    "case_id",
                    "",
                )
            ),

        "agent_id":
            trajectory.get(
                "agent_id"
            ),

        "base_model":
            trajectory.get(
                "base_model"
            ),

        "instruction_tool_failure":
            instruction_tool_failure,

        "failure_reasons":
            failure_reasons,

        "n_tool_calls":
            n_tool_calls,

        "max_tool_calls":
            max_tool_calls,

        "exceeded_max_tool_calls":
            exceeded_max_tool_calls,

        "n_unknown_tool_calls":
            len(
                unknown_tools
            ),

        "n_non_object_tool_inputs":
            len(
                non_object_tool_inputs
            ),

        "n_invalid_tool_inputs":
            len(
                schema_invalid_tool_inputs
            ),

        "n_malformed_tool_calls":
            len(
                malformed_calls
            ),

        "n_tool_execution_errors":
            len(
                execution_errors
            ),

        "final_response_valid":
            final_validation[
                "valid"
            ],

        "final_response_errors":
            final_validation[
                "errors"
            ],

        # ---------------------------------------------
        # Audit details
        # ---------------------------------------------

        "unknown_tool_calls":
            unknown_tools,

        "non_object_tool_inputs":
            non_object_tool_inputs,

        "invalid_tool_inputs":
            schema_invalid_tool_inputs,

        "malformed_tool_calls":
            malformed_calls,

        "tool_execution_errors":
            execution_errors,

        "parsed_tool_calls":
            tool_calls,
    }


# =========================================================
# File interface
# =========================================================

def evaluate_trajectory_file(
    trajectory_path: Path,
    output_path: Optional[Path] = None,
    max_tool_calls: int = 40,
) -> Dict[str, Any]:
    trajectory_path = Path(
        trajectory_path
    )

    trajectory = _load_json(
        trajectory_path
    )

    result = (
        evaluate_instruction_tool(
            trajectory=trajectory,
            max_tool_calls=max_tool_calls,
        )
    )

    if output_path is not None:
        _save_json(
            result,
            Path(
                output_path
            ),
        )

    return result


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate instruction/tool failures for one "
            "Clinical HyperAgent Experiment 1 trajectory."
        )
    )

    parser.add_argument(
        "trajectory",
        help=(
            "Trajectory JSON generated by run_exp1.py"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=40,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    trajectory_path = Path(
        args.trajectory
    )

    if args.output:
        output_path = Path(
            args.output
        )

    else:
        agent_id = (
            trajectory_path
            .parent
            .name
        )

        case_id = (
            trajectory_path
            .stem
        )

        output_path = (
            PROJECT_ROOT
            / "outputs"
            / "exp1"
            / "instruction_tool_eval"
            / agent_id
            / f"{case_id}.json"
        )

    result = (
        evaluate_trajectory_file(
            trajectory_path=trajectory_path,
            output_path=output_path,
            max_tool_calls=args.max_tool_calls,
        )
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()