import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional


CLINICAL_TOOLS = [
    "get_history",
    "get_physical_exam",
    "get_lab_results",
    "get_imaging_results",
    "get_microbiology_results",
]


# =========================================================
# Basic helpers
# =========================================================

def _get_text(message: Dict[str, Any]) -> str:
    """
    Support both HyperAgent history formats:

        {"role": "...", "text": "..."}
        {"role": "...", "content": "..."}
    """

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


# =========================================================
# Tool call parsing
# =========================================================

def extract_tool_calls(
    msg_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Parse tool calls from the prompt-based tool protocol used
    by agent/llm_withtools.py.

    Expected model output may contain:

    <json>
    {
        "tool_name": "get_lab_results",
        "tool_input": {}
    }
    </json>

    Returns:
        [
            {
                "message_index": 3,
                "tool_name": "get_lab_results",
                "tool_input": {},
                "valid": True,
            },
            ...
        ]
    """

    calls = []

    for message_index, message in enumerate(
        msg_history
    ):
        # Only assistant messages can initiate tool calls.
        # Tool-call JSON may be repeated inside later user/tool
        # context, so counting all messages would double count.
        if _get_role(message) != "assistant":
            continue
    
        text = _get_text(message)
    
        if "tool_name" not in text:
            continue

        # ---------------------------------------------
        # First try extracting JSON objects.
        # ---------------------------------------------

        json_candidates = re.findall(
            r"\{.*?\}",
            text,
            flags=re.DOTALL,
        )

        found_valid_json = False

        for candidate in json_candidates:
            try:
                obj = json.loads(candidate)
            except Exception:
                continue

            if not isinstance(obj, dict):
                continue

            if "tool_name" not in obj:
                continue

            found_valid_json = True

            tool_name = str(
                obj.get(
                    "tool_name",
                    "",
                )
            ).strip()

            tool_input = obj.get(
                "tool_input",
                {},
            )

            valid = (
                tool_name
                in CLINICAL_TOOLS
            )

            calls.append(
                {
                    "message_index":
                        message_index,

                    "tool_name":
                        tool_name,

                    "tool_input":
                        tool_input,

                    "valid":
                        valid,
                }
            )

        # ---------------------------------------------
        # Fallback regex when JSON is malformed.
        # ---------------------------------------------

        if not found_valid_json:
            matches = re.findall(
                r'"tool_name"\s*:\s*"([^"]+)"',
                text,
            )

            for tool_name in matches:
                tool_name = tool_name.strip()

                calls.append(
                    {
                        "message_index":
                            message_index,

                        "tool_name":
                            tool_name,

                        "tool_input":
                            None,

                        "valid":
                            tool_name
                            in CLINICAL_TOOLS,
                    }
                )

    return calls


# =========================================================
# Repeated calls
# =========================================================

def count_repeated_calls(
    tool_calls: List[Dict[str, Any]],
) -> int:
    """
    Counts every call after the first call to the same tool.

    Example:

        history
        labs
        labs
        imaging
        labs

    repeated = 2
    """

    counts = Counter(
        call["tool_name"]
        for call in tool_calls
        if call.get("valid")
    )

    return sum(
        max(
            count - 1,
            0,
        )
        for count
        in counts.values()
    )


# =========================================================
# Assistant rounds
# =========================================================

def count_rounds(
    msg_history: List[Dict[str, Any]],
) -> int:
    """
    Number of assistant responses.

    This is used as an approximate interaction depth.
    """

    return sum(
        1
        for message in msg_history
        if _get_role(message)
        == "assistant"
    )


# =========================================================
# Final diagnosis round
# =========================================================

def find_final_diagnosis_round(
    msg_history: List[Dict[str, Any]],
) -> Optional[int]:
    """
    Find the assistant round where the final structured
    diagnosis first appears.

    Expected final format:

    {
        "response": {
            "diagnosis": "...",
            ...
        }
    }
    """

    assistant_round = 0

    for message in msg_history:
        if (
            _get_role(message)
            != "assistant"
        ):
            continue

        assistant_round += 1

        text = _get_text(
            message
        )

        if (
            '"diagnosis"' in text
            and '"response"' in text
        ):
            return assistant_round

    return None


# =========================================================
# Stop reason
# =========================================================

def extract_stop_reason(
    prediction: Any,
) -> str:
    if isinstance(
        prediction,
        dict,
    ):
        value = prediction.get(
            "stop_reason",
            "",
        )

        if value is None:
            return ""

        return str(
            value
        ).strip()

    return ""


# =========================================================
# Invalid calls
# =========================================================

def count_invalid_calls(
    tool_calls: List[Dict[str, Any]],
) -> int:
    return sum(
        1
        for call in tool_calls
        if not call.get(
            "valid",
            False,
        )
    )


# =========================================================
# Tool-specific counts
# =========================================================

def build_tool_counts(
    tool_calls: List[Dict[str, Any]],
) -> Dict[str, int]:

    valid_names = [
        call["tool_name"]
        for call in tool_calls
        if call.get(
            "valid",
            False,
        )
    ]

    counter = Counter(
        valid_names
    )

    known_total = sum(
        counter.get(
            tool,
            0,
        )
        for tool in CLINICAL_TOOLS
    )

    return {
        "n_history_calls":
            counter.get(
                "get_history",
                0,
            ),

        "n_exam_calls":
            counter.get(
                "get_physical_exam",
                0,
            ),

        "n_lab_calls":
            counter.get(
                "get_lab_results",
                0,
            ),

        "n_imaging_calls":
            counter.get(
                "get_imaging_results",
                0,
            ),

        "n_microbiology_calls":
            counter.get(
                "get_microbiology_results",
                0,
            ),

        "n_other_calls":
            max(
                len(valid_names)
                - known_total,
                0,
            ),
    }


# =========================================================
# Optional token extraction
# =========================================================

def extract_token_usage(
    trajectory: Dict[str, Any],
) -> Dict[str, Optional[int]]:
    """
    At the moment your llm.py returns info={} / limited metadata.

    This function is future-compatible. If later you save
    token usage into the trajectory, it will automatically
    extract it.

    Supported possible locations:

        trajectory["usage"]
        trajectory["llm_usage"]
        trajectory["metrics"]["usage"]
    """

    candidates = [
        trajectory.get(
            "usage"
        ),
        trajectory.get(
            "llm_usage"
        ),
        trajectory.get(
            "metrics",
            {},
        ).get(
            "usage"
        )
        if isinstance(
            trajectory.get(
                "metrics"
            ),
            dict,
        )
        else None,
    ]

    usage = None

    for candidate in candidates:
        if isinstance(
            candidate,
            dict,
        ):
            usage = candidate
            break

    if usage is None:
        return {
            "prompt_tokens":
                None,

            "completion_tokens":
                None,

            "total_tokens":
                None,
        }

    prompt_tokens = usage.get(
        "prompt_tokens",
        usage.get(
            "input_tokens"
        ),
    )

    completion_tokens = usage.get(
        "completion_tokens",
        usage.get(
            "output_tokens"
        ),
    )

    total_tokens = usage.get(
        "total_tokens"
    )

    if (
        total_tokens is None
        and prompt_tokens is not None
        and completion_tokens is not None
    ):
        total_tokens = (
            prompt_tokens
            + completion_tokens
        )

    return {
        "prompt_tokens":
            prompt_tokens,

        "completion_tokens":
            completion_tokens,

        "total_tokens":
            total_tokens,
    }


# =========================================================
# Main metric extraction
# =========================================================

def extract_trajectory_metrics(
    msg_history: List[Dict[str, Any]],
    prediction: Any = None,
    latency_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Main function to use from run_exp1.py.
    """

    if msg_history is None:
        msg_history = []

    tool_calls = extract_tool_calls(
        msg_history
    )

    valid_tool_calls = [
        call
        for call
        in tool_calls
        if call.get(
            "valid",
            False,
        )
    ]

    tool_counts = build_tool_counts(
        tool_calls
    )

    unique_tools = set(
        call["tool_name"]
        for call
        in valid_tool_calls
    )

    metrics = {
        # ---------------------------------------------
        # Conversation
        # ---------------------------------------------

        "n_messages":
            len(
                msg_history
            ),

        "n_rounds":
            count_rounds(
                msg_history
            ),

        # ---------------------------------------------
        # Tools
        # ---------------------------------------------

        "n_tool_calls":
            len(
                valid_tool_calls
            ),

        "n_invalid_calls":
            count_invalid_calls(
                tool_calls
            ),

        "n_repeated_calls":
            count_repeated_calls(
                tool_calls
            ),

        "unique_tools_used":
            len(
                unique_tools
            ),

        # ---------------------------------------------
        # Diagnosis process
        # ---------------------------------------------

        "time_to_diagnosis_round":
            find_final_diagnosis_round(
                msg_history
            ),

        "stop_reason":
            extract_stop_reason(
                prediction
            ),

        # ---------------------------------------------
        # Runtime
        # ---------------------------------------------

        "latency_seconds":
            latency_seconds,

        # ---------------------------------------------
        # Raw sequence for later analysis
        # ---------------------------------------------

        "tool_sequence": [
            call[
                "tool_name"
            ]
            for call
            in valid_tool_calls
        ],
    }

    metrics.update(
        tool_counts
    )

    return metrics


# =========================================================
# Full trajectory file
# =========================================================

def extract_metrics_from_trajectory(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Parse one saved run_exp1 trajectory JSON.
    """

    msg_history = (
        trajectory.get(
            "message_history",
            [],
        )
    )

    prediction = (
        trajectory.get(
            "prediction"
        )
    )

    latency = (
        trajectory.get(
            "latency_seconds"
        )
    )

    metrics = extract_trajectory_metrics(
        msg_history=
            msg_history,

        prediction=
            prediction,

        latency_seconds=
            latency,
    )

    token_usage = (
        extract_token_usage(
            trajectory
        )
    )

    metrics.update(
        token_usage
    )

    return metrics


# =========================================================
# Command-line test
# =========================================================

def main():
    """
    Usage:

        python -m clinevolve.endpoints.trajectory_metrics path/to/trajectory.json
    """

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "trajectory",
        help=(
            "Path to one trajectory JSON "
            "generated by run_exp1.py"
        ),
    )

    args = parser.parse_args()

    with open(
        args.trajectory,
        "r",
        encoding="utf-8",
    ) as f:
        trajectory = json.load(
            f
        )

    metrics = (
        extract_metrics_from_trajectory(
            trajectory
        )
    )

    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()