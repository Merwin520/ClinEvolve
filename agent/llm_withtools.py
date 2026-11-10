import inspect
import json
import re

from agent.llm import get_response_from_llm
from agent.tools import load_tools


def get_tooluse_prompt(tool_infos=None):
    """
    Build the prompt that describes available tools and the required
    tool-call format.
    """
    if tool_infos is None:
        tool_infos = []

    if not tool_infos:
        return ""

    tools_available = "\n\n".join(
        str(tool_info)
        for tool_info in tool_infos
    )

    tooluse_prompt = f"""
Here are the available tools:

{tools_available}

Use only one tool at a time, if needed, in exactly this format:

<json>
{{
    "tool_name": "...",
    "tool_input": {{
        ...
    }}
}}
</json>

IMPORTANT RULES:

1. ONLY USE ONE TOOL PER RESPONSE.

2. STRICTLY FOLLOW THE TOOL_NAME AND TOOL_INPUT FORMAT.

3. tool_input MUST be a JSON object.

4. Only use tools listed above.

5. Do not invent tool outputs.

6. After receiving a tool result, decide whether another tool is needed.

7. When sufficient evidence is available, stop calling tools and return
   the final answer requested by the task prompt.

8. Do not repeatedly deliberate about which tool to use. Once you decide
   another tool is needed, immediately output the tool call.
"""

    return tooluse_prompt.strip()


def should_retry_tool_use(
    response,
    tool_uses=None,
    info=None,
):
    """
    Retry when the model output was truncated before a complete
    tool call or final response could be produced.
    """
    if tool_uses:
        return False

    if info is None:
        info = {}

    finish_reason = info.get("finish_reason")

    if finish_reason == "length":
        return True

    if response is None:
        return False

    response = str(response)

    json_pos = response.find("<json>")
    tool_name_pos = response.find("tool_name")
    tool_input_pos = response.find("tool_input")

    return (
        json_pos != -1
        and tool_name_pos != -1
        and tool_input_pos != -1
        and json_pos < tool_name_pos < tool_input_pos
    )


def check_for_tool_uses(response):
    """
    Extract valid tool calls from <json>...</json> blocks.

    Returns:
        list[dict] or None
    """
    if response is None:
        return None

    response = str(response)

    pattern = r"<json>\s*(\{.*?\})\s*</json>"
    matches = re.findall(
        pattern,
        response,
        flags=re.DOTALL,
    )

    tool_uses = []

    for match in matches:
        try:
            tool_use = json.loads(match)
        except json.JSONDecodeError:
            continue

        if not isinstance(tool_use, dict):
            continue

        if "tool_name" not in tool_use:
            continue

        if "tool_input" not in tool_use:
            continue

        tool_uses.append(tool_use)

    return tool_uses if tool_uses else None


def process_tool_call(
    tools_dict,
    tool_name,
    tool_input,
):
    """
    Execute one registered tool call.
    """
    try:
        if tool_name not in tools_dict:
            return f"Error: Tool '{tool_name}' not found"

        if not isinstance(tool_input, dict):
            return "Error: tool_input must be a JSON object"

        tool_function = tools_dict[tool_name]["function"]

        return tool_function(**tool_input)

    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"


def _call_llm(
    get_response_fn,
    msg,
    model,
    msg_history,
    temperature,
    max_tokens,
):
    """
    Call get_response_from_llm and pass temperature/max_tokens
    only if its current signature supports them.
    """
    signature = inspect.signature(get_response_fn)

    kwargs = {
        "msg": msg,
        "model": model,
        "msg_history": msg_history,
    }

    if "temperature" in signature.parameters:
        kwargs["temperature"] = temperature

    if "max_tokens" in signature.parameters:
        kwargs["max_tokens"] = max_tokens

    return get_response_fn(**kwargs)


def _make_tool_result_message(
    tool_name,
    tool_input,
    tool_output,
):
    """
    Build a valid JSON tool-result message.
    """
    payload = {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_output": tool_output,
    }

    return (
        "<json>\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n</json>"
    )


def _attach_llm_info(
    msg_history,
    info,
):
    """
    Attach LLM metadata to the most recent assistant message.
    """
    if not isinstance(info, dict):
        return

    llm_info = {
        "finish_reason":
            info.get("finish_reason"),

        "prompt_tokens":
            info.get("prompt_tokens"),

        "completion_tokens":
            info.get("completion_tokens"),

        "total_tokens":
            info.get("total_tokens"),
    }

    for message in reversed(msg_history):
        if not isinstance(message, dict):
            continue

        role = str(
            message.get("role", "")
        ).lower()

        if role == "assistant":
            message["llm_info"] = llm_info
            return


def chat_with_agent(
    msg,
    model="claude-4-sonnet-genai",
    msg_history=None,
    logging=print,
    tools_available=None,
    multiple_tool_calls=False,
    max_tool_calls=40,
    temperature=0.0,
    max_tokens=4096,
):
    """
    Run an LLM + tool interaction loop.

    Args:
        msg:
            Main task instruction.

        model:
            Model name passed to get_response_from_llm.

        msg_history:
            Existing conversation history.

        logging:
            Logging function.

        tools_available:
            List of tool module names.

        multiple_tool_calls:
            If False, only the first tool call in each assistant response
            is executed.

        max_tool_calls:
            Maximum number of tool executions.

        temperature:
            Sampling temperature.

        max_tokens:
            Maximum model output tokens.

    Returns:
        list:
            Updated full message history.
    """
    if tools_available is None:
        tools_available = []

    if msg_history is None:
        msg_history = []

    get_response_fn = get_response_from_llm
    new_msg_history = msg_history

    try:
        # -------------------------------------------------
        # Load available tools
        # -------------------------------------------------
        all_tools = load_tools(
            logging=logging,
            names=tools_available,
        )

        tools_dict = {
            tool["info"]["name"]: tool
            for tool in all_tools
        }

        tool_prompt = get_tooluse_prompt(
            [
                tool["info"]
                for tool in all_tools
            ]
        )

        system_msg = (
            tool_prompt + "\n\n"
            if tool_prompt
            else ""
        )

        num_tool_calls = 0

        # -------------------------------------------------
        # Initial LLM call
        # -------------------------------------------------
        logging(f"Input: {repr(msg)}")

        logging(
            "[PROMPT_DEBUG] "
            f"instruction_chars={len(str(msg))}, "
            f"tool_prompt_chars={len(system_msg)}, "
            f"final_initial_chars={len(system_msg + str(msg))}"
        )

        response, new_msg_history, info = _call_llm(
            get_response_fn=get_response_fn,
            msg=system_msg + msg,
            model=model,
            msg_history=new_msg_history,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        _attach_llm_info(
            new_msg_history,
            info,
        )

        logging(f"Output: {repr(response)}")

        # -------------------------------------------------
        # Tool interaction loop
        # -------------------------------------------------
        tool_uses = check_for_tool_uses(response)

        retry_tool_use = should_retry_tool_use(
            response,
            tool_uses,
            info,
        )

        while tool_uses or retry_tool_use:
            # ---------------------------------------------
            # Tool-call limit
            # ---------------------------------------------
            if (
                max_tool_calls > 0
                and num_tool_calls >= max_tool_calls
            ):
                logging(
                    "Error: Maximum number of tool calls reached."
                )
                break

            tool_msgs = []

            # ---------------------------------------------
            # Execute requested tool
            # ---------------------------------------------
            if tool_uses:
                if not multiple_tool_calls:
                    tool_uses = tool_uses[:1]

                for tool_use in tool_uses:
                    if (
                        max_tool_calls > 0
                        and num_tool_calls >= max_tool_calls
                    ):
                        logging(
                            "Error: Maximum number of tool calls reached."
                        )
                        break

                    tool_name = str(
                        tool_use.get(
                            "tool_name",
                            "",
                        )
                    ).strip()

                    tool_input = tool_use.get(
                        "tool_input"
                    )

                    tool_output = process_tool_call(
                        tools_dict=tools_dict,
                        tool_name=tool_name,
                        tool_input=tool_input,
                    )

                    num_tool_calls += 1

                    tool_msg = _make_tool_result_message(
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_output=tool_output,
                    )

                    logging(
                        f"Tool output: {repr(tool_msg)}"
                    )

                    tool_msgs.append(tool_msg)

            # ---------------------------------------------
            # Retry incomplete tool syntax
            # ---------------------------------------------
            if retry_tool_use:
                retry_message = (
                    "Your previous response was truncated before completing "
                    "the next action. "
                    "Do not repeat your reasoning. "
                    "Immediately output exactly one of the following:\n"
                    "1. One complete tool call in the required <json> format, "
                    "if more evidence is needed; or\n"
                    "2. The final response JSON in the required schema, "
                    "if no more evidence is needed."
                )

                logging(retry_message)
                tool_msgs.append(retry_message)

            if not tool_msgs:
                break

            # ---------------------------------------------
            # Follow-up LLM call
            # ---------------------------------------------
            response, new_msg_history, info = _call_llm(
                get_response_fn=get_response_fn,
                msg=system_msg + "\n\n".join(tool_msgs),
                model=model,
                msg_history=new_msg_history,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            _attach_llm_info(
                new_msg_history,
                info,
            )

            logging(f"Output: {repr(response)}")

            tool_uses = check_for_tool_uses(response)

            retry_tool_use = should_retry_tool_use(
                response,
                tool_uses,
                info,
            )

    except Exception as e:
        logging(f"Error: {str(e)}")
        raise

    return new_msg_history


if __name__ == "__main__":
    msg = "hello"

    new_msg_history = chat_with_agent(msg)

    print(new_msg_history)