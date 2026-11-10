import backoff
import json
import os
from typing import Tuple

import litellm
import requests
from dotenv import load_dotenv


load_dotenv()

MAX_TOKENS = 16384

litellm.drop_params = True


# =========================================================
# Remote API models
# =========================================================

CLAUDE_MODEL = "anthropic/claude-sonnet-4-5-20250929"
CLAUDE_HAIKU_MODEL = "anthropic/claude-3-haiku-20240307"
CLAUDE_35NEW_MODEL = "anthropic/claude-3-5-sonnet-20241022"

OPENAI_MODEL = "openai/gpt-4o"
OPENAI_MINI_MODEL = "openai/gpt-4o-mini"
OPENAI_O3_MODEL = "openai/o3"
OPENAI_O3MINI_MODEL = "openai/o3-mini"
OPENAI_O4MINI_MODEL = "openai/o4-mini"
OPENAI_GPT52_MODEL = "openai/gpt-5.2"
OPENAI_GPT5_MODEL = "openai/gpt-5"
OPENAI_GPT5MINI_MODEL = "openai/gpt-5-mini"

GEMINI_3_MODEL = "gemini/gemini-3-pro-preview"
GEMINI_MODEL = "gemini/gemini-2.5-pro"
GEMINI_FLASH_MODEL = "gemini/gemini-2.5-flash"


# =========================================================
# Local models
#
# These names must match the --served-model-name passed
# to vLLM.
# =========================================================

GEMMA4_12B_IT_MODEL = "openai/gemma4-12b-it"

GLM_46V_FLASH_MODEL = "openai/glm-4.6v-flash"
GLM_47_FLASH_MODEL = "openai/glm-4.7-flash"

GPT_OSS_20B_MODEL = "openai/gpt-oss-20b"

MINISTRAL_3_14B_REASONING_MODEL = (
    "openai/ministral-3-14b-reasoning-2512"
)

QWEN3_8B_MODEL = "openai/qwen3-8b"
QWEN35_9B_MODEL = "openai/qwen3.5-9b"
QWEN35_27B_MODEL = "openai/qwen3.5-27b"


LOCAL_MODELS = {
    GEMMA4_12B_IT_MODEL,
    GLM_46V_FLASH_MODEL,
    GLM_47_FLASH_MODEL,
    GPT_OSS_20B_MODEL,
    MINISTRAL_3_14B_REASONING_MODEL,
    QWEN3_8B_MODEL,
    QWEN35_9B_MODEL,
    QWEN35_27B_MODEL,
}


# =========================================================
# Local server configuration
# =========================================================

# Default local vLLM endpoint.
#
# If only one model is running at a time, this is all you need.
#
# export VLLM_API_BASE=http://127.0.0.1:8000/v1
#
VLLM_API_BASE = os.environ.get(
    "VLLM_API_BASE",
    "http://127.0.0.1:8000/v1",
)

VLLM_API_KEY = os.environ.get(
    "VLLM_API_KEY",
    "dummy",
)


# =========================================================
# Optional: multiple local model servers
#
# Example:
#
# export LOCAL_MODEL_ENDPOINTS='{
#   "qwen3-8b": "http://127.0.0.1:8001/v1",
#   "qwen3.5-9b": "http://127.0.0.1:8002/v1"
# }'
#
# If not set, every local model uses VLLM_API_BASE.
# =========================================================

def _load_local_endpoints():
    raw = os.environ.get(
        "LOCAL_MODEL_ENDPOINTS",
        "",
    )

    if not raw:
        return {}

    try:
        endpoints = json.loads(raw)

        if not isinstance(endpoints, dict):
            raise ValueError(
                "LOCAL_MODEL_ENDPOINTS must be a JSON object."
            )

        return endpoints

    except Exception as e:
        raise ValueError(
            f"Invalid LOCAL_MODEL_ENDPOINTS: {e}"
        )


LOCAL_MODEL_ENDPOINTS = _load_local_endpoints()


# =========================================================
# Default model
# =========================================================

# Clinical experiment can override:
#
# export DEFAULT_LLM_MODEL="openai/qwen3.5-9b"
#
DEFAULT_MODEL = os.environ.get(
    "DEFAULT_LLM_MODEL",
    QWEN3_8B_MODEL,
)


# =========================================================
# Fail-fast context guard
#
# This guard does NOT truncate history, change max_tokens, or
# alter model outputs. It only avoids sending a request that
# is already guaranteed to exceed the vLLM context window.
#
# It is intentionally enabled only for GLM-4.7-Flash because
# that is the model currently served with a 98,304-token
# context window in Experiment 1.
# =========================================================

GLM47_CONTEXT_LIMIT = int(
    os.environ.get(
        "GLM47_CONTEXT_LIMIT",
        "98304",
    )
)

GLM47_TOKENIZER_PATH = os.environ.get(
    "GLM47_TOKENIZER_PATH",
    "zai-org/GLM-4.7-Flash",
)

_CONTEXT_GUARD_TOKENIZERS = {}


class ContextWindowExceededError(RuntimeError):
    """Local fail-fast equivalent of a server context overflow."""


def _uses_glm47_context_guard(model: str) -> bool:
    return (
        _get_served_model_name(model).lower()
        == "glm-4.7-flash"
    )


def _get_context_guard_tokenizer(model: str):
    """Load the local tokenizer lazily and cache it."""

    if not _uses_glm47_context_guard(model):
        return None

    cache_key = "glm-4.7-flash"

    if cache_key not in _CONTEXT_GUARD_TOKENIZERS:
        from transformers import AutoTokenizer

        _CONTEXT_GUARD_TOKENIZERS[cache_key] = (
            AutoTokenizer.from_pretrained(
                GLM47_TOKENIZER_PATH,
                trust_remote_code=True,
                local_files_only=True,
            )
        )

    return _CONTEXT_GUARD_TOKENIZERS[cache_key]


def _count_prompt_tokens_for_context_guard(
    model: str,
    messages,
):
    """
    Ask the running vLLM server to tokenize the exact chat request.

    This is more reliable than calling the local Hugging Face tokenizer
    directly because vLLM applies the same server-side chat template and
    default chat-template kwargs that will be used by /v1/chat/completions.

    Returns:
        (prompt_tokens, server_max_model_len)

    If the lightweight /tokenize preflight unexpectedly fails, return
    (None, None) and let vLLM perform the authoritative check during the
    real completion request. The guard therefore remains fail-open.
    """

    if not _uses_glm47_context_guard(model):
        return None, None

    api_base = _get_local_api_base(model).rstrip("/")

    if api_base.endswith("/v1"):
        server_root = api_base[:-3]
    else:
        server_root = api_base

    tokenize_url = f"{server_root}/tokenize"

    payload = {
        "model": _get_served_model_name(model),
        "messages": messages,
        "add_generation_prompt": True,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VLLM_API_KEY}",
    }

    try:
        response = requests.post(
            tokenize_url,
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        prompt_tokens = data.get("count")
        server_max_model_len = data.get("max_model_len")

        if prompt_tokens is None:
            return None, server_max_model_len

        return int(prompt_tokens), (
            int(server_max_model_len)
            if server_max_model_len is not None
            else None
        )

    except Exception as exc:
        print(
            "WARNING: vLLM /tokenize context preflight failed; "
            f"falling back to server completion check: {exc}"
        )
        return None, None


def _fail_fast_if_context_will_overflow(
    model: str,
    messages,
    max_tokens: int,
):
    """Reject only requests already guaranteed to exceed 98,304."""

    if not _uses_glm47_context_guard(model):
        return

    (
        prompt_tokens,
        server_max_model_len,
    ) = _count_prompt_tokens_for_context_guard(
        model,
        messages,
    )

    if prompt_tokens is None:
        return

    context_limit = (
        server_max_model_len
        if server_max_model_len is not None
        else GLM47_CONTEXT_LIMIT
    )

    requested_total = prompt_tokens + int(max_tokens)

    if requested_total > context_limit:
        raise ContextWindowExceededError(
            "Fail-fast context guard: "
            f"model={_get_served_model_name(model)}, "
            f"context_limit={context_limit}, "
            f"prompt_tokens={prompt_tokens}, "
            f"requested_output_tokens={int(max_tokens)}, "
            f"requested_total_tokens={requested_total}. "
            "Request was not sent to /v1/chat/completions."
        )


# =========================================================
# Helpers
# =========================================================
def _is_local_model(model: str) -> bool:
    served_name = _get_served_model_name(model)

    return (
        served_name in LOCAL_MODEL_ENDPOINTS
        or model.startswith("openai/")
    )


def _get_served_model_name(model: str) -> str:
    """
    LiteLLM model:
        openai/qwen3-8b

    vLLM served model:
        qwen3-8b
    """

    if model.startswith("openai/"):
        return model[len("openai/"):]

    return model


def _get_local_api_base(model: str) -> str:
    """
    Return model-specific endpoint when configured.

    Otherwise use the default VLLM endpoint.
    """

    served_name = _get_served_model_name(model)

    return LOCAL_MODEL_ENDPOINTS.get(
        served_name,
        VLLM_API_BASE,
    )


def _history_text_to_content(msg_history):
    """
    HyperAgent:

        {"role": "user", "text": "..."}

    -> LiteLLM:

        {"role": "user", "content": "..."}
    """

    converted = []

    for msg in msg_history:
        new_msg = dict(msg)

        if (
            "text" in new_msg
            and "content" not in new_msg
        ):
            new_msg["content"] = new_msg.pop("text")

        converted.append(new_msg)

    return converted


def _history_content_to_text(msg_history):
    """
    LiteLLM:

        {"role": "assistant", "content": "..."}

    -> HyperAgent:

        {"role": "assistant", "text": "..."}
    """

    converted = []

    for msg in msg_history:
        new_msg = dict(msg)

        if (
            "content" in new_msg
            and "text" not in new_msg
        ):
            new_msg["text"] = new_msg.pop("content")

        converted.append(new_msg)

    return converted


# =========================================================
# Main LLM function
# =========================================================

@backoff.on_exception(
    backoff.expo,
    (
        requests.exceptions.RequestException,
        json.JSONDecodeError,
        KeyError,
    ),
    max_time=600,
    max_value=60,
)
def get_response_from_llm(
    msg: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    msg_history=None,
) -> Tuple[str, list, dict]:

    if msg_history is None:
        msg_history = []

    # -----------------------------------------------------
    # Convert HyperAgent history to LiteLLM history
    # -----------------------------------------------------

    converted_history = _history_text_to_content(
        msg_history
    )

    new_msg_history = converted_history + [
        {
            "role": "user",
            "content": msg,
        }
    ]

    completion_kwargs = {
        "model": model,
        "messages": new_msg_history,
    }

    # =====================================================
    # Local models
    # =====================================================

    if _is_local_model(model):

        completion_kwargs["api_base"] = (
            _get_local_api_base(model)
        )

        completion_kwargs["api_key"] = (
            VLLM_API_KEY
        )

        completion_kwargs["temperature"] = temperature
        completion_kwargs["max_tokens"] = max_tokens

    # =====================================================
    # Remote models
    # =====================================================

    else:

        # GPT-5 / GPT-5-mini
        if model not in [
            OPENAI_GPT5_MODEL,
            OPENAI_GPT5MINI_MODEL,
        ]:
            completion_kwargs["temperature"] = temperature

        if "gpt-5" in model:
            completion_kwargs[
                "max_completion_tokens"
            ] = max_tokens

        elif "claude-3-haiku" in model:
            completion_kwargs["max_tokens"] = min(
                max_tokens,
                4096,
            )

        else:
            completion_kwargs["max_tokens"] = max_tokens

    # =====================================================
    # Fail fast on a guaranteed GLM-4.7 context overflow.
    #
    # Important: this does not truncate messages or reduce
    # max_tokens. Requests that fit are sent unchanged.
    # =====================================================

    _fail_fast_if_context_will_overflow(
        model=model,
        messages=new_msg_history,
        max_tokens=max_tokens,
    )

    # =====================================================
    # LLM inference
    # =====================================================

    response = litellm.completion(
        **completion_kwargs
    )

    choice = response["choices"][0]

    response_text = (
        choice["message"]["content"]
    )

    if response_text is None:
        response_text = ""

    # -----------------------------------------------------
    # Finish reason
    # -----------------------------------------------------

    finish_reason = choice.get(
        "finish_reason"
    )

    # -----------------------------------------------------
    # Token usage
    # -----------------------------------------------------

    usage = response.get(
        "usage"
    )

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    if usage is not None:
        try:
            prompt_tokens = usage.get(
                "prompt_tokens"
            )
            completion_tokens = usage.get(
                "completion_tokens"
            )
            total_tokens = usage.get(
                "total_tokens"
            )
        except Exception:
            pass

    # -----------------------------------------------------
    # Add assistant response
    # -----------------------------------------------------

    new_msg_history.append(
        {
            "role": "assistant",
            "content": response_text,
        }
    )

    # -----------------------------------------------------
    # Convert back to HyperAgent format
    # -----------------------------------------------------

    new_msg_history = _history_content_to_text(
        new_msg_history
    )

    info = {
        "model": model,

        "local": _is_local_model(
            model
        ),

        "served_model": (
            _get_served_model_name(model)
            if _is_local_model(model)
            else model
        ),

        "api_base": (
            _get_local_api_base(model)
            if _is_local_model(model)
            else None
        ),

        "finish_reason":
            finish_reason,

        "prompt_tokens":
            prompt_tokens,

        "completion_tokens":
            completion_tokens,

        "total_tokens":
            total_tokens,

        "requested_max_tokens":
            max_tokens,

        "temperature":
            temperature,
    }

    return (
        response_text,
        new_msg_history,
        info,
    )

# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("HyperAgent LLM Test")
    print("=" * 70)

    print(f"Model: {DEFAULT_MODEL}")

    if _is_local_model(DEFAULT_MODEL):
        print(
            f"Served model: "
            f"{_get_served_model_name(DEFAULT_MODEL)}"
        )

        print(
            f"API base: "
            f"{_get_local_api_base(DEFAULT_MODEL)}"
        )

    print("=" * 70)

    try:

        output_msg, msg_history, info = (
            get_response_from_llm(
                "Hello. Briefly introduce yourself.",
                model=DEFAULT_MODEL,
                max_tokens=512,
            )
        )

        print("\nOK")
        print("\nResponse:")
        print(output_msg)

        print("\nInfo:")
        print(
            json.dumps(
                info,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as e:

        print("\nFAIL")
        print(repr(e))