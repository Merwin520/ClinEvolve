from contextvars import ContextVar
from typing import Optional


# Thread/task-local current patient context.
# ContextVar is safer than a global variable when evaluations run concurrently.
_current_hadm_id: ContextVar[Optional[str]] = ContextVar(
    "current_hadm_id",
    default=None,
)


def set_current_hadm_id(hadm_id) -> None:
    """
    Set the current patient admission ID for the active evaluation context.

    Args:
        hadm_id:
            MIMIC-CDM admission ID. It is normalized to string internally.

    Notes:
        This should be called by the evaluation harness before TaskAgent.forward().
        The LLM itself should never be allowed to set this value.
    """
    if hadm_id is None:
        raise ValueError("hadm_id cannot be None")

    _current_hadm_id.set(str(hadm_id))


def get_current_hadm_id() -> str:
    """
    Return the current patient admission ID.

    Raises:
        RuntimeError:
            If no patient context has been initialized.
    """
    hadm_id = _current_hadm_id.get()

    if hadm_id is None:
        raise RuntimeError(
            "Clinical patient context has not been initialized. "
            "Call set_current_hadm_id(hadm_id) before using clinical tools."
        )

    return hadm_id


def clear_current_hadm_id() -> None:
    """
    Clear the current patient context.

    This should normally be called after one patient episode finishes.
    """
    _current_hadm_id.set(None)


def has_current_hadm_id() -> bool:
    """
    Return True if a current patient context has been initialized.
    """
    return _current_hadm_id.get() is not None