import os
import pandas as pd

from agent.clinical_context import get_current_hadm_id
from agent.paths import mimic_cdm_dir


DATA_PATH = os.environ.get(
    "MIMIC_CDM_HPI_PATH",
    str(mimic_cdm_dir() / "history_of_present_illness.csv"),
)

_df = None


def _load_data():
    global _df

    if _df is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"history_of_present_illness.csv not found: {DATA_PATH}"
            )

        _df = pd.read_csv(DATA_PATH)

        required_columns = {"hadm_id", "hpi"}
        missing = required_columns - set(_df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns in HPI file: {sorted(missing)}"
            )

        _df["hadm_id"] = _df["hadm_id"].astype(str)

    return _df


def tool_info():
    return {
        "name": "get_history",
        "description": (
            "Retrieve the history of present illness for the current patient."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


def tool_function():
    hadm_id = get_current_hadm_id()
    df = _load_data()

    rows = df[df["hadm_id"] == str(hadm_id)]

    if rows.empty:
        return "No history of present illness is available for the current patient."

    histories = (
        rows["hpi"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    histories = [x for x in histories if x]

    if not histories:
        return "No history of present illness is available for the current patient."

    return "\n\n".join(histories)