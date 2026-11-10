import os
import pandas as pd

from agent.clinical_context import get_current_hadm_id
from agent.paths import mimic_cdm_dir


DATA_PATH = os.environ.get(
    "MIMIC_CDM_PE_PATH",
   str(mimic_cdm_dir() / "physical_examination.csv"),
)

_df = None


def _load_data():
    global _df

    if _df is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"physical_examination.csv not found: {DATA_PATH}"
            )

        _df = pd.read_csv(DATA_PATH)

        required_columns = {"hadm_id", "pe"}
        missing = required_columns - set(_df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns in physical examination file: "
                f"{sorted(missing)}"
            )

        _df["hadm_id"] = _df["hadm_id"].astype(str)

    return _df


def tool_info():
    return {
        "name": "get_physical_exam",
        "description": (
            "Retrieve the physical examination findings for the current patient."
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
        return "No physical examination findings are available for the current patient."

    findings = (
        rows["pe"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    findings = [x for x in findings if x]

    if not findings:
        return "No physical examination findings are available for the current patient."

    return "\n\n".join(findings)