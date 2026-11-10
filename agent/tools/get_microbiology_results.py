import os
import pandas as pd

from agent.clinical_context import get_current_hadm_id
from agent.paths import mimic_cdm_dir


MICROBIOLOGY_DATA_PATH = str(mimic_cdm_dir() / "microbiology.csv")

_microbiology_df = None


def _load_data():
    global _microbiology_df

    if _microbiology_df is None:
        if not os.path.exists(MICROBIOLOGY_DATA_PATH):
            raise FileNotFoundError(
                f"microbiology.csv not found: {MICROBIOLOGY_DATA_PATH}"
            )

        _microbiology_df = pd.read_csv(MICROBIOLOGY_DATA_PATH)

        required_columns = {
            "hadm_id",
            "test_itemid",
            "valuestr",
            "spec_itemid",
        }

        missing = required_columns - set(_microbiology_df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns in microbiology.csv: "
                f"{sorted(missing)}"
            )

        _microbiology_df["hadm_id"] = (
            _microbiology_df["hadm_id"]
            .astype(str)
        )

    return _microbiology_df


def tool_info():
    return {
        "name": "get_microbiology_results",
        "description": (
            "Retrieve available microbiology test results for the current "
            "patient, including test ID, specimen ID, and reported result."
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

    rows = df[
        df["hadm_id"] == str(hadm_id)
    ]

    if rows.empty:
        return (
            "No microbiology results are available "
            "for the current patient."
        )

    output_lines = []

    for _, row in rows.iterrows():
        test_itemid = row["test_itemid"]
        spec_itemid = row["spec_itemid"]
        value = row["valuestr"]

        test_itemid = (
            str(int(test_itemid))
            if pd.notna(test_itemid)
            else "Unknown"
        )

        spec_itemid = (
            str(int(spec_itemid))
            if pd.notna(spec_itemid)
            else "Unknown"
        )

        value = (
            str(value).strip()
            if pd.notna(value)
            else "Not available"
        )

        output_lines.append(
            f"- Test ID: {test_itemid} | "
            f"Specimen ID: {spec_itemid} | "
            f"Result: {value}"
        )

    return "\n".join(output_lines)