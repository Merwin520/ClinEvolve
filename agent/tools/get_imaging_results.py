import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from agent.clinical_context import get_current_hadm_id
from agent.paths import mimic_cdm_dir


# ============================================================
# Paths
# ============================================================

MIMIC_ROOT = mimic_cdm_dir()
RADIOLOGY_PATH = MIMIC_ROOT / "radiology_reports.csv"


# ============================================================
# HyperAgents tool interface
# ============================================================

def tool_info():
    return {
        "name": "get_imaging_results",
        "description": (
            "Request imaging reports for the current patient by modality "
            "and anatomical region. Only reports matching the requested "
            "modality and region are returned. A result with status "
            "'no_matching_report' means no matching report was found; "
            "it does NOT mean the imaging study was normal or negative. "
            "Examples: modality='CT', region='Abdomen'; "
            "modality='CT', region='Abdomen/Pelvis'; "
            "modality='Ultrasound', region='Pelvis'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "modality": {
                    "type": "string",
                    "description": (
                        "Requested imaging modality, for example "
                        "'CT', 'Ultrasound', 'MRI', 'X-ray', or 'HIDA'."
                    ),
                },
                "region": {
                    "type": "string",
                    "description": (
                        "Requested anatomical region, for example "
                        "'Abdomen', 'Pelvis', or 'Abdomen/Pelvis'."
                    ),
                },
            },
            "required": [
                "modality",
                "region",
            ],
        },
    }


# ============================================================
# Cache
# ============================================================

_RADIOLOGY: Optional[pd.DataFrame] = None


# ============================================================
# Normalization
# ============================================================

def _normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _clean_value(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


# ============================================================
# Modality aliases
# ============================================================

MODALITY_ALIASES = {
    "ct": [
        "ct",
        "computed tomography",
        "cat scan",
    ],

    "computed tomography": [
        "ct",
        "computed tomography",
        "cat scan",
    ],

    "ultrasound": [
        "ultrasound",
        "us",
        "sonogram",
        "sonography",
    ],

    "us": [
        "ultrasound",
        "us",
        "sonogram",
        "sonography",
    ],

    "mri": [
        "mri",
        "mr",
        "magnetic resonance",
    ],

    "mr": [
        "mri",
        "mr",
        "magnetic resonance",
    ],

    "x ray": [
        "x ray",
        "xray",
        "radiograph",
        "radiography",
    ],

    "xray": [
        "x ray",
        "xray",
        "radiograph",
        "radiography",
    ],

    "hida": [
        "hida",
        "hepatobiliary",
        "cholescintigraphy",
        "nuclear medicine",
    ],

    "eus": [
        "eus",
        "endoscopic ultrasound",
    ],
}


# ============================================================
# Region aliases
# ============================================================

REGION_ALIASES = {
    "abdomen": [
        "abdomen",
        "abdominal",
    ],

    "pelvis": [
        "pelvis",
        "pelvic",
    ],

    "abdomen pelvis": [
        "abdomen pelvis",
        "abdomen and pelvis",
        "abdominal pelvic",
        "abdominopelvic",
    ],

    "abdomen and pelvis": [
        "abdomen pelvis",
        "abdomen and pelvis",
        "abdominal pelvic",
        "abdominopelvic",
    ],

    "abdominopelvic": [
        "abdomen pelvis",
        "abdomen and pelvis",
        "abdominal pelvic",
        "abdominopelvic",
    ],

    "abdominal pelvic": [
        "abdomen pelvis",
        "abdomen and pelvis",
        "abdominal pelvic",
        "abdominopelvic",
    ],

    "right upper quadrant": [
        "right upper quadrant",
        "ruq",
    ],

    "ruq": [
        "right upper quadrant",
        "ruq",
    ],
}


# ============================================================
# Data loading
# ============================================================

def _load_data():
    global _RADIOLOGY

    if _RADIOLOGY is not None:
        return

    if not RADIOLOGY_PATH.exists():
        raise FileNotFoundError(
            f"Radiology file not found: {RADIOLOGY_PATH}"
        )

    df = pd.read_csv(
        RADIOLOGY_PATH,
        low_memory=False,
    )

    required_columns = {
        "hadm_id",
        "note_id",
        "modality",
        "region",
        "exam_name",
        "text",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "radiology_reports.csv missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    df = df.copy()

    df["hadm_id"] = (
        df["hadm_id"]
        .astype(str)
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    _RADIOLOGY = df


# ============================================================
# Matching helpers
# ============================================================

def _expand_modality_terms(
    modality: str,
) -> List[str]:
    normalized = _normalize_text(
        modality
    )

    terms = [normalized]

    for alias in MODALITY_ALIASES.get(
        normalized,
        [],
    ):
        alias_norm = _normalize_text(
            alias
        )

        if alias_norm:
            terms.append(
                alias_norm
            )

    return list(
        dict.fromkeys(
            terms
        )
    )


def _expand_region_terms(
    region: str,
) -> List[str]:
    normalized = _normalize_text(
        region
    )

    terms = [normalized]

    for alias in REGION_ALIASES.get(
        normalized,
        [],
    ):
        alias_norm = _normalize_text(
            alias
        )

        if alias_norm:
            terms.append(
                alias_norm
            )

    # Combined abdomen/pelvis requests should match reports
    # indexed as abdomen, pelvis, or combined abdomen/pelvis.
    if normalized in {
        "abdomen pelvis",
        "abdomen and pelvis",
        "abdominopelvic",
        "abdominal pelvic",
    }:
        terms.extend([
            "abdomen",
            "abdominal",
            "pelvis",
            "pelvic",
            "abdomen pelvis",
            "abdomen and pelvis",
            "abdominal pelvic",
            "abdominopelvic",
        ])

    return list(
        dict.fromkeys(
            terms
        )
    )


def _matches_any_term(
    value: Any,
    terms: List[str],
) -> bool:
    normalized_value = _normalize_text(
        value
    )

    if not normalized_value:
        return False

    for term in terms:
        if not term:
            continue

        if normalized_value == term:
            return True

        if term in normalized_value:
            return True

    return False


def _modality_matches(
    row: pd.Series,
    requested_modality: str,
) -> bool:
    terms = _expand_modality_terms(
        requested_modality
    )

    # Primary modality field.
    if _matches_any_term(
        row.get("modality"),
        terms,
    ):
        return True

    # Fallback to exam name.
    if _matches_any_term(
        row.get("exam_name"),
        terms,
    ):
        return True

    return False


def _region_matches(
    row: pd.Series,
    requested_region: str,
) -> bool:
    terms = _expand_region_terms(
        requested_region
    )

    # Primary region field.
    if _matches_any_term(
        row.get("region"),
        terms,
    ):
        return True

    # Fallback to exam name.
    if _matches_any_term(
        row.get("exam_name"),
        terms,
    ):
        return True

    return False


# ============================================================
# Output formatting
# ============================================================

def _build_record(
    row: pd.Series,
) -> Dict[str, Any]:
    return {
        "note_id": _clean_value(
            row.get("note_id")
        ),

        "modality": _clean_value(
            row.get("modality")
        ),

        "region": _clean_value(
            row.get("region")
        ),

        "exam_name": _clean_value(
            row.get("exam_name")
        ),

        "report": _clean_value(
            row.get("text")
        ),
    }


def _no_matching_report_response(
    modality: str,
    region: str,
    message: str,
) -> str:
    return json.dumps(
        {
            "status": "no_matching_report",
            "requested_modality": modality,
            "requested_region": region,
            "n_results": 0,
            "results": [],
            "message": message,
        },
        ensure_ascii=False,
    )


# ============================================================
# HyperAgents tool function
# ============================================================

def tool_function(
    modality: str,
    region: str,
):
    """
    Return only imaging reports matching the explicitly
    requested modality and anatomical region for the
    current patient.

    Important:
    no_matching_report does NOT mean that a study was
    performed and interpreted as normal.
    """
    try:
        _load_data()

        hadm_id = str(
            get_current_hadm_id()
        ).strip()

        if not hadm_id:
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        "No current patient context "
                        "has been set."
                    ),
                },
                ensure_ascii=False,
            )

        if modality is None:
            modality = ""

        if region is None:
            region = ""

        modality = str(
            modality
        ).strip()

        region = str(
            region
        ).strip()

        if not modality:
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        "A specific imaging modality "
                        "must be requested."
                    ),
                },
                ensure_ascii=False,
            )

        if not region:
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        "A specific anatomical region "
                        "must be requested."
                    ),
                },
                ensure_ascii=False,
            )

        assert _RADIOLOGY is not None

        patient_reports = _RADIOLOGY[
            _RADIOLOGY["hadm_id"] == hadm_id
        ].copy()

        # Patient has no radiology reports at all.
        if patient_reports.empty:
            return _no_matching_report_response(
                modality=modality,
                region=region,
                message=(
                    "No radiology report is available for the current "
                    "patient. This does NOT mean that the requested imaging "
                    "study was performed and normal or negative."
                ),
            )

        matched_rows = []

        for _, row in patient_reports.iterrows():
            modality_ok = _modality_matches(
                row,
                modality,
            )

            region_ok = _region_matches(
                row,
                region,
            )

            if (
                modality_ok
                and region_ok
            ):
                matched_rows.append(
                    row
                )

        records = [
            _build_record(row)
            for row in matched_rows
        ]

        # Remove exact duplicates while preserving order.
        deduplicated = []
        seen = set()

        for record in records:
            key = json.dumps(
                record,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )

            if key in seen:
                continue

            seen.add(key)
            deduplicated.append(
                record
            )

        # Patient has radiology data, but nothing matched the
        # requested modality + region.
        if not deduplicated:
            return _no_matching_report_response(
                modality=modality,
                region=region,
                message=(
                    "No radiology report matching the requested modality "
                    "and anatomical region was found. This does NOT mean "
                    "that the imaging study was normal or negative. "
                    "If clinically appropriate, consider requesting another "
                    "relevant modality or anatomical region."
                ),
            )

        return json.dumps(
            {
                "status": "ok",
                "requested_modality": modality,
                "requested_region": region,
                "n_results": len(
                    deduplicated
                ),
                "results": deduplicated,
            },
            ensure_ascii=False,
            default=str,
        )

    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error": str(e),
            },
            ensure_ascii=False,
        )


# ============================================================
# Manual smoke test
# ============================================================
