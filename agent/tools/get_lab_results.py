import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from agent.clinical_context import get_current_hadm_id
from agent.paths import mimic_cdm_dir


# ============================================================
# Paths
# ============================================================

MIMIC_ROOT = mimic_cdm_dir()

LAB_PATH = MIMIC_ROOT / "laboratory_tests.csv"
LAB_MAPPING_PATH = MIMIC_ROOT / "lab_test_mapping.csv"


# ============================================================
# HyperAgents tool interface
# IMPORTANT:
# Your loader calls module.tool_info(), so this MUST be a function.
# ============================================================

def tool_info():
    return {
        "name": "get_lab_results",
        "description": (
            "Request specific laboratory tests for the current patient. "
            "Only explicitly requested laboratory tests are returned. "
            "Examples: ['WBC', 'Lipase'], ['AST', 'ALT', 'Bilirubin'], "
            "['CBC'], ['BUN', 'Creatinine']."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "test_names": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "minItems": 1,
                    "description": (
                        "Specific laboratory tests or common laboratory "
                        "panels to request."
                    ),
                }
            },
            "required": ["test_names"],
        },
    }


# ============================================================
# Cache
# ============================================================

_LABS: Optional[pd.DataFrame] = None
_ITEM_META: Optional[Dict[int, Dict[str, Any]]] = None


# ============================================================
# Aliases
# ============================================================

ALIASES = {
    "wbc": [
        "wbc",
        "wbc count",
        "white blood cell",
        "white blood cell count",
    ],
    "white blood cell count": [
        "wbc",
        "wbc count",
        "white blood cell",
        "white blood cell count",
    ],

    "hemoglobin": [
        "hemoglobin",
        "hgb",
    ],
    "hematocrit": [
        "hematocrit",
        "hct",
    ],
    "platelet": [
        "platelet",
        "platelet count",
    ],

    "lipase": [
        "lipase",
    ],
    "amylase": [
        "amylase",
    ],

    "ast": [
        "ast",
        "aspartate aminotransferase",
        "asparate aminotransferase",
        "sgot",
    ],
    "alt": [
        "alt",
        "alanine aminotransferase",
        "sgpt",
    ],

    "alkaline phosphatase": [
        "alkaline phosphatase",
        "alk phos",
    ],
    "alp": [
        "alkaline phosphatase",
        "alk phos",
    ],

    "bilirubin": [
        "bilirubin",
        "bilirubin total",
        "bilirubin direct",
        "total bilirubin",
        "direct bilirubin",
    ],

    "ggt": [
        "ggt",
        "gamma glutamyl",
        "gamma glutamyltransferase",
    ],

    "albumin": [
        "albumin",
    ],

    "bun": [
        "bun",
        "urea nitrogen",
        "blood urea nitrogen",
    ],
    "creatinine": [
        "creatinine",
    ],
    "sodium": [
        "sodium",
    ],
    "potassium": [
        "potassium",
    ],
    "chloride": [
        "chloride",
    ],
    "bicarbonate": [
        "bicarbonate",
        "total co2",
        "carbon dioxide",
    ],
    "calcium": [
        "calcium",
    ],
    "glucose": [
        "glucose",
    ],

    "triglycerides": [
        "triglyceride",
        "triglycerides",
    ],

    "crp": [
        "crp",
        "c reactive protein",
    ],

    "esr": [
        "esr",
        "erythrocyte sedimentation",
    ],

    "lactate": [
        "lactate",
        "lactic acid",
    ],
}


PANEL_ALIASES = {
    "cbc": [
        "wbc",
        "hemoglobin",
        "hematocrit",
        "platelet",
    ],
    "complete blood count": [
        "wbc",
        "hemoglobin",
        "hematocrit",
        "platelet",
    ],

    "lft": [
        "ast",
        "alt",
        "alkaline phosphatase",
        "bilirubin",
        "ggt",
        "albumin",
    ],
    "liver function tests": [
        "ast",
        "alt",
        "alkaline phosphatase",
        "bilirubin",
        "ggt",
        "albumin",
    ],
    "liver panel": [
        "ast",
        "alt",
        "alkaline phosphatase",
        "bilirubin",
        "ggt",
        "albumin",
    ],

    "pancreatic enzymes": [
        "lipase",
        "amylase",
    ],

    "bmp": [
        "sodium",
        "potassium",
        "chloride",
        "bicarbonate",
        "bun",
        "creatinine",
        "glucose",
        "calcium",
    ],
    "basic metabolic panel": [
        "sodium",
        "potassium",
        "chloride",
        "bicarbonate",
        "bun",
        "creatinine",
        "glucose",
        "calcium",
    ],
}


# ============================================================
# Helpers
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


def _normalize_itemid(value: Any) -> Optional[int]:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return int(float(value))
    except Exception:
        return None


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


def _parse_corresponding_ids(value: Any) -> List[int]:
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)

    else:
        text = str(value).strip()

        if not text:
            return []

        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None

        if isinstance(parsed, (list, tuple, set)):
            raw_values = list(parsed)

        elif parsed is not None:
            raw_values = [parsed]

        else:
            raw_values = re.findall(
                r"\d+(?:\.\d+)?",
                text,
            )

    output = []

    for raw in raw_values:
        itemid = _normalize_itemid(raw)

        if itemid is not None:
            output.append(itemid)

    return output


# ============================================================
# Load data
# ============================================================

def _load_data():
    global _LABS
    global _ITEM_META

    if (
        _LABS is not None
        and _ITEM_META is not None
    ):
        return

    if not LAB_PATH.exists():
        raise FileNotFoundError(
            f"Laboratory file not found: {LAB_PATH}"
        )

    if not LAB_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Lab mapping file not found: {LAB_MAPPING_PATH}"
        )

    labs = pd.read_csv(
        LAB_PATH,
        low_memory=False,
    )

    mapping = pd.read_csv(
        LAB_MAPPING_PATH,
        low_memory=False,
    )

    required_lab_columns = {
        "hadm_id",
        "itemid",
        "valuestr",
        "ref_range_lower",
        "ref_range_upper",
    }

    missing_lab_columns = (
        required_lab_columns
        - set(labs.columns)
    )

    if missing_lab_columns:
        raise ValueError(
            "laboratory_tests.csv missing columns: "
            + ", ".join(
                sorted(missing_lab_columns)
            )
        )

    required_mapping_columns = {
        "itemid",
        "label",
        "fluid",
        "category",
        "corresponding_ids",
    }

    missing_mapping_columns = (
        required_mapping_columns
        - set(mapping.columns)
    )

    if missing_mapping_columns:
        raise ValueError(
            "lab_test_mapping.csv missing columns: "
            + ", ".join(
                sorted(missing_mapping_columns)
            )
        )

    labs = labs.copy()

    labs["hadm_id"] = (
        labs["hadm_id"]
        .astype(str)
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    labs["_itemid_norm"] = (
        labs["itemid"]
        .apply(_normalize_itemid)
    )

    item_meta: Dict[int, Dict[str, Any]] = {}

    for _, row in mapping.iterrows():

        main_itemid = _normalize_itemid(
            row.get("itemid")
        )

        corresponding_ids = (
            _parse_corresponding_ids(
                row.get("corresponding_ids")
            )
        )

        metadata = {
            "label": _clean_value(
                row.get("label")
            ),
            "fluid": _clean_value(
                row.get("fluid")
            ),
            "category": _clean_value(
                row.get("category")
            ),
        }

        all_ids = []

        if main_itemid is not None:
            all_ids.append(main_itemid)

        all_ids.extend(
            corresponding_ids
        )

        for itemid in all_ids:
            if itemid not in item_meta:
                item_meta[itemid] = metadata.copy()

    _LABS = labs
    _ITEM_META = item_meta


# ============================================================
# Request expansion
# ============================================================

def _expand_request(requested: str) -> List[str]:
    normalized = _normalize_text(
        requested
    )

    if not normalized:
        return []

    if normalized in PANEL_ALIASES:
        return list(
            PANEL_ALIASES[normalized]
        )

    return [normalized]


def _search_terms_for_test(
    test_name: str,
) -> List[str]:

    normalized = _normalize_text(
        test_name
    )

    terms = [normalized]

    for alias in ALIASES.get(
        normalized,
        [],
    ):
        alias_norm = _normalize_text(alias)

        if alias_norm:
            terms.append(alias_norm)

    return list(
        dict.fromkeys(terms)
    )


# ============================================================
# Matching
# ============================================================

def _metadata_matches_test(
    metadata: Dict[str, Any],
    test_name: str,
) -> bool:

    label = _normalize_text(
        metadata.get("label")
    )

    if not label:
        return False

    search_terms = (
        _search_terms_for_test(
            test_name
        )
    )

    # Exact match first.
    for term in search_terms:
        if term == label:
            return True

    # One-way substring only.
    #
    # IMPORTANT:
    # Do NOT use "label in term".
    # Otherwise:
    #   label="blood"
    #   term="white blood cell"
    # incorrectly matches urine "Blood".
    for term in search_terms:

        if len(term) < 3:
            continue

        if term in label:
            return True

    return False


def _find_matching_itemids(
    test_name: str,
) -> Set[int]:

    assert _ITEM_META is not None

    normalized = _normalize_text(
        test_name
    )

    # Explicit itemid request.
    if re.fullmatch(
        r"\d+",
        normalized,
    ):
        itemid = _normalize_itemid(
            normalized
        )

        if (
            itemid is not None
            and itemid in _ITEM_META
        ):
            return {itemid}

    matches: Set[int] = set()

    for itemid, metadata in _ITEM_META.items():

        fluid = _normalize_text(
            metadata.get("fluid")
        )

        # For the current Exp1 diagnostic laboratory tool,
        # exclude urinalysis fields to prevent false matches
        # such as urine "Blood" when requesting WBC.
        if fluid == "urine":
            continue

        if _metadata_matches_test(
            metadata,
            normalized,
        ):
            matches.add(itemid)

    return matches


# ============================================================
# Format results
# ============================================================

def _build_result_record(
    row: pd.Series,
) -> Dict[str, Any]:

    assert _ITEM_META is not None

    itemid = row.get(
        "_itemid_norm"
    )

    metadata = _ITEM_META.get(
        itemid,
        {},
    )

    return {
        "itemid": itemid,
        "label": metadata.get("label"),
        "fluid": metadata.get("fluid"),
        "category": metadata.get("category"),
        "value": _clean_value(
            row.get("valuestr")
        ),
        "ref_range_lower": _clean_value(
            row.get("ref_range_lower")
        ),
        "ref_range_upper": _clean_value(
            row.get("ref_range_upper")
        ),
    }


# ============================================================
# HyperAgents tool function
# ============================================================

def tool_function(
    test_names: List[str],
):
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

        # Be tolerant if model provides a single string.
        if isinstance(
            test_names,
            str,
        ):
            test_names = [test_names]

        if not isinstance(
            test_names,
            list,
        ):
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        "test_names must be a "
                        "non-empty list of strings."
                    ),
                },
                ensure_ascii=False,
            )

        requested_tests = []

        for test in test_names:

            if test is None:
                continue

            test = str(test).strip()

            if test:
                requested_tests.append(test)

        if not requested_tests:
            return json.dumps(
                {
                    "status": "error",
                    "error": (
                        "At least one specific "
                        "laboratory test must be requested."
                    ),
                },
                ensure_ascii=False,
            )

        assert _LABS is not None
        assert _ITEM_META is not None

        patient_labs = _LABS[
            _LABS["hadm_id"]
            == hadm_id
        ].copy()

        if patient_labs.empty:
            return json.dumps(
                {
                    "status": "ok",
                    "requested_tests":
                        requested_tests,
                    "request_resolution": [],
                    "n_results": 0,
                    "results": [],
                    "unmatched_requests":
                        requested_tests,
                },
                ensure_ascii=False,
            )

        patient_itemids = set(
            patient_labs[
                "_itemid_norm"
            ]
            .dropna()
            .astype(int)
            .tolist()
        )

        all_matching_itemids: Set[int] = set()

        request_resolution = []
        unmatched_requests = []

        for original_request in requested_tests:

            expanded_tests = (
                _expand_request(
                    original_request
                )
            )

            request_itemids: Set[int] = set()

            for expanded_test in expanded_tests:

                itemids = (
                    _find_matching_itemids(
                        expanded_test
                    )
                )

                request_itemids.update(
                    itemids
                )

            available_itemids = (
                request_itemids
                & patient_itemids
            )

            matched_labels = []

            for itemid in sorted(
                available_itemids
            ):

                label = (
                    _ITEM_META
                    .get(
                        itemid,
                        {},
                    )
                    .get("label")
                )

                if label is not None:
                    matched_labels.append(
                        str(label)
                    )

            matched_labels = sorted(
                set(matched_labels)
            )

            if not available_itemids:
                unmatched_requests.append(
                    original_request
                )

            all_matching_itemids.update(
                available_itemids
            )

            request_resolution.append(
                {
                    "requested":
                        original_request,
                    "expanded_tests":
                        expanded_tests,
                    "matched_labels":
                        matched_labels,
                }
            )

        if not all_matching_itemids:

            return json.dumps(
                {
                    "status": "ok",
                    "requested_tests":
                        requested_tests,
                    "request_resolution":
                        request_resolution,
                    "n_results": 0,
                    "results": [],
                    "unmatched_requests":
                        unmatched_requests,
                },
                ensure_ascii=False,
            )

        selected = patient_labs[
            patient_labs[
                "_itemid_norm"
            ].isin(
                all_matching_itemids
            )
        ].copy()

        records = []

        for _, row in selected.iterrows():
            records.append(
                _build_result_record(row)
            )

        # Deduplicate identical rows.
        deduplicated_records = []
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
            deduplicated_records.append(
                record
            )

        return json.dumps(
            {
                "status": "ok",
                "requested_tests":
                    requested_tests,
                "request_resolution":
                    request_resolution,
                "n_results":
                    len(
                        deduplicated_records
                    ),
                "results":
                    deduplicated_records,
                "unmatched_requests":
                    unmatched_requests,
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
# Manual test
# ============================================================
