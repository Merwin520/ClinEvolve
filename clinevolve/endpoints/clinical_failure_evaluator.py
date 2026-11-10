import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from agent.paths import mimic_cdm_dir


# =========================================================
# Project configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MIMIC_DIR = Path(
    os.environ.get(
        "MIMIC_CDM_DATA_DIR",
        str(mimic_cdm_dir()),
    )
)


# =========================================================
# Official MIMIC-CDM laboratory definitions
# =========================================================

INFLAMMATION_LAB_IDS = {
    51301,  # White Blood Cells
    51755,
    51300,  # WBC Count
    50889,  # C-Reactive Protein
    51652,
}

CHOLECYSTITIS_LIVER_IDS = {
    50861,  # ALT
    50878,  # AST
}

CHOLECYSTITIS_GALLBLADDER_IDS = {
    50883,  # Bilirubin
    50927,  # GGT
}

PANCREATITIS_PANCREAS_IDS = {
    50867,  # Amylase
    50956,  # Lipase
}

PANCREATITIS_SERIOUSNESS_IDS = {
    51480,
    50810,
    51221,
    51638,  # Hematocrit

    51006,
    52647,  # Urea nitrogen

    51000,  # Triglycerides

    50893,  # Calcium

    50824,
    52623,
    50983,  # Sodium

    52610,
    50971,
    50822,  # Potassium
}


REQUIRED_LAB_CATEGORIES = {
    "appendicitis": {
        "Inflammation":
            INFLAMMATION_LAB_IDS,
    },

    "diverticulitis": {
        "Inflammation":
            INFLAMMATION_LAB_IDS,
    },

    "cholecystitis": {
        "Inflammation":
            INFLAMMATION_LAB_IDS,

        "Liver":
            CHOLECYSTITIS_LIVER_IDS,

        "Gallbladder":
            CHOLECYSTITIS_GALLBLADDER_IDS,
    },

    "pancreatitis": {
        "Inflammation":
            INFLAMMATION_LAB_IDS,

        "Pancreas":
            PANCREATITIS_PANCREAS_IDS,

        "Seriousness":
            PANCREATITIS_SERIOUSNESS_IDS,
    },
}


# =========================================================
# Laboratory request aliases
#
# These bridge the current HyperAgent string-based interface
# to the official itemid/category definitions.
# =========================================================

LAB_CATEGORY_ALIASES = {
    "Inflammation": {
        "wbc",
        "wbc count",
        "white blood cell",
        "white blood cells",
        "white blood cell count",
        "crp",
        "c reactive protein",
        "c-reactive protein",
        "cbc",
        "complete blood count",
    },

    "Liver": {
        "alt",
        "alanine aminotransferase",
        "alanine transaminase",
        "ast",
        "aspartate aminotransferase",
        "asparate aminotransferase",
        "aspartate transaminase",
        "lft",
        "lfts",
        "liver function test",
        "liver function tests",
        "liver function panel",
        "cmp",
        "comprehensive metabolic panel",
    },

    "Gallbladder": {
        "bilirubin",
        "total bilirubin",
        "direct bilirubin",
        "bilirubin direct",
        "ggt",
        "gamma glutamyltransferase",
        "gamma-glutamyltransferase",
        "lft",
        "lfts",
        "liver function test",
        "liver function tests",
        "liver function panel",
        "cmp",
        "comprehensive metabolic panel",
    },

    "Pancreas": {
        "amylase",
        "serum amylase",
        "amylase serum",
        "lipase",
    },

    "Seriousness": {
        "hematocrit",
        "hct",
        "bun",
        "blood urea nitrogen",
        "urea nitrogen",
        "triglyceride",
        "triglycerides",
        "calcium",
        "sodium",
        "potassium",

        # Panels containing at least one official seriousness item.
        "cbc",
        "complete blood count",
        "bmp",
        "basic metabolic panel",
        "cmp",
        "comprehensive metabolic panel",
        "rfp",
        "renal function panel",
        "renal function test",
        "electrolyte panel",
        "electrolytes",
    },
}


# =========================================================
# Official imaging rules
# =========================================================

VALID_IMAGING = {
    "appendicitis": {
        ("Abdomen", "Ultrasound"),
        ("Abdomen", "CT"),
        ("Abdomen", "MRI"),
    },

    "cholecystitis": {
        ("Abdomen", "Ultrasound"),
        ("Abdomen", "HIDA"),
        ("Abdomen", "MRI"),
        ("Abdomen", "EUS"),
    },

    "diverticulitis": {
        ("Abdomen", "CT"),
        ("Abdomen", "Ultrasound"),
        ("Abdomen", "MRI"),
    },

    "pancreatitis": {
        ("Abdomen", "Ultrasound"),
        ("Abdomen", "CT"),
        ("Abdomen", "EUS"),
    },
}


# =========================================================
# Official conditional procedure mappings
# =========================================================

APPENDECTOMY_ICD9 = {
    "4701",
    "4709",
}

APPENDECTOMY_ICD10 = {
    "0DTJ4ZZ",
    "0DTJ0ZZ",
}

APPENDECTOMY_KEYWORDS = {
    "appendectomy",
}


CHOLECYSTECTOMY_ICD9 = {
    "5123",
    "5122",
    "5121",
    "5102",
    "5124",
    "5103",
}

CHOLECYSTECTOMY_ICD10 = {
    "0FB44ZZ",
    "0FB40ZZ",
}

CHOLECYSTECTOMY_KEYWORDS = {
    "cholecystectomy",
    "cholecystecotmy",
    "cholecsytectomy",
    "cholecystecomy",
    "cholecytectomy",
    "laparoscopic cholecystitis",
    "cholecyctectomy",
}


COLECTOMY_ICD9 = {
    "4575",
    "4576",
    "4863",
    "4562",
    "4542",
    "4579",
    "4531",
    "4530",
    "4533",
    "1739",
    "1734",
    "1736",
    "4572",
    "4573",
    "1733",
    "4541",
    "1732",
    "5459",
    "5451",
    "4574",
    "9624",
}

COLECTOMY_ICD10 = {
    "0DBM8ZZ",
    "0DBH8ZZ",
    "0DB90ZZ",
    "0DB94ZZ",
    "0DBB0ZZ",
    "0DTK4ZZ",
    "0DTM4ZZ",
    "0DTL4ZZ",
    "0DTN4ZZ",
    "0DBG4ZZ",
    "0DTH0ZZ",
    "0DTF0ZZ",
    "0DTN0ZZ",
    "0DB80ZZ",
    "0DBA0ZZ",
    "0DB84ZZ",
    "0DBB4ZZ",
    "0DTF4ZZ",
    "0DBH0ZZ",
    "0DTH4ZZ",
    "0DNN0ZZ",
    "0DBM4ZZ",
    "0DTM0ZZ",
    "0DNB0ZZ",
    "0DBM0ZZ",
    "0DBE0ZZ",
    "0DBG0ZZ",
    "0DNE0ZZ",
    "0DNE4ZZ",
    "0DN90ZZ",
    "0DNL4ZZ",
    "0DNA0ZZ",
    "0DBH4ZZ",
    "0DTK0ZZ",
    "0DTL0ZZ",
    "0DNB4ZZ",
    "0DNL0ZZ",
    "0DNN4ZZ",
    "0DBA8ZZ",
    "0DBA4ZZ",
    "0DNB7ZZ",
    "0DBE4ZZ",
    "0DTNFZZ",
    "0DNA4ZZ",
    "0DN94ZZ",
    "0DB98ZZ",
    "0DTH8ZZ",
    "0DTN7ZZ",
    "0DBB3ZZ",
    "0DBN0ZZ",
    "0DBN4ZZ",
    "0DT80ZZ",
}

COLECTOMY_KEYWORDS = {
    "low anterior resection",
    "colectomy",
    "colonic resection",
    "colostomy",
    "resection of rectosigmoid colon",
    "resection of sigmoid colon",
    "rectosigmoid resection",
    "resection of colon",
    "sigmoidectomy",
    "sigmoid resection",
    "small bowel resection",
}


DRAINAGE_ICD9 = {
    "5491",
}

DRAINAGE_ALL_ICD10 = {
    "0W2JX0Z",
    "0W9J30Z",
    "0W9J3ZZ",
    "0W9J3ZX",
    "0W9G30Z",
    "0W9G3ZZ",
    "0W9G3ZX",
}

DRAINAGE_PANCREATITIS_ICD10 = {
    "0F2BX0Z",
    "0F9430Z",
    "0F9G30Z",
    "0F998ZZ",
    "0F998ZX",
}

DRAINAGE_KEYWORDS = {
    "drain",
    "pigtail",
    "catheter",
    "aspiration",
}

DRAINAGE_LOCATIONS_DIVERTICULITIS = {
    "abscess",
    "abdom",
    "pelvic",
    "peritoneal",
    "pericolonic",
    "sigmoid",
    "diverticular",
    "pararectal",
}

DRAINAGE_LOCATIONS_PANCREATITIS = {
    "abscess",
    "abdom",
    "pelvic",
    "peritoneal",
    "pancrea",
    "gallbladder",
    "biliary",
    "bile duct",
    "perirectal",
}


ERCP_ICD9 = {
    "8766",
    "5181",
    "5184",
    "5221",
    "5187",
    "5293",
    "5188",
    "5110",
    "5185",
    "8754",
    "5114",
    "9705",
}

ERCP_ICD10 = {
    "0F798DZ",
    "0F798ZZ",
    "0F7D8DZ",
    "0FB98ZX",
    "0FBD8ZX",
    "0FC98ZZ",
    "0FD98ZX",
    "0FDG8ZX",
    "BF101ZZ",
    "BF10YZZ",
    "BF11YZZ",
    "BF141ZZ",
    "BF14YZZ",
    "0FJB8ZZ",
    "0FJD8ZZ",
}

ERCP_KEYWORDS = {
    "biliary stent",
    "biliary cannulation",
    "ercp",
    "endoscopic retrograde cholangiography",
    "endoscopic retrograde cholangiopancreatography",
    "cholangiogram",
    "cbd stent",
    "pancreatic stent",
    "sphincterotomy",
    "sphinctertomy",
}


# =========================================================
# Generic helpers
# =========================================================

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


def _get_text(
    message: Dict[str, Any],
) -> str:
    if "text" in message:
        value = message.get(
            "text",
            "",
        )
    else:
        value = message.get(
            "content",
            "",
        )

    if value is None:
        return ""

    return str(value)


def _role(
    message: Dict[str, Any],
) -> str:
    return str(
        message.get(
            "role",
            "",
        )
    ).lower()


def _normalize_text(
    value: Any,
) -> str:
    text = str(
        value or ""
    ).lower()

    text = text.replace(
        "_",
        " ",
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _normalize_code(
    value: Any,
) -> str:
    """
    Normalize ICD code formatting.

    Examples:
        47.01   -> 4701
        4701.0  -> 4701
        0DTJ4ZZ -> 0DTJ4ZZ
    """
    if value is None:
        return ""

    text = str(
        value
    ).strip().upper()

    if not text:
        return ""

    # pandas may convert numeric ICD9 values to 4701.0
    if re.fullmatch(
        r"\d+\.0",
        text,
    ):
        text = text[:-2]

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text,
    )

    return text


def _contains_any(
    text: str,
    keywords,
) -> bool:
    lowered = str(
        text or ""
    ).lower()

    return any(
        str(keyword).lower()
        in lowered
        for keyword in keywords
    )


def _positive_keyword(
    text: str,
    keyword: str,
) -> bool:
    """
    Lightweight deterministic approximation of the official
    keyword-positive check.

    It rejects a small set of explicit local negations.
    """
    text = str(
        text or ""
    )

    keyword_pattern = re.escape(
        keyword
    )

    matches = list(
        re.finditer(
            keyword_pattern,
            text,
            flags=re.IGNORECASE,
        )
    )

    if not matches:
        return False

    negations = {
        "no",
        "not",
        "without",
        "avoid",
        "avoiding",
        "defer",
        "deferred",
        "decline",
        "declined",
    }

    for match in matches:
        start = max(
            0,
            match.start() - 40,
        )

        prefix = (
            text[start:match.start()]
            .lower()
        )

        words = re.findall(
            r"[a-z]+",
            prefix,
        )

        local_words = words[-4:]

        if not any(
            word in negations
            for word in local_words
        ):
            return True

    return False


# =========================================================
# Tool-call parsing
# =========================================================

def _extract_json_blocks(
    text: str,
) -> List[str]:
    return re.findall(
        r"<json>\s*(.*?)\s*</json>",
        text,
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    )


def extract_tool_calls(
    msg_history: List[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    calls = []

    for message_index, message in enumerate(
        msg_history
    ):
        if _role(
            message
        ) != "assistant":
            continue

        text = _get_text(
            message
        )

        if "tool_name" not in text:
            continue

        blocks = _extract_json_blocks(
            text
        )

        for block in blocks:
            try:
                obj = json.loads(
                    block
                )
            except Exception:
                continue

            if not isinstance(
                obj,
                dict,
            ):
                continue

            tool_name = obj.get(
                "tool_name"
            )

            if not tool_name:
                continue

            calls.append(
                {
                    "message_index":
                        message_index,

                    "tool_name":
                        str(
                            tool_name
                        ),

                    "tool_input":
                        obj.get(
                            "tool_input",
                            {},
                        ),
                }
            )

    return calls


# =========================================================
# Tool-output parsing
# =========================================================

def extract_lab_outputs(
    msg_history: List[
        Dict[str, Any]
    ],
) -> List[Dict[str, Any]]:
    outputs = []

    for message_index, message in enumerate(
        msg_history
    ):
        if _role(
            message
        ) != "user":
            continue

        text = _get_text(
            message
        )

        if (
            "get_lab_results"
            not in text
        ):
            continue

        for block in _extract_json_blocks(
            text
        ):
            try:
                outer = json.loads(
                    block
                )
            except Exception:
                continue

            if not isinstance(
                outer,
                dict,
            ):
                continue

            if (
                outer.get(
                    "tool_name"
                )
                != "get_lab_results"
            ):
                continue

            tool_output = outer.get(
                "tool_output"
            )

            if isinstance(
                tool_output,
                str,
            ):
                try:
                    payload = json.loads(
                        tool_output
                    )
                except Exception:
                    continue

            elif isinstance(
                tool_output,
                dict,
            ):
                payload = tool_output

            else:
                continue

            outputs.append(
                {
                    "message_index":
                        message_index,

                    "payload":
                        payload,
                }
            )

    return outputs


# =========================================================
# Diagnostic process evaluator
# =========================================================

def _requested_lab_categories(
    tool_calls: List[
        Dict[str, Any]
    ],
) -> Tuple[
    set,
    Dict[str, List[str]],
]:
    categories = set()

    evidence = {}

    for call in tool_calls:
        if (
            call.get(
                "tool_name"
            )
            != "get_lab_results"
        ):
            continue

        tool_input = call.get(
            "tool_input",
            {},
        )

        if not isinstance(
            tool_input,
            dict,
        ):
            continue

        test_names = tool_input.get(
            "test_names",
            [],
        )

        if not isinstance(
            test_names,
            list,
        ):
            continue

        for requested_test in test_names:
            normalized = _normalize_text(
                requested_test
            )

            for category, aliases in (
                LAB_CATEGORY_ALIASES.items()
            ):
                normalized_aliases = {
                    _normalize_text(
                        alias
                    )
                    for alias in aliases
                }

                if normalized in (
                    normalized_aliases
                ):
                    categories.add(
                        category
                    )

                    evidence.setdefault(
                        category,
                        [],
                    ).append(
                        str(
                            requested_test
                        )
                    )

    return (
        categories,
        evidence,
    )


def _returned_lab_categories(
    lab_outputs: List[
        Dict[str, Any]
    ],
) -> Tuple[
    set,
    Dict[str, List[int]],
]:
    categories = set()

    evidence = {}

    all_category_ids = {}

    for disease_categories in (
        REQUIRED_LAB_CATEGORIES.values()
    ):
        for category, ids in (
            disease_categories.items()
        ):
            all_category_ids.setdefault(
                category,
                set(),
            ).update(
                ids
            )

    for output in lab_outputs:
        payload = output.get(
            "payload",
            {},
        )

        for result in payload.get(
            "results",
            [],
        ):
            try:
                itemid = int(
                    result.get(
                        "itemid"
                    )
                )
            except Exception:
                continue

            for category, ids in (
                all_category_ids.items()
            ):
                if itemid in ids:
                    categories.add(
                        category
                    )

                    evidence.setdefault(
                        category,
                        [],
                    ).append(
                        itemid
                    )

    return (
        categories,
        evidence,
    )


def _normalize_region(
    region: Any,
) -> str:
    normalized = _normalize_text(
        region
    )

    abdomen_terms = {
        "abdomen",
        "abdomen pelvis",
        "abd pelvis",
        "abdominal",
        "pelvis",
        "pelvic",
    }

    if normalized in abdomen_terms:
        return "Abdomen"

    if (
        "abdomen" in normalized
        or "abdominal" in normalized
        or "pelvis" in normalized
        or "pelvic" in normalized
    ):
        return "Abdomen"

    return str(
        region or ""
    ).strip()


def _normalize_modality(
    modality: Any,
) -> str:
    normalized = _normalize_text(
        modality
    )

    if (
        normalized == "ct"
        or normalized.startswith(
            "ct "
        )
    ):
        return "CT"

    if normalized in {
        "ultrasound",
        "us",
        "sonography",
        "sonogram",
    }:
        return "Ultrasound"

    if normalized in {
        "mri",
        "mr",
        "mrcp",
    }:
        return "MRI"

    if normalized in {
        "hida",
        "hida scan",
    }:
        return "HIDA"

    if normalized in {
        "eus",
        "endoscopic ultrasound",
    }:
        return "EUS"

    return str(
        modality or ""
    ).strip()


def evaluate_diagnostic_process(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    disease = str(
        trajectory.get(
            "disease",
            "",
        )
    ).strip().lower()

    msg_history = trajectory.get(
        "message_history",
        [],
    )

    if disease not in (
        REQUIRED_LAB_CATEGORIES
    ):
        return {
            "value": None,
            "applicable": False,
            "status": "unsupported_disease",
            "reason": (
                f"Unsupported disease: {disease}"
            ),
        }

    tool_calls = extract_tool_calls(
        msg_history
    )

    lab_outputs = extract_lab_outputs(
        msg_history
    )

    # -----------------------------------------------------
    # Physical examination
    #
    # Official evaluator gives full Physical Examination
    # score only when it is the first action.
    # -----------------------------------------------------

    first_tool = (
        tool_calls[0].get(
            "tool_name"
        )
        if tool_calls
        else None
    )

    physical_exam_first = (
        first_tool
        == "get_physical_exam"
    )

    # -----------------------------------------------------
    # Required laboratory categories
    # -----------------------------------------------------

    requested_categories, request_evidence = (
        _requested_lab_categories(
            tool_calls
        )
    )

    returned_categories, result_evidence = (
        _returned_lab_categories(
            lab_outputs
        )
    )

    required_categories = set(
        REQUIRED_LAB_CATEGORIES[
            disease
        ].keys()
    )
    
    covered_categories = (
        requested_categories
        | returned_categories
    ) & required_categories
    

    missing_lab_categories = sorted(
        required_categories
        - covered_categories
    )

    labs_complete = (
        len(
            missing_lab_categories
        )
        == 0
    )

    # -----------------------------------------------------
    # Disease-valid imaging
    # -----------------------------------------------------

    valid_imaging_calls = []

    imaging_calls = []

    for call in tool_calls:
        if (
            call.get(
                "tool_name"
            )
            != "get_imaging_results"
        ):
            continue

        tool_input = call.get(
            "tool_input",
            {},
        )

        if not isinstance(
            tool_input,
            dict,
        ):
            continue

        region = _normalize_region(
            tool_input.get(
                "region"
            )
        )

        modality = _normalize_modality(
            tool_input.get(
                "modality"
            )
        )

        record = {
            "message_index":
                call.get(
                    "message_index"
                ),

            "region":
                region,

            "modality":
                modality,
        }

        imaging_calls.append(
            record
        )

        if (
            region,
            modality,
        ) in VALID_IMAGING[
            disease
        ]:
            valid_imaging_calls.append(
                record
            )

    imaging_complete = bool(
        valid_imaging_calls
    )

    # -----------------------------------------------------
    # Final process failure
    # -----------------------------------------------------

    components = {
        "physical_exam_first":
            physical_exam_first,

        "required_labs_complete":
            labs_complete,

        "valid_imaging_requested":
            imaging_complete,
    }

    failure = int(
        not all(
            components.values()
        )
    )

    return {
        "value":
            failure,

        "applicable":
            True,

        "status":
            "evaluated",

        "components":
            components,

        "required_lab_categories":
            sorted(
                required_categories
            ),

        "covered_lab_categories":
            sorted(
                covered_categories
            ),

        "missing_lab_categories":
            missing_lab_categories,

        "lab_request_evidence":
            request_evidence,

        "lab_result_evidence":
            result_evidence,

        "imaging_calls":
            imaging_calls,

        "valid_imaging_calls":
            valid_imaging_calls,
    }


# =========================================================
# Laboratory interpretation evaluator
# =========================================================

def _numeric_value(
    value: Any,
) -> Optional[float]:
    if value is None:
        return None

    text = str(
        value
    ).strip()

    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        text,
    )

    if not match:
        return None

    try:
        return float(
            match.group(0)
        )
    except Exception:
        return None


def _reference_class(
    value: float,
    lower: float,
    upper: float,
) -> str:
    """
    Official MIMIC-CDM binning rule.
    """
    if value < lower:
        return "low"

    if value > upper:
        return "high"

    return "normal"


def _lab_aliases(
    label: str,
) -> List[str]:
    normalized = _normalize_text(
        label
    )

    aliases = {
        normalized,
    }

    if (
        "white blood" in normalized
        or normalized == "wbc count"
    ):
        aliases.update(
            {
                "wbc",
                "wbc count",
                "white blood cell",
                "white blood cells",
            }
        )

    if "urea nitrogen" in normalized:
        aliases.update(
            {
                "bun",
                "urea nitrogen",
            }
        )

    if "creatinine" in normalized:
        aliases.add(
            "creatinine"
        )

    if "platelet" in normalized:
        aliases.update(
            {
                "platelet",
                "platelets",
                "platelet count",
            }
        )

    if "hemoglobin" in normalized:
        aliases.update(
            {
                "hemoglobin",
                "hgb",
            }
        )

    if "hematocrit" in normalized:
        aliases.update(
            {
                "hematocrit",
                "hct",
            }
        )

    if "lipase" in normalized:
        aliases.add(
            "lipase"
        )

    if "amylase" in normalized:
        aliases.add(
            "amylase"
        )

    if "bilirubin" in normalized:
        aliases.add(
            "bilirubin"
        )

    if (
        "alanine aminotransferase"
        in normalized
    ):
        aliases.add(
            "alt"
        )

    if (
        "aspartate aminotransferase"
        in normalized
        or
        "asparate aminotransferase"
        in normalized
    ):
        aliases.add(
            "ast"
        )

    if "sodium" in normalized:
        aliases.add(
            "sodium"
        )

    if "potassium" in normalized:
        aliases.add(
            "potassium"
        )

    if "calcium" in normalized:
        aliases.add(
            "calcium"
        )

    if "triglyceride" in normalized:
        aliases.update(
            {
                "triglyceride",
                "triglycerides",
            }
        )

    return sorted(
        aliases,
        key=len,
        reverse=True,
    )


def _sentence_split(
    text: str,
) -> List[str]:
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+|\n+",
            text,
        )
        if sentence.strip()
    ]


def _find_lab_mentions(
    sentence: str,
    lab_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Find all laboratory-test mentions in one sentence.

    Each returned mention contains:
        result_index
        alias
        start
        end

    Mentions are sorted by character position.
    """
    normalized_sentence = _normalize_text(
        sentence
    )

    mentions = []

    for result_index, result in enumerate(
        lab_results
    ):
        aliases = result.get(
            "aliases",
            [],
        )

        for alias in aliases:
            alias_norm = _normalize_text(
                alias
            )

            if not alias_norm:
                continue

            pattern = (
                r"\b"
                + re.escape(alias_norm)
                + r"\b"
            )

            for match in re.finditer(
                pattern,
                normalized_sentence,
            ):
                mentions.append(
                    {
                        "result_index":
                            result_index,

                        "alias":
                            alias_norm,

                        "start":
                            match.start(),

                        "end":
                            match.end(),
                    }
                )

    mentions.sort(
        key=lambda x: (
            x["start"],
            -(x["end"] - x["start"]),
        )
    )

    # Remove overlapping duplicate aliases for the same lab.
    cleaned = []

    for mention in mentions:
        duplicate = False

        for existing in cleaned:
            if (
                existing[
                    "result_index"
                ]
                == mention[
                    "result_index"
                ]
                and not (
                    mention[
                        "end"
                    ]
                    <= existing[
                        "start"
                    ]
                    or
                    mention[
                        "start"
                    ]
                    >= existing[
                        "end"
                    ]
                )
            ):
                duplicate = True
                break

        if not duplicate:
            cleaned.append(
                mention
            )

    return cleaned


def _clause_for_lab(
    sentence: str,
    mention: Dict[str, Any],
    all_mentions: List[Dict[str, Any]],
) -> str:
    """
    Extract a local clause around one laboratory mention.

    The goal is to prevent interpretation words belonging to
    another laboratory test from leaking into this test.

    Example:

        "WBC is elevated, and lipase is within normal range"

    For lipase we should not see the WBC-specific "elevated".
    """
    normalized = _normalize_text(
        sentence
    )

    start = mention[
        "start"
    ]

    end = mention[
        "end"
    ]

    # -----------------------------------------------------
    # Boundaries from neighboring lab mentions
    # -----------------------------------------------------

    previous_end = 0
    next_start = len(
        normalized
    )

    for other in all_mentions:
        if other is mention:
            continue

        if (
            other[
                "end"
            ]
            <= start
        ):
            previous_end = max(
                previous_end,
                other[
                    "end"
                ],
            )

        elif (
            other[
                "start"
            ]
            >= end
        ):
            next_start = min(
                next_start,
                other[
                    "start"
                ],
            )

    # -----------------------------------------------------
    # Also use punctuation / conjunction boundaries.
    # -----------------------------------------------------

    left_window = normalized[
        previous_end:start
    ]

    right_window = normalized[
        end:next_start
    ]

    # Strong clause boundaries.
    separators = [
        ",",
        ";",
        " but ",
        " however ",
        " whereas ",
        " while ",
        " and ",
    ]

    left_boundary = previous_end

    for separator in separators:
        pos = left_window.rfind(
            separator.strip()
        )

        if pos >= 0:
            candidate = (
                previous_end
                + pos
                + len(
                    separator.strip()
                )
            )

            left_boundary = max(
                left_boundary,
                candidate,
            )

    right_boundary = next_start

    for separator in separators:
        token = separator.strip()

        pos = right_window.find(
            token
        )

        if pos >= 0:
            candidate = (
                end
                + pos
            )

            right_boundary = min(
                right_boundary,
                candidate,
            )

    # Keep enough left context for:
    #   "elevated WBC"
    # and enough right context for:
    #   "WBC is elevated"
    left_boundary = max(
        previous_end,
        min(
            left_boundary,
            start,
        ),
    )

    right_boundary = min(
        next_start,
        max(
            right_boundary,
            end,
        ),
    )

    clause = normalized[
        left_boundary:
        right_boundary
    ].strip()

    # Fallback if boundaries become too aggressive.
    if len(
        clause
    ) < len(
        mention[
            "alias"
        ]
    ):
        left = max(
            previous_end,
            start - 50,
        )

        right = min(
            next_start,
            end + 80,
        )

        clause = normalized[
            left:right
        ].strip()

    return clause


def _has_negated_direction(
    clause: str,
    direction_terms: List[str],
) -> bool:
    """
    Detect local negation of directional interpretation.

    Examples:
        isn't elevated
        is not elevated
        not high
        wasn't low
        no elevation
    """
    clause = _normalize_text(
        clause
    )

    negation_patterns = [
        r"\bnot\s+{}",
        r"\bisnt\s+{}",
        r"\bisn t\s+{}",
        r"\bwasnt\s+{}",
        r"\bwasn t\s+{}",
        r"\bwerent\s+{}",
        r"\bweren t\s+{}",
        r"\bwithout\s+{}",
        r"\bno\s+{}",
    ]

    for term in direction_terms:
        term_pattern = re.escape(
            _normalize_text(
                term
            )
        )

        for pattern in negation_patterns:
            if re.search(
                pattern.format(
                    term_pattern
                ),
                clause,
            ):
                return True

    return False

def _interpretation_claim(
    clause: str,
    alias: str,
) -> Optional[str]:
    """
    Infer explicit Low / Normal / High interpretation for one
    laboratory test from a clause localized to that test.

    Important:
    panel-level statements such as
        "CBC within normal range except WBC"
    must not be propagated to individual labs such as
    creatinine or WBC.
    """
    normalized = _normalize_text(
        clause
    )

    alias_norm = _normalize_text(
        alias
    )

    if (
        alias_norm
        and alias_norm not in normalized
    ):
        return None

    # =====================================================
    # Remove panel-level normal statements.
    #
    # Example:
    #
    #   "Creatinine 1.9, CBC within normal range except WBC"
    #
    # "within normal range" belongs to CBC, not creatinine.
    #
    # Likewise:
    #
    #   "CBC within normal range except WBC"
    #
    # does not mean WBC itself is normal.
    # =====================================================

    panel_normal_patterns = [
        r"\bcbc\s+(?:is\s+)?(?:mostly\s+)?within\s+(?:the\s+)?normal\s+range\b",
        r"\bcbc\s+(?:is\s+)?(?:mostly\s+)?within\s+(?:the\s+)?reference\s+range\b",
        r"\bcbc\s+(?:is\s+)?normal\b",
        r"\bnormal\s+cbc\b",
        r"\bcbc\s+otherwise\s+normal\b",
        r"\bcbc\s+other\s+values\s+normal\b",
        r"\bnormal\s+cbc\s+other\s+values\b",
    ]

    for pattern in panel_normal_patterns:
        normalized = re.sub(
            pattern,
            " ",
            normalized,
        )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    # =====================================================
    # Direction terms
    # =====================================================

    high_terms = [
        "elevated",
        "high",
        "increased",
        "above normal",
        "above the normal",
        "above reference",
        "above the reference",
        "above normal range",
        "above the normal range",
        "above reference range",
        "above the reference range",
    ]

    low_terms = [
        "low",
        "decreased",
        "reduced",
        "below normal",
        "below the normal",
        "below reference",
        "below the reference",
        "below normal range",
        "below the normal range",
        "below reference range",
        "below the reference range",
    ]

    normal_terms = [
        "normal",
        "within normal",
        "within the normal",
        "within reference",
        "within the reference",
        "within normal limits",
        "within normal range",
        "within the normal range",
        "within reference range",
        "within the reference range",
        "upper end of normal",
        "upper end of the normal",
        "upper end of the normal range",
        "lower end of normal",
        "lower end of the normal",
        "lower end of the normal range",
    ]

    # =====================================================
    # Clinical synonyms
    # =====================================================

    if alias_norm in {
        "wbc",
        "wbc count",
        "white blood cell",
        "white blood cells",
    }:
        if "leukocytosis" in normalized:
            return "high"

        if "leukopenia" in normalized:
            return "low"

    # =====================================================
    # Explicit negation
    #
    # "not elevated" / "isn't elevated" should not become
    # High merely because the word "elevated" is present.
    # =====================================================

    high_negated = _has_negated_direction(
        normalized,
        [
            "elevated",
            "high",
            "increased",
        ],
    )

    low_negated = _has_negated_direction(
        normalized,
        [
            "low",
            "decreased",
            "reduced",
        ],
    )

    if high_negated:
        return "normal"

    if low_negated:
        return "normal"

    # =====================================================
    # Explicit directional interpretation
    # =====================================================

    if any(
        term in normalized
        for term in high_terms
    ):
        return "high"

    if any(
        term in normalized
        for term in low_terms
    ):
        return "low"

    # =====================================================
    # Explicit normal interpretation
    # =====================================================

    if any(
        term in normalized
        for term in normal_terms
    ):
        return "normal"

    # No explicit interpretation for this individual lab.
    return None



def evaluate_lab_interpretation(
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    msg_history = trajectory.get(
        "message_history",
        [],
    )

    lab_outputs = extract_lab_outputs(
        msg_history
    )

    evaluable_results = []

    # =====================================================
    # Build evaluable lab reference set
    # =====================================================

    for output in lab_outputs:
        payload = output.get(
            "payload",
            {},
        )

        source_message_index = (
            output.get(
                "message_index"
            )
        )

        for result in payload.get(
            "results",
            [],
        ):
            value = _numeric_value(
                result.get(
                    "value"
                )
            )

            lower = _numeric_value(
                result.get(
                    "ref_range_lower"
                )
            )

            upper = _numeric_value(
                result.get(
                    "ref_range_upper"
                )
            )

            if (
                value is None
                or lower is None
                or upper is None
            ):
                continue

            label = str(
                result.get(
                    "label",
                    "",
                )
            )

            evaluable_results.append(
                {
                    "source_message_index":
                        source_message_index,

                    "itemid":
                        result.get(
                            "itemid"
                        ),

                    "label":
                        label,

                    "value":
                        value,

                    "ref_range_lower":
                        lower,

                    "ref_range_upper":
                        upper,

                    "reference_class":
                        _reference_class(
                            value,
                            lower,
                            upper,
                        ),

                    "aliases":
                        _lab_aliases(
                            label
                        ),
                }
            )

    if not evaluable_results:
        return {
            "value":
                None,

            "applicable":
                False,

            "status":
                "no_evaluable_labs",

            "n_evaluable_labs":
                0,

            "n_interpretation_claims":
                0,

            "n_incorrect_claims":
                0,

            "claims":
                [],
        }

    claims = []

    # =====================================================
    # Evaluate assistant statements after lab availability
    # =====================================================

    for message_index, message in enumerate(
        msg_history
    ):
        if _role(
            message
        ) != "assistant":
            continue

        text = _get_text(
            message
        )

        for sentence in _sentence_split(
            text
        ):
            # Only consider results already available before
            # this assistant message.
            available_results = [
                result
                for result
                in evaluable_results
                if (
                    result[
                        "source_message_index"
                    ]
                    < message_index
                )
            ]

            if not available_results:
                continue

            mentions = _find_lab_mentions(
                sentence,
                available_results,
            )

            if not mentions:
                continue

            for mention in mentions:
                result = available_results[
                    mention[
                        "result_index"
                    ]
                ]

                clause = _clause_for_lab(
                    sentence=sentence,
                    mention=mention,
                    all_mentions=mentions,
                )

                predicted_class = (
                    _interpretation_claim(
                        clause=clause,
                        alias=mention[
                            "alias"
                        ],
                    )
                )

                if predicted_class is None:
                    continue

                reference_class = result[
                    "reference_class"
                ]

                claims.append(
                    {
                        "message_index":
                            message_index,

                        "itemid":
                            result[
                                "itemid"
                            ],

                        "label":
                            result[
                                "label"
                            ],

                        "value":
                            result[
                                "value"
                            ],

                        "ref_range_lower":
                            result[
                                "ref_range_lower"
                            ],

                        "ref_range_upper":
                            result[
                                "ref_range_upper"
                            ],

                        "reference_class":
                            reference_class,

                        "agent_class":
                            predicted_class,

                        "correct":
                            (
                                predicted_class
                                == reference_class
                            ),

                        "alias":
                            mention[
                                "alias"
                            ],

                        "clause":
                            clause[:1000],

                        "sentence":
                            sentence[:1000],
                    }
                )

    # =====================================================
    # Deduplicate identical interpretations
    # =====================================================

    deduplicated_claims = []

    seen = set()

    for claim in claims:
        key = (
            claim[
                "message_index"
            ],
            claim[
                "itemid"
            ],
            claim[
                "agent_class"
            ],
            claim[
                "sentence"
            ],
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        deduplicated_claims.append(
            claim
        )

    claims = deduplicated_claims

    # =====================================================
    # Applicability
    # =====================================================

    if not claims:
        return {
            "value":
                None,

            "applicable":
                False,

            "status":
                "no_explicit_interpretation",

            "n_evaluable_labs":
                len(
                    evaluable_results
                ),

            "n_interpretation_claims":
                0,

            "n_incorrect_claims":
                0,

            "claims":
                [],
        }

    incorrect_claims = [
        claim
        for claim in claims
        if not claim[
            "correct"
        ]
    ]

    failure = int(
        bool(
            incorrect_claims
        )
    )

    return {
        "value":
            failure,

        "applicable":
            True,

        "status":
            "evaluated",

        "n_evaluable_labs":
            len(
                evaluable_results
            ),

        "n_interpretation_claims":
            len(
                claims
            ),

        "n_incorrect_claims":
            len(
                incorrect_claims
            ),

        "claims":
            claims,

        "incorrect_claims":
            incorrect_claims,
    }

# =========================================================
# MIMIC ground-truth loader for conditional treatment
# =========================================================

def _find_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    lower_map = {
        str(column).lower():
            column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[
                candidate.lower()
            ]

    return None


class ClinicalGroundTruth:
    def __init__(
        self,
        data_dir: Path = DEFAULT_MIMIC_DIR,
    ):
        self.data_dir = Path(
            data_dir
        )

        self.icd_procedures = self._read_optional(
            "icd_procedures.csv"
        )

        self.discharge_procedures = self._read_optional(
            "discharge_procedures.csv"
        )

        self.icd_diagnosis = self._read_optional(
            "icd_diagnosis.csv"
        )

        self.discharge_diagnosis = self._read_optional(
            "discharge_diagnosis.csv"
        )

    def _read_optional(
        self,
        filename: str,
    ) -> Optional[pd.DataFrame]:
        path = (
            self.data_dir
            / filename
        )

        if not path.exists():
            return None

        return pd.read_csv(
            path,
            low_memory=False,
        )

    @staticmethod
    def _case_rows(
        df: Optional[
            pd.DataFrame
        ],
        case_id: str,
    ) -> Optional[
        pd.DataFrame
    ]:
        if df is None:
            return None

        hadm_col = _find_column(
            df,
            [
                "hadm_id",
                "case_id",
            ],
        )

        if hadm_col is None:
            return None

        mask = (
            df[
                hadm_col
            ]
            .astype(str)
            .str.replace(
                r"\.0$",
                "",
                regex=True,
            )
            == str(
                case_id
            )
        )

        return df.loc[
            mask
        ]

    def get_procedure_reference(
        self,
        case_id: str,
    ) -> Dict[str, Any]:
        icd9 = []
        icd10 = []
        discharge = []

        # ---------------------------------------------
        # ICD procedures
        # ---------------------------------------------

        rows = self._case_rows(
            self.icd_procedures,
            case_id,
        )

        if (
            rows is not None
            and not rows.empty
        ):
            code_col = _find_column(
                rows,
                [
                    "icd_code",
                    "procedure_code",
                    "code",
                ],
            )

            version_col = _find_column(
                rows,
                [
                    "icd_version",
                    "version",
                ],
            )

            if code_col is not None:
                for _, row in rows.iterrows():
                    code = _normalize_code(
                        row[
                            code_col
                        ]
                    )

                    if not code:
                        continue

                    version = None

                    if version_col is not None:
                        try:
                            version = int(
                                float(
                                    row[
                                        version_col
                                    ]
                                )
                            )
                        except Exception:
                            version = None

                    if version == 9:
                        icd9.append(
                            code
                        )

                    elif version == 10:
                        icd10.append(
                            code
                        )

                    else:
                        # Fallback based on alphanumeric code form.
                        if re.fullmatch(
                            r"\d+",
                            code,
                        ):
                            icd9.append(
                                code
                            )
                        else:
                            icd10.append(
                                code
                            )

        # ---------------------------------------------
        # Discharge procedure text
        # ---------------------------------------------

        rows = self._case_rows(
            self.discharge_procedures,
            case_id,
        )

        if (
            rows is not None
            and not rows.empty
        ):
            hadm_col = _find_column(
                rows,
                [
                    "hadm_id",
                    "case_id",
                ],
            )

            preferred = [
                "discharge_procedures",
                "procedure",
                "procedure_name",
                "text",
                "description",
                "long_title",
            ]

            text_cols = [
                column
                for column in preferred
                if column in rows.columns
            ]

            if not text_cols:
                text_cols = [
                    column
                    for column in rows.columns
                    if column != hadm_col
                ]

            for _, row in rows.iterrows():
                for column in text_cols:
                    value = row[
                        column
                    ]

                    if pd.isna(
                        value
                    ):
                        continue

                    text = str(
                        value
                    ).strip()

                    if text:
                        discharge.append(
                            text
                        )

        return {
            "icd9":
                sorted(
                    set(
                        icd9
                    )
                ),

            "icd10":
                sorted(
                    set(
                        icd10
                    )
                ),

            "discharge":
                discharge,
        }

    def get_diagnosis_text(
        self,
        case_id: str,
    ) -> List[str]:
        values = []

        for df in [
            self.discharge_diagnosis,
            self.icd_diagnosis,
        ]:
            rows = self._case_rows(
                df,
                case_id,
            )

            if (
                rows is None
                or rows.empty
            ):
                continue

            hadm_col = _find_column(
                rows,
                [
                    "hadm_id",
                    "case_id",
                ],
            )

            preferred = [
                "discharge_diagnosis",
                "long_title",
                "diagnosis",
                "description",
                "text",
            ]

            columns = [
                column
                for column in preferred
                if column in rows.columns
            ]

            if not columns:
                columns = [
                    column
                    for column in rows.columns
                    if column != hadm_col
                ]

            for _, row in rows.iterrows():
                for column in columns:
                    value = row[
                        column
                    ]

                    if pd.isna(
                        value
                    ):
                        continue

                    text = str(
                        value
                    ).strip()

                    if text:
                        values.append(
                            text
                        )

        return values


# =========================================================
# Treatment helpers
# =========================================================

def _procedure_present(
    reference: Dict[str, Any],
    icd9_codes,
    icd10_codes,
    keywords,
) -> bool:
    case_icd9 = {
        _normalize_code(
            code
        )
        for code in reference.get(
            "icd9",
            []
        )
    }

    case_icd10 = {
        _normalize_code(
            code
        )
        for code in reference.get(
            "icd10",
            []
        )
    }

    if (
        case_icd9
        & {
            _normalize_code(
                code
            )
            for code in icd9_codes
        }
    ):
        return True

    if (
        case_icd10
        & {
            _normalize_code(
                code
            )
            for code in icd10_codes
        }
    ):
        return True

    discharge_text = " ".join(
        reference.get(
            "discharge",
            []
        )
    )

    return _contains_any(
        discharge_text,
        keywords,
    )


def _alternative_operation_requested(
    text: str,
    location: str,
    modifiers,
) -> bool:
    sentences = re.split(
        r"[.\n]",
        text,
    )

    for sentence in sentences:
        if not _positive_keyword(
            sentence,
            location,
        ):
            continue

        for modifier in modifiers:
            if _positive_keyword(
                sentence,
                modifier,
            ):
                return True

    return False


def _drainage_requested(
    treatment: str,
    locations,
) -> bool:
    for location in locations:
        if _alternative_operation_requested(
            treatment,
            location,
            DRAINAGE_KEYWORDS,
        ):
            return True

    return False


def _support_requested_standard(
    treatment: str,
) -> bool:
    return (
        _positive_keyword(
            treatment,
            "fluid",
        )
        or _positive_keyword(
            treatment,
            "analgesi",
        )
        or _positive_keyword(
            treatment,
            "pain",
        )
    )


def _support_requested_pancreatitis(
    treatment: str,
) -> bool:
    return (
        _positive_keyword(
            treatment,
            "fluid",
        )
        and (
            _positive_keyword(
                treatment,
                "analgesi",
            )
            or _positive_keyword(
                treatment,
                "pain",
            )
        )
        and _positive_keyword(
            treatment,
            "monitor",
        )
    )


def evaluate_treatment_guideline(
    trajectory: Dict[str, Any],
    ground_truth: ClinicalGroundTruth,
) -> Dict[str, Any]:
    disease = str(
        trajectory.get(
            "disease",
            "",
        )
    ).strip().lower()

    case_id = str(
        trajectory.get(
            "case_id",
            "",
        )
    )

    prediction = trajectory.get(
        "prediction",
        {},
    )

    if not isinstance(
        prediction,
        dict,
    ):
        prediction = {}

    treatment = str(
        prediction.get(
            "treatment_plan",
            "",
        )
        or ""
    )

    if disease not in {
        "appendicitis",
        "cholecystitis",
        "diverticulitis",
        "pancreatitis",
    }:
        return {
            "value":
                None,

            "applicable":
                False,

            "status":
                "unsupported_disease",
        }

    reference = (
        ground_truth
        .get_procedure_reference(
            case_id
        )
    )

    diagnosis_text = (
        ground_truth
        .get_diagnosis_text(
            case_id
        )
    )

    requested = {}

    required = {}

    # =====================================================
    # Appendicitis
    # =====================================================

    if disease == "appendicitis":
        required = {
            "Appendectomy":
                _procedure_present(
                    reference,
                    APPENDECTOMY_ICD9,
                    APPENDECTOMY_ICD10,
                    APPENDECTOMY_KEYWORDS,
                ),

            "Antibiotics":
                True,

            "Support":
                True,
        }

        requested = {
            "Appendectomy":
                (
                    _contains_any(
                        treatment,
                        APPENDECTOMY_KEYWORDS,
                    )
                    or
                    _alternative_operation_requested(
                        treatment,
                        "appendix",
                        {
                            "surgery",
                            "surgical",
                            "removal",
                            "remove",
                        },
                    )
                ),

            "Antibiotics":
                _positive_keyword(
                    treatment,
                    "antibiotic",
                ),

            "Support":
                _support_requested_standard(
                    treatment
                ),
        }

    # =====================================================
    # Cholecystitis
    # =====================================================

    elif disease == "cholecystitis":
        required = {
            "Cholecystectomy":
                _procedure_present(
                    reference,
                    CHOLECYSTECTOMY_ICD9,
                    CHOLECYSTECTOMY_ICD10,
                    CHOLECYSTECTOMY_KEYWORDS,
                ),

            "Antibiotics":
                True,

            "Support":
                True,
        }

        requested = {
            "Cholecystectomy":
                (
                    _contains_any(
                        treatment,
                        CHOLECYSTECTOMY_KEYWORDS,
                    )
                    or
                    _alternative_operation_requested(
                        treatment,
                        "gallbladder",
                        {
                            "surgery",
                            "surgical",
                            "removal",
                            "remove",
                        },
                    )
                ),

            "Antibiotics":
                _positive_keyword(
                    treatment,
                    "antibiotic",
                ),

            "Support":
                _support_requested_standard(
                    treatment
                ),
        }

    # =====================================================
    # Diverticulitis
    # =====================================================

    elif disease == "diverticulitis":
        drainage_required = (
            _procedure_present(
                reference,
                DRAINAGE_ICD9,
                DRAINAGE_ALL_ICD10,
                set(),
            )
            or (
                _contains_any(
                    " ".join(
                        reference.get(
                            "discharge",
                            [],
                        )
                    ),
                    DRAINAGE_KEYWORDS,
                )
                and
                _contains_any(
                    " ".join(
                        reference.get(
                            "discharge",
                            [],
                        )
                    ),
                    DRAINAGE_LOCATIONS_DIVERTICULITIS,
                )
            )
        )

        colectomy_required = (
            _procedure_present(
                reference,
                COLECTOMY_ICD9,
                COLECTOMY_ICD10,
                COLECTOMY_KEYWORDS,
            )
        )

        required = {
            "Colonoscopy":
                True,

            "Antibiotics":
                True,

            "Support":
                True,

            "Drainage":
                drainage_required,

            "Colectomy":
                colectomy_required,
        }

        requested = {
            "Colonoscopy":
                _positive_keyword(
                    treatment,
                    "colonoscopy",
                ),

            "Antibiotics":
                _positive_keyword(
                    treatment,
                    "antibiotic",
                ),

            "Support":
                _support_requested_standard(
                    treatment
                ),

            "Drainage":
                _drainage_requested(
                    treatment,
                    DRAINAGE_LOCATIONS_DIVERTICULITIS,
                ),

            "Colectomy":
                (
                    _contains_any(
                        treatment,
                        COLECTOMY_KEYWORDS,
                    )
                    or
                    _alternative_operation_requested(
                        treatment,
                        "colon",
                        {
                            "surgery",
                            "surgical",
                            "removal",
                            "remove",
                        },
                    )
                ),
        }

    # =====================================================
    # Pancreatitis
    # =====================================================

    else:
        combined_diagnoses = " ".join(
            diagnosis_text
        )
        
        biliary = any(
            _positive_keyword(
                combined_diagnoses,
                keyword,
            )
            for keyword in (
                "biliary",
                "gallstone",
                "cholelithiasis",
                "calculus of gallbladder",
            )
        )

        drainage_required = (
            _procedure_present(
                reference,
                DRAINAGE_ICD9,
                (
                    DRAINAGE_ALL_ICD10
                    | DRAINAGE_PANCREATITIS_ICD10
                ),
                set(),
            )
            or (
                _contains_any(
                    " ".join(
                        reference.get(
                            "discharge",
                            [],
                        )
                    ),
                    DRAINAGE_KEYWORDS,
                )
                and
                _contains_any(
                    " ".join(
                        reference.get(
                            "discharge",
                            [],
                        )
                    ),
                    DRAINAGE_LOCATIONS_PANCREATITIS,
                )
            )
        )

        ercp_required = (
            _procedure_present(
                reference,
                ERCP_ICD9,
                ERCP_ICD10,
                ERCP_KEYWORDS,
            )
        )

        required = {
            "Support":
                True,

            "Drainage":
                drainage_required,

            "ERCP":
                ercp_required,

            "Cholecystectomy":
                biliary,
        }

        requested = {
            "Support":
                _support_requested_pancreatitis(
                    treatment
                ),

            "Drainage":
                _drainage_requested(
                    treatment,
                    DRAINAGE_LOCATIONS_PANCREATITIS,
                ),

            "ERCP":
                _contains_any(
                    treatment,
                    ERCP_KEYWORDS,
                ),

            "Cholecystectomy":
                (
                    _contains_any(
                        treatment,
                        CHOLECYSTECTOMY_KEYWORDS,
                    )
                    or
                    _alternative_operation_requested(
                        treatment,
                        "gallbladder",
                        {
                            "surgery",
                            "surgical",
                            "removal",
                            "remove",
                        },
                    )
                ),
        }

    missing_required = [
        treatment_name
        for treatment_name, is_required
        in required.items()
        if (
            is_required
            and not requested.get(
                treatment_name,
                False,
            )
        )
    ]

    failure = int(
        bool(
            missing_required
        )
    )

    return {
        "value":
            failure,

        "applicable":
            True,

        "status":
            "evaluated",

        "treatment_plan":
            treatment,

        "treatment_required":
            required,

        "treatment_requested":
            requested,

        "missing_required_treatments":
            missing_required,

        "procedure_reference":
            reference,

        "biliary_pancreatitis":
            (
                biliary
                if disease
                == "pancreatitis"
                else None
            ),
    }


# =========================================================
# Combined clinical evaluator
# =========================================================

def evaluate_clinical_failures(
    trajectory: Dict[str, Any],
    ground_truth: Optional[
        ClinicalGroundTruth
    ] = None,
) -> Dict[str, Any]:
    if ground_truth is None:
        ground_truth = (
            ClinicalGroundTruth()
        )

    diagnostic_process = (
        evaluate_diagnostic_process(
            trajectory
        )
    )

    lab_interpretation = (
        evaluate_lab_interpretation(
            trajectory
        )
    )

    treatment_guideline = (
        evaluate_treatment_guideline(
            trajectory,
            ground_truth,
        )
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

        "disease":
            trajectory.get(
                "disease"
            ),

        # -------------------------------------------------
        # Three clinical failure domains
        # -------------------------------------------------

        "diagnostic_process_failure":
            diagnostic_process.get(
                "value"
            ),

        "lab_interpretation_failure":
            lab_interpretation.get(
                "value"
            ),

        "treatment_guideline_failure":
            treatment_guideline.get(
                "value"
            ),

        # -------------------------------------------------
        # Applicability
        # -------------------------------------------------

        "diagnostic_process_applicable":
            diagnostic_process.get(
                "applicable"
            ),

        "lab_interpretation_applicable":
            lab_interpretation.get(
                "applicable"
            ),

        "treatment_guideline_applicable":
            treatment_guideline.get(
                "applicable"
            ),

        # -------------------------------------------------
        # Status
        # -------------------------------------------------

        "diagnostic_process_status":
            diagnostic_process.get(
                "status"
            ),

        "lab_interpretation_status":
            lab_interpretation.get(
                "status"
            ),

        "treatment_guideline_status":
            treatment_guideline.get(
                "status"
            ),

        # -------------------------------------------------
        # Audit
        # -------------------------------------------------

        "diagnostic_process_details":
            diagnostic_process,

        "lab_interpretation_details":
            lab_interpretation,

        "treatment_guideline_details":
            treatment_guideline,
    }


# =========================================================
# File interface
# =========================================================

def evaluate_trajectory_file(
    trajectory_path: Path,
    output_path: Optional[
        Path
    ] = None,
    data_dir: Path = DEFAULT_MIMIC_DIR,
) -> Dict[str, Any]:
    trajectory_path = Path(
        trajectory_path
    )

    trajectory = _load_json(
        trajectory_path
    )

    ground_truth = (
        ClinicalGroundTruth(
            data_dir=data_dir
        )
    )

    result = (
        evaluate_clinical_failures(
            trajectory=trajectory,
            ground_truth=ground_truth,
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
            "Evaluate the three clinical failure domains "
            "for one Clinical HyperAgent Experiment 1 "
            "trajectory."
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
        "--data-dir",
        default=str(
            DEFAULT_MIMIC_DIR
        ),
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
            / "clinical_failure_eval"
            / agent_id
            / f"{case_id}.json"
        )

    result = (
        evaluate_trajectory_file(
            trajectory_path=
                trajectory_path,

            output_path=
                output_path,

            data_dir=
                Path(
                    args.data_dir
                ),
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