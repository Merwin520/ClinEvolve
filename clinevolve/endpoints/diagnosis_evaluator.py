import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from agent.paths import mimic_cdm_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIMIC_ROOT = mimic_cdm_dir()

DISCHARGE_DIAGNOSIS_PATH = (
    MIMIC_ROOT / "discharge_diagnosis.csv"
)

ICD_DIAGNOSIS_PATH = (
    MIMIC_ROOT / "icd_diagnosis.csv"
)


TARGET_DISEASES = [
    "appendicitis",
    "cholecystitis",
    "diverticulitis",
    "pancreatitis",
]


# =========================================================
# Normalization
# =========================================================

def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    text = str(text).lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_disease_name(
    text: Any,
) -> Optional[str]:
    """
    Map free-text diagnosis into one of the four
    Experiment 1 disease classes.

    Returns:
        appendicitis
        cholecystitis
        diverticulitis
        pancreatitis
        None
    """

    text = normalize_text(text)

    if not text:
        return None

    # -----------------------------------------------------
    # Appendicitis
    # -----------------------------------------------------

    appendicitis_patterns = [
        r"\bappendicitis\b",
        r"\bacute appendicitis\b",
        r"\bperforated appendicitis\b",
        r"\bgangrenous appendicitis\b",
    ]

    for pattern in appendicitis_patterns:
        if re.search(
            pattern,
            text,
        ):
            return "appendicitis"

    # -----------------------------------------------------
    # Cholecystitis
    # -----------------------------------------------------

    cholecystitis_patterns = [
        r"\bcholecystitis\b",
        r"\bacute cholecystitis\b",
        r"\bcalculous cholecystitis\b",
        r"\bacalculous cholecystitis\b",
    ]

    for pattern in cholecystitis_patterns:
        if re.search(
            pattern,
            text,
        ):
            return "cholecystitis"

    # -----------------------------------------------------
    # Diverticulitis
    # -----------------------------------------------------

    diverticulitis_patterns = [
        r"\bdiverticulitis\b",
        r"\bsigmoid diverticulitis\b",
        r"\bacute diverticulitis\b",
        r"\bperforated diverticulitis\b",
    ]

    for pattern in diverticulitis_patterns:
        if re.search(
            pattern,
            text,
        ):
            return "diverticulitis"

    # -----------------------------------------------------
    # Pancreatitis
    # -----------------------------------------------------

    pancreatitis_patterns = [
        r"\bpancreatitis\b",
        r"\bacute pancreatitis\b",
        r"\bgallstone pancreatitis\b",
        r"\bbiliary pancreatitis\b",
        r"\balcoholic pancreatitis\b",
    ]

    for pattern in pancreatitis_patterns:
        if re.search(
            pattern,
            text,
        ):
            return "pancreatitis"

    return None


# =========================================================
# Ground truth loader
# =========================================================

class DiagnosisGroundTruth:
    def __init__(
        self,
        discharge_path: Path = DISCHARGE_DIAGNOSIS_PATH,
        icd_path: Path = ICD_DIAGNOSIS_PATH,
    ):
        self.discharge_path = Path(
            discharge_path
        )

        self.icd_path = Path(
            icd_path
        )

        self.discharge_lookup = {}
        self.icd_lookup = {}

        self._load_discharge()
        self._load_icd()

    def _load_discharge(self):
        if not self.discharge_path.exists():
            raise FileNotFoundError(
                f"Missing discharge diagnosis file: "
                f"{self.discharge_path}"
            )

        df = pd.read_csv(
            self.discharge_path
        )

        required = {
            "hadm_id",
            "discharge_diagnosis",
        }

        missing = (
            required
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                "Missing columns in "
                "discharge_diagnosis.csv: "
                f"{sorted(missing)}"
            )

        df["hadm_id"] = (
            df["hadm_id"]
            .astype(str)
        )

        for _, row in df.iterrows():

            hadm_id = row["hadm_id"]

            raw_text = row[
                "discharge_diagnosis"
            ]

            normalized = (
                normalize_disease_name(
                    raw_text
                )
            )

            self.discharge_lookup[
                hadm_id
            ] = {
                "raw":
                    ""
                    if pd.isna(raw_text)
                    else str(raw_text),

                "normalized":
                    normalized,
            }

    def _load_icd(self):
        """
        ICD file is stored for auxiliary verification only.

        Since the exact schema may vary, this loader is
        intentionally permissive.
        """

        if not self.icd_path.exists():
            return

        df = pd.read_csv(
            self.icd_path
        )

        if "hadm_id" not in df.columns:
            return

        df["hadm_id"] = (
            df["hadm_id"]
            .astype(str)
        )

        text_columns = [
            col
            for col in df.columns
            if col != "hadm_id"
        ]

        for hadm_id, group in df.groupby(
            "hadm_id"
        ):

            texts = []

            for _, row in group.iterrows():

                for col in text_columns:

                    value = row[col]

                    if pd.notna(value):
                        texts.append(
                            str(value)
                        )

            joined = " ".join(
                texts
            )

            normalized = (
                normalize_disease_name(
                    joined
                )
            )

            self.icd_lookup[
                str(hadm_id)
            ] = {
                "raw":
                    joined,

                "normalized":
                    normalized,
            }

    def get_ground_truth(
        self,
        hadm_id: str,
    ) -> Dict[str, Any]:

        hadm_id = str(
            hadm_id
        )

        discharge = (
            self.discharge_lookup.get(
                hadm_id,
                {},
            )
        )

        icd = (
            self.icd_lookup.get(
                hadm_id,
                {},
            )
        )

        discharge_class = (
            discharge.get(
                "normalized"
            )
        )

        icd_class = (
            icd.get(
                "normalized"
            )
        )

        # Primary source:
        # discharge diagnosis.
        #
        # ICD is only used if discharge text could not
        # be normalized.
        if discharge_class is not None:

            final_class = (
                discharge_class
            )

            source = (
                "discharge_diagnosis"
            )

        elif icd_class is not None:

            final_class = (
                icd_class
            )

            source = (
                "icd_diagnosis"
            )

        else:

            final_class = None
            source = "unresolved"

        return {
            "hadm_id":
                hadm_id,

            "ground_truth_disease":
                final_class,

            "ground_truth_source":
                source,

            "discharge_diagnosis_raw":
                discharge.get(
                    "raw",
                    "",
                ),

            "discharge_diagnosis_class":
                discharge_class,

            "icd_diagnosis_class":
                icd_class,
        }


# =========================================================
# Prediction extraction
# =========================================================

def extract_prediction_diagnosis(
    trajectory: Dict[str, Any],
) -> str:

    prediction = trajectory.get(
        "prediction"
    )

    if isinstance(
        prediction,
        dict,
    ):

        diagnosis = (
            prediction.get(
                "diagnosis",
                "",
            )
        )

        return (
            ""
            if diagnosis is None
            else str(diagnosis)
        )

    if prediction is None:
        return ""

    return str(
        prediction
    )


# =========================================================
# Main evaluator
# =========================================================

def evaluate_diagnosis(
    hadm_id: str,
    predicted_diagnosis: str,
    ground_truth: DiagnosisGroundTruth,
) -> Dict[str, Any]:

    truth = (
        ground_truth.get_ground_truth(
            hadm_id
        )
    )

    predicted_class = (
        normalize_disease_name(
            predicted_diagnosis
        )
    )

    true_class = (
        truth[
            "ground_truth_disease"
        ]
    )

    if true_class is None:

        diagnosis_correct = None
        final_diagnosis_failure = None

        status = (
            "ground_truth_unresolved"
        )

    elif predicted_class is None:

        diagnosis_correct = 0
        final_diagnosis_failure = 1

        status = (
            "prediction_unresolved"
        )

    else:

        diagnosis_correct = int(
            predicted_class
            == true_class
        )

        final_diagnosis_failure = int(
            not diagnosis_correct
        )

        status = "evaluated"

    return {
        "hadm_id":
            str(hadm_id),

        "predicted_diagnosis_raw":
            predicted_diagnosis,

        "predicted_disease":
            predicted_class,

        "ground_truth_disease":
            true_class,

        "ground_truth_source":
            truth[
                "ground_truth_source"
            ],

        "diagnosis_correct":
            diagnosis_correct,

        "final_diagnosis_failure":
            final_diagnosis_failure,

        "evaluation_status":
            status,

        # Keep raw GT for audit only.
        # Never expose this back to TaskAgent.
        "discharge_diagnosis_raw":
            truth[
                "discharge_diagnosis_raw"
            ],
    }


# =========================================================
# Trajectory evaluator
# =========================================================

def evaluate_trajectory(
    trajectory: Dict[str, Any],
    ground_truth: DiagnosisGroundTruth,
) -> Dict[str, Any]:

    hadm_id = str(
        trajectory.get(
            "case_id",
            "",
        )
    )

    predicted_diagnosis = (
        extract_prediction_diagnosis(
            trajectory
        )
    )

    result = evaluate_diagnosis(
        hadm_id=
            hadm_id,

        predicted_diagnosis=
            predicted_diagnosis,

        ground_truth=
            ground_truth,
    )

    result.update(
        {
            "run_id":
                trajectory.get(
                    "run_id"
                ),

            "agent_id":
                trajectory.get(
                    "agent_id"
                ),

            "base_model":
                trajectory.get(
                    "base_model"
                ),
        }
    )

    return result


# =========================================================
# File interface
# =========================================================

def evaluate_trajectory_file(
    trajectory_path: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:

    trajectory_path = Path(
        trajectory_path
    )

    with open(
        trajectory_path,
        "r",
        encoding="utf-8",
    ) as f:
        trajectory = json.load(
            f
        )

    if (
        trajectory.get(
            "status"
        )
        != "success"
    ):
        raise ValueError(
            "Cannot evaluate unsuccessful "
            f"trajectory: {trajectory_path}"
        )

    ground_truth = (
        DiagnosisGroundTruth()
    )

    result = (
        evaluate_trajectory(
            trajectory=
                trajectory,

            ground_truth=
                ground_truth,
        )
    )

    if output_path is not None:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = (
            output_path.with_suffix(
                output_path.suffix
                + ".tmp"
            )
        )

        with open(
            temp_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False,
            )

        temp_path.replace(
            output_path
        )

    return result


# =========================================================
# CLI
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate final diagnosis accuracy "
            "for one Clinical HyperAgent "
            "Experiment 1 trajectory."
        )
    )

    parser.add_argument(
        "trajectory",
        help=(
            "Path to one trajectory JSON "
            "generated by run_exp1.py"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
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
            / "diagnosis_eval"
            / agent_id
            / f"{case_id}.json"
        )

    result = (
        evaluate_trajectory_file(
            trajectory_path=
                trajectory_path,

            output_path=
                output_path,
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