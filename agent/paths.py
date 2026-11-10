"""Data locations. Set MIMIC_CDM_DATA_DIR to the directory holding the MIMIC-CDM CSV files
(history_of_present_illness.csv, physical_examination.csv, laboratory_tests.csv, lab_test_mapping.csv,
microbiology.csv, radiology_reports.csv, discharge_diagnosis.csv, icd_diagnosis.csv, ...)."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def mimic_cdm_dir() -> Path:
    return Path(os.environ.get("MIMIC_CDM_DATA_DIR", str(REPO_ROOT / "data" / "mimic-cdm"))).expanduser().resolve()
