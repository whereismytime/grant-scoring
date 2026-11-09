"""Project paths, resolved from the repository root so scripts work from any CWD."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

SCENARIOS_CSV = RAW_DIR / "generated_applicants.csv"    # labeled (approved + bucket)
UNLABELED_CSV = RAW_DIR / "unlabeled_applicants.csv"    # features only
HOLDOUT_CSV = PROCESSED_DIR / "holdout.csv"             # test split written by train.py
