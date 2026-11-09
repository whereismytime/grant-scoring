"""Feature engineering shared by training and inference.

``build_features`` is the first step of the saved pipeline, so the model takes
the raw applicant columns (``RAW_COLS``) and derives everything else itself.
Training and prediction therefore cannot drift apart.
"""

import numpy as np
import pandas as pd

from app.domain.rules import (
    DISTANCE_THRESHOLD_KM,
    IMMIGRANT_RESIDENCY_LIMIT_YEARS,
    rule_approve,
)

RAW_COLS = ["distance_km", "residency_years_ie", "is_immigrant", "parents_status"]
CAT_COLS = ["parents_status"]
NUM_COLS = [
    "distance_km",
    "residency_years_ie",
    "is_immigrant",
    "dist_gt_30",
    "imm_ge_3",
    "mid_x_dist",
]


def build_features(X: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from the raw applicant columns."""
    X = X[RAW_COLS].copy()
    X["is_immigrant"] = X["is_immigrant"].astype(int)
    far = X["distance_km"] > DISTANCE_THRESHOLD_KM
    X["dist_gt_30"] = far.astype(int)
    X["imm_ge_3"] = (X["residency_years_ie"] >= IMMIGRANT_RESIDENCY_LIMIT_YEARS).astype(int)
    X["mid_x_dist"] = ((X["parents_status"] == "middle") & far).astype(int)
    return X[CAT_COLS + NUM_COLS]


def rule_labels(df: pd.DataFrame) -> np.ndarray:
    """Label rows with the business rules from ``app.domain.rules``."""
    rows = df[RAW_COLS].itertuples(index=False)
    return np.fromiter((int(rule_approve(r)) for r in rows), dtype=int, count=len(df))
