"""Evaluate the model on a labeled CSV with bucketed metrics.

Defaults to the holdout split written by ``app.ml.train``, so the model is not
scored on rows it was trained on. Writes
``data/processed/{metrics_from_csv.json,preds_from_csv.csv}``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from app.domain.rules import DISTANCE_THRESHOLD_KM, IMMIGRANT_RESIDENCY_LIMIT_YEARS, hard_decline
from app.ml.features import rule_labels
from app.ml.paths import HOLDOUT_CSV, PROCESSED_DIR
from app.services.ml_model import MLModel, model_path

BORDER_KM = 1.2      # half-width of the "around 30 km" band
BORDER_YEARS = 0.2   # half-width of the "around 3 years" band


def bucket_row(r) -> str:
    """Fallback bucket for CSVs without a `bucket` column."""
    declined, _ = hard_decline(r)
    if declined:
        return "DECLINE_high_income" if r.parents_status == "high" else "DECLINE_immigrant>=3y"
    if (r.parents_status == "low" and not r.is_immigrant
            and r.residency_years_ie < IMMIGRANT_RESIDENCY_LIMIT_YEARS
            and r.distance_km <= DISTANCE_THRESHOLD_KM):
        return "APPROVE_easy"
    if r.parents_status == "middle" and abs(r.distance_km - DISTANCE_THRESHOLD_KM) <= BORDER_KM:
        return "BORDER_middle_around_30"
    if r.is_immigrant and abs(r.residency_years_ie - IMMIGRANT_RESIDENCY_LIMIT_YEARS) <= BORDER_YEARS:
        return "BORDER_immigrant_around_3y"
    return "MIX_random"


def sweep_best_threshold(y, p):
    thrs = np.linspace(0.05, 0.95, 19)
    accs = [(t, accuracy_score(y, (p >= t).astype(int))) for t in thrs]
    return max(accs, key=lambda x: x[1])


def metrics(y_true, proba, mask, threshold=0.5) -> dict:
    p, yt = proba[mask], y_true[mask]
    y_hat = (p >= threshold).astype(int)
    two_classes = len(np.unique(yt)) > 1
    best_t, best_acc = sweep_best_threshold(yt, p) if two_classes else (None, None)
    return {
        "N": int(mask.sum()),
        "approve_rate_true": float(yt.mean()),
        "approve_rate_ML": float(y_hat.mean()),
        f"ACC@{threshold:.2f}": float(accuracy_score(yt, y_hat)),
        "AUC": float(roc_auc_score(yt, p)) if two_classes else None,
        "best_thr_ACC": float(best_t) if best_t is not None else None,
        "best_ACC": float(best_acc) if best_acc is not None else None,
        f"confusion@{threshold:.2f}": confusion_matrix(yt, y_hat, labels=[0, 1]).tolist(),
    }


def evaluate(df: pd.DataFrame, model: MLModel, threshold=0.5):
    """Return (summary dict, per-row predictions DataFrame)."""
    df = df.copy()
    if "approved" in df.columns:
        y_true = df["approved"].astype(int).values
    else:
        y_true = rule_labels(df)  # fallback: labels from the business rules
    if "bucket" not in df.columns:
        df["bucket"] = [bucket_row(r) for r in df.itertuples(index=False)]

    proba = model.prob_batch(df)
    summary = {"overall": metrics(y_true, proba, np.ones(len(df), dtype=bool), threshold)}
    for b in sorted(df["bucket"].unique()):
        summary[b] = metrics(y_true, proba, df["bucket"].values == b, threshold)

    df["p_ml"] = proba
    df[f"y_ml@{threshold:.2f}"] = (proba >= threshold).astype(int)
    return summary, df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=HOLDOUT_CSV)
    p.add_argument("--model", type=Path, default=model_path())
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args(argv)

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found. Run `python -m app.ml.train` first.")

    summary, preds = evaluate(pd.read_csv(args.data), MLModel.load(args.model), args.threshold)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out_dir / "preds_from_csv.csv", index=False)
    (args.out_dir / "metrics_from_csv.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("saved CSV ->", args.out_dir / "preds_from_csv.csv")
    print("saved JSON ->", args.out_dir / "metrics_from_csv.json")


if __name__ == "__main__":
    main()
