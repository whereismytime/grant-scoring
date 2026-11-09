"""Train the calibrated logistic regression and save it with a holdout split.

The saved pipeline starts with ``build_features``, so it is called with the raw
applicant columns only. The test split is written to ``data/processed/holdout.csv``
for an honest evaluation with ``app.ml.eval_from_csv``.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from app.ml.features import CAT_COLS, NUM_COLS, RAW_COLS, build_features, rule_labels
from app.ml.paths import HOLDOUT_CSV, SCENARIOS_CSV
from app.services.ml_model import model_path


def make_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", "passthrough", NUM_COLS),
    ])
    base = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    return Pipeline([
        ("features", FunctionTransformer(build_features)),
        ("pre", pre),
        ("clf", clf),
    ])


def labels(df: pd.DataFrame):
    if "approved" in df.columns:
        return df["approved"].astype(int).values
    return rule_labels(df)


def train(df: pd.DataFrame, test_size=0.2, seed=42):
    """Fit on a train split; return (pipeline, holdout_df, test AUC)."""
    y = labels(df)
    df_tr, df_te, y_tr, y_te = train_test_split(
        df, y, test_size=test_size, random_state=seed, stratify=y
    )
    pipe = make_pipeline()
    pipe.fit(df_tr[RAW_COLS], y_tr)
    auc = roc_auc_score(y_te, pipe.predict_proba(df_te[RAW_COLS])[:, 1])
    holdout = df_te.copy()
    holdout["approved"] = y_te
    return pipe, holdout, auc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=SCENARIOS_CSV)
    p.add_argument("--model", type=Path, default=model_path())
    p.add_argument("--holdout", type=Path, default=HOLDOUT_CSV)
    args = p.parse_args(argv)

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found. Run `python -m app.ml.generate_scenarios` first.")

    pipe, holdout, auc = train(pd.read_csv(args.data))
    print(f"AUC (holdout) = {auc:.3f}")

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.model)
    print(f"saved -> {args.model}")

    args.holdout.parent.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(args.holdout, index=False)
    print(f"saved holdout ({len(holdout):,} rows) -> {args.holdout}")


if __name__ == "__main__":
    main()
