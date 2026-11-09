"""Compare the ML model with the business rules on a fresh random population.

The reference here is the rules themselves, so the numbers show how closely
the model reproduces them (agreement), not real-world accuracy.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from app.ml.features import rule_labels
from app.services.ml_model import MLModel, model_path


def gen_dataset(n=10000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "distance_km": rng.integers(0, 81, n).astype(float),
        "residency_years_ie": np.round(rng.uniform(0, 6, n), 1),
        "is_immigrant": rng.random(n) < 0.35,
        "parents_status": rng.choice(["low", "middle", "high"], size=n, p=[0.5, 0.35, 0.15]),
    })


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", type=int, default=20000)
    p.add_argument("--model", type=Path, default=model_path())
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args(argv)

    df = gen_dataset(args.n)
    y_rule = rule_labels(df)
    p_ml = MLModel.load(args.model).prob_batch(df)
    y_ml = (p_ml >= args.threshold).astype(int)

    print("AUC (ML probability vs rule decisions):", round(roc_auc_score(y_rule, p_ml), 3))
    print("Agreement ML vs rules:", round(accuracy_score(y_rule, y_ml), 3))
    diff = y_rule != y_ml
    print("Share of disagreements (ML vs rules):", round(diff.mean(), 3))
    print("Confusion matrix (rows = rules, cols = ML):\n", confusion_matrix(y_rule, y_ml, labels=[0, 1]))

    show = df.loc[diff].copy()
    show["rule"] = y_rule[diff]
    show["ml"] = y_ml[diff]
    show["p_ml"] = np.round(p_ml[diff], 3)
    print("\nExamples of disagreements (first 5):")
    print(show.head(5).to_string(index=False) if len(show) else "none")


if __name__ == "__main__":
    main()
