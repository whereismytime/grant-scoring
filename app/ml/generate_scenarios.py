"""Generate a labeled scenario dataset (approved + bucket).

Each block only shapes the feature distribution. Labels are computed afterwards
from the rounded values with ``app.domain.rules``, so they always match the
business rules (apart from the deliberate label noise).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.features import rule_labels
from app.ml.paths import SCENARIOS_CSV


def block_approve(rng, n):
    return pd.DataFrame({
        "distance_km": rng.uniform(0, 30, n).round(1),
        "residency_years_ie": rng.uniform(0, 3, n).round(1),
        "is_immigrant": False,
        "parents_status": "low",
        "bucket": "APPROVE_easy",
    })


def block_decline_parents_high(rng, n):
    return pd.DataFrame({
        "distance_km": rng.uniform(0, 80, n).round(1),
        "residency_years_ie": rng.uniform(0, 6, n).round(1),
        "is_immigrant": rng.choice([True, False], n),
        "parents_status": "high",
        "bucket": "DECLINE_high_income",
    })


def block_decline_immigrants(rng, n):
    return pd.DataFrame({
        "distance_km": rng.uniform(0, 80, n).round(1),
        "residency_years_ie": rng.uniform(3.0, 6.0, n).round(1),
        "is_immigrant": True,
        "parents_status": rng.choice(["low", "middle"], n, p=[0.6, 0.4]),
        "bucket": "DECLINE_immigrant>=3y",
    })


def block_borderline_middle(rng, n):
    d = np.clip(rng.normal(30.0, 1.2, n), 0, 80)
    return pd.DataFrame({
        "distance_km": d.round(1),
        "residency_years_ie": rng.uniform(0, 6, n).round(1),
        "is_immigrant": rng.choice([True, False], n, p=[0.2, 0.8]),
        "parents_status": "middle",
        "bucket": "BORDER_middle_around_30",
    })


def block_borderline_imm(rng, n):
    y = np.clip(rng.normal(3.0, 0.15, n), 0, 6)
    return pd.DataFrame({
        "distance_km": rng.uniform(0, 80, n).round(1),
        "residency_years_ie": y.round(1),
        "is_immigrant": True,
        "parents_status": rng.choice(["low", "middle"], n, p=[0.7, 0.3]),
        "bucket": "BORDER_immigrant_around_3y",
    })


def block_random(rng, n):
    return pd.DataFrame({
        "distance_km": rng.uniform(0, 80, n).round(1),
        "residency_years_ie": rng.uniform(0, 6, n).round(1),
        "is_immigrant": rng.choice([True, False], n, p=[0.35, 0.65]),
        "parents_status": rng.choice(["low", "middle", "high"], n, p=[0.5, 0.35, 0.15]),
        "bucket": "MIX_random",
    })


BLOCKS = [
    block_approve,
    block_decline_parents_high,
    block_decline_immigrants,
    block_borderline_middle,
    block_borderline_imm,
    block_random,
]


def make_dataset(n_each=4000, noise=0.03, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.concat([block(rng, n_each) for block in BLOCKS], ignore_index=True)
    df["approved"] = rule_labels(df)
    if noise > 0:
        flip = rng.random(len(df)) < noise
        df.loc[flip, "approved"] = 1 - df.loc[flip, "approved"]
    cols = ["distance_km", "residency_years_ie", "is_immigrant", "parents_status", "approved", "bucket"]
    return df[cols]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=SCENARIOS_CSV)
    p.add_argument("--n-each", type=int, default=4000)
    p.add_argument("--noise", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = make_dataset(n_each=args.n_each, noise=args.noise, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"saved {len(df):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
