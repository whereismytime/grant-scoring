"""Generate an *unlabeled* applicant CSV (features only, ~10% duplicates).

It writes to its own file so it never overwrites the labeled scenario dataset.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.paths import UNLABELED_CSV


def make_unlabeled(n=50000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    distance = rng.uniform(0, 100, n)
    residency = rng.uniform(0, 10, n)
    is_immigrant = rng.choice([True, False], n)
    parents_status = rng.choice(["low", "middle", "high"], n)

    # add duplicates (about 10%)
    dup_idx = rng.choice(n, n // 10)
    return pd.DataFrame({
        "distance_km": np.round(np.concatenate([distance, distance[dup_idx]]), 1),
        "residency_years_ie": np.round(np.concatenate([residency, residency[dup_idx]]), 1),
        "is_immigrant": np.concatenate([is_immigrant, is_immigrant[dup_idx]]),
        "parents_status": np.concatenate([parents_status, parents_status[dup_idx]]),
    })


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=UNLABELED_CSV)
    p.add_argument("-n", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = make_unlabeled(n=args.n, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"✅ Saved {len(df):,} rows to {args.out}")
    print(df.head())


if __name__ == "__main__":
    main()
