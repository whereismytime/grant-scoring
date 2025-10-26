# app/ml/generate_scenarios.py
import numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(42)

def block_approve(n):
    return pd.DataFrame({
        "distance_km": rng.uniform(0, 30, n).round(1),
        "residency_years_ie": rng.uniform(0, 3, n).round(1),
        "is_immigrant": False,
        "parents_status": "low",
        "approved": 1,
        "bucket": "APPROVE_easy",
    })

def block_decline_parents_high(n):
    return pd.DataFrame({
        "distance_km": rng.uniform(0, 80, n).round(1),
        "residency_years_ie": rng.uniform(0, 6, n).round(1),
        "is_immigrant": rng.choice([True, False], n),
        "parents_status": "high",
        "approved": 0,
        "bucket": "DECLINE_high_income",
    })

def block_decline_immigrants(n):
    return pd.DataFrame({
        "distance_km": rng.uniform(0, 80, n).round(1),
        "residency_years_ie": rng.uniform(3.0, 6.0, n).round(1),
        "is_immigrant": True,
        "parents_status": rng.choice(["low","middle"], n, p=[0.6,0.4]),
        "approved": 0,
        "bucket": "DECLINE_immigrant>=3y",
    })

def block_borderline_middle(n):
    d = np.clip(rng.normal(30.0, 1.2, n), 0, 80)
    return pd.DataFrame({
        "distance_km": d.round(1),
        "residency_years_ie": rng.uniform(0, 6, n).round(1),
        "is_immigrant": rng.choice([True, False], n, p=[0.2,0.8]),
        "parents_status": "middle",
        "approved": (d > 30).astype(int),   # >30 одобряем
        "bucket": "BORDER_middle_around_30",
    })

def block_borderline_imm(n):
    y = np.clip(rng.normal(3.0, 0.15, n), 0, 6)
    return pd.DataFrame({
        "distance_km": rng.uniform(0, 80, n).round(1),
        "residency_years_ie": y.round(1),
        "is_immigrant": True,
        "parents_status": rng.choice(["low","middle"], n, p=[0.7,0.3]),
        "approved": (y < 3.0).astype(int),  # <3 лет одобряем
        "bucket": "BORDER_immigrant_around_3y",
    })

def block_random(n):
    df = pd.DataFrame({
        "distance_km": rng.uniform(0, 80, n).round(1),
        "residency_years_ie": rng.uniform(0, 6, n).round(1),
        "is_immigrant": rng.choice([True, False], n, p=[0.35,0.65]),
        "parents_status": rng.choice(["low","middle","high"], n, p=[0.5,0.35,0.15]),
    })
    appr = (
        (df.parents_status == "low") |
        ((df.parents_status == "middle") & (df.distance_km > 30))
    ) & ~(df.parents_status == "high") & ~(
        df.is_immigrant & (df.residency_years_ie >= 3.0)
    )
    df["approved"] = appr.astype(int)
    df["bucket"] = "MIX_random"
    return df

def make_dataset(n_each=4000, noise=0.03):
    parts = [
        block_approve(n_each),
        block_decline_parents_high(n_each),
        block_decline_immigrants(n_each),
        block_borderline_middle(n_each),
        block_borderline_imm(n_each),
        block_random(n_each),
    ]
    df = pd.concat(parts, ignore_index=True)
    if noise > 0:
        flip = rng.random(len(df)) < noise
        df.loc[flip, "approved"] = 1 - df.loc[flip, "approved"].astype(int)
    return df

if __name__ == "__main__":
    out = Path("data/raw/generated_applicants.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make_dataset(n_each=4000, noise=0.03)  # ~24k строк
    df.to_csv(out, index=False)
    print(f"saved {len(df):,} rows -> {out}")
