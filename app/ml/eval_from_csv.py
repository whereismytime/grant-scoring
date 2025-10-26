import json, numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

IN = Path("data/raw/generated_applicants.csv")
OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

# Фоллбек: если approved нет, размечаем по правилам
if "approved" in df.columns:
    y_true = df["approved"].astype(int).values
else:
    def rule_label(r):
        if bool(r.is_immigrant) and r.residency_years_ie >= 3: return 0
        if r.parents_status == "high": return 0
        if r.parents_status == "middle": return int(r.distance_km > 30)
        return 1
    y_true = df.apply(rule_label, axis=1).astype(int).values

X = df[["distance_km","residency_years_ie","is_immigrant","parents_status"]]

model = joblib.load("app/models/model.pkl")
proba = model.predict_proba(X)[:, 1]

def bucket_row(r):
    if "bucket" in r and isinstance(r["bucket"], str): return r["bucket"]
    if r.parents_status == "high": return "DECLINE_high_income"
    if r.is_immigrant and r.residency_years_ie >= 3.0: return "DECLINE_immigrant>=3y"
    if r.parents_status == "low" and (not r.is_immigrant) and r.residency_years_ie < 3.0 and r.distance_km <= 30:
        return "APPROVE_easy"
    if r.parents_status == "middle" and 28.8 <= r.distance_km <= 31.2:
        return "BORDER_middle_around_30"
    if r.is_immigrant and 2.8 <= r.residency_years_ie <= 3.2:
        return "BORDER_immigrant_around_3y"
    return "MIX_random"

df["bucket"] = df.apply(bucket_row, axis=1)

def sweep_best_threshold(y, p):
    thrs = np.linspace(0.05, 0.95, 19)
    accs = [(t, accuracy_score(y, (p >= t).astype(int))) for t in thrs]
    return max(accs, key=lambda x: x[1])

def metrics(mask):
    p = proba[mask]; y_hat = (p >= 0.5).astype(int); yt = y_true[mask]
    best_t, best_acc = sweep_best_threshold(yt, p) if len(np.unique(yt))>1 else (None, None)
    return {
        "N": int(mask.sum()),
        "approve_rate_ML": float(y_hat.mean()),
        "ACC@0.50": float(accuracy_score(yt, y_hat)),
        "AUC": float(roc_auc_score(yt, p)) if len(np.unique(yt))>1 else None,
        "best_thr_ACC": float(best_t) if best_t is not None else None,
        "best_ACC": float(best_acc) if best_acc is not None else None,
        "confusion@0.50": confusion_matrix(yt, y_hat).tolist(),
    }

summary = {"overall": metrics(np.ones(len(df), dtype=bool))}
for b in df["bucket"].unique():
    summary[b] = metrics(df["bucket"].values == b)

df_out = df.copy()
df_out["p_ml"] = proba
df_out["y_ml@0.50"] = (proba >= 0.5).astype(int)
df_out.to_csv(OUT/"preds_from_csv.csv", index=False)
Path(OUT/"metrics_from_csv.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("saved CSV ->", OUT/"preds_from_csv.csv"); print("saved JSON ->", OUT/"metrics_from_csv.json")
