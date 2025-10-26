import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

Path("app/models").mkdir(parents=True, exist_ok=True)
CSV = "data/raw/generated_applicants.csv"
df = pd.read_csv(CSV)

if "approved" in df.columns:
    y = df["approved"].astype(int).values
else:
    def rule_label(r):
        if bool(r.is_immigrant) and r.residency_years_ie >= 3: return 0
        if r.parents_status == "high": return 0
        if r.parents_status == "middle": return int(r.distance_km > 30)
        return 1
    y = df.apply(rule_label, axis=1).astype(int).values

df["is_immigrant"] = df["is_immigrant"].astype(int)

X = df[["distance_km","residency_years_ie","is_immigrant","parents_status"]].copy()
X["dist_gt_30"] = (X["distance_km"] > 30).astype(int)
X["imm_ge_3"]   = (X["residency_years_ie"] >= 3).astype(int)
X["mid_x_dist"] = ((X["parents_status"]=="middle") & (X["distance_km"]>30)).astype(int)

num_cols = ["distance_km","residency_years_ie","is_immigrant",
            "dist_gt_30","imm_ge_3","mid_x_dist"]
cat_cols = ["parents_status"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X[cat_cols+num_cols], y, test_size=0.2, random_state=42, stratify=y
)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols),
])

base = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
clf = CalibratedClassifierCV(base, method="isotonic", cv=3)

pipe = Pipeline([
    ("pre", pre),
    ("clf", clf),
])

pipe.fit(X_tr, y_tr)
auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
print(f"AUC = {auc:.3f}")

joblib.dump(pipe, "app/models/model.pkl")
print("saved -> app/models/model.pkl")
