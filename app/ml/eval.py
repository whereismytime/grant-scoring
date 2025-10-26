import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from app.domain.schemas import Applicant
from app.domain.rules import hard_decline, rule_decision
from app.services.ml_model import MLModel

np.random.seed(42)

def gen_dataset(n=10000):
    distance_km = np.random.randint(0, 81, n)
    residency_years_ie = np.round(np.random.uniform(0, 6, n), 1)
    is_immigrant = (np.random.rand(n) < 0.35)
    parents_status = np.random.choice(["low", "middle", "high"], size=n, p=[0.5, 0.35, 0.15])
    df = pd.DataFrame({
        "distance_km": distance_km,
        "residency_years_ie": residency_years_ie,
        "is_immigrant": is_immigrant,
        "parents_status": parents_status
    })
    return df

def rule_approve(a: Applicant) -> int:
    hd, _ = hard_decline(a)
    if hd:
        return 0
    ok, _ = rule_decision(a)
    return int(ok)

def main():
    df = gen_dataset(20000)
    ml = MLModel("app/models/model.pkl")
    # The “true” label is based on what train.py generates (rules).
    # If you have a real dataset, replace this with the actual approved column.
    y_true = []
    y_rule = []
    y_ml = []
    p_ml = []
    for r in df.itertuples(index=False):
        a = Applicant(
            distance_km=float(r.distance_km),
            residency_years_ie=float(r.residency_years_ie),
            is_immigrant=bool(r.is_immigrant),
            parents_status=r.parents_status
        )
        y_r = rule_approve(a)
        y_rule.append(y_r)
        x = {
            "distance_km": a.distance_km,
            "residency_years_ie": a.residency_years_ie,
            "is_immigrant": int(a.is_immigrant),
            "parents_status": a.parents_status
        }
        p = ml.prob(x)
        p_ml.append(p)
        y_ml.append(int(p >= 0.5))
        y_true.append(y_r)  # replace with real labels if available

    y_true = np.array(y_true)
    y_rule = np.array(y_rule)
    y_ml = np.array(y_ml)
    p_ml = np.array(p_ml)

    print("AUC (ML vs y_true):", round(roc_auc_score(y_true, p_ml), 3))
    print("Accuracy (Rules):", round(accuracy_score(y_true, y_rule), 3),
          "Accuracy (ML):", round(accuracy_score(y_true, y_ml), 3))
    diff = (y_rule != y_ml)
    print("Share of disagreements (ML vs Rules):", round(diff.mean(), 3))
    print("Confusion Matrix (ML vs Rules):\n", confusion_matrix(y_rule, y_ml))

    # Display 5 examples where ML ≠ rule
    show = df.loc[diff].copy()
    show["rule"] = y_rule[diff]
    show["ml"] = y_ml[diff]
    show["p_ml"] = np.round(p_ml[diff], 3)
    print("\nExamples of discrepancies (first 5):")
    print(show.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
