import joblib
import pandas as pd
from functools import lru_cache

@lru_cache(maxsize=1)
def _load_model(path="app/models/model.pkl"):
    return joblib.load(path)

class MLModel:
    def __init__(self, path="app/models/model.pkl"):
        self.path = path
        self.m = _load_model(path)

    def prob(self, x: dict) -> float:
        X = pd.DataFrame([x])
        p = self.m.predict_proba(X)[0, 1]
        return float(p)
