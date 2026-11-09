import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.ml.features import RAW_COLS

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
MODEL_PATH_ENV = "GRANT_MODEL_PATH"


class ModelNotAvailableError(RuntimeError):
    """The trained model file is missing, so ML scoring is unavailable."""


def model_path() -> Path:
    return Path(os.getenv(MODEL_PATH_ENV) or DEFAULT_MODEL_PATH)


def _to_frame(x) -> pd.DataFrame:
    if isinstance(x, BaseModel):
        x = x.model_dump()
    if isinstance(x, dict):
        x = pd.DataFrame([x])
    return x[RAW_COLS]


class MLModel:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def load(cls, path: str | Path) -> "MLModel":
        path = Path(path)
        if not path.exists():
            raise ModelNotAvailableError(
                f"ML model not found at {path}. Train it with `python -m app.ml.train` "
                "or score in rules mode (use_ml=false)."
            )
        return cls(joblib.load(path))

    def prob(self, x) -> float:
        """Approval probability for one applicant (Applicant or dict of raw fields)."""
        return float(self.prob_batch(_to_frame(x))[0])

    def prob_batch(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(_to_frame(df))[:, 1]


@lru_cache(maxsize=None)
def _load_cached(path: str) -> MLModel:
    # A failed load raises and is not cached, so a model trained later is picked up.
    return MLModel.load(path)


def get_model() -> MLModel:
    """Load the model lazily on first use (path from GRANT_MODEL_PATH or the default)."""
    return _load_cached(str(model_path()))
