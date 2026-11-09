import joblib
import pytest

from app.ml.generate_scenarios import make_dataset
from app.ml.train import train
from app.services.ml_model import MODEL_PATH_ENV


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory):
    """Train a small model once per test session (never touches app/models)."""
    df = make_dataset(n_each=500, noise=0.0, seed=0)
    pipe, _, _ = train(df)
    path = tmp_path_factory.mktemp("model") / "model.pkl"
    joblib.dump(pipe, path)
    return path


@pytest.fixture
def with_model(monkeypatch, trained_model_path):
    monkeypatch.setenv(MODEL_PATH_ENV, str(trained_model_path))
    return trained_model_path


@pytest.fixture
def without_model(monkeypatch, tmp_path):
    monkeypatch.setenv(MODEL_PATH_ENV, str(tmp_path / "missing.pkl"))
