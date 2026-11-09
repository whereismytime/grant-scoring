import numpy as np
import pandas as pd
import pytest

from app.domain.schemas import Applicant
from app.ml import eval_from_csv, generate_scenarios, train
from app.ml.features import NUM_COLS, RAW_COLS, build_features, rule_labels
from app.ml.generate_scenarios import make_dataset
from app.services.ml_model import MLModel, ModelNotAvailableError


def test_build_features_does_not_mutate_input():
    df = pd.DataFrame([{"distance_km": 31.0, "residency_years_ie": 3.0,
                        "is_immigrant": True, "parents_status": "middle"}])
    before = df.copy()
    out = build_features(df)
    pd.testing.assert_frame_equal(df, before)
    assert out.loc[0, ["dist_gt_30", "imm_ge_3", "mid_x_dist", "is_immigrant"]].tolist() == [1, 1, 1, 1]
    assert set(NUM_COLS) <= set(out.columns)


def test_model_scores_raw_applicant(trained_model_path):
    """Regression: the saved model must accept only the raw applicant fields."""
    model = MLModel.load(trained_model_path)
    a = Applicant(distance_km=20, residency_years_ie=2.0, is_immigrant=True, parents_status="low")
    p = model.prob(a)
    assert 0.0 <= p <= 1.0
    assert p == pytest.approx(model.prob(a.model_dump()))


def test_model_reproduces_rules(trained_model_path):
    df = make_dataset(n_each=300, noise=0.0, seed=1)
    p = MLModel.load(trained_model_path).prob_batch(df)
    assert ((p >= 0.5).astype(int) == rule_labels(df)).mean() > 0.97


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelNotAvailableError):
        MLModel.load(tmp_path / "nope.pkl")


def test_scenario_labels_match_rules_without_noise():
    """Regression: every bucket must be labeled by the business rules."""
    df = make_dataset(n_each=1000, noise=0.0)
    assert (df["approved"].values == rule_labels(df)).all()


def test_scenario_noise_flips_roughly_the_requested_share():
    clean = make_dataset(n_each=1000, noise=0.0)
    noisy = make_dataset(n_each=1000, noise=0.05)
    pd.testing.assert_frame_equal(clean[RAW_COLS], noisy[RAW_COLS])
    flipped = (clean["approved"] != noisy["approved"]).mean()
    assert 0.03 < flipped < 0.07


def test_pipeline_end_to_end(tmp_path):
    data, model, holdout, out = (tmp_path / n for n in ("data.csv", "model.pkl", "holdout.csv", "out"))
    generate_scenarios.main(["--out", str(data), "--n-each", "300"])
    train.main(["--data", str(data), "--model", str(model), "--holdout", str(holdout)])

    full, held = pd.read_csv(data), pd.read_csv(holdout)
    assert len(held) == pytest.approx(0.2 * len(full), abs=1)

    eval_from_csv.main(["--data", str(holdout), "--model", str(model), "--out-dir", str(out)])
    assert (out / "metrics_from_csv.json").exists()
    preds = pd.read_csv(out / "preds_from_csv.csv")
    assert len(preds) == len(held)
    assert np.all((preds["p_ml"] >= 0) & (preds["p_ml"] <= 1))


def test_eval_without_labels_or_buckets_falls_back_to_rules(trained_model_path):
    df = make_dataset(n_each=200, noise=0.0)[RAW_COLS]
    summary, preds = eval_from_csv.evaluate(df, MLModel.load(trained_model_path))
    assert summary["overall"]["N"] == len(df)
    assert set(preds["bucket"]) <= {
        "DECLINE_high_income", "DECLINE_immigrant>=3y", "APPROVE_easy",
        "BORDER_middle_around_30", "BORDER_immigrant_around_3y", "MIX_random",
    }
