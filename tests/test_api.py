import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)

LOW_NEAR = {"distance_km": 20, "residency_years_ie": 2.0, "is_immigrant": True, "parents_status": "low"}
MIDDLE_NEAR = {"distance_km": 20, "residency_years_ie": 1.0, "is_immigrant": False, "parents_status": "middle"}
HIGH = {"distance_km": 50, "residency_years_ie": 1.0, "is_immigrant": False, "parents_status": "high"}


def test_health():
    assert client.get("/").json() == {"status": "ok"}


def test_rules_mode_works_without_model(without_model):
    r = client.post("/score", params={"use_ml": False}, json=LOW_NEAR)
    assert r.status_code == 200
    assert r.json() == {
        "approved": True, "decision": "approve", "amount": 2560, "prob": None, "reasons": ["low income"],
    }


def test_ml_mode_without_model_returns_503(without_model):
    r = client.post("/score", json=LOW_NEAR)
    assert r.status_code == 503
    assert "python -m app.ml.train" in r.json()["detail"]


def test_ml_mode_scores_with_model(with_model):
    r = client.post("/score", json=LOW_NEAR)
    assert r.status_code == 200
    body = r.json()
    assert body["approved"] is True
    assert body["amount"] == 2560
    assert 0.5 <= body["prob"] <= 1.0
    assert body["reasons"] == ["low income"]


def test_hard_decline_overrides_ml(with_model):
    body = client.post("/score", params={"threshold": 0.0}, json=HIGH).json()
    assert body["decision"] == "decline"
    assert body["amount"] == 0
    assert body["prob"] is not None
    assert body["reasons"] == ["parents high income", "rule override"]


def test_threshold_lets_ml_override_rules(with_model):
    body = client.post("/score", params={"threshold": 0.0}, json=MIDDLE_NEAR).json()
    assert body["approved"] is True
    assert body["amount"] == 2560
    assert body["reasons"] == ["middle & distance ≤30km", "ml ≥ thr"]


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_threshold_out_of_range_is_rejected(threshold):
    assert client.post("/score", params={"threshold": threshold}, json=LOW_NEAR).status_code == 422


def test_invalid_applicant_is_rejected():
    assert client.post("/score", json={**LOW_NEAR, "parents_status": "rich"}).status_code == 422
