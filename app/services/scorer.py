from app.domain.schemas import Applicant, ScoreResponse
from app.domain.rules import amount_by_distance, hard_decline, rule_decision
from app.services.ml_model import MLModel

_ml = MLModel(path="app/models/model.pkl")

def score(a: Applicant, use_ml: bool = True, threshold: float = 0.5) -> ScoreResponse:
    # 1) ML вероятность заранее, если включён ML
    prob = None
    if use_ml:
        feats = {
            "distance_km": a.distance_km,
            "residency_years_ie": a.residency_years_ie,
            "is_immigrant": int(a.is_immigrant),
            "parents_status": a.parents_status,
        }
        prob = _ml.prob(feats)

    # 2) Жёсткие отказы всегда сильнее, но prob не скрываем
    hd, r = hard_decline(a)
    if hd:
        return ScoreResponse(
            approved=False, decision="decline", amount=0,
            prob=prob, reasons=[r, "rule override"]
        )

    # 3) Базовое правило + возможное ML-вeto/approve
    ok_rule, reason_rule = rule_decision(a)
    ok = ok_rule
    reasons = [reason_rule]

    if use_ml:
        ok_ml = (prob is not None) and (prob >= threshold)
        if ok_ml != ok_rule:
            # Явно фиксируем, кто переиграл
            reasons.append("ml ≥ thr" if ok_ml else "ml < thr")
        ok = ok_ml

    amount = amount_by_distance(a.distance_km) if ok else 0
    return ScoreResponse(
        approved=ok, decision="approve" if ok else "decline",
        amount=amount, prob=prob, reasons=reasons
    )
