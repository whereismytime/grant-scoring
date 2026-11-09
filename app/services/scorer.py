from app.domain.schemas import Applicant, ScoreResponse
from app.domain.rules import amount_by_distance, hard_decline, rule_decision
from app.services.ml_model import MLModel, get_model


def score(
    a: Applicant,
    use_ml: bool = True,
    threshold: float = 0.5,
    model: MLModel | None = None,
) -> ScoreResponse:
    # 1) Compute ML probability first, if ML mode is enabled.
    #    The model is loaded lazily, so rules mode works without model.pkl.
    prob = None
    if use_ml:
        prob = (model or get_model()).prob(a)

    # 2) Hard declines always take precedence, but we still show the ML probability
    hd, r = hard_decline(a)
    if hd:
        return ScoreResponse(
            approved=False, decision="decline", amount=0,
            prob=prob, reasons=[r, "rule override"]
        )

    # 3) Apply rule-based decision + possible ML override
    ok_rule, reason_rule = rule_decision(a)
    ok = ok_rule
    reasons = [reason_rule]

    if use_ml:
        ok_ml = prob >= threshold
        if ok_ml != ok_rule:
            # Explicitly record who overrode the decision
            reasons.append("ml ≥ thr" if ok_ml else "ml < thr")
        ok = ok_ml

    amount = amount_by_distance(a.distance_km) if ok else 0
    return ScoreResponse(
        approved=ok, decision="approve" if ok else "decline",
        amount=amount, prob=prob, reasons=reasons
    )
