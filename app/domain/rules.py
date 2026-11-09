"""Business rules: the single source of truth for decisions and labels.

The functions only read attributes (``parents_status``, ``is_immigrant``,
``residency_years_ie``, ``distance_km``), so they accept an ``Applicant`` as
well as any row-like object with the same fields (e.g. a pandas ``itertuples``
row). Data generators and evaluation scripts reuse them instead of copying
the logic.
"""

from .schemas import Applicant

DISTANCE_THRESHOLD_KM = 30.0
IMMIGRANT_RESIDENCY_LIMIT_YEARS = 3.0
AMOUNT_NEAR = 2560
AMOUNT_FAR = 5697


def amount_by_distance(d: float) -> int:
    return AMOUNT_NEAR if d <= DISTANCE_THRESHOLD_KM else AMOUNT_FAR


def hard_decline(a: Applicant) -> tuple[bool, str | None]:
    if a.parents_status == "high":
        return True, "parents high income"
    if a.is_immigrant and a.residency_years_ie >= IMMIGRANT_RESIDENCY_LIMIT_YEARS:
        return True, "immigrant ≥3y"
    return False, None


def rule_decision(a: Applicant) -> tuple[bool, str]:
    if a.parents_status == "low":
        return True, "low income"
    if a.parents_status == "middle":
        far = a.distance_km > DISTANCE_THRESHOLD_KM
        return far, "middle & distance >30km" if far else "middle & distance ≤30km"
    return False, "invalid parents_status"


def rule_approve(a: Applicant) -> bool:
    """Final rules-only decision: hard declines first, then the base rule."""
    declined, _ = hard_decline(a)
    if declined:
        return False
    ok, _ = rule_decision(a)
    return ok
