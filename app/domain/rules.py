from .schemas import Applicant

def amount_by_distance(d: float) -> int:
    return 2560 if d <= 30 else 5697

def hard_decline(a: Applicant) -> tuple[bool, str | None]:
    if a.parents_status == "high":
        return True, "parents high income"
    if a.is_immigrant and a.residency_years_ie >= 3.0:
        return True, "immigrant ≥3y"
    return False, None

def rule_decision(a: Applicant) -> tuple[bool, str]:
    if a.parents_status == "low":
        return True, "low income"
    if a.parents_status == "middle":
        return (a.distance_km > 30,
                "middle & distance >30km" if a.distance_km > 30 else "middle & distance ≤30km")
    return False, "invalid parents_status"
