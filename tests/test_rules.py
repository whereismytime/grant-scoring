import pytest
from pydantic import ValidationError

from app.domain.rules import amount_by_distance, hard_decline, rule_approve, rule_decision
from app.domain.schemas import Applicant


def applicant(distance_km=10.0, residency_years_ie=1.0, is_immigrant=False, parents_status="low"):
    return Applicant(
        distance_km=distance_km,
        residency_years_ie=residency_years_ie,
        is_immigrant=is_immigrant,
        parents_status=parents_status,
    )


@pytest.mark.parametrize("d, amount", [(0, 2560), (30, 2560), (30.01, 5697), (80, 5697)])
def test_amount_by_distance_boundary(d, amount):
    assert amount_by_distance(d) == amount


def test_hard_decline_high_income():
    assert hard_decline(applicant(parents_status="high")) == (True, "parents high income")


@pytest.mark.parametrize("years, declined", [(2.99, False), (3.0, True), (5.0, True)])
def test_hard_decline_immigrant_residency_boundary(years, declined):
    assert hard_decline(applicant(is_immigrant=True, residency_years_ie=years))[0] is declined


def test_residency_does_not_matter_for_non_immigrants():
    assert hard_decline(applicant(is_immigrant=False, residency_years_ie=5.0)) == (False, None)


def test_low_income_approved():
    assert rule_decision(applicant(parents_status="low")) == (True, "low income")


@pytest.mark.parametrize("d, approved", [(30, False), (30.01, True)])
def test_middle_income_distance_boundary(d, approved):
    assert rule_decision(applicant(parents_status="middle", distance_km=d))[0] is approved


@pytest.mark.parametrize(
    "kwargs, approved",
    [
        (dict(parents_status="low"), True),
        (dict(parents_status="high"), False),
        (dict(parents_status="middle", distance_km=31), True),
        (dict(parents_status="middle", distance_km=31, is_immigrant=True, residency_years_ie=3), False),
        (dict(parents_status="low", is_immigrant=True, residency_years_ie=2.9), True),
    ],
)
def test_rule_approve(kwargs, approved):
    assert rule_approve(applicant(**kwargs)) is approved


@pytest.mark.parametrize(
    "kwargs",
    [dict(distance_km=-1), dict(residency_years_ie=-0.1), dict(parents_status="unknown")],
)
def test_schema_rejects_invalid_input(kwargs):
    with pytest.raises(ValidationError):
        applicant(**kwargs)
