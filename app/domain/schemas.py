from pydantic import BaseModel, Field
from typing import Literal

class Applicant(BaseModel):
    distance_km: float = Field(ge=0)
    residency_years_ie: float = Field(ge=0)
    is_immigrant: bool
    parents_status: Literal["low","middle","high"]

class ScoreResponse(BaseModel):
    approved: bool
    decision: Literal["approve","decline"]
    amount: int
    prob: float | None = None
    reasons: list[str]
