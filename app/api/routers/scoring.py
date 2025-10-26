from fastapi import APIRouter, Query
from app.domain.schemas import Applicant, ScoreResponse
from app.services.scorer import score

router = APIRouter()

@router.post("/score", response_model=ScoreResponse)
def score_endpoint(applicant: Applicant, use_ml: bool = Query(True)):
    return score(applicant, use_ml=use_ml, threshold=0.5)
