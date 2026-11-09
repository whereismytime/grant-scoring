from fastapi import APIRouter, HTTPException, Query
from app.domain.schemas import Applicant, ScoreResponse
from app.services.ml_model import ModelNotAvailableError
from app.services.scorer import score

router = APIRouter()


@router.post("/score", response_model=ScoreResponse)
def score_endpoint(
    applicant: Applicant,
    use_ml: bool = Query(True),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    try:
        return score(applicant, use_ml=use_ml, threshold=threshold)
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
