"""HTTP-роут подсчёта людей: POST /detect/counting."""

from fastapi import APIRouter, HTTPException, Request, status

from app.counting.detector import InvalidImageError
from app.counting.schemas import CountingRequest, CountingResponse

router = APIRouter(tags=["counting"])


@router.post("/detect/counting", response_model=CountingResponse)
def detect_counting(payload: CountingRequest, request: Request) -> CountingResponse:
    """Считает людей в кадре (детектор person, count = число боксов)."""
    detector = request.app.state.detectors["counting"]
    try:
        result = detector.predict(payload.frame)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid base64 image",
        ) from exc

    return CountingResponse(**result)
