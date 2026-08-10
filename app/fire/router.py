"""HTTP-роут детекции огня/дыма: POST /detect/fire."""

from fastapi import APIRouter, HTTPException, Request, status

from app.fire.detector import InvalidImageError
from app.fire.schemas import FireRequest, FireResponse

router = APIRouter(tags=["fire"])


@router.post("/detect/fire", response_model=FireResponse)
def detect_fire(payload: FireRequest, request: Request) -> FireResponse:
    """Классифицирует кадр как `fire`, `smoke` или `normal`."""
    detector = request.app.state.detectors["fire"]
    try:
        result = detector.predict(payload.frame)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid base64 image",
        ) from exc

    return FireResponse(**result)
