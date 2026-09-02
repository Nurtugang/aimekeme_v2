"""HTTP-роут вовлечённости на лекции: POST /detect/attention."""

from fastapi import APIRouter, HTTPException, Request, status

from app.attention.detector import InvalidImageError
from app.attention.schemas import AttentionRequest, AttentionResponse

router = APIRouter(tags=["attention"])


@router.post("/detect/attention", response_model=AttentionResponse)
def detect_attention(payload: AttentionRequest, request: Request) -> AttentionResponse:
    """Оценивает вовлечённость каждого человека в кадре по взгляду, позе и времени."""
    detector = request.app.state.detectors["attention"]
    try:
        result = detector.predict(payload.frame, payload.camera_id)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid base64 image",
        ) from exc

    return AttentionResponse(**result)
