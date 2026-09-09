"""HTTP-роут детекции драк: POST /detect/fight."""

from fastapi import APIRouter, HTTPException, Request, status

from app.fight.detector import InvalidFrameError
from app.fight.schemas import DetectionRequest, DetectionResponse
from app.config import settings

router = APIRouter(tags=["fight"])


@router.post("/detect/fight", response_model=DetectionResponse)
def detect_fight(payload: DetectionRequest, request: Request) -> DetectionResponse:
    """Классифицирует клип из 16 кадров как `fight` или `normal`."""
    n_frames = len(payload.frames)
    if n_frames != settings.expected_frames:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Expected {settings.expected_frames} frames, got {n_frames}",
        )

    detector = request.app.state.detectors["fight"]
    try:
        result = detector.predict(payload.frames)
    except InvalidFrameError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid base64 in frame {exc.index}",
        ) from exc

    return DetectionResponse(**result)
