"""HTTP-роут распознавания лиц: POST /detect/face."""

from fastapi import APIRouter, HTTPException, Request, status

from app.face.detector import InvalidImageError
from app.face.schemas import FaceRequest, FaceResponse

router = APIRouter(tags=["face"])


@router.post("/detect/face", response_model=FaceResponse)
def detect_face(payload: FaceRequest, request: Request) -> FaceResponse:
    """Находит лица в кадре и сопоставляет с мини-базой known_faces/."""
    detector = request.app.state.detectors["face"]
    try:
        result = detector.predict(payload.frame)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid base64 image",
        ) from exc

    return FaceResponse(**result)
