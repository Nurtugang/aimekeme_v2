"""HTTP-роут детекции людей: POST /detect/persons."""

from fastapi import APIRouter, HTTPException, Request, status

from app.persons.detector import InvalidFrameError
from app.persons.schemas import PersonsRequest, PersonsResponse

router = APIRouter(tags=["persons"])


@router.post("/detect/persons", response_model=PersonsResponse)
def detect_persons(payload: PersonsRequest, request: Request) -> PersonsResponse:
    """Находит людей на каждом из N кадров, определяет цвет верха/низа."""
    detector = request.app.state.detectors["persons"]
    try:
        result = detector.predict(payload.frames)
    except InvalidFrameError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid base64 in frame {exc.index}",
        ) from exc

    return PersonsResponse(**result)
