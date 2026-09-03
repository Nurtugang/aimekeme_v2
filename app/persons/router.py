"""HTTP-роут детекции людей: POST /detect/persons."""

from fastapi import APIRouter, HTTPException, Request, status

from app.persons.detector import InvalidFrameError
from app.persons.schemas import PersonsRequest, PersonsResponse

router = APIRouter(tags=["persons"])


@router.post("/detect/persons", response_model=PersonsResponse)
def detect_persons(payload: PersonsRequest, request: Request) -> PersonsResponse:
    """Находит людей на каждом из N кадров, определяет цвет верха/низа.

    Опциональный `query` (`{"top": "blue", "bottom": "black"}`) фильтрует
    людей по названию цвета -- в ответе остаются только совпадения.
    """
    detector = request.app.state.detectors["persons"]
    query = payload.query.model_dump() if payload.query else None
    try:
        result = detector.predict(payload.frames, query)
    except InvalidFrameError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid base64 in frame {exc.index}",
        ) from exc

    return PersonsResponse(**result)
