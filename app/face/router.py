"""HTTP-роуты лиц.

- POST   /detect/face         — распознать лица в кадре (поток от брокера, base64);
- POST   /faces               — записать человека (multipart, 1..N фото);
- POST   /faces/{id}/images   — догрузить ещё фото существующему человеку;
- GET    /faces               — список записанных (метаданные + кол-во фото);
- GET    /faces/{id}/image    — первичное фото человека;
- DELETE /faces/{id}          — удалить из базы.
"""

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from app.config import settings
from app.face.detector import (
    DuplicateNameError,
    EnrollmentError,
    FaceNotFoundError,
    InvalidImageError,
)
from app.face.schemas import FaceRecord, FaceRequest, FaceResponse

router = APIRouter(tags=["face"])


def _read_images(files: list[UploadFile]) -> list[bytes]:
    """Читает файлы, проверяет размер каждого (413 при превышении)."""
    data = []
    for f in files:
        raw = f.file.read()
        if len(raw) > settings.face_max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"image is larger than {settings.face_max_upload_bytes} bytes",
            )
        data.append(raw)
    return data


@router.post("/detect/face", response_model=FaceResponse)
def detect_face(payload: FaceRequest, request: Request) -> FaceResponse:
    """Находит лица в кадре и сопоставляет с базой known_faces/."""
    detector = request.app.state.detectors["face"]
    try:
        result = detector.predict(payload.frame)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid base64 image",
        ) from exc

    return FaceResponse(**result)


@router.post("/faces", response_model=FaceRecord, status_code=status.HTTP_201_CREATED)
def enroll_face(
    request: Request,
    name: str = Form(..., description="Имя человека."),
    images: list[UploadFile] = File(..., description="1..N фото, на каждом одно чёткое лицо."),
) -> FaceRecord:
    """Записывает человека в базу: на каждом годном фото (ровно одно лицо) считает эмбеддинг."""
    data = _read_images(images)
    detector = request.app.state.detectors["face"]
    try:
        record = detector.add_face(name, data)
    except DuplicateNameError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (InvalidImageError, EnrollmentError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc) or "invalid image",
        ) from exc
    return FaceRecord(**record)


@router.post("/faces/{face_id}/images", response_model=FaceRecord)
def add_face_images(
    face_id: int,
    request: Request,
    images: list[UploadFile] = File(..., description="Дополнительные фото человека."),
) -> FaceRecord:
    """Догружает эталонные фото существующему человеку (другие ракурсы/условия)."""
    data = _read_images(images)
    detector = request.app.state.detectors["face"]
    try:
        record = detector.add_images(face_id, data)
    except FaceNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (InvalidImageError, EnrollmentError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc) or "invalid image",
        ) from exc
    return FaceRecord(**record)


@router.get("/faces", response_model=list[FaceRecord])
def list_faces(request: Request) -> list[FaceRecord]:
    """Список записанных людей (метаданные + кол-во эталонных фото)."""
    detector = request.app.state.detectors["face"]
    return [FaceRecord(**r) for r in detector.list_faces()]


@router.get("/faces/{face_id}/image")
def get_face_image(face_id: int, request: Request) -> FileResponse:
    """Отдаёт первичное (первое) фото записанного человека."""
    detector = request.app.state.detectors["face"]
    try:
        path = detector.get_face_image(face_id)
    except FaceNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FileResponse(path)


@router.delete("/faces/{face_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_face(face_id: int, request: Request) -> Response:
    """Удаляет человека из базы (все фото + эмбеддинги)."""
    detector = request.app.state.detectors["face"]
    try:
        detector.delete_face(face_id)
    except FaceNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
