"""Pydantic-модели запроса/ответа для распознавания и базы лиц."""

from pydantic import BaseModel, Field


class FaceBox(BaseModel):
    box: list[float] = Field(..., description="Рамка лица [x1, y1, x2, y2].")
    det_confidence: float = Field(..., description="Уверенность детектора лица.")
    identity: str = Field(..., description="Имя из базы или 'unknown'.")
    identity_id: int | None = Field(None, description="ID записи из базы или null, если 'unknown'.")
    similarity: float = Field(..., description="Косинусная близость к ближайшему лицу из базы.")


class FaceRequest(BaseModel):
    frame: str = Field(..., description="Один base64-JPEG кадр.")


class FaceResponse(BaseModel):
    faces: list[FaceBox]
    count: int
    processing_ms: float


class FaceRecord(BaseModel):
    """Запись из базы лиц (без байтов картинки — она отдаётся отдельным эндпоинтом)."""
    id: int = Field(..., description="Стабильный ID человека в базе.")
    name: str = Field(..., description="Имя человека.")
    created_at: str = Field(..., description="Когда добавлен (ISO 8601, UTC).")
    photos: int = Field(..., description="Сколько эталонных фото у человека.")
