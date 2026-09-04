"""Pydantic-модели запроса/ответа для распознавания и базы лиц."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FaceBox(BaseModel):
    box: list[float] = Field(..., description="Рамка лица [x1, y1, x2, y2].")
    det_confidence: float = Field(..., description="Уверенность детектора лица.")
    identity: str = Field(..., description="Имя из базы или 'unknown'.")
    identity_id: int | None = Field(None, description="ID записи из базы или null, если 'unknown'.")
    similarity: float = Field(..., description="Косинусная близость к ближайшему лицу из базы.")
    sex: Literal["M", "F"] | None = Field(
        None, description="Пол по кадру (genderage). null, если модель не дала ответ."
    )
    age: int | None = Field(
        None, description="Возраст по кадру (genderage), годы. Оценка грубая, ±5-10 лет."
    )


class FaceRequest(BaseModel):
    """Один кадр (`frame`) ЛИБО пачка кадров (`frames`) — ровно одно из двух полей."""

    frame: str | None = Field(None, description="Один base64-JPEG кадр.")
    frames: list[str] | None = Field(
        None, description="Пачка base64-JPEG кадров (напр. зоны обхода PTZ)."
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "FaceRequest":
        if (self.frame is None) == (self.frames is None):
            raise ValueError("provide exactly one of 'frame' or 'frames'")
        if self.frames is not None and not self.frames:
            raise ValueError("'frames' must not be empty")
        return self


class FaceResponse(BaseModel):
    """Ответ на одиночный кадр (`frame`)."""

    faces: list[FaceBox]
    count: int
    processing_ms: float


class FaceFrameResult(BaseModel):
    """Результат по одному кадру пачки (время — общее на запрос, здесь его нет)."""

    faces: list[FaceBox]
    count: int


class FaceBatchResponse(BaseModel):
    """Ответ на пачку (`frames`). Порядок results = порядок присланных кадров."""

    results: list[FaceFrameResult]
    processing_ms: float


class FaceRecord(BaseModel):
    """Запись из базы лиц (без байтов картинки — она отдаётся отдельным эндпоинтом)."""
    id: int = Field(..., description="Стабильный ID человека в базе.")
    name: str = Field(..., description="Имя человека.")
    created_at: str = Field(..., description="Когда добавлен (ISO 8601, UTC).")
    photos: int = Field(..., description="Сколько эталонных фото у человека.")
