"""Pydantic-модели запроса/ответа для распознавания лиц."""

from pydantic import BaseModel, Field


class FaceBox(BaseModel):
    box: list[float] = Field(..., description="Рамка лица [x1, y1, x2, y2].")
    det_confidence: float = Field(..., description="Уверенность детектора лица.")
    identity: str = Field(..., description="Имя из базы или 'unknown'.")
    similarity: float = Field(..., description="Косинусная близость к ближайшему лицу из базы.")


class FaceRequest(BaseModel):
    frame: str = Field(..., description="Один base64-JPEG кадр.")


class FaceResponse(BaseModel):
    faces: list[FaceBox]
    count: int
    processing_ms: float
