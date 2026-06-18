"""Pydantic-модели запроса/ответа для детекции драк."""

from typing import Literal

from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    frames: list[str] = Field(
        ...,
        description="Ровно 16 base64-JPEG кадров в хронологическом порядке.",
        examples=[["<base64_jpg>", "...", "<base64_jpg>"]],
    )


class DetectionResponse(BaseModel):
    label: Literal["fight", "normal"] = Field(..., description="Класс клипа.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Вероятность предсказанного класса (0..1)."
    )
    processing_ms: float = Field(
        ..., description="Время сервера на decode + inference, мс."
    )
