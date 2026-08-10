"""Pydantic-модели запроса/ответа для детекции огня/дыма."""

from typing import Literal

from pydantic import BaseModel, Field


class FireRequest(BaseModel):
    frame: str = Field(..., description="Один base64-JPEG кадр.")


class FireResponse(BaseModel):
    label: Literal["fire", "smoke", "normal"] = Field(..., description="Класс кадра.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Вероятность предсказанного класса (0..1)."
    )
    processing_ms: float = Field(
        ..., description="Время сервера на decode + inference, мс."
    )
