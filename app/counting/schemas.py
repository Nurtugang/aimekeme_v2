"""Pydantic-модели запроса/ответа для подсчёта людей."""

from typing import Literal

from pydantic import BaseModel, Field


class CountingRequest(BaseModel):
    frame: str = Field(..., description="Один base64-JPEG кадр.")


class CountingResponse(BaseModel):
    label: Literal["person"] = Field("person", description="Что считаем.")
    count: int = Field(..., ge=0, description="Число обнаруженных людей на кадре.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Средний score детектора по найденным людям (0.0, если никого).",
    )
    processing_ms: float = Field(
        ..., description="Время сервера на decode + inference, мс."
    )
