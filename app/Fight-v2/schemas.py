"""Pydantic-модели запроса/ответа для детекции драк."""

from typing import Literal

from pydantic import BaseModel, Field

from app.config import settings

_FRAMES_DESCRIPTION = (
    f"Ровно {settings.expected_frames} base64-JPEG кадров в хронологическом порядке, "
    f"РАВНОМЕРНО распределённых по временному окну ~{settings.fight_window_seconds:g} секунд "
    f"(например, np.linspace(0, N-1, {settings.expected_frames}).astype(int) по индексам кадров "
    f"буфера, покрывающего последние {settings.fight_window_seconds:g}с потока). "
    f"НЕ отправляйте {settings.expected_frames} последовательных кадров подряд (~0.5с при 30fps) — "
    f"модель X3D-M на таком узком окне даёт плоский, не информативный score независимо от "
    f"содержимого клипа (см. app/config.py, комментарий у fight_window_seconds)."
)


class DetectionRequest(BaseModel):
    frames: list[str] = Field(
        ...,
        description=_FRAMES_DESCRIPTION,
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
