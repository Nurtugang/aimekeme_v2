"""Pydantic-модели запроса/ответа для детекции людей + цвета одежды."""

from pydantic import BaseModel, Field


class PersonBox(BaseModel):
    track_id: int | None = Field(
        None,
        description="Идентификатор человека, общий для всех кадров запроса -- по нему "
                    "платформа группирует и усредняет цвет за окно. null, если трекер "
                    "ещё не подтвердил след. ВАЖНО: сшивать людей между разными "
                    "запросами по track_id нельзя -- номера не повторяются, в каждом "
                    "ответе они новые даже для тех же самых людей.",
    )
    box: list[float] = Field(
        ..., description="Рамка человека [x1, y1, x2, y2], доли кадра 0..1."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность детектора.")
    top_color: str = Field(..., description="Ближайшее название цвета верха из палитры.")
    top_hsv: list[float] = Field(
        ..., description="Сырой HSV верха: [hue 0-360, saturation 0-1, value 0-1]."
    )
    bottom_visible: bool = Field(
        ..., description="Виден ли низ (колени в кадре). false -- человек сидит или обрезан."
    )
    bottom_color: str | None = Field(
        None, description="Название цвета низа из палитры; null, если низ не виден."
    )
    bottom_hsv: list[float] | None = Field(
        None,
        description="Сырой HSV низа: [hue 0-360, saturation 0-1, value 0-1]; null, если не виден.",
    )


class FrameResult(BaseModel):
    persons: list[PersonBox]
    count: int


class PersonsRequest(BaseModel):
    frames: list[str] = Field(
        ...,
        description="1..N base64-JPEG кадров -- подряд идущие кадры ОДНОЙ камеры (окно). "
                    "Несколько камер = несколько параллельных запросов.",
    )


class PersonsResponse(BaseModel):
    results: list[FrameResult] = Field(..., description="По одному результату на кадр, в том же порядке.")
    processing_ms: float = Field(..., description="Время сервера на decode + inference, мс.")
