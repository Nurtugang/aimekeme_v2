"""Pydantic-модели запроса/ответа для детекции людей + цвета одежды."""

from pydantic import BaseModel, Field


class PersonBox(BaseModel):
    box: list[float] = Field(
        ..., description="Рамка человека [x1, y1, x2, y2], доли кадра 0..1."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность детектора.")
    top_color: str = Field(..., description="Ближайшее название цвета верха из палитры.")
    top_hsv: list[float] = Field(
        ..., description="Сырой HSV верха: [hue 0-360, saturation 0-1, value 0-1]."
    )
    bottom_color: str = Field(..., description="Ближайшее название цвета низа из палитры.")
    bottom_hsv: list[float] = Field(
        ..., description="Сырой HSV низа: [hue 0-360, saturation 0-1, value 0-1]."
    )


class FrameResult(BaseModel):
    persons: list[PersonBox]
    count: int


class PersonsQuery(BaseModel):
    top: str | None = Field(None, description="Оставить только людей с таким цветом верха.")
    bottom: str | None = Field(None, description="Оставить только людей с таким цветом низа.")


class PersonsRequest(BaseModel):
    frames: list[str] = Field(..., description="1..N base64-JPEG кадров.")
    query: PersonsQuery | None = Field(
        None, description="Необязательный фильтр по цвету -- в ответе останутся только совпадения."
    )


class PersonsResponse(BaseModel):
    results: list[FrameResult] = Field(..., description="По одному результату на кадр, в том же порядке.")
    processing_ms: float = Field(..., description="Время сервера на decode + inference, мс.")
