"""Pydantic-модели запроса/ответа для оценки вовлечённости на лекции."""

from typing import Literal

from pydantic import BaseModel, Field

State = Literal["engaged", "writing", "phone", "sleeping", "distracted", "unknown"]
Activity = Literal["neutral", "writing", "phone", "slumped"]


class AttentionRequest(BaseModel):
    frame: str = Field(..., description="Один base64-JPEG кадр.")
    camera_id: str = Field(
        "default",
        max_length=64,
        description="Идентификатор камеры. Треки и временные окна ведутся "
                    "отдельно на каждую камеру — на разных потоках он ДОЛЖЕН отличаться.",
    )


class Person(BaseModel):
    track_id: int = Field(..., description="Стабильный id человека в пределах камеры.")
    box: list[float] = Field(..., description="Рамка человека [x1, y1, x2, y2].")
    keypoints: list[list[float]] = Field(
        ..., description="17 точек COCO: [x, y, confidence]. Один скелет на человека.")

    state: State = Field(..., description="Итоговое состояние за временное окно.")
    engagement: float = Field(..., ge=0.0, le=1.0, description="Вовлечённость 0..1.")

    gaze_yaw: float | None = Field(None, description="Взгляд влево/вправо, градусы.")
    gaze_pitch: float | None = Field(None, description="Взгляд вверх/вниз, градусы.")
    head_yaw: float | None = Field(None, description="Поворот головы, градусы.")
    head_pitch: float | None = Field(None, description="Наклон головы, градусы.")
    eye_closure: float | None = Field(
        None, ge=0.0, le=1.0, description="Закрытость глаз: 0 открыты, 1 закрыты.")

    attention_score: float = Field(..., ge=0.0, le=1.0, description="Скор взгляда сейчас.")
    looking_now: bool = Field(..., description="Смотрит на доску в этом кадре.")
    gaze_hold_s: float = Field(..., description="Длительность текущего непрерывного взгляда, с.")
    looking_ratio: float | None = Field(None, ge=0.0, le=1.0, description="Доля окна со взглядом.")
    perclos: float | None = Field(
        None, ge=0.0, le=1.0, description="Доля времени с закрытыми глазами за окно.")
    eyes_closed_s: float = Field(..., description="Длительность текущей серии закрытых глаз, с.")

    activity: Activity = Field(..., description="Активность за окно.")
    activity_share: float = Field(..., ge=0.0, le=1.0, description="Доля голосов за активность.")
    held_objects: list[str] = Field(
        default_factory=list, description="Предметы у кистей: cell phone / book / laptop.")

    window_s: float = Field(..., description="Сколько секунд реально накоплено.")
    warming_up: bool = Field(..., description="Окно ещё не набралось, выводам верить рано.")


class AttentionResponse(BaseModel):
    people: list[Person]
    count: int = Field(..., ge=0, description="Сколько людей найдено.")
    engaged_count: int = Field(..., ge=0, description="Сколько вовлечены (engaged + writing).")
    engagement_rate: float | None = Field(
        None, ge=0.0, le=1.0, description="Доля вовлечённых; null, если людей нет.")
    mean_engagement: float | None = Field(
        None, ge=0.0, le=1.0, description="Средний скор вовлечённости; null, если людей нет.")
    states: dict[str, int] = Field(
        default_factory=dict, description="Сколько человек в каждом состоянии.")
    processing_ms: float = Field(..., description="Время сервера на кадр, мс.")
