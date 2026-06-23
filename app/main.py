"""FastAPI-приложение: модульный монолит системы видеонаблюдения.

Один процесс, один общий torch. Все модели грузятся ОДИН раз при старте
(lifespan) и складываются в app.state.detectors; роутеры берут готовые
детекторы оттуда через request.app.state — без загрузки на запрос.
"""

import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from app.face.detector import FaceDetector
from app.face.router import router as face_router
from app.fight.detector import FightDetector
from app.fight.router import router as fight_router
from app.config import settings

# Глобальная настройка логирования.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def resolve_device(preference: str) -> torch.device:
    """auto -> cuda при наличии, иначе cpu; либо явное значение (cuda/cpu/cuda:0)."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = resolve_device(settings.device)

    fight = FightDetector(settings, device)
    fight.load()

    face = FaceDetector(settings)
    face.load()

    # Кладём готовые модели на app.state — отсюда их берут роутеры.
    app.state.detectors = {"fight": fight, "face": face}

    yield


app = FastAPI(
    title="Intelligent Surveillance API",
    description="Алгоритмы видеонаблюдения: детекция драк и распознавание лиц.",
    version=settings.api_version,
    lifespan=lifespan,
)

app.include_router(fight_router)
app.include_router(face_router)


class ModelHealth(BaseModel):
    name: str
    ready: bool
    device: str


class HealthResponse(BaseModel):
    status: str
    models: list[ModelHealth]
    version: str


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Готовность: какие модели загружены и на каком устройстве."""
    models = [
        ModelHealth(name=name, ready=det.is_ready, device=det.device)
        for name, det in app.state.detectors.items()
    ]
    all_ready = all(m.ready for m in models) and bool(models)
    return HealthResponse(
        status="ok" if all_ready else "loading",
        models=models,
        version=settings.api_version,
    )
