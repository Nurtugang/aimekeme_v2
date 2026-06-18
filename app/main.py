"""FastAPI application: HTTP layer for the violence detector.

The model is loaded exactly once, at startup, via the lifespan handler.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from app.config import settings
from app.detector import InvalidFrameError, ViolenceDetector
from app.schemas import DetectionRequest, DetectionResponse, HealthResponse

# Глобальная настройка логирования.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Single detector instance shared across requests.
detector = ViolenceDetector(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    detector.load()  # грузим модель один раз при старте
    yield


app = FastAPI(
    title="Violence Detection API",
    description="Классификация 16-кадрового клипа как `fight` или `normal`.",
    version=settings.api_version,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness/readiness probe for orchestration (k8s, load balancers)."""
    return HealthResponse(
        status="ok" if detector.is_ready else "loading",
        model_ready=detector.is_ready,
        device=detector.device,
        version=settings.api_version,
    )


@app.post("/detect_violence", response_model=DetectionResponse, tags=["detection"])
def detect_violence(request: DetectionRequest) -> DetectionResponse:
    """Classify a 16-frame clip as `fight` or `normal`."""
    n_frames = len(request.frames)
    if n_frames != settings.expected_frames:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Expected {settings.expected_frames} frames, got {n_frames}",
        )

    try:
        result = detector.predict(request.frames)
    except InvalidFrameError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid base64 in frame {exc.index}",
        ) from exc

    return DetectionResponse(**result)
