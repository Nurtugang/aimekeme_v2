"""Детектор подсчёта людей. Backend выбирается настройкой `count_model`:

- `frcnn`      — torchvision Faster R-CNN, класс person (BSD, ноль доп. зависимостей);
- `yolo_head`  — YOLOv8-детектор голов (SCUT-HEAD), точнее в толпе (ultralytics, AGPL).

Backend грузится один раз при старте и реализует единый интерфейс
`load()`/`is_ready`/`predict(bgr) -> (boxes, scores)`. Детектор поверх него делает
decode base64, лок вокруг GPU-инференса, тайминг и формат ответа. Контракт ответа
одинаков для обеих моделей: `{ label:"person", count, confidence, processing_ms }`.
"""

from __future__ import annotations

import base64
import binascii
import logging
import threading
import time

import cv2
import numpy as np
import torch

from app.counting.model_frcnn import FrcnnCounter
from app.counting.model_yolo_head import YoloHeadCounter
from app.config import Settings

logger = logging.getLogger("surveillance.counting")

_DATA_URI_MARKER = "base64,"
_BACKENDS = {"frcnn": FrcnnCounter, "yolo_head": YoloHeadCounter}


class InvalidImageError(ValueError):
    """Кадр не удалось декодировать как изображение."""


class CountingDetector:
    def __init__(self, settings: Settings, device: torch.device):
        self._settings = settings
        self._device = device
        try:
            backend_cls = _BACKENDS[settings.count_model]
        except KeyError as exc:
            raise ValueError(
                f"Unknown count_model={settings.count_model!r}, "
                f"choose from {list(_BACKENDS)}") from exc
        self._backend = backend_cls(device, settings)
        # Сериализуем доступ к одной модели/GPU между потоками воркеров.
        self._lock = threading.Lock()

    # --- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        logger.info("Loading counting backend '%s' on device=%s ...",
                    self._settings.count_model, self._device)
        self._backend.load()
        logger.info("Counting model ready.")

    @property
    def is_ready(self) -> bool:
        return self._backend.is_ready

    @property
    def device(self) -> str:
        return str(self._device)

    # --- inference ---------------------------------------------------------

    def predict(self, frame: str) -> dict:
        """Декод + подсчёт людей на одном кадре.

        Raises:
            InvalidImageError: если кадр не валидный base64/JPEG.
        """
        if not self._backend.is_ready:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()

        bgr = self._decode(frame)
        with self._lock:
            _, scores = self._backend.predict(bgr)

        count = int(scores.shape[0])
        confidence = float(scores.mean()) if count else 0.0

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.debug("people=%d (%.1f ms)", count, elapsed_ms)

        return {
            "label": "person",
            "count": count,
            "confidence": round(confidence, 4),
            "processing_ms": round(elapsed_ms, 2),
        }

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _decode(raw: str) -> np.ndarray:
        """base64 JPEG -> BGR numpy array (H, W, 3) uint8 (как отдаёт OpenCV)."""
        if _DATA_URI_MARKER in raw:
            raw = raw.split(_DATA_URI_MARKER, 1)[1]
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidImageError() from exc
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise InvalidImageError()
        return bgr
