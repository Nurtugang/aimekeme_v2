"""Детектор драк (X3D-M).

Модель загружается один раз при старте; predict декодирует 16 base64-кадров,
препроцессит и классифицирует клип. GPU-инференс под локом (один процесс,
одна видеокарта).
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
import torch.nn.functional as F

from app.fight.model import load_model, preprocess_frames
from app.config import Settings

logger = logging.getLogger("surveillance.fight")

# Индекс класса "драка" в логитах модели.
_FIGHT_CLASS_IDX = 1
# Терпим необязательный data-URI префикс, напр. "data:image/jpeg;base64,...".
_DATA_URI_MARKER = "base64,"


class InvalidFrameError(ValueError):
    """Кадр не удалось декодировать. Несёт 0-based индекс кадра для сообщения."""

    def __init__(self, index: int):
        self.index = index
        super().__init__(f"Invalid base64 in frame {index}")


class FightDetector:
    def __init__(self, settings: Settings, device: torch.device):
        self._settings = settings
        self._device = device
        self._model: torch.nn.Module | None = None
        # Сериализуем доступ к одной модели/GPU между потоками воркеров.
        self._lock = threading.Lock()

    # --- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        logger.info("Loading X3D-M on device=%s ...", self._device)
        self._model = load_model(self._device)
        logger.info("Fight model ready.")

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return str(self._device)

    # --- inference ---------------------------------------------------------

    def predict(self, frames: list[str]) -> dict:
        """Декод + препроцесс + классификация клипа из base64-JPEG кадров.

        Raises:
            InvalidFrameError: если какой-то кадр не валидный base64/JPEG.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()

        decoded = [self._decode_frame(raw, i) for i, raw in enumerate(frames)]
        tensor = preprocess_frames(decoded).to(self._device)

        with self._lock, torch.inference_mode():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        prob_fight = probs[_FIGHT_CLASS_IDX].item()
        is_fight = prob_fight >= self._settings.fight_threshold
        label = "fight" if is_fight else "normal"
        confidence = prob_fight if is_fight else 1.0 - prob_fight

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Важное событие — драка. Обычные кадры пишем в DEBUG (по умолчанию скрыт).
        if is_fight:
            logger.warning("FIGHT detected: confidence=%.3f (%.1f ms)", confidence, elapsed_ms)
        else:
            logger.debug("normal: confidence=%.3f (%.1f ms)", confidence, elapsed_ms)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "processing_ms": round(elapsed_ms, 2),
        }

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _decode_frame(raw: str, index: int) -> np.ndarray:
        """base64 JPEG -> BGR numpy array (H, W, 3), как отдаёт OpenCV."""
        if _DATA_URI_MARKER in raw:
            raw = raw.split(_DATA_URI_MARKER, 1)[1]

        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidFrameError(index) from exc

        buffer = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidFrameError(index)
        return image
