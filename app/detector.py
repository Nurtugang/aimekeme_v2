"""Violence detector service.

Wraps the X3D-M model loaded once at startup and exposes a single
``predict`` entry point. Frame decoding, preprocessing (from ``app.model``)
and GPU inference all live here, keeping the HTTP layer in ``main.py`` thin.
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

from app.config import Settings
from app.model import load_model, preprocess_frames

logger = logging.getLogger("violence_api.detector")

# Index of the "violent" class in the model's output logits.
_FIGHT_CLASS_IDX = 1
# Optional data-URI prefix we tolerate, e.g. "data:image/jpeg;base64,...".
_DATA_URI_MARKER = "base64,"


class InvalidFrameError(ValueError):
    """Raised when a frame cannot be decoded into an image.

    Carries the 0-based ``index`` of the offending frame so the API layer
    can build a precise error message.
    """

    def __init__(self, index: int):
        self.index = index
        super().__init__(f"Invalid base64 in frame {index}")


class ViolenceDetector:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._device: torch.device | None = None
        self._model: torch.nn.Module | None = None
        # Serialize access to a single GPU/model across worker threads.
        self._lock = threading.Lock()

    # --- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Resolve the device and load model weights. Call once at startup."""
        self._device = self._resolve_device(self._settings.device)
        logger.info("Loading X3D-M on device=%s ...", self._device)
        self._model = load_model(self._device)
        logger.info("Model loaded and ready.")

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return str(self._device) if self._device is not None else "uninitialized"

    @staticmethod
    def _resolve_device(preference: str) -> torch.device:
        if preference == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(preference)

    # --- inference ---------------------------------------------------------

    def predict(self, frames: list[str]) -> dict:
        """Decode, preprocess and classify a clip of base64 JPEG frames.

        Raises:
            InvalidFrameError: if any frame is not valid base64/JPEG.
        """
        if self._model is None or self._device is None:
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

        # Важное событие — драка. Обычные кадры пишем в DEBUG (по умолчанию скрыт),
        # чтобы лог не зарастал «водой».
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
        """base64 JPEG string -> BGR numpy array (H, W, 3), as OpenCV produces."""
        if _DATA_URI_MARKER in raw:
            raw = raw.split(_DATA_URI_MARKER, 1)[1]

        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidFrameError(index) from exc

        buffer = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            # Valid base64 but not a decodable image — same client-side fix.
            raise InvalidFrameError(index)
        return image
