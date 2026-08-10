"""Детектор огня/дыма. Backend выбирается настройкой `fire_model`:

- `siglip2`     — SigLIP2-классификатор кадра целиком, без боксов (transformers);
- `yolo_dfire`  — YOLOv8n, дообученный на D-Fire: боксы вокруг очага (ultralytics, AGPL);

Backend грузится один раз при старте и реализует единый интерфейс
`load()`/`is_ready`/`predict(images) -> [(scores, boxes)]`. Детектор поверх него
делает decode base64, тайлинг, лок вокруг GPU-инференса, порог и формат ответа.
Контракт ответа одинаков для обеих моделей: `{ label, confidence, processing_ms }`.

Тайлинг (2x2 + кадр целиком, `use_tiling`) даёт мелкому очагу занять заметную
долю кадра. Решение берём с тайла с максимальной тревогой; боксы — со всех
тайлов, приведённые к координатам полного кадра.
"""

from __future__ import annotations

import base64
import binascii
import logging
import threading
import time
from io import BytesIO

import torch
from PIL import Image

from app.config import Settings
from app.fire.model_siglip2 import Siglip2FireClassifier
from app.fire.model_yolo_dfire import YoloDfireDetector

logger = logging.getLogger("surveillance.fire")

_DATA_URI_MARKER = "base64,"
_NORMAL = "normal"
_BACKENDS = {
    "siglip2": Siglip2FireClassifier,
    "yolo_dfire": YoloDfireDetector,
}


class InvalidImageError(ValueError):
    """Кадр не удалось декодировать как изображение."""


class FireDetector:
    def __init__(self, settings: Settings, device: torch.device):
        self._settings = settings
        self._device = device
        try:
            backend_cls = _BACKENDS[settings.fire_model]
        except KeyError as exc:
            raise ValueError(
                f"Unknown fire_model={settings.fire_model!r}, "
                f"choose from {list(_BACKENDS)}") from exc
        self._backend = backend_cls(device, settings)
        # Сериализуем доступ к одной модели/GPU между потоками воркеров.
        self._lock = threading.Lock()

    # --- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        logger.info("Loading fire backend '%s' on device=%s ...",
                    self._settings.fire_model, self._device)
        self._backend.load()
        logger.info("Fire model ready (use_tiling=%s).", self._settings.use_tiling)

    @property
    def is_ready(self) -> bool:
        return self._backend.is_ready

    @property
    def device(self) -> str:
        return str(self._device)

    # --- inference ---------------------------------------------------------

    def predict(self, frame: str) -> dict:
        """Декод + препроцесс + классификация одного кадра.

        Raises:
            InvalidImageError: если кадр не валидный base64/JPEG.
        """
        if not self._backend.is_ready:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()

        image = self._decode(frame)
        with self._lock:
            scores, _ = self.analyze(image)

        label, confidence = self.decide(scores)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Важное событие — огонь/дым. Обычные кадры пишем в DEBUG.
        if label != _NORMAL:
            logger.warning("FIRE/SMOKE detected: %s confidence=%.3f (%.1f ms)",
                           label, confidence, elapsed_ms)
        else:
            logger.debug("normal: confidence=%.3f (%.1f ms)", confidence, elapsed_ms)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "processing_ms": round(elapsed_ms, 2),
        }

    def analyze(self, image: Image.Image) -> tuple[dict[str, float], list]:
        """Картинка -> (скоры по классам, боксы в координатах полного кадра).

        Публичный метод: им же пользуется офлайн-скрипт, чтобы рисовать боксы.
        Без лока — вызывающий сам решает, нужна ли синхронизация.
        """
        tiles = self._tiles(image)
        outputs = self._backend.predict([img for img, _ in tiles])

        boxes: list = []
        best_scores, best_hazard = outputs[0][0], -1.0
        for (scores, tile_boxes), (_, (dx, dy)) in zip(outputs, tiles):
            hazard = max(scores["fire"], scores["smoke"])
            if hazard > best_hazard:
                best_hazard, best_scores = hazard, scores
            boxes.extend(
                (x1 + dx, y1 + dy, x2 + dx, y2 + dy, label, conf)
                for x1, y1, x2, y2, label, conf in tile_boxes
            )
        return best_scores, boxes

    def decide(self, scores: dict[str, float]) -> tuple[str, float]:
        """Скоры -> (итоговая метка, уверенность) с учётом fire_threshold."""
        top_label = max(scores, key=scores.get)
        top_prob = scores[top_label]
        if top_label != _NORMAL and top_prob < self._settings.fire_threshold:
            return _NORMAL, scores.get(_NORMAL, 1.0 - top_prob)
        return top_label, top_prob

    # --- helpers -----------------------------------------------------------

    def _tiles(self, image: Image.Image) -> list[tuple[Image.Image, tuple[int, int]]]:
        """[(кроп, смещение в полном кадре), ...] — 2x2 + кадр целиком."""
        if not self._settings.use_tiling:
            return [(image, (0, 0))]

        w, h = image.size
        mw, mh = w // 2, h // 2
        return [
            (image.crop((0, 0, mw, mh)), (0, 0)),        # Верхний-левый
            (image.crop((mw, 0, w, mh)), (mw, 0)),       # Верхний-правый
            (image.crop((0, mh, mw, h)), (0, mh)),       # Нижний-левый
            (image.crop((mw, mh, w, h)), (mw, mh)),      # Нижний-правый
            (image, (0, 0)),                             # Кадр целиком
        ]

    @staticmethod
    def _decode(raw: str) -> Image.Image:
        """base64 JPEG -> RGB PIL-картинка (формат, который ждут бэкенды)."""
        if _DATA_URI_MARKER in raw:
            raw = raw.split(_DATA_URI_MARKER, 1)[1]
        try:
            data = base64.b64decode(raw, validate=True)
            image = Image.open(BytesIO(data)).convert("RGB")
        except (binascii.Error, ValueError, OSError) as exc:
            raise InvalidImageError() from exc
        return image
