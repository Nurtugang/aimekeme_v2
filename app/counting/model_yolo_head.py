"""Counting backend: YOLOv8-детектор голов (SCUT-HEAD, Abcfsa/YOLOv8_head_detector).

Точнее в толпе/при перекрытиях, чем person-детектор: считает головы, а не силуэты.
Один класс — "head", поэтому count = число боксов выше порога conf.

Зависимость: `ultralytics` (AGPL-3.0). Инференс идёт через наш torch (cu128),
на sm_120 работает нативно, без onnx.

Веса НЕ коммитим — качаются один раз в кеш (~/.cache/aimekeme/head_detector/):
- medium.pt (~52 МБ, точнее) | nano.pt (~6 МБ, быстрее).
Источник — репозиторий модели на GitHub (raw-файлы).
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import torch

from app.config import Settings

logger = logging.getLogger("surveillance.counting.yolo_head")

_BASE_URL = "https://raw.githubusercontent.com/Abcfsa/YOLOv8_head_detector/main"
_VARIANTS = {"medium": "medium.pt", "nano": "nano.pt"}
_CACHE_DIR = Path.home() / ".cache" / "aimekeme" / "head_detector"


class YoloHeadCounter:
    def __init__(self, device: torch.device, settings: Settings):
        self._device = device
        self._settings = settings
        self._model = None
        # ultralytics ждёт индекс GPU (0) или "cpu".
        self._ul_device = (
            (self._device.index or 0) if self._device.type == "cuda" else "cpu")

    def load(self) -> None:
        from ultralytics import YOLO

        weights = self._ensure_weights(self._settings.count_head_variant)
        self._model = YOLO(str(weights))
        self._model.to(self._device)
        logger.info("YOLOv8 head detector ready: %s", weights.name)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """BGR-кадр -> (boxes_xyxy, scores) голов выше порога conf.

        ultralytics ждёт BGR numpy (как OpenCV) — конвертация цвета не нужна.
        """
        result = self._model(
            bgr,
            conf=self._settings.count_score_thresh,
            device=self._ul_device,
            verbose=False,
        )[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return boxes, scores

    # --- weights -----------------------------------------------------------

    def _ensure_weights(self, variant: str) -> Path:
        if variant not in _VARIANTS:
            raise ValueError(
                f"Unknown count_head_variant={variant!r}, choose from {list(_VARIANTS)}")
        filename = _VARIANTS[variant]
        path = _CACHE_DIR / filename
        if not path.exists():
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            url = f"{_BASE_URL}/{filename}"
            logger.info("Downloading head detector weights: %s -> %s", url, path)
            tmp = path.with_suffix(".pt.part")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 -- фиксированный доверенный URL
            tmp.rename(path)
        return path
