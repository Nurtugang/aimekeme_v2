"""Предметы в руках: YOLO11 (COCO) — телефон, книга, ноутбук.

Отличить «пишет» от «сидит в телефоне» по одному скелету трудно: в обоих
случаях голова наклонена, руки внизу. Надёжный признак — САМ ПРЕДМЕТ. COCO
содержит готовые классы `cell phone`, `book`, `laptop`, поэтому обычный
предобученный детектор решает задачу прямо, без обучения и без эвристик.

Найденные предметы привязываются к людям по близости к кистям (detector.py):
телефон у кисти — это телефон, а не «поза, похожая на телефон».

На 4К телефон занимает десятки пикселей, поэтому imgsz здесь свой и больше,
чем у детектора поз (OBJECT_IMGSZ). На слабой карте бэкенд можно выключить
(OBJECT_MODEL_ENABLED=false) — тогда активность определяется только по позе,
менее точно, но сервис работает.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from app.attention.weights import cached_yolo
from app.config import Settings

logger = logging.getLogger("surveillance.attention.objects")

# Классы COCO, которые говорят об активности студента.
PHONE, BOOK, LAPTOP = "cell phone", "book", "laptop"
_WANTED = (PHONE, BOOK, LAPTOP)


class ObjectModel:
    def __init__(self, device: torch.device, settings: Settings):
        self._device = device
        self._settings = settings
        self._model = None
        self._class_ids: list[int] = []
        self._names: dict[int, str] = {}
        self._ul_device = (device.index or 0) if device.type == "cuda" else "cpu"

    def load(self) -> None:
        variant = self._settings.object_variant
        self._model = cached_yolo(f"yolo11{variant}.pt", "objects")
        self._model.to(self._device)

        # Номера классов берём ИЗ МОДЕЛИ по именам, а не хардкодим индексы COCO.
        self._names = {int(i): str(n) for i, n in self._model.names.items()}
        by_name = {n: i for i, n in self._names.items()}
        self._class_ids = [by_name[n] for n in _WANTED if n in by_name]
        missing = [n for n in _WANTED if n not in by_name]
        if missing:
            logger.warning("В модели нет классов %s — они не будут находиться", missing)
        logger.info("YOLO11%s (COCO) готов на %s, классы: %s",
                    variant, self._device, [self._names[i] for i in self._class_ids])

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, bgr: np.ndarray) -> list[tuple[str, tuple[float, float, float, float], float]]:
        """BGR-кадр -> [(имя класса, (x1,y1,x2,y2), conf), ...]."""
        if not self._class_ids:
            return []
        result = self._model(
            bgr,
            classes=self._class_ids,
            conf=self._settings.object_conf,
            imgsz=self._settings.object_imgsz,
            device=self._ul_device,
            verbose=False,
        )[0]

        out = []
        xyxy = result.boxes.xyxy.cpu().numpy()
        for box, cls, conf in zip(xyxy, result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            out.append((self._names.get(int(cls), str(int(cls))),
                        tuple(float(v) for v in box), float(conf)))
        return out
