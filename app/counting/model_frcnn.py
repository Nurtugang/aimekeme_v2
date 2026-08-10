"""Counting backend: torchvision Faster R-CNN (COCO, класс person).

Чистый torchvision (BSD) — нативно работает на Blackwell (sm_120), ноль новых
зависимостей. Веса COCO качаются один раз в кеш torch
(~/.cache/torch/hub/checkpoints/).

Единый интерфейс backend'а подсчёта: `load()`, `is_ready`, `predict(bgr)`.
`predict` принимает BGR-кадр (как отдаёт OpenCV) и возвращает (boxes_xyxy, scores)
только для людей выше порога score.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

from app.config import Settings

logger = logging.getLogger("surveillance.counting.frcnn")

# В COCO класс "person" имеет индекс 1.
_PERSON_LABEL = 1


class FrcnnCounter:
    def __init__(self, device: torch.device, settings: Settings):
        self._device = device
        self._settings = settings
        self._model = None
        self._transform = None

    def load(self) -> None:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self._model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self._model.eval()
        self._model.to(self._device)
        self._transform = weights.transforms()
        logger.info("Faster R-CNN ResNet50 FPN v2 ready.")

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """BGR-кадр -> (boxes_xyxy, scores) людей выше порога score."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # (C, H, W) uint8
        image = self._transform(tensor).to(self._device)

        with torch.inference_mode():
            output = self._model([image])[0]

        keep = (output["labels"] == _PERSON_LABEL) & (
            output["scores"] >= self._settings.count_score_thresh)
        boxes = output["boxes"][keep].cpu().numpy()
        scores = output["scores"][keep].cpu().numpy()
        return boxes, scores
