"""Скелеты людей: YOLO11-pose (ultralytics), COCO-17 точек.

Готовая предобученная модель, ничего не дообучаем. Работает в режиме predict
(без встроенного трекинга) — треки ведёт app/attention/tracking.py, потому что
состояние ultralytics-трекера живёт внутри модели и на нескольких камерах
перемешалось бы.

Дубли скелетов исключены здесь же: NMS внутри YOLO оставляет по одной рамке
на человека, а трекер сводит их к одному стабильному id.

Вариант весов задаётся POSE_VARIANT (n/s/m/l/x). Для 4К и мощной
карты берите x — он заметно лучше видит мелкие фигуры на дальних рядах.
Веса качаются один раз в ~/.cache/aimekeme/pose/.

Зависимость: ultralytics (AGPL-3.0) — уже нужна модулю counting.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from app.attention.weights import cached_yolo
from app.config import Settings

logger = logging.getLogger("surveillance.attention.pose")

_VARIANTS = ("n", "s", "m", "l", "x")
_PERSON_CLASS = 0


class PoseModel:
    def __init__(self, device: torch.device, settings: Settings):
        self._device = device
        self._settings = settings
        self._model = None
        self._ul_device = (
            (device.index or 0) if device.type == "cuda" else "cpu")

    def load(self) -> None:
        variant = self._settings.pose_variant
        if variant not in _VARIANTS:
            raise ValueError(
                f"Unknown pose_variant={variant!r}, choose from {list(_VARIANTS)}")

        self._model = cached_yolo(f"yolo11{variant}-pose.pt", "pose")
        self._model.to(self._device)
        logger.info("YOLO11%s-pose готов на %s", variant, self._device)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """BGR-кадр -> (boxes_xyxy [N,4], keypoints [N,17,2], kpt_conf [N,17]).

        ultralytics ждёт BGR numpy — конвертация цвета не нужна.
        """
        result = self._model(
            bgr,
            classes=[_PERSON_CLASS],
            conf=self._settings.pose_conf,
            iou=self._settings.pose_nms_iou,
            imgsz=self._settings.pose_imgsz,
            device=self._ul_device,
            verbose=False,
        )[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        if result.keypoints is None or len(boxes) == 0:
            return boxes, np.zeros((len(boxes), 17, 2)), np.zeros((len(boxes), 17))
        kpts = result.keypoints.xy.cpu().numpy()
        conf = (result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None else np.ones(kpts.shape[:2]))
        return boxes, kpts, conf
