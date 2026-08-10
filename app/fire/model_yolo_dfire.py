"""Fire backend: YOLOv8n, дообученный на D-Fire (rabahdev/fire-smoke-yolov8n).

Детектор БОКСОВ: классы 0=smoke, 1=fire. В отличие от классификатора, находит
локальный очаг, а не «настроение кадра» — по docs/fire.md это принципиально
для раннего/мелкого дыма.

Датасет D-Fire — 21 527 изображений, из них 9 838 негативов (лампы, блики) и
съёмка с камер наблюдения, поэтому домен ближе к нашему, чем wildfire-модели.
Заявлено на тестовом сплите D-Fire: mAP@50 0.754, recall 0.688.

Зависимость: ultralytics (AGPL-3.0) — уже стоит ради COUNT_MODEL=yolo_head.
Веса (~6 МБ) качаются один раз в HF cache (~/.cache/huggingface/).

ВНИМАНИЕ: модель обучена на outdoor+surveillance, но не на наши аудитории.
На светлых стенах/проходах даёт ложный smoke — порог подбирайте по своим
кадрам (FIRE_THRESHOLD), а не по умолчанию.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

from app.config import Settings

logger = logging.getLogger("surveillance.fire.yolo_dfire")

_REPO_ID = "rabahdev/fire-smoke-yolov8n"
_WEIGHTS = "best.pt"
# Нижний пол для выдачи боксов. Решающий порог — fire_threshold в детекторе,
# чтобы вся логика решения жила в одном месте.
_BOX_FLOOR = 0.05


class YoloDfireDetector:
    def __init__(self, device: torch.device, settings: Settings):
        self._device = device
        self._settings = settings
        self._model = None
        self._names: dict[int, str] = {}
        # ultralytics ждёт индекс GPU (0) или "cpu".
        self._ul_device = (
            (self._device.index or 0) if self._device.type == "cuda" else "cpu")

    def load(self) -> None:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        weights = hf_hub_download(repo_id=_REPO_ID, filename=_WEIGHTS)
        self._model = YOLO(weights)
        self._model.to(self._device)
        self._names = {i: str(n).lower() for i, n in self._model.names.items()}
        logger.info("D-Fire YOLOv8n ready: names=%s", self._names)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, images: list[Image.Image]) -> list[tuple[dict[str, float], list]]:
        """Батч картинок -> для каждой (скоры по классам, боксы).

        Скор класса = уверенность самого уверенного бокса этого класса;
        normal = 1 - max(fire, smoke). Боксы в пикселях своей картинки:
        (x1, y1, x2, y2, label, conf).
        """
        # ultralytics ждёт BGR numpy (как OpenCV), а у нас PIL RGB.
        frames = [np.asarray(img.convert("RGB"))[:, :, ::-1] for img in images]
        outputs = self._model(
            frames, conf=_BOX_FLOOR, device=self._ul_device, verbose=False)

        results = []
        for out in outputs:
            boxes = []
            scores = {"fire": 0.0, "smoke": 0.0, "normal": 0.0}
            xyxy = out.boxes.xyxy.cpu().numpy()
            for (x1, y1, x2, y2), cls, conf in zip(
                xyxy, out.boxes.cls.tolist(), out.boxes.conf.tolist()
            ):
                label = self._names.get(int(cls), str(int(cls)))
                boxes.append((float(x1), float(y1), float(x2), float(y2), label, float(conf)))
                if label in scores:
                    scores[label] = max(scores[label], float(conf))
            scores["normal"] = 1.0 - max(scores["fire"], scores["smoke"])
            results.append((scores, boxes))
        return results
