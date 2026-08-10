"""Fire backend: SigLIP2-классификатор (prithivMLmods/Fire-Detection-Siglip2).

Классифицирует кадр ЦЕЛИКОМ на fire/smoke/normal — боксов не даёт. Классы
модели читаем из config.id2label, порядок не хардкодим.

Ограничение (см. docs/fire.md): у классификатора нет пространственной
локализации, поэтому мелкий/ранний дым растворяется в глобальном признаке.
Для раннего задымления берите детекторный бэкенд.

Чистый PyTorch: на Blackwell (sm_120) torch cu128 работает нативно, без onnx.
Веса качаются один раз в HF cache (~/.cache/huggingface/).
"""

from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from app.config import Settings

logger = logging.getLogger("surveillance.fire.siglip2")

_REPO_ID = "prithivMLmods/Fire-Detection-Siglip2"


class Siglip2FireClassifier:
    def __init__(self, device: torch.device, settings: Settings):
        self._device = device
        self._settings = settings
        self._model = None
        self._processor = None
        self._id2label: dict[int, str] = {}

    def load(self) -> None:
        self._processor = AutoImageProcessor.from_pretrained(_REPO_ID)
        model = AutoModelForImageClassification.from_pretrained(_REPO_ID)
        model.eval()
        self._model = model.to(self._device)
        self._id2label = {i: str(lbl).lower() for i, lbl in model.config.id2label.items()}
        logger.info("SigLIP2 fire classifier ready: labels=%s", self._id2label)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, images: list[Image.Image]) -> list[tuple[dict[str, float], list]]:
        """Батч картинок -> для каждой (вероятности по классам, [] боксов)."""
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            probs = torch.softmax(self._model(**inputs).logits, dim=1)

        results = []
        for row in probs:
            scores = {"fire": 0.0, "smoke": 0.0, "normal": 0.0}
            for idx, label in self._id2label.items():
                if label in scores:
                    scores[label] = float(row[idx].item())
            results.append((scores, []))
        return results
