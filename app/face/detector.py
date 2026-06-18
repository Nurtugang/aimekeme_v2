"""Распознавание лиц: MTCNN (детекция) + InceptionResnetV1 (эмбеддинг) + мини-база.

При старте строит эмбеддинги известных лиц из папки known_faces/ (<имя>.jpg).
predict на 1 кадр: находит лица -> эмбеддинг каждого -> сравнение с базой по
косинусной близости (эмбеддинги L2-нормированы, поэтому косинус = скалярное произв.).
"""

from __future__ import annotations

import base64
import binascii
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from app.face.model import load_face_models
from app.config import Settings

logger = logging.getLogger("surveillance.face")

_DATA_URI_MARKER = "base64,"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_UNKNOWN = "unknown"


class InvalidImageError(ValueError):
    """Кадр не удалось декодировать как изображение."""


class FaceDetector:
    def __init__(self, settings: Settings, device: torch.device):
        self._settings = settings
        self._device = device
        self._mtcnn = None
        self._resnet = None
        self._known_names: list[str] = []
        self._known_embeddings: torch.Tensor | None = None  # (k, 512) на устройстве
        self._lock = threading.Lock()

    # --- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        logger.info("Loading face models on device=%s ...", self._device)
        self._mtcnn, self._resnet = load_face_models(self._device)
        self._load_known_faces()
        logger.info("Face detector ready. Known faces: %d %s",
                    len(self._known_names), self._known_names)

    @property
    def is_ready(self) -> bool:
        return self._resnet is not None

    @property
    def device(self) -> str:
        return str(self._device)

    # --- known faces (мини-база) ------------------------------------------

    def _load_known_faces(self) -> None:
        directory = Path(self._settings.known_faces_dir)
        if not directory.exists():
            logger.warning("known_faces dir not found: %s", directory.resolve())
            return

        names, embeddings = [], []
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path))
            if image is None:
                logger.warning("Cannot read %s", path)
                continue
            emb = self._embed_best_face(image)
            if emb is None:
                logger.warning("No face found in %s", path)
                continue
            names.append(path.stem)
            embeddings.append(emb)

        if embeddings:
            self._known_names = names
            self._known_embeddings = torch.cat(embeddings, dim=0)

    def _embed_best_face(self, bgr_image: np.ndarray) -> torch.Tensor | None:
        """Самое уверенное лицо на картинке -> эмбеддинг (1, 512) или None."""
        image = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        with self._lock, torch.inference_mode():
            boxes, probs = self._mtcnn.detect(image)
            if boxes is None:
                return None
            best = int(np.argmax(probs))
            faces = self._mtcnn.extract(image, boxes[best : best + 1], None)
            if faces is None:
                return None
            return self._resnet(faces.to(self._device))

    # --- inference ---------------------------------------------------------

    def predict(self, frame: str) -> dict:
        if self._resnet is None or self._mtcnn is None:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()
        image = Image.fromarray(cv2.cvtColor(self._decode(frame), cv2.COLOR_BGR2RGB))

        faces_out: list[dict] = []
        with self._lock, torch.inference_mode():
            boxes, probs = self._mtcnn.detect(image)
            if boxes is not None:
                crops = self._mtcnn.extract(image, boxes, None)  # (n, 3, 160, 160)
                embeddings = self._resnet(crops.to(self._device))  # (n, 512)
                for i in range(len(boxes)):
                    identity, similarity = self._match(embeddings[i : i + 1])
                    faces_out.append({
                        "box": [float(v) for v in boxes[i].tolist()],
                        "det_confidence": round(float(probs[i]), 4),
                        "identity": identity,
                        "similarity": round(similarity, 4),
                    })

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        recognized = [f["identity"] for f in faces_out if f["identity"] != _UNKNOWN]
        if recognized:
            logger.info("Recognized %s (%.1f ms)", recognized, elapsed_ms)
        else:
            logger.debug("faces=%d, none recognized (%.1f ms)", len(faces_out), elapsed_ms)

        return {
            "faces": faces_out,
            "count": len(faces_out),
            "processing_ms": round(elapsed_ms, 2),
        }

    def _match(self, embedding: torch.Tensor) -> tuple[str, float]:
        """Ближайшее лицо из базы по косинусной близости (эмбеддинги нормированы)."""
        if self._known_embeddings is None:
            return _UNKNOWN, 0.0
        sims = (self._known_embeddings @ embedding.T).squeeze(1)  # (k,)
        best = int(torch.argmax(sims))
        score = float(sims[best])
        if score >= self._settings.face_match_threshold:
            return self._known_names[best], score
        return _UNKNOWN, score

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _decode(raw: str) -> np.ndarray:
        if _DATA_URI_MARKER in raw:
            raw = raw.split(_DATA_URI_MARKER, 1)[1]
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidImageError() from exc
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError()
        return image
