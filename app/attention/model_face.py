"""Лицо, глаза и взгляд: MediaPipe FaceLandmarker (готовая модель Google).

Почему именно она, а не insightface, который уже есть в проекте: FaceLandmarker
отдаёт blendshapes — 52 ИМЕНОВАННЫХ коэффициента мимики. Закрытость глаза
приходит как `eyeBlinkLeft`, направление зрачка — как `eyeLookInLeft` и соседи.
Это снимает главный риск такого модуля: не нужно угадывать номера точек
контура глаза в безымянном массиве ландмарков и не нужно опытом подбирать
знак угла. Всё берётся по имени.

Плюс модель отдаёт матрицу трансформации головы, из которой углы извлекаются
нашим кодом (geometry.euler_from_matrix) — значит соглашение по осям и знакам
известно точно и записано, а не выведено экспериментом.

Работает на CPU через XNNPACK. Модель вызывается на КРОПЕ ГОЛОВЫ одного трека
с num_faces=1, поэтому на человека приходится ровно одно лицо, а стоимость
не зависит от разрешения кадра — только от числа людей.

Бандл (~3.7 МБ) качается один раз в ~/.cache/aimekeme/mediapipe/.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger("surveillance.attention.face")

_CACHE_DIR = Path.home() / ".cache" / "aimekeme" / "mediapipe"
_BUNDLE = "face_landmarker.task"
_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task")

# Имена blendshapes, которые нам нужны. Если модель отдаст другой набор,
# load() скажет об этом сразу, а не выдаст молча нули на весь поток.
_BLINK = ("eyeBlinkLeft", "eyeBlinkRight")
_LOOK = ("eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
         "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight")


class FaceSignals:
    """Сигналы по одному лицу. None означает «модель лица не нашла»."""

    __slots__ = ("pitch", "yaw", "eye_closure", "gaze_h", "gaze_v")

    def __init__(self, pitch, yaw, eye_closure, gaze_h, gaze_v):
        self.pitch = pitch
        self.yaw = yaw
        self.eye_closure = eye_closure
        self.gaze_h = gaze_h
        self.gaze_v = gaze_v


class FaceModel:
    def __init__(self, settings):
        self._settings = settings
        self._landmarker = None
        self._checked_names = False

    # --- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        import mediapipe as mp

        path = self._ensure_bundle()
        vision = mp.tasks.vision
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,                       # кроп головы одного трека
            min_face_detection_confidence=self._settings.face_min_confidence,
            min_face_presence_confidence=self._settings.face_min_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp = mp
        logger.info("MediaPipe FaceLandmarker готов (%s)", path.name)

    @staticmethod
    def _ensure_bundle() -> Path:
        path = _CACHE_DIR / _BUNDLE
        if not path.exists():
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Качаю бандл FaceLandmarker: %s -> %s", _URL, path)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(_URL, tmp)  # noqa: S310 -- фиксированный URL Google
            tmp.rename(path)
        return path

    @property
    def is_ready(self) -> bool:
        return self._landmarker is not None

    # --- inference ---------------------------------------------------------

    def predict(self, head_rgb: np.ndarray) -> FaceSignals | None:
        """RGB-кроп головы -> сигналы лица, либо None если лица не видно."""
        from app.attention.geometry import euler_from_matrix

        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(head_rgb))
        result = self._landmarker.detect(image)

        if not result.face_blendshapes or not result.facial_transformation_matrixes:
            return None

        shapes = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        self._check_names(shapes)

        # roll (завал головы) на решения не влияет, поэтому не храним.
        pitch, yaw, _ = euler_from_matrix(
            np.asarray(result.facial_transformation_matrixes[0]))

        # Закрытость глаза: 0 — открыт, 1 — закрыт. Берём среднее по двум,
        # чтобы прищур с одной стороны (свет из окна) не читался как сон.
        closure = 0.5 * (shapes.get("eyeBlinkLeft", 0.0) + shapes.get("eyeBlinkRight", 0.0))

        # Смещение зрачков в орбите, -1..1. "In" — к носу, "Out" — от носа,
        # поэтому для левого и правого глаза знак горизонтали разный.
        gaze_h = 0.5 * ((shapes.get("eyeLookOutLeft", 0.0) - shapes.get("eyeLookInLeft", 0.0))
                        + (shapes.get("eyeLookInRight", 0.0) - shapes.get("eyeLookOutRight", 0.0)))
        gaze_v = 0.5 * ((shapes.get("eyeLookDownLeft", 0.0) - shapes.get("eyeLookUpLeft", 0.0))
                        + (shapes.get("eyeLookDownRight", 0.0) - shapes.get("eyeLookUpRight", 0.0)))

        return FaceSignals(pitch, yaw, closure, gaze_h, gaze_v)

    def _check_names(self, shapes: dict) -> None:
        """Один раз сверяем, что модель отдаёт ожидаемые имена коэффициентов."""
        if self._checked_names:
            return
        self._checked_names = True
        missing = [n for n in (*_BLINK, *_LOOK) if n not in shapes]
        if missing:
            logger.error(
                "FaceLandmarker не отдал blendshapes %s. Есть: %s. "
                "Взгляд и сон считаться не будут — проверьте версию бандла.",
                missing, sorted(shapes))
        else:
            logger.info("Blendshapes на месте: %d коэффициентов", len(shapes))
