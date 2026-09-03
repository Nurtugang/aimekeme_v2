"""Модели вовлечённости: три готовые сети, ничего не дообучается.

- PoseModel   — YOLO11-pose: люди и скелеты COCO-17 (ultralytics, AGPL-3.0);
- ObjectModel — YOLO11 COCO: телефон, книга, ноутбук (ultralytics, AGPL-3.0);
- FaceModel   — MediaPipe FaceLandmarker: углы головы, глаза, зрачки (Apache-2.0).

Почему для лица MediaPipe, а не insightface, который уже есть в проекте:
FaceLandmarker отдаёт blendshapes — 52 ИМЕНОВАННЫХ коэффициента мимики.
Закрытость глаза приходит как `eyeBlinkLeft`, направление зрачка — как
`eyeLookInLeft` и соседи. Это снимает главный риск такого модуля: не нужно
угадывать номера точек контура глаза в безымянном массиве ландмарков.
Углы головы берутся из матрицы трансформации нашим кодом (engagement.
euler_from_matrix), поэтому соглашение по осям известно точно и записано.

Стоимость лица не зависит от разрешения кадра: модель зовут на КРОПЕ ГОЛОВЫ
одного трека с num_faces=1 — отсюда ровно одно лицо на человека.

Веса качаются один раз в ~/.cache/aimekeme/ и туда же переносятся, если
ultralytics уронил их в рабочую директорию.
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import torch

from app.attention.engagement import euler_from_matrix
from app.config import Settings

# =====================================================================
# Кеш весов YOLO
# =====================================================================

_log_weights = logging.getLogger("surveillance.attention.weights")

CACHE_ROOT = Path.home() / ".cache" / "aimekeme"


def cached_yolo(name: str, subdir: str):
    """Загружает YOLO по имени веса, держа сам файл в кеше."""
    from ultralytics import YOLO

    cache_dir = CACHE_ROOT / subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / name

    if target.exists():
        return YOLO(str(target))

    model = YOLO(name)          # качает в текущую директорию
    downloaded = Path(name)
    if downloaded.is_file():
        try:
            # shutil.move, а не Path.replace: репозиторий и кеш нередко лежат
            # на разных дисках, а os.replace через границу тома не работает
            # (на Windows это WinError 17).
            shutil.move(str(downloaded), str(target))
            _log_weights.info("Веса перенесены в кеш: %s", target)
        except OSError as exc:  # права, занятый файл — не критично
            _log_weights.warning("Не удалось перенести %s в %s: %s", downloaded, target, exc)
    return model


# =====================================================================
# Скелеты: YOLO11-pose
# =====================================================================

_log_pose = logging.getLogger("surveillance.attention.pose")

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
        _log_pose.info("YOLO11%s-pose готов на %s", variant, self._device)

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


# =====================================================================
# Предметы в руках: YOLO11 COCO
# =====================================================================

_log_objects = logging.getLogger("surveillance.attention.objects")

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
            _log_objects.warning("В модели нет классов %s — они не будут находиться", missing)
        _log_objects.info("YOLO11%s (COCO) готов на %s, классы: %s",
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


# =====================================================================
# Лицо, глаза, взгляд: MediaPipe FaceLandmarker
# =====================================================================

_log_face = logging.getLogger("surveillance.attention.face")

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
        _log_face.info("MediaPipe FaceLandmarker готов (%s)", path.name)

    @staticmethod
    def _ensure_bundle() -> Path:
        path = _CACHE_DIR / _BUNDLE
        if not path.exists():
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _log_face.info("Качаю бандл FaceLandmarker: %s -> %s", _URL, path)
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
            _log_face.error(
                "FaceLandmarker не отдал blendshapes %s. Есть: %s. "
                "Взгляд и сон считаться не будут — проверьте версию бандла.",
                missing, sorted(shapes))
        else:
            _log_face.info("Blendshapes на месте: %d коэффициентов", len(shapes))
