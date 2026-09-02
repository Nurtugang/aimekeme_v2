"""Детектор вовлечённости на лекции: скелет + взгляд + предметы + время.

Конвейер на один кадр:

    1. YOLO11-pose        -> люди и скелеты (COCO-17), дубли срезает NMS
    2. IouTracker         -> стабильный track_id на человека (своё состояние на камеру)
    3. кроп головы по скелету -> MediaPipe FaceLandmarker: углы головы,
                             закрытость глаз, смещение зрачков
    4. YOLO11 (COCO)      -> телефон/книга/ноутбук, привязка к кистям
    5. geometry           -> направление взгляда, поза (пишет/телефон/упал)
    6. temporal           -> окно по треку: длительность взгляда, PERCLOS,
                             активность, итоговая вовлечённость

ТРЕК — ПЕРВИЧНЫЙ КЛЮЧ. Лицо, взгляд и активность считаются НА ТРЕК, а не
отдельными детекторами по всему кадру. Поэтому один человек не может получить
два скелета или два лица: и то и другое привязано к одной записи трека.

СОСТОЯНИЕ. Модуль stateful (см. temporal.py): держит окно отсчётов на
(camera_id, track_id). Это осознанный отход от правила 3 CONVENTIONS.md —
длительность взгляда по одному кадру не вычислима. camera_id приходит в
запросе; на каждую камеру свой трекер и свои окна, TTL ограничен.
"""

from __future__ import annotations

import base64
import binascii
import logging
import threading
import time

import cv2
import numpy as np
import torch

from app.attention import engagement as g
from app.attention.engagement import EngagementWindow, IouTracker, Sample
from app.attention.model import BOOK, LAPTOP, PHONE, FaceModel, ObjectModel, PoseModel
from app.config import Settings

logger = logging.getLogger("surveillance.attention")

_DATA_URI_MARKER = "base64,"


class InvalidImageError(ValueError):
    """Кадр не удалось декодировать как изображение."""


class _CameraState:
    """Трекер и временные окна одной камеры."""

    def __init__(self, cfg):
        self.tracker = IouTracker(
            cfg.track_iou_threshold, cfg.track_max_age_s, cfg.track_min_hits)
        self.windows: dict[int, EngagementWindow] = {}
        self.last_used = time.perf_counter()


class AttentionDetector:
    def __init__(self, settings: Settings, device: torch.device):
        self._settings = settings
        self._device = device
        self._pose = PoseModel(device, settings)
        self._face = FaceModel(settings)
        self._objects = ObjectModel(device, settings) if settings.object_enabled else None
        self._cameras: dict[str, _CameraState] = {}
        # Один лок на модели И на состояние камер: инференс всё равно
        # сериализован одной картой, отдельные локи ничего не ускорят.
        self._lock = threading.Lock()
        self._validate(settings)

    @staticmethod
    def _validate(cfg: Settings) -> None:
        if cfg.attention_yaw_tolerance <= 0 or cfg.attention_pitch_tolerance <= 0:
            raise ValueError("attention_yaw_tolerance и attention_pitch_tolerance должны быть > 0")
        if cfg.engagement_window_s <= 0:
            raise ValueError("engagement_window_s должен быть > 0")
        if not 0.0 <= cfg.sleep_perclos <= 1.0:
            raise ValueError("sleep_perclos должен лежать в 0..1")

    # --- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        logger.info("Загружаю модели вовлечённости на %s ...", self._device)
        self._pose.load()
        self._face.load()
        if self._objects is not None:
            self._objects.load()
        else:
            logger.warning("Детектор предметов выключен: телефон определяется "
                           "только по позе, точность ниже")
        logger.info("Детектор вовлечённости готов (окно %.0f с).",
                    self._settings.engagement_window_s)

    @property
    def is_ready(self) -> bool:
        return self._pose.is_ready and self._face.is_ready and (
            self._objects is None or self._objects.is_ready)

    @property
    def device(self) -> str:
        return str(self._device)

    # --- inference ---------------------------------------------------------

    def predict(self, frame: str, camera_id: str) -> dict:
        """Кадр (base64) + id камеры -> состояние каждого человека и аудитории.

        Raises:
            InvalidImageError: если кадр не валидный base64/JPEG.
        """
        start = time.perf_counter()
        bgr = self._decode(frame)
        with self._lock:
            people = self.analyze(bgr, camera_id)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return {**self.aggregate(people),
                "people": people,
                "processing_ms": round(elapsed_ms, 2)}

    def analyze(self, bgr: np.ndarray, camera_id: str = "offline") -> list[dict]:
        """BGR-кадр -> список людей с состоянием. Публичный: им же живут скрипты.

        Без лока — вызывающий сам решает, нужна ли синхронизация.
        """
        now = time.perf_counter()
        cam = self._camera(camera_id, now)
        h, w = bgr.shape[:2]

        boxes, kpts, kconf = self._pose.predict(bgr)
        track_ids = cam.tracker.update([tuple(b) for b in boxes], now)
        objects = self._objects.predict(bgr) if self._objects is not None else []

        people = []
        for box, kp, kc, tid in zip(boxes, kpts, kconf, track_ids):
            if not cam.tracker.is_confirmed(tid):
                continue    # трек ещё не подтверждён — не показываем мигающие ложняки
            people.append(self._person(bgr, w, h, box, kp, kc, tid, objects, cam, now))

        self._drop_dead_windows(cam)
        return people

    def _person(self, bgr, w, h, box, kp, kc, tid, objects, cam, now) -> dict:
        cfg = self._settings

        # --- лицо: кроп по скелету, поэтому ровно одно лицо на трек ---
        face = None
        crop = g.head_box(kp, kc, w, h, cfg.pose_kpt_conf)
        if crop is not None:
            x1, y1, x2, y2 = crop
            face = self._face.predict(cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))

        # --- взгляд: голова + поправка на зрачки ---
        if face is not None:
            gaze_pitch, gaze_yaw = g.fuse_gaze(
                face.pitch, face.yaw, face.gaze_h, face.gaze_v, cfg.gaze_eye_gain)
            score = g.attention_score(gaze_pitch, gaze_yaw, cfg)
            looking = score >= cfg.attention_threshold
            eyes_closed = face.eye_closure >= cfg.sleep_eye_closure
        else:
            gaze_pitch = gaze_yaw = None
            score, looking, eyes_closed = 0.0, False, False

        # --- поза и предметы в руках ---
        feats = g.pose_features(kp, kc, cfg.pose_kpt_conf)
        posture, posture_conf = g.classify_posture(
            feats, None if face is None else face.pitch, cfg)
        held = self._objects_in_hands(kp, kc, feats, objects)
        posture, posture_conf = self._apply_objects(posture, posture_conf, held)

        # --- временное окно трека ---
        window = cam.windows.get(tid)
        if window is None:
            window = cam.windows[tid] = EngagementWindow(cfg)
        window.add(Sample(now, looking, eyes_closed,
                          posture, posture_conf, face is not None))
        resolved = window.resolve()

        return {
            "track_id": int(tid),
            "box": [float(v) for v in box],
            "keypoints": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                          for (x, y), c in zip(kp, kc)],
            "gaze_yaw": None if gaze_yaw is None else round(gaze_yaw, 2),
            "gaze_pitch": None if gaze_pitch is None else round(gaze_pitch, 2),
            "head_yaw": None if face is None else round(face.yaw, 2),
            "head_pitch": None if face is None else round(face.pitch, 2),
            "eye_closure": None if face is None else round(face.eye_closure, 4),
            "attention_score": round(score, 4),
            "looking_now": looking,
            "held_objects": held,
            **resolved,
        }

    # --- предметы в руках --------------------------------------------------

    def _objects_in_hands(self, kp, kc, feats, objects) -> list[str]:
        """Какие предметы лежат рядом с кистями этого человека.

        Радиус привязки — в долях ширины плеч, чтобы не зависеть от расстояния
        до камеры: телефон соседа по парте не должен считаться нашим.
        """
        if not objects or feats is None:
            return []
        radius = feats["shoulder_width"] * self._settings.object_hand_radius
        wrists = [kp[i] for i in (g.LEFT_WRIST, g.RIGHT_WRIST)
                  if kc[i] >= self._settings.pose_kpt_conf]
        if not wrists:
            return []

        found = []
        for name, (x1, y1, x2, y2), _conf in objects:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if any(float(np.hypot(cx - wx, cy - wy)) <= radius for wx, wy in wrists):
                found.append(name)
        return sorted(set(found))

    @staticmethod
    def _apply_objects(posture, posture_conf, held) -> tuple[str, float]:
        """Найденный предмет перебивает эвристику по позе — он однозначнее.

        Упавшую голову не трогаем: если человек спит, телефон в руке этого
        не отменяет.
        """
        if posture == g.POSTURE_SLUMPED:
            return posture, posture_conf
        if PHONE in held:
            return g.POSTURE_PHONE, 0.95
        if BOOK in held or LAPTOP in held:
            return g.POSTURE_WRITING, 0.85
        return posture, posture_conf

    # --- агрегат по аудитории ----------------------------------------------

    @staticmethod
    def aggregate(people: list[dict]) -> dict:
        count = len(people)
        if not count:
            # Никого в кадре — это отсутствие замера, а не нулевая вовлечённость.
            return {"count": 0, "engaged_count": 0, "engagement_rate": None,
                    "mean_engagement": None, "states": {}}

        states: dict[str, int] = {}
        for p in people:
            states[p["state"]] = states.get(p["state"], 0) + 1
        engaged = states.get("engaged", 0) + states.get("writing", 0)
        return {
            "count": count,
            "engaged_count": engaged,
            "engagement_rate": round(engaged / count, 4),
            "mean_engagement": round(sum(p["engagement"] for p in people) / count, 4),
            "states": states,
        }

    # --- состояние камер ---------------------------------------------------

    def _camera(self, camera_id: str, now: float) -> _CameraState:
        cam = self._cameras.get(camera_id)
        if cam is None:
            if len(self._cameras) >= self._settings.max_cameras:
                oldest = min(self._cameras, key=lambda k: self._cameras[k].last_used)
                logger.warning("Лимит камер (%d) исчерпан, вытесняю %r",
                               self._settings.max_cameras, oldest)
                del self._cameras[oldest]
            cam = self._cameras[camera_id] = _CameraState(self._settings)
            logger.info("Новая камера %r, активных: %d", camera_id, len(self._cameras))
        cam.last_used = now
        return cam

    @staticmethod
    def _drop_dead_windows(cam: _CameraState) -> None:
        """Окна умерших треков не должны копиться в памяти."""
        for tid in [t for t in cam.windows if cam.tracker.get(t) is None]:
            del cam.windows[tid]

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _decode(raw: str) -> np.ndarray:
        """base64 JPEG -> BGR numpy array (H, W, 3) uint8 (как отдаёт OpenCV)."""
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
