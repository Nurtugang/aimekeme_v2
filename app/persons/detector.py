"""Детекция людей + оценка цвета одежды (верх/низ) по окну кадров одной камеры.

`frames` -- подряд идущие кадры ОДНОЙ камеры (окно). YOLOv8n-pose + BoT-SORT
(model.py) находят людей, скелет и сшивают одного человека между кадрами:
у каждого есть `track_id`, по которому платформа усредняет цвет одежды за окно
(покадровая оценка шумит из-за теней и автоэкспозиции) и берёт бокс с
последнего кадра для скриншота. Усреднение и фильтрация -- на стороне
платформы, здесь только то, для чего нужны пиксели.

Бокс каждого человека делится на торс/ноги по плечам/бёдрам/коленям (для торса
fallback — фиксированные доли высоты, если keypoints не уверенные).
Если коленей в кадре нет — человек сидит за партой или обрезан — низ считаем
невидимым и отдаём `bottom_color: null`, а не цвет парты.

Перед оценкой доминирующего цвета из каждой зоны исключаются пиксели, похожие
на голую кожу (YCrCb-эвристика) — иначе загорелые руки на короткой рубашке или
голые ноги в шортах тянут цвет в сторону тона кожи. Если кожа занимает почти
всю зону — маска игнорируется (на крохах пикселей k-means может зацепиться за
случайную деталь, а не за одежду).

Цвет отдаётся и как ближайшее название из фиксированной палитры, и как сырой
HSV. Фильтрация по цвету — не наша забота: в ответе есть всё, что нужно
платформе, чтобы отобрать людей самой.
"""

from __future__ import annotations

import base64
import binascii
import colorsys
import logging
import threading
import time

import cv2
import numpy as np
import torch

from app.persons.model import detect, load_persons_model
from app.config import Settings

logger = logging.getLogger("surveillance.persons")

_DATA_URI_MARKER = "base64,"
_MIN_CROP_H, _MIN_CROP_W = 40, 20
_MIN_REGION_PX = 6  # зона тоньше -- пикселей не хватит на осмысленный цвет
_MAX_HIP_REL = 0.75   # бёдра ниже этой доли высоты бокса -> ног в кадре нет
_MAX_KNEE_REL = 0.95  # колени у нижней кромки бокса -> они «дорисованы» под партой
_KMEANS_K = 3
_SKIN_FALLBACK_FRACTION = 0.75  # кожа >= этой доли зоны -> маска игнорируется

# (название, представительный RGB) -- палитра для ближайшего названия цвета.
_PALETTE = [
    ("black", (20, 20, 20)),
    ("white", (240, 240, 240)),
    ("gray", (130, 130, 130)),
    ("red", (200, 30, 30)),
    ("maroon", (115, 15, 40)),
    ("orange", (230, 120, 20)),
    ("yellow", (230, 200, 40)),
    ("green", (40, 130, 60)),
    ("cyan", (60, 170, 210)),
    ("blue", (40, 70, 190)),
    ("navy", (20, 25, 80)),
    ("purple", (110, 40, 160)),
    ("pink", (230, 110, 160)),
    ("brown", (110, 70, 40)),
    ("beige", (210, 180, 140)),
]


def _rgb_to_lab(rgb) -> np.ndarray:
    arr = np.uint8([[list(rgb)]])
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0][0].astype(np.float32)


_PALETTE_LAB = [(label, _rgb_to_lab(rgb)) for label, rgb in _PALETTE]


def _nearest_color_label(rgb) -> str:
    lab = _rgb_to_lab(rgb)
    return min(_PALETTE_LAB, key=lambda p: float(np.linalg.norm(p[1] - lab)))[0]


def _rgb_to_hsv_out(rgb) -> list[float]:
    h, s, v = colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    return [round(h * 360, 1), round(s, 3), round(v, 3)]


def _skin_mask(region_bgr: np.ndarray) -> np.ndarray:
    """True где пиксель похож на голую кожу (стандартный диапазон YCrCb)."""
    ycrcb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    return cv2.inRange(ycrcb, lower, upper) > 0


def _dominant_color(region_bgr: np.ndarray) -> tuple[int, int, int]:
    if region_bgr is None or region_bgr.size == 0:
        return (128, 128, 128)

    rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
    mask = _skin_mask(region_bgr)
    pixels = rgb[~mask] if mask.sum() < _SKIN_FALLBACK_FRACTION * mask.size else rgb.reshape(-1, 3)

    pixels = pixels.astype(np.float32)
    if len(pixels) > 1500:
        idx = np.random.choice(len(pixels), 1500, replace=False)
        pixels = pixels[idx]

    k = max(1, min(_KMEANS_K, len(pixels)))
    if k == 1:
        return tuple(int(c) for c in pixels.mean(axis=0))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    return tuple(int(c) for c in centers[np.argmax(counts)])


def _region_bounds(
    crop_h: int, body_lines_rel: dict | None
) -> tuple[tuple[float, float], tuple[float, float] | None]:
    """Границы (верх, низ) зоны в пикселях кропа. Низ = None, если его не видно.

    Низ считаем видимым только когда ноги реально попали в кадр. Одних
    keypoints мало: pose-модель дорисовывает скрытые суставы, и у сидящего за
    партой студента колени «предсказываются» под столом -- ниже бокса детекции,
    который построен по видимой части тела. Поэтому требуем, чтобы бёдра и
    колени лежали внутри бокса (замеры на аудитории: сидящие -- бёдра 0.8-1.3,
    колени 0.93-1.12 высоты бокса; стоящие -- 0.6 и 0.85). Иначе там парта, и
    честнее вернуть None, чем правдоподобное враньё.
    Верх при отсутствии keypoints ещё можно взять по долям высоты (торс всегда
    в верхней части бокса), низ -- уже нет.
    """
    if body_lines_rel is not None:
        shoulder_y = body_lines_rel["shoulder_y"]
        hip_y = body_lines_rel["hip_y"]
        knee_y = body_lines_rel["knee_y"]
        if shoulder_y is not None and hip_y is not None and hip_y > shoulder_y:
            top = (max(0.0, shoulder_y + 0.06 * crop_h), min(crop_h, hip_y - 0.03 * crop_h))
            bottom = None
            legs_in_frame = (
                knee_y is not None
                and knee_y > hip_y
                and hip_y <= _MAX_HIP_REL * crop_h
                and knee_y <= _MAX_KNEE_REL * crop_h
            )
            if legs_in_frame:
                bottom = (max(0.0, hip_y + 0.02 * crop_h), knee_y)
                if bottom[1] - bottom[0] < _MIN_REGION_PX:
                    bottom = None
            if top[1] - top[0] >= _MIN_REGION_PX:
                return top, bottom
    return (crop_h * 0.22, crop_h * 0.52), None


class InvalidFrameError(ValueError):
    """Кадр не удалось декодировать. Несёт 0-based индекс кадра для сообщения."""

    def __init__(self, index: int):
        self.index = index
        super().__init__(f"Invalid base64 in frame {index}")


class PersonsDetector:
    def __init__(self, settings: Settings, device: torch.device):
        self._settings = settings
        self._device = device
        self._model = None
        # Сериализуем доступ к одной модели/GPU между потоками воркеров.
        self._lock = threading.Lock()
        # Сквозной счётчик id: трекер в каждом запросе нумерует с 1, и платформа,
        # склеив ответы двух запросов по track_id, склеила бы разных людей.
        # Наружу отдаём номера, которые в рамках процесса не повторяются.
        self._next_track_id = 1

    # --- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        logger.info("Loading %s on device=%s ...",
                    self._settings.persons_weights, self._device)
        self._model = load_persons_model(self._settings.persons_weights, self._device)
        logger.info("Persons detector ready.")

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return str(self._device)

    # --- inference ---------------------------------------------------------

    def predict(self, frames: list[str]) -> dict:
        """Декод + детекция людей + цвет верха/низа на каждом кадре окна.

        Raises:
            InvalidFrameError: если какой-то кадр не валидный base64/JPEG.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()
        decoded = [self._decode_frame(raw, i) for i, raw in enumerate(frames)]

        with self._lock:
            per_frame_dets = detect(
                self._model, decoded, self._settings.persons_conf_thresh, self._device
            )
            self._renumber_tracks(per_frame_dets)

        results = [
            self._build_frame_result(bgr, dets)
            for bgr, dets in zip(decoded, per_frame_dets)
        ]

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        total_persons = sum(r["count"] for r in results)
        logger.debug("frames=%d, persons=%d (%.1f ms)", len(frames), total_persons, elapsed_ms)

        return {"results": results, "processing_ms": round(elapsed_ms, 2)}

    def _renumber_tracks(self, per_frame_dets: list[list[dict]]) -> None:
        """Локальные id трекера (1, 2, 3... в каждом запросе) -> сквозные по процессу."""
        mapping: dict[int, int] = {}
        for dets in per_frame_dets:
            for det in dets:
                raw = det["track_id"]
                if raw is None:
                    continue
                if raw not in mapping:
                    mapping[raw] = self._next_track_id
                    self._next_track_id += 1
                det["track_id"] = mapping[raw]

    def _build_frame_result(self, bgr: np.ndarray, dets: list[dict]) -> dict:
        h, w = bgr.shape[:2]
        persons = [
            person for det in dets
            if (person := self._describe_person(bgr, w, h, det)) is not None
        ]
        return {"persons": persons, "count": len(persons)}

    def _describe_person(self, bgr: np.ndarray, w: int, h: int, det: dict) -> dict | None:
        x1, y1, x2, y2 = det["box_px"]
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        crop = bgr[int(y1):int(y2), int(x1):int(x2)]
        crop_h, crop_w = crop.shape[:2]
        if crop_h < _MIN_CROP_H or crop_w < _MIN_CROP_W:
            return None

        body_lines_rel = None
        if det["body_lines"] is not None:
            bl = det["body_lines"]
            body_lines_rel = {
                "shoulder_y": bl["shoulder_y"] - y1,
                "hip_y": bl["hip_y"] - y1,
                "knee_y": (bl["knee_y"] - y1) if bl["knee_y"] is not None else None,
            }

        top_bounds, bottom_bounds = _region_bounds(crop_h, body_lines_rel)
        margin_x = max(1, int(crop_w * 0.2))

        t0, t1 = top_bounds
        top_rgb = _dominant_color(crop[int(t0):int(t1), margin_x:crop_w - margin_x])

        bottom_rgb = None
        if bottom_bounds is not None:
            b0, b1 = bottom_bounds
            bottom_rgb = _dominant_color(crop[int(b0):int(b1), margin_x:crop_w - margin_x])

        return {
            "track_id": det["track_id"],
            "box": [x1 / w, y1 / h, x2 / w, y2 / h],
            "confidence": round(det["conf"], 4),
            "top_color": _nearest_color_label(top_rgb),
            "top_hsv": _rgb_to_hsv_out(top_rgb),
            "bottom_visible": bottom_rgb is not None,
            "bottom_color": _nearest_color_label(bottom_rgb) if bottom_rgb else None,
            "bottom_hsv": _rgb_to_hsv_out(bottom_rgb) if bottom_rgb else None,
        }

    # --- helpers -------------------------------------------------------

    @staticmethod
    def _decode_frame(raw: str, index: int) -> np.ndarray:
        """base64 JPEG -> BGR numpy array (H, W, 3), как отдаёт OpenCV."""
        if _DATA_URI_MARKER in raw:
            raw = raw.split(_DATA_URI_MARKER, 1)[1]
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidFrameError(index) from exc
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidFrameError(index)
        return image
