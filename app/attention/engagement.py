"""Логика вовлечённости: геометрия, трекинг и время — без единой модели.

Здесь всё, что можно посчитать и проверить без GPU и без весов: углы головы из
матрицы, кроп головы по скелету, признаки позы, сопоставление треков и окно
наблюдений по человеку. Модели отдают сырые числа, этот модуль превращает их
в смысл, а detector.py только оркестрирует.

Отдельный файл (сверх model/detector/schemas/router из CONVENTIONS.md) нужен
ровно ради этого: пороги и правила ниже — самая хрупкая часть модуля, и они
покрыты тестами, которые не поднимают ни одной модели (tests/).

Три слоя, в порядке зависимости:

1. ГЕОМЕТРИЯ — мгновенные измерения по одному кадру.
2. ТРЕКИНГ   — превращает кадры в временной ряд по человеку.
3. ОКНО      — читает вовлечённость из статистики этого ряда.

СИСТЕМА КООРДИНАТ. Пиксели кадра: x вправо, y ВНИЗ (как отдаёт OpenCV).
Углы головы в градусах: yaw>0 — голова повёрнута вправо от камеры,
pitch>0 — НАКЛОНЕНА ВНИЗ, roll>0 — завалена к правому плечу.
Знак pitch задан явно (см. euler_from_matrix), а не угадывается.

МАСШТАБ. Все расстояния по скелету нормируются на ШИРИНУ ПЛЕЧ. В аудитории
бёдра почти всегда закрыты партой, а плечи видны — поэтому единица измерения
именно плечи, и пороги не зависят ни от роста, ни от расстояния до камеры.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


# =====================================================================
# 1. ГЕОМЕТРИЯ — измерения по одному кадру
# =====================================================================

# --- COCO-17: порядок точек, который отдаёт YOLO-pose ---------------------
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12

# Рёбра скелета для отрисовки (только верх тела: ниже пояса в аудитории парта).
SKELETON_EDGES = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
)

_HEAD_POINTS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
# Во сколько раз рамка головы шире разброса видимых точек лица.
_HEAD_MARGIN = 1.6


def euler_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Матрица трансформации лица (4x4 от MediaPipe) -> (pitch, yaw, roll), градусы.

    Углы берём из НАПРАВЛЯЮЩИХ ВЕКТОРОВ, а не разложением по углам Эйлера:
    у Эйлера результат зависит от выбранного порядка осей, и перепутать его
    легко, а вектор «куда смотрит лицо» имеет один-единственный смысл.

    Принятая система (каноническая модель лица MediaPipe):
        X — вправо, Y — ВВЕРХ, Z — из лица в сторону камеры.
    Единичная матрица = лицо смотрит прямо в камеру = (0, 0, 0).

    Знаки на выходе:
        pitch > 0 — голова опущена ВНИЗ;
        yaw   > 0 — голова повёрнута в свою левую сторону;
        roll  > 0 — голова завалена к плечу (только для диагностики,
                    на решения модуля не влияет).
    """
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]

    forward = r @ np.array([0.0, 0.0, 1.0])   # куда смотрит лицо
    right = r @ np.array([1.0, 0.0, 0.0])     # ось «вправо» головы

    horizontal = math.hypot(float(forward[0]), float(forward[2]))
    yaw = math.atan2(float(forward[0]), float(forward[2]))
    # Y смотрит вверх, а у нас вниз — отсюда минус: наклон вниз даёт плюс.
    pitch = math.atan2(-float(forward[1]), horizontal)
    roll = math.atan2(float(right[1]), float(right[0]))

    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def head_box(kpts: np.ndarray, conf: np.ndarray, frame_w: int, frame_h: int,
             min_conf: float) -> tuple[int, int, int, int] | None:
    """Рамка головы по точкам лица скелета -> (x1, y1, x2, y2) в пикселях кадра.

    Кроп берём от скелета, а НЕ отдельным детектором лиц: так на каждый трек
    приходится ровно одна голова, и дублирование лиц невозможно по построению.

    Размер оценивается по разбросу видимых точек лица, а если видна одна-две
    (человек сидит боком) — по ширине плеч, которая почти всегда есть.
    """
    visible = [i for i in _HEAD_POINTS if conf[i] >= min_conf]
    if not visible:
        return None

    pts = kpts[visible]
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())

    spread = 0.0
    if len(visible) >= 2:
        # np.ptp(), а не метод массива: ndarray.ptp() убрали в NumPy 2.0.
        spread = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))

    shoulder = shoulder_width(kpts, conf, min_conf)
    # Голова примерно в треть ширины плеч; берём максимум из двух оценок,
    # чтобы не срезать лицо, когда видна только одна точка.
    size = max(spread * _HEAD_MARGIN, (shoulder or 0.0) * 0.6, 24.0)
    half = size / 2.0

    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(frame_w, int(cx + half))
    y2 = min(frame_h, int(cy + half))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    return x1, y1, x2, y2


def shoulder_width(kpts: np.ndarray, conf: np.ndarray, min_conf: float) -> float | None:
    """Ширина плеч в пикселях — единица масштаба для всех правил позы."""
    if conf[LEFT_SHOULDER] < min_conf or conf[RIGHT_SHOULDER] < min_conf:
        return None
    d = float(np.linalg.norm(kpts[LEFT_SHOULDER] - kpts[RIGHT_SHOULDER]))
    return d if d > 1.0 else None


def _midpoint(kpts, conf, a, b, min_conf):
    ok_a, ok_b = conf[a] >= min_conf, conf[b] >= min_conf
    if ok_a and ok_b:
        return (kpts[a] + kpts[b]) / 2.0
    if ok_a:
        return kpts[a]
    if ok_b:
        return kpts[b]
    return None


def pose_features(kpts: np.ndarray, conf: np.ndarray, min_conf: float) -> dict | None:
    """Скелет -> безразмерные признаки позы (в единицах ширины плеч).

    Всё, что ниже, не зависит от роста человека и расстояния до камеры, потому
    что поделено на ширину плеч. Ось y направлена вниз, поэтому «выше» = меньше y.
    """
    sw = shoulder_width(kpts, conf, min_conf)
    if sw is None:
        return None

    shoulders = _midpoint(kpts, conf, LEFT_SHOULDER, RIGHT_SHOULDER, min_conf)
    if shoulders is None:
        return None

    feats = {"shoulder_width": sw}

    # Насколько нос опущен относительно линии плеч. В норме нос ВЫШЕ плеч,
    # поэтому значение отрицательное (~-0.5). Рост к нулю и выше = голова падает.
    if conf[NOSE] >= min_conf:
        feats["head_drop"] = float((kpts[NOSE][1] - shoulders[1]) / sw)
    else:
        feats["head_drop"] = None

    hands = []
    for wrist, elbow in ((LEFT_WRIST, LEFT_ELBOW), (RIGHT_WRIST, RIGHT_ELBOW)):
        if conf[wrist] < min_conf:
            continue
        w = kpts[wrist]
        hand = {
            # >0 — кисть ВЫШЕ линии плеч (поднята к лицу).
            "lift": float((shoulders[1] - w[1]) / sw),
            # Отклонение кисти вбок от центра корпуса.
            "lateral": float(abs(w[0] - shoulders[0]) / sw),
        }
        # Предплечье: кисть выше локтя = рука поднята, ниже = лежит на парте.
        hand["above_elbow"] = (
            bool(w[1] < kpts[elbow][1]) if conf[elbow] >= min_conf else None)
        if conf[NOSE] >= min_conf:
            hand["to_face"] = float(np.linalg.norm(w - kpts[NOSE]) / sw)
        else:
            hand["to_face"] = None
        hands.append(hand)

    feats["hands"] = hands
    if len(hands) == 2 and conf[LEFT_WRIST] >= min_conf and conf[RIGHT_WRIST] >= min_conf:
        feats["hands_apart"] = float(
            np.linalg.norm(kpts[LEFT_WRIST] - kpts[RIGHT_WRIST]) / sw)
    else:
        feats["hands_apart"] = None
    return feats


# Позы, различимые по одному кадру. «Спит» здесь НЕТ: сон — величина временная
# (закрытые глаза, держащиеся секундами), его решает temporal.py.
POSTURE_NEUTRAL = "neutral"
POSTURE_WRITING = "writing"
POSTURE_PHONE = "phone"
POSTURE_SLUMPED = "slumped"


def classify_posture(feats: dict | None, pitch: float | None, cfg) -> tuple[str, float]:
    """Признаки позы + наклон головы -> (поза, уверенность 0..1).

    Порядок проверок = порядок важности: упавшая голова важнее телефона,
    телефон важнее письма. Уверенность — насколько уверенно сработало правило,
    она нужна временному слою, чтобы взвешивать голоса, а не считать их поровну.

    ВНИМАНИЕ: пороги ниже — эвристики, выведенные из геометрии сидящего человека,
    а не обученная модель. Их НАДО подстроить под свою аудиторию по реальной
    записи (scripts/attention_tune_posture.py). Надёжный признак телефона —
    не поза, а сам телефон, найденный детектором объектов; поза остаётся
    запасным вариантом, когда телефон в кадре не виден.
    """
    if feats is None:
        return POSTURE_NEUTRAL, 0.0

    drop = feats.get("head_drop")
    hands = feats.get("hands") or []
    near_face = [h["to_face"] for h in hands if h["to_face"] is not None]

    # 1) Голова упала к парте или лежит на руках.
    if drop is not None and drop >= cfg.posture_slump_drop:
        return POSTURE_SLUMPED, _ramp(drop, cfg.posture_slump_drop, cfg.posture_slump_drop + 0.3)
    if near_face and min(near_face) < cfg.posture_head_on_hands:
        return POSTURE_SLUMPED, 0.6

    head_down = pitch is not None and pitch >= cfg.posture_head_down_deg

    # 2) Телефон: кисть поднята к груди/лицу, близко к центру корпуса,
    #    голова наклонена вниз — взгляд уходит в руки, а не на доску.
    for h in hands:
        if (h["lift"] > cfg.posture_phone_min_lift
                and h["to_face"] is not None and h["to_face"] < cfg.posture_phone_to_face
                and h["lateral"] < cfg.posture_phone_lateral):
            conf = 0.75 if head_down else 0.45
            return POSTURE_PHONE, conf

    # 3) Письмо: кисть лежит низко (на парте), ниже локтя, в стороне от корпуса.
    for h in hands:
        if (h["lift"] < cfg.posture_write_max_lift
                and h["above_elbow"] is False
                and (h["to_face"] is None or h["to_face"] >= cfg.posture_write_min_to_face)):
            conf = 0.7 if head_down else 0.4
            return POSTURE_WRITING, conf

    return POSTURE_NEUTRAL, 0.5


def _ramp(value: float, low: float, high: float) -> float:
    """Линейная 0..1 между low и high (за границами — 0 и 1)."""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


def fuse_gaze(pitch: float, yaw: float, eye_h: float, eye_v: float,
              gain: float) -> tuple[float, float]:
    """Углы головы + смещение зрачков -> направление ВЗГЛЯДА, градусы.

    Голова даёт основную часть направления, глаза — поправку: человек может
    держать голову прямо и коситься в телефон. `gain` переводит безразмерное
    смещение зрачка (-1..1 от blendshape) в градусы; типичный диапазон
    подвижности глаза в орбите — около +-25 градусов, отсюда дефолт.
    """
    return pitch + eye_v * gain, yaw + eye_h * gain


def attention_score(pitch: float, yaw: float, cfg) -> float:
    """Направление взгляда -> скор 0..1 относительно «направления на доску».

    1.0 точно по центру, 0.5 на границе допуска (эллипс dyaw^2+dpitch^2=1).
    Формула та же, что была в модуле изначально, — порог 0.5 сохраняет смысл
    «внутри допуска» и остаётся сравнимым с прошлыми настройками.
    """
    dyaw = (yaw - cfg.attention_yaw_center) / cfg.attention_yaw_tolerance
    dpitch = (pitch - cfg.attention_pitch_center) / cfg.attention_pitch_tolerance
    return 1.0 / (1.0 + dyaw * dyaw + dpitch * dpitch)


def iou(box_a, box_b) -> float:
    """Пересечение над объединением для двух рамок (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# =====================================================================
# 2. ТРЕКИНГ — один человек, один стабильный id
# =====================================================================

@dataclass
class Track:
    track_id: int
    box: tuple[float, float, float, float]
    last_seen: float
    hits: int = 1


class IouTracker:
    """Трекер на одну камеру. Не потокобезопасен — вызывающий держит лок."""

    def __init__(self, iou_threshold: float, max_age_s: float, min_hits: int):
        self._iou_threshold = iou_threshold
        self._max_age_s = max_age_s
        self._min_hits = min_hits
        self._tracks: dict[int, Track] = {}
        self._ids = itertools.count(1)

    def update(self, boxes: list[tuple[float, float, float, float]],
               now: float | None = None) -> list[int]:
        """Рамки текущего кадра -> id трека для каждой рамки (в том же порядке)."""
        # perf_counter, а не monotonic: у monotonic на Windows шаг ~15 мс,
        # и подряд идущие кадры получают одинаковую метку времени.
        now = time.perf_counter() if now is None else now
        self._expire(now)

        track_ids = list(self._tracks)
        assigned: list[int | None] = [None] * len(boxes)

        if track_ids and boxes:
            cost = np.ones((len(boxes), len(track_ids)), dtype=np.float64)
            for i, box in enumerate(boxes):
                for j, tid in enumerate(track_ids):
                    cost[i, j] = 1.0 - iou(box, self._tracks[tid].box)

            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] <= 1.0 - self._iou_threshold:
                    tid = track_ids[j]
                    assigned[i] = tid
                    track = self._tracks[tid]
                    track.box = boxes[i]
                    track.last_seen = now
                    track.hits += 1

        for i, tid in enumerate(assigned):
            if tid is None:
                new_id = next(self._ids)
                self._tracks[new_id] = Track(new_id, boxes[i], now)
                assigned[i] = new_id

        return [tid for tid in assigned]      # type: ignore[misc]

    def get(self, track_id: int) -> Track | None:
        return self._tracks.get(track_id)

    def is_confirmed(self, track_id: int) -> bool:
        """Трек считается настоящим после min_hits кадров — режет одиночные ложняки."""
        track = self._tracks.get(track_id)
        return track is not None and track.hits >= self._min_hits

    def _expire(self, now: float) -> None:
        dead = [tid for tid, t in self._tracks.items()
                if now - t.last_seen > self._max_age_s]
        for tid in dead:
            del self._tracks[tid]

    @property
    def active(self) -> int:
        return len(self._tracks)


# =====================================================================
# 3. ОКНО — вовлечённость из статистики по времени
# =====================================================================

# Итоговые состояния человека, которые видит платформа.
STATE_ENGAGED = "engaged"        # смотрит на доску
STATE_WRITING = "writing"        # пишет (это тоже вовлечённость, но другого рода)
STATE_PHONE = "phone"            # в телефоне
STATE_SLEEPING = "sleeping"      # спит
STATE_DISTRACTED = "distracted"  # отвлёкся: не спит, не пишет, но и не смотрит
STATE_UNKNOWN = "unknown"        # лица не видно, судить не по чему


class Sample:
    """Один отсчёт по человеку. __slots__ — их тысячи на камеру."""

    __slots__ = ("t", "looking", "eyes_closed", "posture", "posture_conf", "has_face")

    def __init__(self, t, looking, eyes_closed, posture, posture_conf, has_face):
        self.t = t
        self.looking = looking
        self.eyes_closed = eyes_closed
        self.posture = posture
        self.posture_conf = posture_conf
        self.has_face = has_face


class EngagementWindow:
    """Кольцевой буфер отсчётов одного человека и выводы из него."""

    def __init__(self, cfg):
        self._cfg = cfg
        self._samples: deque[Sample] = deque()
        # Длительность ТЕКУЩЕГО непрерывного взгляда на доску и текущей серии
        # закрытых глаз. Считаются нарастающе, а не пересчётом буфера: так
        # непрерывный взгляд длиннее окна не обрезается окном.
        self._gaze_start: float | None = None
        self._closed_start: float | None = None
        self._gaze_hold_s = 0.0
        self._closed_run_s = 0.0

    def add(self, sample: Sample) -> None:
        self._samples.append(sample)
        horizon = sample.t - self._cfg.engagement_window_s
        while self._samples and self._samples[0].t < horizon:
            self._samples.popleft()

        if sample.looking:
            self._gaze_start = sample.t if self._gaze_start is None else self._gaze_start
            self._gaze_hold_s = sample.t - self._gaze_start
        else:
            self._gaze_start = None
            self._gaze_hold_s = 0.0

        if sample.eyes_closed:
            self._closed_start = sample.t if self._closed_start is None else self._closed_start
            self._closed_run_s = sample.t - self._closed_start
        else:
            self._closed_start = None
            self._closed_run_s = 0.0

    # --- производные величины ---------------------------------------------

    @property
    def gaze_hold_s(self) -> float:
        """Сколько секунд длится ТЕКУЩИЙ непрерывный взгляд на доску."""
        return round(self._gaze_hold_s, 2)

    @property
    def span_s(self) -> float:
        """Сколько реально накоплено — пока меньше min_span, выводы ненадёжны."""
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1].t - self._samples[0].t

    def looking_ratio(self) -> float | None:
        """Доля отсчётов с лицом, где человек смотрел на доску."""
        seen = [s for s in self._samples if s.has_face]
        if not seen:
            return None
        return sum(1 for s in seen if s.looking) / len(seen)

    def perclos(self) -> float | None:
        """Доля времени с закрытыми глазами за окно (0..1)."""
        seen = [s for s in self._samples if s.has_face]
        if len(seen) < self._cfg.engagement_min_samples:
            return None
        return sum(1 for s in seen if s.eyes_closed) / len(seen)

    def posture_vote(self) -> tuple[str, float]:
        """Активность за окно: голоса, взвешенные уверенностью правила."""
        if not self._samples:
            return POSTURE_NEUTRAL, 0.0
        weights: dict[str, float] = {}
        for s in self._samples:
            weights[s.posture] = weights.get(s.posture, 0.0) + s.posture_conf
        total = sum(weights.values())
        if total <= 0:
            return POSTURE_NEUTRAL, 0.0
        best = max(weights, key=weights.get)
        return best, weights[best] / total

    def resolve(self) -> dict:
        """Окно -> итог: состояние, скор вовлечённости и величины, из которых он вышел.

        Порядок проверок = приоритет: сон важнее телефона, телефон важнее письма.
        Письмо считается вовлечённостью — человек работает, просто не на доску.
        """
        cfg = self._cfg
        ratio = self.looking_ratio()
        perclos = self.perclos()
        posture, posture_share = self.posture_vote()
        faces = sum(1 for s in self._samples if s.has_face)
        warmup = self.span_s < cfg.engagement_min_span_s

        # Базовый скор: сколько смотрел + надбавка за НЕПРЕРЫВНОСТЬ взгляда.
        # Без второго слагаемого «десять раз мазнул взглядом» равнялось бы
        # «одному внимательному взгляду в полминуты», а это разные вещи.
        hold = min(1.0, self._gaze_hold_s / cfg.engagement_hold_target_s)
        score = (cfg.engagement_look_weight * (ratio or 0.0)
                 + cfg.engagement_hold_weight * hold)

        # 1) Сон. Два независимых пути: глаза видно (PERCLOS + непрерывная серия)
        #    и глаза НЕ видно (голова лежит на парте — лица просто нет в кадре).
        sleeping_by_eyes = (
            perclos is not None and perclos >= cfg.sleep_perclos
            and self._closed_run_s >= cfg.sleep_min_closed_s)
        sleeping_by_pose = (
            posture == POSTURE_SLUMPED and posture_share >= cfg.posture_vote_share
            and not warmup)
        if sleeping_by_eyes or sleeping_by_pose:
            state, score = STATE_SLEEPING, 0.0
        elif posture == POSTURE_PHONE and posture_share >= cfg.posture_vote_share:
            state = STATE_PHONE
            score *= cfg.engagement_phone_factor
        elif posture == POSTURE_WRITING and posture_share >= cfg.posture_vote_share:
            state = STATE_WRITING
            score = max(score, cfg.engagement_writing_floor)
        elif faces == 0:
            state = STATE_UNKNOWN
        elif ratio is not None and ratio >= cfg.engagement_looking_ratio:
            state = STATE_ENGAGED
        else:
            state = STATE_DISTRACTED

        return {
            "state": state,
            "engagement": round(min(1.0, max(0.0, score)), 4),
            "gaze_hold_s": self.gaze_hold_s,
            "looking_ratio": None if ratio is None else round(ratio, 4),
            "perclos": None if perclos is None else round(perclos, 4),
            "eyes_closed_s": round(self._closed_run_s, 2),
            "activity": posture,
            "activity_share": round(posture_share, 4),
            "window_s": round(self.span_s, 2),
            "warming_up": warmup,
        }
