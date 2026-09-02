"""Геометрия вовлечённости: чистая математика без моделей и тяжёлых зависимостей.

Здесь живёт всё, что можно посчитать и проверить без GPU: углы головы из матрицы
трансформации, кроп головы по скелету, слияние взгляда, классификация позы
(пишет / телефон / спит). Модели дают сырые числа — этот модуль превращает их
в смысл, а detector.py только оркестрирует.

Разделение сделано ради тестируемости: правила ниже — эвристики, их придётся
подкручивать под конкретную аудиторию, и делать это надо на юнит-тестах, а не
гоняя 4К-видео.

СИСТЕМА КООРДИНАТ. Пиксели кадра: x вправо, y ВНИЗ (как отдаёт OpenCV).
Углы головы в градусах: yaw>0 — голова повёрнута вправо от камеры,
pitch>0 — НАКЛОНЕНА ВНИЗ, roll>0 — завалена к правому плечу.
Знак pitch задан здесь явно (см. euler_from_matrix), а не угадывается.

МАСШТАБ. Все расстояния по скелету нормируются на ШИРИНУ ПЛЕЧ. В аудитории
бёдра почти всегда закрыты партой, а плечи видны — поэтому единица измерения
именно плечи, и пороги не зависят ни от роста, ни от расстояния до камеры.
"""

from __future__ import annotations

import math

import numpy as np

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
