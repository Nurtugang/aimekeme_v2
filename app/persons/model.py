"""YOLOv8-pose + BoT-SORT: детекция людей, keypoints скелета и track_id.

Keypoints (плечи/бёдра/колени) нужны детектору, чтобы делить бокс человека на
верх/низ по фактической анатомии, а не пополам. Трекер (встроенный в ultralytics)
даёт track_id — им платформа сшивает одного человека между кадрами окна и
усредняет цвет одежды. Один класс на выходе — person (COCO id 0).
Размер модели задаётся настройкой `persons_weights` (по умолчанию yolov8l-pose:
нано теряет треть аудитории, замеры в config.py). Веса не коммитим — качаются
один раз при load(), как для counting `yolo_head` / fire `yolo_dfire`.
"""

from __future__ import annotations

import numpy as np
import torch

# COCO-17 keypoint indices.
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_MIN_KEYPOINT_CONF = 0.3
_PERSON_CLASS_ID = 0
# Трекер задаём явно: по умолчанию ultralytics 8.4 берёт свой TRACKTRACK, а он
# в плотной аудитории отдаёт id только 9 людям из 15 (замерено на test/01-09).
_TRACKER_CFG = "botsort.yaml"


def load_persons_model(weights: str, device: torch.device):
    from ultralytics import YOLO

    model = YOLO(weights)
    model.to(device)
    return model


def detect(model, frames_bgr: list[np.ndarray], conf: float, device: torch.device) -> list[list[dict]]:
    """Кадры одной камеры (подряд) -> для каждого кадра список детекций людей:

    {"box_px": (x1, y1, x2, y2), "conf": float, "track_id": int | None,
     "body_lines": dict | None}

    box_px и body_lines (shoulder_y/hip_y/knee_y) — в пикселях исходного кадра
    (не кропа). body_lines = None, если плечи/бёдра не распознались уверенно —
    вызывающий код сам решает, каким запасным вариантом делить бокс.

    Кадры прогоняются по одному через трекер BoT-SORT: он сшивает одного и того
    же человека между кадрами и выдаёт track_id, по которому платформа усредняет
    цвет одежды за окно. Батчем это не сделать — трекер по своей природе идёт
    кадр за кадром.

    Перед каждым окном состояние трекера сбрасывается явно (reset), иначе следы
    от предыдущего запроса дотягиваются до нового — проверено, persist=False на
    первом кадре от этого не спасает. Так сервис остаётся stateless: track_id
    живёт только внутри одного вызова.
    """
    if not frames_bgr:
        return []

    trackers = getattr(getattr(model, "predictor", None), "trackers", None)
    for tracker in trackers or []:
        tracker.reset()

    ul_device = (device.index or 0) if device.type == "cuda" else "cpu"
    return [
        _parse_result(
            model.track(
                frame, classes=[_PERSON_CLASS_ID], conf=conf, device=ul_device,
                tracker=_TRACKER_CFG, persist=(bool(trackers) or i > 0), verbose=False,
            )[0]
        )
        for i, frame in enumerate(frames_bgr)
    ]


def _parse_result(r) -> list[dict]:
    dets: list[dict] = []
    if r.boxes is None:
        return dets

    kpts_xy = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
    kpts_conf = (
        r.keypoints.conf.cpu().numpy()
        if (r.keypoints is not None and r.keypoints.conf is not None)
        else None
    )
    for i, box in enumerate(r.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        body_lines = None
        if kpts_xy is not None and i < len(kpts_xy):
            body_lines = _body_lines(kpts_xy[i], kpts_conf[i] if kpts_conf is not None else None)
        dets.append({
            "box_px": (x1, y1, x2, y2),
            "conf": float(box.conf[0]),
            # None -- трекер ещё не подтвердил след (обычно первые кадры).
            "track_id": int(box.id[0]) if box.id is not None else None,
            "body_lines": body_lines,
        })
    return dets


def _avg_y(xy, conf, idx_a: int, idx_b: int) -> float | None:
    ys = []
    for idx in (idx_a, idx_b):
        if conf is not None and conf[idx] < _MIN_KEYPOINT_CONF:
            continue
        x, y = xy[idx]
        if x > 0 or y > 0:  # (0, 0) = keypoint не предсказан
            ys.append(y)
    return sum(ys) / len(ys) if ys else None


def _body_lines(xy, conf) -> dict | None:
    """Плечи/бёдра/колени в абсолютных координатах кадра (не кропа)."""
    shoulder_y = _avg_y(xy, conf, _L_SHOULDER, _R_SHOULDER)
    hip_y = _avg_y(xy, conf, _L_HIP, _R_HIP)
    if shoulder_y is None or hip_y is None:
        return None
    knee_y = _avg_y(xy, conf, _L_KNEE, _R_KNEE)
    return {"shoulder_y": shoulder_y, "hip_y": hip_y, "knee_y": knee_y}
