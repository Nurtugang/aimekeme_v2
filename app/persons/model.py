"""YOLOv8n-pose: детекция людей + keypoints скелета.

Keypoints (плечи/бёдра/колени) нужны детектору, чтобы делить бокс человека на
верх/низ по фактической анатомии, а не пополам. Один класс на выходе — person
(COCO id 0). Веса (~6.5 МБ) не коммитим — качаются один раз при load() в кэш
ultralytics, как и для counting `yolo_head` / fire `yolo_dfire`.
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


def load_persons_model(device: torch.device):
    from ultralytics import YOLO

    model = YOLO("yolov8n-pose.pt")
    model.to(device)
    return model


def detect(model, frames_bgr: list[np.ndarray], conf: float, device: torch.device) -> list[list[dict]]:
    """Батч BGR-кадров -> для каждого кадра список детекций людей:

    {"box_px": (x1, y1, x2, y2), "conf": float, "body_lines": dict | None}

    box_px и body_lines (shoulder_y/hip_y/knee_y) — в пикселях исходного кадра
    (не кропа). body_lines = None, если плечи/бёдра не распознались уверенно —
    вызывающий код сам решает, каким запасным вариантом делить бокс.
    """
    if not frames_bgr:
        return []

    ul_device = (device.index or 0) if device.type == "cuda" else "cpu"
    results = model(
        frames_bgr, classes=[_PERSON_CLASS_ID], conf=conf, device=ul_device, verbose=False
    )

    per_frame: list[list[dict]] = []
    for r in results:
        dets: list[dict] = []
        if r.boxes is None:
            per_frame.append(dets)
            continue

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
                "body_lines": body_lines,
            })
        per_frame.append(dets)
    return per_frame


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
