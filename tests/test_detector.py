"""Детектор целиком, с подменёнными моделями.

Модели здесь фальшивые: проверяется СБОРКА конвейера — что трек один на
человека, что скелет не дублируется, что состояние камер изолировано и не течёт.
Именно эти свойства ломаются при правках, и именно их нельзя проверить, глядя
на картинку глазами.
"""

import numpy as np
import pytest
import torch

from app.attention.detector import AttentionDetector
from app.attention.model_face import FaceSignals


class FakePose:
    """Отдаёт заранее заданные детекции. Скелет — сидящий человек."""

    def __init__(self, layout):
        self.layout = layout
        self.is_ready = True

    def predict(self, bgr):
        boxes, kpts, confs = [], [], []
        for cx, cy in self.layout:
            boxes.append([cx - 50, cy - 120, cx + 50, cy + 160])
            kp = np.zeros((17, 2), dtype=float)
            cf = np.zeros(17, dtype=float)
            for idx, xy in ((0, (cx, cy - 80)), (1, (cx - 12, cy - 85)), (2, (cx + 12, cy - 85)),
                            (5, (cx - 50, cy)), (6, (cx + 50, cy)),
                            (7, (cx - 60, cy + 70)), (8, (cx + 60, cy + 70)),
                            (9, (cx - 55, cy + 130)), (10, (cx + 55, cy + 130))):
                kp[idx] = xy
                cf[idx] = 0.9
            kpts.append(kp)
            confs.append(cf)
        return np.array(boxes, dtype=float), np.array(kpts), np.array(confs)


class FakeFace:
    """Лицо, смотрящее прямо; глаза открыты."""

    def __init__(self, signals=None):
        self.signals = signals if signals is not None else FaceSignals(
            0.0, 0.0, 0.05, 0.0, 0.0)
        self.is_ready = True
        self.calls = 0

    def predict(self, rgb):
        self.calls += 1
        return self.signals


@pytest.fixture
def make(tune):
    def _make(layout, face=None, **over):
        cfg = tune(object_enabled=False, track_min_hits=1, **over)
        det = AttentionDetector(cfg, torch.device("cpu"))
        det._pose = FakePose(layout)
        det._face = face or FakeFace()
        det._objects = None
        return det
    return _make


FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


class TestNoDuplicates:
    def test_one_skeleton_per_person(self, make):
        det = make([(300, 300), (700, 300), (1000, 300)])
        people = det.analyze(FRAME, "cam")
        assert len(people) == 3
        assert all(len(p["keypoints"]) == 17 for p in people)

    def test_track_ids_are_unique_within_a_frame(self, make):
        det = make([(300, 300), (700, 300), (1000, 300)])
        for _ in range(30):
            ids = [p["track_id"] for p in det.analyze(FRAME, "cam")]
            assert len(ids) == len(set(ids))

    def test_track_ids_are_stable_over_time(self, make):
        """ID не должны прыгать — иначе окно вовлечённости сбрасывается каждый кадр."""
        det = make([(300, 300), (700, 300)])
        first = sorted(p["track_id"] for p in det.analyze(FRAME, "cam"))
        for _ in range(50):
            later = sorted(p["track_id"] for p in det.analyze(FRAME, "cam"))
        assert first == later

    def test_person_count_never_exceeds_detections(self, make):
        det = make([(300, 300), (700, 300)])
        for _ in range(20):
            assert len(det.analyze(FRAME, "cam")) <= 2

    def test_face_is_queried_once_per_person(self, make):
        """Лицо считается на кроп трека: ровно один вызов на человека за кадр."""
        face = FakeFace()
        det = make([(300, 300), (700, 300)], face=face)
        det.analyze(FRAME, "cam")
        assert face.calls == 2


class TestCameraIsolation:
    def test_tracks_do_not_leak_between_cameras(self, make):
        """Разные камеры — разные трекеры; иначе потоки перемешаются."""
        det = make([(300, 300)])
        det.analyze(FRAME, "cam_a")
        det.analyze(FRAME, "cam_a")
        b = det.analyze(FRAME, "cam_b")
        assert b[0]["window_s"] == 0.0        # у cam_b своё, пустое окно

    def test_window_accumulates_per_camera(self, make):
        """Отсчёты копятся в окне своей камеры, а не общего."""
        det = make([(300, 300)])
        for _ in range(10):
            det.analyze(FRAME, "cam_a")
        det.analyze(FRAME, "cam_b")

        def samples(cam):
            windows = det._cameras[cam].windows
            return sum(len(w._samples) for w in windows.values())

        assert samples("cam_a") == 10
        assert samples("cam_b") == 1

    def test_camera_limit_evicts_oldest(self, make):
        det = make([(300, 300)], max_cameras=3)
        for i in range(6):
            det.analyze(FRAME, f"cam{i}")
        assert len(det._cameras) <= 3

    def test_dead_track_windows_are_released(self, make):
        """Человек ушёл — его окно не должно остаться в памяти навсегда."""
        det = make([(300, 300)], track_max_age_s=0.0)
        det.analyze(FRAME, "cam")
        det._pose.layout = []
        det.analyze(FRAME, "cam")
        assert det._cameras["cam"].windows == {}
