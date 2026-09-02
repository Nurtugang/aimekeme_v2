"""Геометрия: углы головы, признаки позы, классификация активности."""

import math

import numpy as np
import pytest

from app.attention import engagement as g


def rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def to4(r):
    m = np.eye(4)
    m[:3, :3] = r
    return m


class TestEuler:
    def test_identity_is_straight_ahead(self):
        assert g.euler_from_matrix(np.eye(4)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)

    @pytest.mark.parametrize("deg", [10, 30, -25])
    def test_yaw_about_y(self, deg):
        pitch, yaw, roll = g.euler_from_matrix(to4(rot_y(deg)))
        assert yaw == pytest.approx(deg, abs=1e-6)
        assert pitch == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("deg", [10, 30, -20])
    def test_pitch_positive_means_head_down(self, deg):
        """Вращение вокруг X на +deg наклоняет лицо вниз => pitch > 0."""
        pitch, yaw, roll = g.euler_from_matrix(to4(rot_x(deg)))
        assert pitch == pytest.approx(deg, abs=1e-6)
        assert yaw == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("deg", [15, -35])
    def test_roll_about_z(self, deg):
        pitch, yaw, roll = g.euler_from_matrix(to4(rot_z(deg)))
        assert roll == pytest.approx(deg, abs=1e-6)

    def test_combined_yaw_and_pitch_stay_separable(self):
        pitch, yaw, _ = g.euler_from_matrix(to4(rot_y(20) @ rot_x(15)))
        assert yaw == pytest.approx(20, abs=0.5)
        assert pitch == pytest.approx(15, abs=0.5)


def skeleton(nose=(200, 140), wrists=(None, None), elbows=((140, 270), (260, 270))):
    """Синтетический сидящий человек. Плечи (150,200)-(250,200) => ширина 100 px."""
    kp = np.zeros((17, 2), dtype=float)
    conf = np.zeros(17, dtype=float)

    def put(idx, xy):
        kp[idx] = xy
        conf[idx] = 0.9

    put(g.NOSE, nose)
    put(g.LEFT_EYE, (nose[0] - 12, nose[1] - 5))
    put(g.RIGHT_EYE, (nose[0] + 12, nose[1] - 5))
    put(g.LEFT_SHOULDER, (150, 200))
    put(g.RIGHT_SHOULDER, (250, 200))
    put(g.LEFT_ELBOW, elbows[0])
    put(g.RIGHT_ELBOW, elbows[1])
    for idx, w in zip((g.LEFT_WRIST, g.RIGHT_WRIST), wrists):
        if w is not None:
            put(idx, w)
    return kp, conf


class TestPoseFeatures:
    def test_scale_is_shoulder_width(self):
        kp, conf = skeleton()
        feats = g.pose_features(kp, conf, 0.35)
        assert feats["shoulder_width"] == pytest.approx(100.0)

    def test_head_drop_negative_when_head_upright(self):
        kp, conf = skeleton()
        feats = g.pose_features(kp, conf, 0.35)
        assert feats["head_drop"] == pytest.approx(-0.6)

    def test_no_shoulders_means_no_features(self):
        kp, conf = skeleton()
        conf[g.LEFT_SHOULDER] = conf[g.RIGHT_SHOULDER] = 0.0
        assert g.pose_features(kp, conf, 0.35) is None

    def test_features_are_scale_invariant(self):
        """Тот же человек вдвое дальше от камеры даёт те же безразмерные числа."""
        kp, conf = skeleton(nose=(200, 140), wrists=((210, 230), None))
        near = g.pose_features(kp, conf, 0.35)
        far = g.pose_features(kp * 0.5, conf, 0.35)
        assert far["head_drop"] == pytest.approx(near["head_drop"])
        assert far["hands"][0]["lift"] == pytest.approx(near["hands"][0]["lift"])


class TestPosture:
    def test_slumped_when_head_falls_to_shoulder_line(self, cfg):
        kp, conf = skeleton(nose=(200, 195))
        feats = g.pose_features(kp, conf, 0.35)
        assert g.classify_posture(feats, 20.0, cfg)[0] == g.POSTURE_SLUMPED

    def test_phone_when_hand_raised_near_face(self, cfg):
        kp, conf = skeleton(wrists=((210, 230), None))
        feats = g.pose_features(kp, conf, 0.35)
        assert g.classify_posture(feats, 25.0, cfg)[0] == g.POSTURE_PHONE

    def test_writing_when_hand_lies_low_on_desk(self, cfg):
        kp, conf = skeleton(wrists=(None, (270, 300)))
        feats = g.pose_features(kp, conf, 0.35)
        assert g.classify_posture(feats, 25.0, cfg)[0] == g.POSTURE_WRITING

    def test_neutral_when_hands_far_and_low_lift(self, cfg):
        kp, conf = skeleton(wrists=(None, (300, 240)))
        feats = g.pose_features(kp, conf, 0.35)
        assert g.classify_posture(feats, 0.0, cfg)[0] == g.POSTURE_NEUTRAL

    def test_head_down_raises_confidence(self, cfg):
        """Один и тот же жест увереннее, когда голова наклонена к рукам."""
        kp, conf = skeleton(wrists=((210, 230), None))
        feats = g.pose_features(kp, conf, 0.35)
        _, up = g.classify_posture(feats, 0.0, cfg)
        _, down = g.classify_posture(feats, 30.0, cfg)
        assert down > up

    def test_no_features_is_neutral_with_zero_confidence(self, cfg):
        assert g.classify_posture(None, 10.0, cfg) == (g.POSTURE_NEUTRAL, 0.0)


class TestGazeAndScore:
    def test_score_is_one_at_center(self, cfg):
        assert g.attention_score(cfg.attention_pitch_center,
                                 cfg.attention_yaw_center, cfg) == pytest.approx(1.0)

    def test_score_is_half_on_tolerance_boundary(self, cfg):
        """0.5 ровно на границе допуска — отсюда осмысленность порога 0.5."""
        yaw = cfg.attention_yaw_center + cfg.attention_yaw_tolerance
        assert g.attention_score(cfg.attention_pitch_center, yaw, cfg) == pytest.approx(0.5)
        pitch = cfg.attention_pitch_center + cfg.attention_pitch_tolerance
        assert g.attention_score(pitch, cfg.attention_yaw_center, cfg) == pytest.approx(0.5)

    @pytest.mark.parametrize("pitch,yaw", [(0, 0), (180, 180), (-90, 90), (1e-6, 0)])
    def test_score_stays_in_unit_range(self, cfg, pitch, yaw):
        """Схема объявляет ge=0 le=1 — выход за диапазон уронил бы ответ."""
        assert 0.0 < g.attention_score(pitch, yaw, cfg) <= 1.0

    def test_eyes_shift_gaze_away_from_head_direction(self):
        """Голова прямо, зрачки вбок — взгляд уже не по центру."""
        pitch, yaw = g.fuse_gaze(0.0, 0.0, eye_h=0.8, eye_v=0.0, gain=25.0)
        assert yaw == pytest.approx(20.0)
        assert pitch == pytest.approx(0.0)

    def test_zero_gain_disables_eye_correction(self):
        assert g.fuse_gaze(5.0, -3.0, 1.0, 1.0, gain=0.0) == (5.0, -3.0)


class TestHeadBox:
    def test_box_is_clamped_to_frame(self):
        kp, conf = skeleton(nose=(5, 5))
        box = g.head_box(kp, conf, 640, 480, 0.35)
        assert box is not None
        x1, y1, x2, y2 = box
        assert x1 >= 0 and y1 >= 0 and x2 <= 640 and y2 <= 480

    def test_none_when_no_face_points_visible(self):
        kp, conf = skeleton()
        for i in (g.NOSE, g.LEFT_EYE, g.RIGHT_EYE, g.LEFT_EAR, g.RIGHT_EAR):
            conf[i] = 0.0
        assert g.head_box(kp, conf, 640, 480, 0.35) is None

    def test_box_covers_the_face_points(self):
        kp, conf = skeleton()
        x1, y1, x2, y2 = g.head_box(kp, conf, 640, 480, 0.35)
        nx, ny = kp[g.NOSE]
        assert x1 <= nx <= x2 and y1 <= ny <= y2


class TestIou:
    def test_identical_boxes(self):
        assert g.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert g.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_half_overlap(self):
        # Пересечение 50, объединение 150.
        assert g.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)

    def test_touching_boxes_do_not_overlap(self):
        assert g.iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0
