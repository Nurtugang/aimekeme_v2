"""Временной слой: длительность взгляда, PERCLOS, сон, итоговое состояние."""

import pytest

from app.attention.engagement import (
    POSTURE_NEUTRAL, POSTURE_PHONE, POSTURE_SLUMPED, POSTURE_WRITING,
    STATE_DISTRACTED, STATE_ENGAGED, STATE_PHONE, STATE_SLEEPING,
    STATE_UNKNOWN, STATE_WRITING, EngagementWindow, Sample)


def feed(window, n, *, t0=0.0, step=0.5, looking=True, eyes_closed=False,
         posture=POSTURE_NEUTRAL, posture_conf=0.5, has_face=True):
    """Залить n одинаковых отсчётов; возвращает время последнего."""
    t = t0
    for i in range(n):
        t = t0 + i * step
        window.add(Sample(t, looking, eyes_closed, posture, posture_conf, has_face))
    return t


class TestGazeDuration:
    def test_hold_grows_while_looking(self, cfg):
        w = EngagementWindow(cfg)
        feed(w, 11, step=0.5)              # 0.0 .. 5.0 c
        assert w.gaze_hold_s == pytest.approx(5.0)

    def test_hold_resets_when_gaze_breaks(self, cfg):
        w = EngagementWindow(cfg)
        feed(w, 11, step=0.5)
        w.add(Sample(5.5, False, False, POSTURE_NEUTRAL, 0.5, True))
        assert w.gaze_hold_s == 0.0

    def test_hold_survives_longer_than_the_window(self, cfg):
        """Непрерывный взгляд длиннее окна не должен обрезаться окном."""
        w = EngagementWindow(cfg)
        last = feed(w, 200, step=0.5)      # 100 c при окне 30 c
        assert last == pytest.approx(99.5)
        assert w.gaze_hold_s == pytest.approx(99.5)

    def test_old_samples_leave_the_window(self, cfg):
        w = EngagementWindow(cfg)
        feed(w, 200, step=0.5)
        assert w.span_s <= cfg.engagement_window_s + 0.5


class TestSleep:
    def test_perclos_counts_closed_share(self, cfg):
        w = EngagementWindow(cfg)
        feed(w, 10, t0=0.0, step=0.5, eyes_closed=False)
        feed(w, 10, t0=5.0, step=0.5, eyes_closed=True)
        assert w.perclos() == pytest.approx(0.5)

    def test_sleeping_when_eyes_closed_long_enough(self, cfg):
        w = EngagementWindow(cfg)
        feed(w, 40, step=0.5, looking=False, eyes_closed=True)
        out = w.resolve()
        assert out["state"] == STATE_SLEEPING
        assert out["engagement"] == 0.0

    def test_blink_is_not_sleep(self, cfg):
        """Моргание — доли секунды, оно не должно давать сон."""
        w = EngagementWindow(cfg)
        feed(w, 40, t0=0.0, step=0.5, eyes_closed=False)
        w.add(Sample(20.0, True, True, POSTURE_NEUTRAL, 0.5, True))
        assert w.resolve()["state"] != STATE_SLEEPING

    def test_sleeping_detected_without_face_when_head_is_down(self, cfg):
        """Спит лицом в парту: лица не видно, но поза говорит сама за себя."""
        w = EngagementWindow(cfg)
        feed(w, 40, step=0.5, looking=False, posture=POSTURE_SLUMPED,
             posture_conf=0.8, has_face=False)
        assert w.resolve()["state"] == STATE_SLEEPING


class TestStates:
    def test_engaged_when_mostly_looking(self, cfg):
        w = EngagementWindow(cfg)
        feed(w, 40, step=0.5, looking=True)
        out = w.resolve()
        assert out["state"] == STATE_ENGAGED
        assert out["engagement"] > 0.9

    def test_distracted_when_rarely_looking(self, cfg):
        w = EngagementWindow(cfg)
        feed(w, 40, step=0.5, looking=False)
        out = w.resolve()
        assert out["state"] == STATE_DISTRACTED
        assert out["engagement"] < 0.2

    def test_phone_beats_looking(self, cfg):
        """Телефон в руках — не вовлечённость, даже если голова к доске."""
        w = EngagementWindow(cfg)
        feed(w, 40, step=0.5, looking=True, posture=POSTURE_PHONE, posture_conf=0.9)
        out = w.resolve()
        assert out["state"] == STATE_PHONE
        assert out["engagement"] < 0.3

    def test_writing_counts_as_engagement(self, cfg):
        """Пишет, на доску не смотрит — но работает, и это вовлечённость."""
        w = EngagementWindow(cfg)
        feed(w, 40, step=0.5, looking=False, posture=POSTURE_WRITING, posture_conf=0.9)
        out = w.resolve()
        assert out["state"] == STATE_WRITING
        assert out["engagement"] >= cfg.engagement_writing_floor

    def test_unknown_when_face_never_seen(self, cfg):
        w = EngagementWindow(cfg)
        feed(w, 40, step=0.5, looking=False, has_face=False)
        assert w.resolve()["state"] == STATE_UNKNOWN

    def test_warming_up_flag_clears_after_min_span(self, cfg):
        w = EngagementWindow(cfg)
        feed(w, 4, step=0.5)                        # 1.5 c — мало
        assert w.resolve()["warming_up"] is True
        feed(w, 30, t0=2.0, step=0.5)               # набрали больше min_span
        assert w.resolve()["warming_up"] is False

    def test_weak_posture_votes_do_not_flip_state(self, cfg):
        """Один-два кадра «похоже на телефон» не должны менять вердикт."""
        w = EngagementWindow(cfg)
        feed(w, 38, step=0.5, looking=True)
        feed(w, 2, t0=19.0, step=0.5, looking=True, posture=POSTURE_PHONE, posture_conf=0.4)
        assert w.resolve()["state"] == STATE_ENGAGED


class TestEngagementScore:
    def test_sustained_gaze_scores_above_flickering_gaze(self, cfg):
        """Смотрел долго подряд > мазнул взглядом столько же раз вразброс."""
        steady = EngagementWindow(cfg)
        feed(steady, 20, step=0.5, looking=True)

        flick = EngagementWindow(cfg)
        for i in range(20):
            flick.add(Sample(i * 0.5, i % 2 == 0, False, POSTURE_NEUTRAL, 0.5, True))

        assert steady.resolve()["engagement"] > flick.resolve()["engagement"]

    @pytest.mark.parametrize("looking,eyes,posture", [
        (True, False, POSTURE_NEUTRAL),
        (False, True, POSTURE_SLUMPED),
        (True, False, POSTURE_PHONE),
        (False, False, POSTURE_WRITING),
    ])
    def test_score_always_in_unit_range(self, cfg, looking, eyes, posture):
        w = EngagementWindow(cfg)
        feed(w, 40, step=0.5, looking=looking, eyes_closed=eyes,
             posture=posture, posture_conf=0.9)
        assert 0.0 <= w.resolve()["engagement"] <= 1.0

    def test_empty_window_is_safe(self, cfg):
        out = EngagementWindow(cfg).resolve()
        assert out["state"] == STATE_UNKNOWN
        assert out["engagement"] == 0.0
        assert out["looking_ratio"] is None
