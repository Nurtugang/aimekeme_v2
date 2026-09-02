"""Трекинг: один человек — один стабильный id, без дублей и без утечек.

Это тесты требования «не дублировать скелет»: скелет рисуется на трек, поэтому
достаточно доказать, что трек на человека ровно один и он не размножается.
"""

import pytest

from app.attention.tracking import IouTracker


@pytest.fixture
def tracker():
    return IouTracker(iou_threshold=0.3, max_age_s=3.0, min_hits=3)


def box(x, y, w=40, h=80):
    return (x, y, x + w, y + h)


class TestIdentity:
    def test_same_person_keeps_id_across_frames(self, tracker):
        """Сидящий человек слегка шевелится — id обязан остаться прежним."""
        ids = [tracker.update([box(100 + i, 200)], now=i * 0.1)[0] for i in range(10)]
        assert len(set(ids)) == 1

    def test_two_people_get_two_stable_ids(self, tracker):
        first = tracker.update([box(100, 200), box(400, 200)], now=0.0)
        second = tracker.update([box(102, 201), box(398, 202)], now=0.1)
        assert first == second
        assert len(set(first)) == 2

    def test_never_returns_duplicate_ids_in_one_frame(self, tracker):
        """Главная гарантия против дублей скелета."""
        boxes = [box(100 * i, 200) for i in range(6)]
        for t in range(5):
            ids = tracker.update(boxes, now=t * 0.1)
            assert len(ids) == len(set(ids)) == len(boxes)

    def test_new_person_gets_new_id(self, tracker):
        tracker.update([box(100, 200)], now=0.0)
        ids = tracker.update([box(100, 200), box(500, 200)], now=0.1)
        assert len(set(ids)) == 2

    def test_crossing_neighbours_do_not_swap_ids(self, tracker):
        """Плотная посадка: соседние рамки почти совпадают по размеру."""
        a, b = tracker.update([box(100, 200), box(150, 200)], now=0.0)
        a2, b2 = tracker.update([box(104, 200), box(154, 200)], now=0.1)
        assert (a, b) == (a2, b2)


class TestLifecycle:
    def test_track_is_unconfirmed_before_min_hits(self, tracker):
        tid = tracker.update([box(100, 200)], now=0.0)[0]
        assert not tracker.is_confirmed(tid)
        tracker.update([box(100, 200)], now=0.1)
        assert not tracker.is_confirmed(tid)
        tracker.update([box(100, 200)], now=0.2)
        assert tracker.is_confirmed(tid)

    def test_track_expires_after_max_age(self, tracker):
        tid = tracker.update([box(100, 200)], now=0.0)[0]
        tracker.update([], now=4.0)
        assert tracker.get(tid) is None
        assert tracker.active == 0

    def test_short_gap_does_not_kill_the_track(self, tracker):
        """Человека на пару кадров закрыли — id должен пережить это."""
        tid = tracker.update([box(100, 200)], now=0.0)[0]
        tracker.update([], now=1.0)
        assert tracker.update([box(100, 200)], now=2.0)[0] == tid

    def test_memory_does_not_grow_on_churn(self, tracker):
        """Люди приходят и уходят — мёртвые треки не должны копиться."""
        for t in range(50):
            tracker.update([box(10 * t, 200)], now=t * 1.0)
        assert tracker.active <= 4

    def test_empty_frame_is_handled(self, tracker):
        assert tracker.update([], now=0.0) == []
