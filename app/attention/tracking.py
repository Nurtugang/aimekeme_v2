"""Трекинг людей по кадрам: один человек — один стабильный ID.

Зачем свой трекер, а не встроенный в ultralytics. Во-первых, `model.track()`
хранит состояние ВНУТРИ объекта модели: с несколькими камерами их треки
перемешаются, а модель у нас одна на сервис. Во-вторых, аудитория — простой
для трекинга случай: люди сидят и почти не двигаются, перекрытий мало. Здесь
достаточно сопоставления по IoU, и оно даёт полный контроль над состоянием
(своё на камеру), сроком жизни трека и стабильностью ID.

Именно трек, а не детекция, — первичный ключ всего модуля. Скелет, лицо, взгляд
и активность считаются НА ТРЕК, поэтому один человек физически не может получить
два скелета: детектор уже отфильтровал дубли своим NMS, а трекер сводит
оставшееся к одной записи с постоянным id.

Венгерское сопоставление (scipy) вместо жадного: жадный алгоритм на плотной
посадке любит перепутать соседей, когда рамки почти одинаковые.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.attention.geometry import iou


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
