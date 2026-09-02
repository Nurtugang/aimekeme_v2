"""Временной слой: из мгновенных сигналов — вовлечённость, сон, активность.

Вовлечённость нельзя измерить по одному кадру. Взгляд на доску длиной 200 мс —
это моргание внимания, а не вовлечённость; закрытые глаза на одном кадре — это
моргание в прямом смысле. Смысл появляется только на окне в десятки секунд,
поэтому здесь на каждый трек живёт кольцевой буфер отсчётов за последние
`window_s` секунд.

ЭТО ОСОЗНАННЫЙ ОТХОД ОТ ПРАВИЛА 3 в docs/CONVENTIONS.md («stateless: запрос
самодостаточен»). Длительность взгляда — величина принципиально временная, её
невозможно вернуть по одному кадру. Отход ограничен: состояние привязано к
(camera_id, track_id), живёт не дольше window_s + TTL трека и умещается в
несколько килобайт на человека. Мгновенные значения в ответе тоже есть, поэтому
брокер при желании может продолжать считать своё поверх сырых сигналов.

СОН меряется по PERCLOS — доле времени с закрытыми глазами за окно. Это
стандартная метрика сонливости из литературы по водителям, а не выдуманный
порог: моргания дают 5-10%, засыпание — десятки процентов. Плюс требуется
непрерывный отрезок закрытых глаз, чтобы длинное моргание не считалось сном.
"""

from __future__ import annotations

from collections import deque

from app.attention.geometry import (
    POSTURE_NEUTRAL,
    POSTURE_PHONE,
    POSTURE_SLUMPED,
    POSTURE_WRITING,
)

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
