"""Офлайн-инференс вовлечённости на лекции. НЕ часть сервиса.

Гоняет тот же детектор, что и API (app/attention/), по камере, видео или фото.
На каждого человека рисует ОДИН скелет и подпись с состоянием: скелет привязан
к треку, а не к детекции, поэтому продублировать его нечем.

Запуск (из корня репозитория):
    python scripts/attention_infer.py <источник> <тип> [опции]

    <источник>  путь к видео/фото ИЛИ индекс камеры (0, 1, ...) для типа camera
    <тип>       camera | video | photo (frame/image — синонимы photo)

Примеры:
    python scripts/attention_infer.py 0 camera
    python scripts/attention_infer.py lecture.mp4 video --pose-variant m
    python scripts/attention_infer.py row3.jpg photo --no-objects

Видео -> <имя>_attention.mp4 рядом с исходником; фото -> <имя>_attention.jpg;
камера -> живое окно (q — выход). В конце — сводка по состояниям и тайминги.

ВАЖНО про фото: вовлечённость измеряется ВРЕМЕНЕМ. На одном кадре окно пустое,
поэтому все будут warming_up, а состояния — предварительными. Фото годится,
чтобы проверить скелеты и углы, но не выводы о вовлечённости.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.attention import engagement as gm          # noqa: E402
from app.attention.detector import AttentionDetector  # noqa: E402
from app.config import settings                     # noqa: E402

# Цвет состояния (BGR).
_COLORS = {
    "engaged": (0, 180, 0),
    "writing": (150, 160, 0),
    "distracted": (0, 140, 230),
    "phone": (0, 80, 240),
    "sleeping": (0, 0, 220),
    "unknown": (140, 140, 140),
}
_GREY = (170, 170, 170)


def draw_person(frame, person, kpt_conf):
    """Один скелет, одна рамка, одна подпись — ровно на один трек."""
    color = _COLORS.get(person["state"], _GREY)
    kpts = person["keypoints"]

    # Кости: рисуем ребро, только если ОБЕ точки видны — иначе линия уедет в угол.
    for a, b in gm.SKELETON_EDGES:
        xa, ya, ca = kpts[a]
        xb, yb, cb = kpts[b]
        if ca >= kpt_conf and cb >= kpt_conf:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2, cv2.LINE_AA)
    for x, y, c in kpts:
        if c >= kpt_conf:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)

    x1, y1, x2, y2 = (int(v) for v in person["box"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    head = f"#{person['track_id']} {person['state']} {person['engagement']:.2f}"
    if person["warming_up"]:
        head += " (набор)"
    cv2.putText(frame, head, (x1, max(16, y1 - 24)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    bits = []
    if person["gaze_yaw"] is not None:
        bits.append(f"y{person['gaze_yaw']:+.0f} p{person['gaze_pitch']:+.0f}")
    if person["gaze_hold_s"] > 0:
        bits.append(f"взгляд {person['gaze_hold_s']:.1f}с")
    if person["perclos"] is not None and person["perclos"] > 0.1:
        bits.append(f"perclos {person['perclos']:.2f}")
    if person["held_objects"]:
        bits.append("+".join(person["held_objects"]))
    if bits:
        cv2.putText(frame, "  ".join(bits), (x1, max(32, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREY, 1, cv2.LINE_AA)
    return frame


def draw_panel(frame, result):
    """Сводка по аудитории в левом верхнем углу."""
    rate = result["engagement_rate"]
    color = _COLORS["engaged"] if (rate or 0) >= 0.5 else _COLORS["distracted"]
    w = 430
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w, 122), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (10, 10), (w, 122), color, 2)

    head = (f"ВОВЛЕЧЕНО: {result['engaged_count']}/{result['count']}"
            if result["count"] else "В кадре никого")
    cv2.putText(frame, head, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    states = result["states"]
    line = "  ".join(f"{k}:{v}" for k, v in sorted(states.items())) or "-"
    cv2.putText(frame, line, (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1,
                cv2.LINE_AA)

    mean = result["mean_engagement"]
    cv2.putText(frame, f"средняя вовлечённость: {'-' if mean is None else f'{mean:.2f}'}",
                (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    bar = int((w - 35) * (mean or 0.0))
    cv2.rectangle(frame, (20, 98), (w - 15, 112), (50, 50, 50), -1)
    if bar > 0:
        cv2.rectangle(frame, (20, 98), (20 + bar, 112), color, -1)
    cv2.rectangle(frame, (20, 98), (w - 15, 112), (150, 150, 150), 1)
    return frame


class Stats:
    """Копит состояния по кадрам для финальной сводки."""

    def __init__(self):
        self.states = Counter()
        self.frames = 0
        self.rates = []
        self.infer_ms = 0.0
        self.tracks = set()

    def add(self, result, ms):
        self.frames += 1
        self.infer_ms += ms
        for p in result["people"]:
            self.states[p["state"]] += 1
            self.tracks.add(p["track_id"])
        if result["mean_engagement"] is not None:
            self.rates.append(result["mean_engagement"])

    def report(self, wall_s):
        print("\n--- Сводка ---")
        print(f"Кадров:                 {self.frames}")
        print(f"Уникальных людей:       {len(self.tracks)}")
        total = sum(self.states.values())
        if total:
            print("Человеко-кадров по состояниям:")
            for state, n in self.states.most_common():
                print(f"  {state:<12} {n:7d}  ({n / total * 100:5.1f}%)")
        else:
            print("Людей не найдено ни на одном кадре.")
        if self.rates:
            print(f"Средняя вовлечённость:  {sum(self.rates) / len(self.rates):.3f}")
        print("\n--- Тайминги ---")
        print(f"Инференс суммарно:      {self.infer_ms / 1000.0:.3f} c")
        if self.frames:
            print(f"Инференс среднее/кадр:  {self.infer_ms / self.frames:.1f} мс")
        print(f"Всего (wall, с I/O):    {wall_s:.3f} c")


def process(detector, frame, camera_id):
    """Кадр -> (результат как у эндпоинта, время инференса в мс)."""
    t0 = time.perf_counter()
    people = detector.analyze(frame, camera_id)
    ms = (time.perf_counter() - t0) * 1000.0
    return {**detector.aggregate(people), "people": people}, ms


def render(frame, result, kpt_conf):
    for person in result["people"]:
        draw_person(frame, person, kpt_conf)
    return draw_panel(frame, result)


def open_source(source: str, kind: str):
    """Источник -> (генератор кадров, подпись, fps, размер кадра).

    Все три режима отличаются только тем, откуда берутся кадры и куда уходит
    картинка. Сам цикл обработки один — дублировать его незачем.
    """
    if kind == "camera":
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            raise RuntimeError(f"Не могу открыть камеру с индексом {source}")
        return _frames(cap), "camera", None, None

    path = Path(source)
    if kind == "video":
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Не могу открыть видео: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return _frames(cap), "video", fps, size

    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Не могу открыть фото: {path}")
    return iter([frame]), "photo", None, None


def _frames(cap):
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def run(detector, source: str, kind: str, kpt_conf: float) -> None:
    frames, mode, fps, size = open_source(source, kind)
    path = Path(source) if mode != "camera" else None
    writer = out_path = None
    if mode == "video":
        out_path = path.with_name(f"{path.stem}_attention.mp4")
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if mode == "camera":
        print()
        print("Окно открыто. q — выход. Первые секунды идёт набор окна.")
        print()

    stats = Stats()
    wall0 = time.perf_counter()
    try:
        for idx, frame in enumerate(frames, 1):
            result, ms = process(detector, frame, mode)
            stats.add(result, ms)
            picture = render(frame, result, kpt_conf)

            if writer is not None:
                writer.write(picture)
                print(f"  Кадр {idx:5d} | людей: {result['count']:3d} | "
                      f"вовлечены: {result['engaged_count']:3d}")
            elif mode == "camera":
                cv2.imshow("attention", picture)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            else:
                out_path = path.with_name(f"{path.stem}_attention.jpg")
                cv2.imwrite(str(out_path), picture)
                report_photo(result)
    finally:
        if writer is not None:
            writer.release()
        if mode == "camera":
            cv2.destroyAllWindows()

    if out_path is not None:
        print(f"Сохранено: {out_path}")
    stats.report(time.perf_counter() - wall0)


def report_photo(result) -> None:
    print()
    print("Одно фото = пустое временное окно: состояния предварительные.")
    for p in result["people"]:
        gaze = "-" if p["gaze_yaw"] is None else f"{p['gaze_yaw']:+.1f}/{p['gaze_pitch']:+.1f}"
        print(f"  #{p['track_id']:<3} {p['state']:<11} вовлечённость={p['engagement']:.2f}  "
              f"взгляд={gaze}  активность={p['activity']}  "
              f"предметы={'+'.join(p['held_objects']) or '-'}")


def main():
    parser = argparse.ArgumentParser(description="Офлайн-инференс вовлечённости на лекции.")
    parser.add_argument("source", help="путь к видео/фото или индекс камеры")
    parser.add_argument("type", choices=["camera", "video", "photo", "frame", "image"])
    parser.add_argument("--pose-variant", choices=list("nsmlx"),
                        default=settings.pose_variant,
                        help="размер детектора поз (по умолчанию из POSE_VARIANT)")
    parser.add_argument("--pose-imgsz", type=int, default=settings.pose_imgsz,
                        help="вход детектора поз; для 4К имеет смысл 1600")
    parser.add_argument("--no-objects", action="store_true",
                        help="не искать телефоны/книги (быстрее, но активность грубее)")
    parser.add_argument("--window", type=float, default=settings.engagement_window_s,
                        help="временное окно вовлечённости, секунды")
    parser.add_argument("--threshold", type=float, default=settings.attention_threshold,
                        help="порог «смотрит на доску»")
    parser.add_argument("--yaw-center", type=float, default=settings.attention_yaw_center)
    parser.add_argument("--pitch-center", type=float, default=settings.attention_pitch_center)
    args = parser.parse_args()

    run_settings = settings.model_copy(update={
        "pose_variant": args.pose_variant,
        "pose_imgsz": args.pose_imgsz,
        "object_enabled": settings.object_enabled and not args.no_objects,
        "engagement_window_s": args.window,
        "attention_threshold": args.threshold,
        "attention_yaw_center": args.yaw_center,
        "attention_pitch_center": args.pitch_center,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Загружаю модели на {device}: поза yolo11{run_settings.pose_variant}-pose "
          f"@{run_settings.pose_imgsz}, предметы="
          f"{'вкл' if run_settings.object_enabled else 'выкл'}, "
          f"окно={run_settings.engagement_window_s:.0f} с ...")
    detector = AttentionDetector(run_settings, device)
    detector.load()

    kind = "photo" if args.type in ("frame", "image") else args.type
    run(detector, args.source, kind, run_settings.pose_kpt_conf)


if __name__ == "__main__":
    main()
