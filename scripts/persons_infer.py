"""Офлайн-инференс детекции людей + цвета одежды. НЕ часть сервиса.

Гоняет ту же цепочку, что и API (app/persons/), по видео или фото — через
PersonsDetector.predict(), то есть ровно тот же путь: decode -> детекция ->
деление на верх/низ по keypoints -> цвет. Рисует боксы + подписи цвета,
печатает тайминги. Необязательно можно проверить и query-фильтр.

Запуск (из корня репозитория):
    python scripts/persons_infer.py <путь> <тип> [--conf 0.4]
                                     [--query-top blue] [--query-bottom black]
где <тип> = video | photo (frame/image — синонимы photo).

Видео -> <имя>_persons.mp4 рядом с исходником; фото -> <имя>_persons.jpg + вывод в консоль.
"""

import argparse
import base64
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Корень репозитория в путь, чтобы работал `import app.*` при запуске из scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                    # noqa: E402
from app.persons.detector import PersonsDetector   # noqa: E402

_BOX_COLOR = (0, 180, 0)


def _encode(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def detect(detector: PersonsDetector, window_bgr: list[np.ndarray], query: dict | None):
    """Окно BGR-кадров одной камеры -> (persons по кадрам, infer_ms).

    Ровно один вызов на окно -- как будет делать платформа: внутри окна track_id
    сшивает человека между кадрами, между окнами трекер сбрасывается.
    """
    t0 = time.perf_counter()
    result = detector.predict([_encode(f) for f in window_bgr], query)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    return [fr["persons"] for fr in result["results"]], infer_ms


def draw_overlay(frame: np.ndarray, persons: list[dict]) -> np.ndarray:
    h, w = frame.shape[:2]
    for p in persons:
        x1, y1, x2, y2 = p["box"]
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        bottom = p["bottom_color"] if p["bottom_visible"] else "?"
        label = f"#{p['track_id']} {p['top_color']}/{bottom}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), _BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (200, 46), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"people: {len(persons)}", (22, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _BOX_COLOR, 2, cv2.LINE_AA)
    return frame


def _report(n_frames: int, infer_total_ms: float, wall_total_s: float) -> None:
    print("\n--- Тайминги ---")
    print(f"Кадров обработано:      {n_frames}")
    print(f"Инференс суммарно:      {infer_total_ms / 1000.0:.3f} c")
    if n_frames:
        print(f"Инференс среднее/кадр:   {infer_total_ms / n_frames:.1f} мс")
    print(f"Всего (wall, с I/O):    {wall_total_s:.3f} c")


def run_photo(detector: PersonsDetector, path: Path, query: dict | None):
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Не могу открыть фото: {path}")

    wall0 = time.perf_counter()
    (persons,), infer_ms = detect(detector, [frame], query)
    out_path = path.with_name(f"{path.stem}_persons.jpg")
    cv2.imwrite(str(out_path), draw_overlay(frame, persons))

    print(f"\nРезультат: людей = {len(persons)}")
    for p in persons:
        print(f"  id={p['track_id']} box={p['box']} conf={p['confidence']} "
              f"top={p['top_color']} {p['top_hsv']} "
              f"bottom={p['bottom_color']} {p['bottom_hsv']} visible={p['bottom_visible']}")
    print(f"Сохранено: {out_path}")
    _report(1, infer_ms, time.perf_counter() - wall0)


def run_video(detector: PersonsDetector, path: Path, query: dict | None, window: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не могу открыть видео: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = path.with_name(f"{path.stem}_persons.mp4")
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    if window <= 0:
        window = total or 10 ** 6
    print(f"\nОбрабатываю: {path}  ({total} кадров, {fps:.1f} fps, окно {window})\n")
    idx = 0
    infer_total_ms = 0.0
    wall0 = time.perf_counter()
    while cap.isOpened():
        batch = []
        while len(batch) < window:
            ret, frame = cap.read()
            if not ret:
                break
            batch.append(frame)
        if not batch:
            break

        per_frame_persons, infer_ms = detect(detector, batch, query)
        infer_total_ms += infer_ms
        ids = {p["track_id"] for persons in per_frame_persons for p in persons}
        for frame, persons in zip(batch, per_frame_persons):
            idx += 1
            writer.write(draw_overlay(frame, persons))
        print(f"  Кадры {idx - len(batch) + 1:4d}-{idx:<4d}/{total}  |  "
              f"людей на последнем кадре: {len(per_frame_persons[-1])}  |  треков в окне: {len(ids)}")

    cap.release()
    writer.release()
    print(f"\nГотово! Сохранено: {out_path}")
    _report(idx, infer_total_ms, time.perf_counter() - wall0)


def main():
    parser = argparse.ArgumentParser(description="Офлайн-инференс детекции людей + цвета одежды.")
    parser.add_argument("path", help="путь к видео или фото")
    parser.add_argument("type", choices=["video", "photo", "frame", "image"],
                        help="тип входа: video или photo")
    parser.add_argument("--weights", default=None,
                        help="веса pose-модели (по умолчанию из config)")
    parser.add_argument("--conf", type=float, default=None,
                        help="порог уверенности детектора (по умолчанию из config)")
    parser.add_argument("--window", type=int, default=0,
                        help="кадров в одном запросе (окно трекинга); 0 = всё видео одним "
                             "окном. Внутри окна track_id стабильны, между окнами -- нет")
    parser.add_argument("--query-top", default=None, help="фильтр: только этот цвет верха")
    parser.add_argument("--query-bottom", default=None, help="фильтр: только этот цвет низа")
    args = parser.parse_args()

    if args.weights:
        settings.persons_weights = args.weights
    if args.conf is not None:
        settings.persons_conf_thresh = args.conf

    query = None
    if args.query_top or args.query_bottom:
        query = {"top": args.query_top, "bottom": args.query_bottom}

    path = Path(args.path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Загружаю {settings.persons_weights} на {device} ...")
    detector = PersonsDetector(settings, device)
    detector.load()

    if args.type == "video":
        run_video(detector, path, query, args.window)
    else:
        run_photo(detector, path, query)


if __name__ == "__main__":
    main()
