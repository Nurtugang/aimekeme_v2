"""Офлайн-инференс подсчёта людей. НЕ часть сервиса.

Гоняет ту же модель, что и API (app/counting/), по видео или фото, рисует боксы +
число в углу и печатает тайминги. Модель выбирается аргументом:
- frcnn      — torchvision Faster R-CNN, класс person;
- yolo_head  — YOLOv8-детектор голов (SCUT-HEAD), точнее в толпе.

Запуск (из корня репозитория):
    python test/counting_infer.py <путь> <тип> [--model frcnn|yolo_head]
                                              [--variant medium|nano] [--conf 0.5]
где <тип> = video | photo (frame/image — синонимы photo).

Видео -> <имя>_count.mp4 рядом с исходником; фото -> <имя>_count.jpg + вывод в консоль.
После обработки печатает: число кадров, суммарное и среднее время инференса на кадр.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Корень репозитория в путь, чтобы работал `import app.*` при запуске из test/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                       # noqa: E402
from app.counting.model_frcnn import FrcnnCounter     # noqa: E402
from app.counting.model_yolo_head import YoloHeadCounter  # noqa: E402

_BACKENDS = {"frcnn": FrcnnCounter, "yolo_head": YoloHeadCounter}
_BOX_COLOR = (0, 180, 0)


def detect(backend, frame_bgr) -> tuple[int, np.ndarray, float]:
    """Один BGR-кадр -> (count, boxes_xyxy, infer_ms)."""
    t0 = time.perf_counter()
    boxes, _ = backend.predict(frame_bgr)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    return len(boxes), boxes, infer_ms


def draw_overlay(frame, count, boxes, model_name):
    """Боксы + число в верхнем левом углу."""
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (260, 78), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (10, 10), (260, 78), _BOX_COLOR, 2)
    cv2.putText(frame, f"people: {count}", (22, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, _BOX_COLOR, 2, cv2.LINE_AA)
    cv2.putText(frame, model_name, (22, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def _report(n_frames: int, infer_total_s: float, wall_total_s: float) -> None:
    print("\n--- Тайминги ---")
    print(f"Кадров обработано:      {n_frames}")
    print(f"Инференс суммарно:      {infer_total_s:.3f} c")
    if n_frames:
        print(f"Инференс среднее/кадр:   {infer_total_s / n_frames * 1000.0:.1f} мс")
    print(f"Всего (wall, с I/O):    {wall_total_s:.3f} c")


def run_photo(backend, path: Path, model_name: str):
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Не могу открыть фото: {path}")

    wall0 = time.perf_counter()
    count, boxes, infer_ms = detect(backend, frame)
    out_path = path.with_name(f"{path.stem}_count.jpg")
    cv2.imwrite(str(out_path), draw_overlay(frame, count, boxes, model_name))

    print(f"\nРезультат: людей = {count}")
    print(f"Сохранено: {out_path}")
    _report(1, infer_ms / 1000.0, time.perf_counter() - wall0)


def run_video(backend, path: Path, model_name: str):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не могу открыть видео: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = path.with_name(f"{path.stem}_count.mp4")
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    print(f"\nОбрабатываю: {path}  ({total} кадров, {fps:.1f} fps)  моделью '{model_name}'\n")
    idx = 0
    infer_total = 0.0
    wall0 = time.perf_counter()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        count, boxes, infer_ms = detect(backend, frame)
        infer_total += infer_ms
        writer.write(draw_overlay(frame, count, boxes, model_name))
        print(f"  Кадр {idx:5d}/{total}  |  людей: {count}")

    cap.release()
    writer.release()
    print(f"\nГотово! Сохранено: {out_path}")
    _report(idx, infer_total / 1000.0, time.perf_counter() - wall0)


def main():
    parser = argparse.ArgumentParser(description="Офлайн-инференс подсчёта людей.")
    parser.add_argument("path", help="путь к видео или фото")
    parser.add_argument("type", choices=["video", "photo", "frame", "image"],
                        help="тип входа: video или photo")
    parser.add_argument("--model", choices=list(_BACKENDS), default="frcnn",
                        help="модель подсчёта (по умолчанию frcnn)")
    parser.add_argument("--variant", choices=["medium", "nano"], default=None,
                        help="для yolo_head: вариант весов (по умолчанию из config)")
    parser.add_argument("--conf", type=float, default=None,
                        help="порог score/conf (по умолчанию из config)")
    args = parser.parse_args()

    # CLI перекрывает настройки из config для этого запуска.
    if args.variant is not None:
        settings.count_head_variant = args.variant
    if args.conf is not None:
        settings.count_score_thresh = args.conf

    path = Path(args.path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Загружаю модель '{args.model}' на {device} ...")
    backend = _BACKENDS[args.model](device, settings)
    backend.load()

    if args.type == "video":
        run_video(backend, path, args.model)
    else:
        run_photo(backend, path, args.model)


if __name__ == "__main__":
    main()
