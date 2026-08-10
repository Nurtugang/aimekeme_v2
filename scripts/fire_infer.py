"""Офлайн-инференс детектора огня/дыма. НЕ часть сервиса.

Гоняет ту же модель, что и API (app/fire/), по видео или фото и рисует
результат: метку (fire/smoke/normal), уверенность по всем классам и боксы,
если бэкенд их даёт.

Запуск (из корня репозитория):
    python scripts/fire_infer.py <путь> <тип> [--model ...] [--crop] [--grid]
        [--threshold 0.5]

    <тип>        video | photo (frame/image — синонимы photo)
    --model      siglip2 | yolo_dfire | rtdetr (по умолчанию — из настроек, FIRE_MODEL)
    --crop       включить тайлинг 2x2 + кадр целиком (мелкие очаги)
    --no-crop    выключить тайлинг, даже если он включён в настройках
    --grid       нарисовать сетку тайлинга
    --threshold  порог решения; по умолчанию — FIRE_THRESHOLD из настроек

Примеры:
    python scripts/fire_infer.py test/fireplace.mp4 video --model yolo_dfire --crop
    python scripts/fire_infer.py photo.jpg photo --model siglip2 --no-crop
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings          # noqa: E402
from app.fire.detector import FireDetector  # noqa: E402

_NORMAL = "normal"
# Цвет метки (BGR): огонь — красный, дым — оранжевый, обычный кадр — зелёный.
_COLORS = {"fire": (0, 0, 220), "smoke": (0, 140, 230), _NORMAL: (0, 180, 0)}


def analyze(detector, frame_bgr):
    """Один BGR-кадр -> (итоговая метка, скоры по классам, боксы)."""
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    scores, boxes = detector.analyze(pil)
    label, _ = detector.decide(scores)
    return label, scores, boxes


def draw_overlay(frame, label, scores, boxes, threshold, draw_grid=False):
    """Инфо-блок в левом верхнем углу + боксы + (опц.) сетка тайлинга."""
    h, w = frame.shape[:2]
    color = _COLORS.get(label, (200, 200, 200))

    if draw_grid:
        cv2.line(frame, (w // 2, 0), (w // 2, h), (100, 100, 100), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (100, 100, 100), 1)

    # Боксы выше порога — только они влияют на решение.
    for x1, y1, x2, y2, box_label, conf in boxes:
        if conf < threshold:
            continue
        box_color = _COLORS.get(box_label, (200, 200, 200))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, box_color, 2)
        cv2.putText(frame, f"{box_label} {conf:.2f}", (p1[0], max(18, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (370, 115), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (10, 10), (370, 115), color, 2)

    cv2.putText(frame, f"STATUS: {label.upper()}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    detail = (f"Fire: {int(scores['fire'] * 100)}% | "
              f"Smoke: {int(scores['smoke'] * 100)}% | "
              f"Norm: {int(scores[_NORMAL] * 100)}%")
    cv2.putText(frame, detail, (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # Шкала преобладающего тревожного класса.
    hazard = "fire" if scores["fire"] >= scores["smoke"] else "smoke"
    bar_w = int(335 * scores[hazard])
    cv2.rectangle(frame, (20, 80), (355, 95), (50, 50, 50), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (20, 80), (20 + bar_w, 95), _COLORS[hazard], -1)
    cv2.rectangle(frame, (20, 80), (355, 95), (150, 150, 150), 1)

    return frame


def run_photo(detector, path: Path, suffix: str, threshold: float, draw_grid: bool):
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Не могу открыть фото: {path}")

    label, scores, boxes = analyze(detector, frame)
    out_path = path.with_name(f"{path.stem}{suffix}.jpg")
    cv2.imwrite(str(out_path), draw_overlay(
        frame, label, scores, boxes, threshold, draw_grid))

    print("\n--- Результат обработки фото ---")
    print(f"Итоговая метка:   {label.upper()}")
    print(f"Вероятности:      Fire={scores['fire']:.3f} | "
          f"Smoke={scores['smoke']:.3f} | Normal={scores[_NORMAL]:.3f}")
    print(f"Боксов выше порога: {sum(1 for b in boxes if b[5] >= threshold)}")
    print(f"Сохранено:        {out_path}\n")


def run_video(detector, path: Path, suffix: str, threshold: float, draw_grid: bool):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не могу открыть видео: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = path.with_name(f"{path.stem}{suffix}.mp4")
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    print(f"\nОбрабатываю видео: {path} ({total} кадров, {fps:.1f} fps)\n")
    idx, alarms = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        label, scores, boxes = analyze(detector, frame)
        writer.write(draw_overlay(
            frame, label, scores, boxes, threshold, draw_grid))

        if label != _NORMAL:
            alarms += 1
            print(f"  Кадр {idx:5d}/{total} | STATUS: {label.upper():6s} | "
                  f"P(fire)={scores['fire']:.3f} | P(smoke)={scores['smoke']:.3f} | "
                  f"P(normal)={scores[_NORMAL]:.3f}")

    cap.release()
    writer.release()
    print(f"\nГотово! Тревожных кадров: {alarms}/{idx}. Сохранено: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Офлайн-инференс детекции огня/дыма.")
    parser.add_argument("path", help="путь к видео или фото")
    parser.add_argument("type", choices=["video", "photo", "frame", "image"],
                        help="тип входа: video или photo")
    parser.add_argument("--model", choices=["siglip2", "yolo_dfire", "rtdetr"],
                        default=settings.fire_model,
                        help="какой бэкенд гонять (по умолчанию из FIRE_MODEL)")
    crop = parser.add_mutually_exclusive_group()
    crop.add_argument("--crop", dest="crop", action="store_true", default=None,
                      help="включить тайлинг 2x2 + кадр целиком")
    crop.add_argument("--no-crop", dest="crop", action="store_false",
                      help="выключить тайлинг")
    parser.add_argument("--grid", action="store_true",
                        help="нарисовать сетку тайлинга")
    parser.add_argument("--threshold", type=float, default=settings.fire_threshold,
                        help="порог решения (по умолчанию из FIRE_THRESHOLD)")
    args = parser.parse_args()

    # Скрипт гоняет ровно тот же детектор, что и API, — просто с настройками
    # из аргументов, чтобы сравнивать модели/режимы без правки .env.
    run_settings = settings.model_copy(update={
        "fire_model": args.model,
        "fire_threshold": args.threshold,
        "use_tiling": settings.use_tiling if args.crop is None else args.crop,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Загружаю бэкенд '{run_settings.fire_model}' на {device} "
          f"(тайлинг={run_settings.use_tiling}, порог={run_settings.fire_threshold}) ...")
    detector = FireDetector(run_settings, device)
    detector.load()

    suffix = f"_fire_{run_settings.fire_model}"
    if run_settings.use_tiling:
        suffix += "_tiled"

    path = Path(args.path)
    if args.type == "video":
        run_video(detector, path, suffix, run_settings.fire_threshold, args.grid)
    else:
        run_photo(detector, path, suffix, run_settings.fire_threshold, args.grid)


if __name__ == "__main__":
    main()
