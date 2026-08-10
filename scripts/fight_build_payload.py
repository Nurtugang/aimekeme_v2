"""Нарезает видео на окна по 16 кадров и для каждого окна сохраняет два файла:
  - payload.json  — готовое тело запроса к POST /detect/fight (16 base64-JPEG);
  - clip.mp4      — эти же 16 кадров как видео, чтобы глазами увидеть момент.

Структура вывода (папка названа по имени видео):
  <video_stem>/
    window_0000_f000000-000015/
        payload.json
        clip.mp4
    window_0001_f000016-000031/
        payload.json
        clip.mp4
    ...

Инструмент для разработчика/тестов; сам сервис его не использует.

Пример:
    python scripts/build_payload.py path/to/video.mp4
    python scripts/build_payload.py video.mp4 -o out --stride 8   # с перекрытием
    curl -X POST http://localhost:8000/detect/fight \\
         -H "Content-Type: application/json" \\
         -d @video/window_0000_f000000-000015/payload.json
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

WINDOW = 16  # модель X3D-M принимает ровно столько кадров


def encode_payload(frames: list[np.ndarray]) -> dict:
    """16 BGR-кадров -> {"frames": [<base64_jpg>, ...]}."""
    encoded = []
    for frame in frames:
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            sys.exit("JPEG encoding failed.")
        encoded.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    return {"frames": encoded}


def write_clip(path: Path, frames: list[np.ndarray], fps: float) -> None:
    """Сохраняет кадры окна как mp4 (для визуальной проверки момента)."""
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def read_all_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video", help="Path to a video file.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Base output dir (default: next to the source video).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=WINDOW,
        help=f"Step between windows in frames (default {WINDOW} = non-overlapping).",
    )
    args = parser.parse_args()

    if args.stride < 1:
        sys.exit("--stride must be >= 1")

    video = Path(args.video)
    frames, fps = read_all_frames(video)
    if len(frames) < WINDOW:
        sys.exit(f"Video has only {len(frames)} frames (< {WINDOW}).")

    # По умолчанию складываем результат рядом с исходным видео.
    output_base = Path(args.output) if args.output else video.parent
    base_dir = output_base / video.stem
    base_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for start in range(0, len(frames) - WINDOW + 1, args.stride):
        window = frames[start : start + WINDOW]
        end = start + WINDOW - 1
        sub = base_dir / f"window_{count:04d}_f{start:06d}-{end:06d}"
        sub.mkdir(exist_ok=True)

        with open(sub / "payload.json", "w", encoding="utf-8") as f:
            json.dump(encode_payload(window), f)
        write_clip(sub / "clip.mp4", window, fps)

        count += 1

    print(f"{count} windows ({WINDOW} frames each, stride={args.stride}) -> {base_dir}/")


if __name__ == "__main__":
    main()
