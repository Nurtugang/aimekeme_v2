"""Офлайн-прототип тепловой карты посещаемости. НЕ часть сервиса.

Идея: та же детекция людей, что и в app/counting/ (backend frcnn/yolo_head,
боксы без доп. постобработки), но вместо счётчика — накопление точек "ног"
человека (низ бокса, а не центр — так точки ложатся на пол/проходы, а не
в грудь) в 2D-аккумулятор. Каждая точка "капает" гауссово пятно; между
кадрами аккумулятор слегка затухает (decay), иначе через пару минут видео
всё станет одного цвета и карта потеряет смысл. Аккумулятор красится
colormap'ом (теплее = чаще бывают люди) и блендится поверх кадра.

Запуск (из корня репозитория):
    python scripts/heatmap_infer.py <путь_к_видео> [--model frcnn|yolo_head]
                                     [--variant medium|nano] [--conf 0.5]
                                     [--decay 0.98] [--radius 40] [--alpha 0.55]

Видео -> <имя>_heatmap.mp4 рядом с исходником (карта нарастает по ходу видео,
как в live-режиме) + <имя>_heatmap.jpg — финальная накопленная карта поверх
последнего кадра.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Корень репозитория в путь, чтобы работал `import app.*` при запуске из scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                          # noqa: E402
from app.counting.model_frcnn import FrcnnCounter        # noqa: E402
from app.counting.model_yolo_head import YoloHeadCounter  # noqa: E402

_BACKENDS = {"frcnn": FrcnnCounter, "yolo_head": YoloHeadCounter}
_COLORMAP = cv2.COLORMAP_JET


class HeatAccumulator:
    """Плотность посещений по кадру: гауссовы пятна в точках ног + decay."""

    def __init__(self, height: int, width: int, radius: int, decay: float):
        self._heat = np.zeros((height, width), dtype=np.float32)
        self._radius = radius
        self._decay = decay
        ksize = radius * 2 + 1
        gauss_1d = cv2.getGaussianKernel(ksize, radius / 2.0)
        self._kernel = (gauss_1d @ gauss_1d.T).astype(np.float32)
        self._kernel /= self._kernel.max()  # пик пятна = 1.0

    def add_points(self, points_xy: list[tuple[int, int]]) -> None:
        h, w = self._heat.shape
        r = self._radius
        for x, y in points_xy:
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            kx0, ky0 = x0 - (x - r), y0 - (y - r)
            patch = self._kernel[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)]
            self._heat[y0:y1, x0:x1] += patch

    def decay_step(self) -> None:
        self._heat *= self._decay

    def render_overlay(self, frame_bgr: np.ndarray, alpha: float) -> np.ndarray:
        peak = self._heat.max()
        if peak < 1e-6:
            return frame_bgr.copy()
        normalized = np.clip(self._heat / peak * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, _COLORMAP)
        # Тихие зоны (почти нулевой heat) не перекрашиваем поверх кадра.
        mask = (normalized > 8).astype(np.float32)[..., None]
        blended = frame_bgr.astype(np.float32) * (1 - alpha * mask) + colored.astype(np.float32) * (alpha * mask)
        return blended.astype(np.uint8)


def _feet_points(boxes: np.ndarray) -> list[tuple[int, int]]:
    """Боксы xyxy -> точка "ног" каждого человека (низ бокса, по центру X).

    Корректно кладёт точку на пол только для боксов всего тела (`frcnn`).
    У `yolo_head` бокс — это голова, и низ бокса окажется на подбородке, а не
    на полу: карта сдвинется вверх на рост человека. Для позиционной карты
    (как на референсе) используйте `--model frcnn`; `yolo_head` тут скорее
    "карта присутствия по головам", чем карта проходимости пола.
    """
    points = []
    for x1, y1, x2, y2 in boxes:
        points.append((int((x1 + x2) / 2), int(y2)))
    return points


def _report(n_frames: int, infer_total_s: float, wall_total_s: float) -> None:
    print("\n--- Тайминги ---")
    print(f"Кадров обработано:      {n_frames}")
    print(f"Инференс суммарно:      {infer_total_s:.3f} c")
    if n_frames:
        print(f"Инференс среднее/кадр:   {infer_total_s / n_frames * 1000.0:.1f} мс")
    print(f"Всего (wall, с I/O):    {wall_total_s:.3f} c")


def run_video(backend, path: Path, radius: int, decay: float, alpha: float) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не могу открыть видео: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = path.with_name(f"{path.stem}_heatmap.mp4")
    writer = cv2.VideoWriter(
        str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    heat = HeatAccumulator(height, width, radius, decay)

    print(f"\nОбрабатываю: {path}  ({total} кадров, {fps:.1f} fps)  "
          f"radius={radius} decay={decay} alpha={alpha}\n")
    idx = 0
    infer_total = 0.0
    last_frame = None
    wall0 = time.perf_counter()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        last_frame = frame

        t0 = time.perf_counter()
        boxes, _ = backend.predict(frame)
        infer_total += (time.perf_counter() - t0) * 1000.0

        heat.decay_step()
        heat.add_points(_feet_points(boxes))
        writer.write(heat.render_overlay(frame, alpha))
        print(f"  Кадр {idx:5d}/{total}  |  людей: {len(boxes)}")

    cap.release()
    writer.release()

    if last_frame is not None:
        out_image_path = path.with_name(f"{path.stem}_heatmap.jpg")
        cv2.imwrite(str(out_image_path), heat.render_overlay(last_frame, alpha))
        print(f"\nГотово! Сохранено: {out_video_path}")
        print(f"Финальная карта:    {out_image_path}")

    _report(idx, infer_total / 1000.0, time.perf_counter() - wall0)


def main():
    parser = argparse.ArgumentParser(description="Офлайн-прототип тепловой карты посещаемости.")
    parser.add_argument("path", help="путь к видео")
    parser.add_argument("--model", choices=list(_BACKENDS), default="frcnn",
                        help="модель детекции людей (по умолчанию frcnn)")
    parser.add_argument("--variant", choices=["medium", "nano"], default=None,
                        help="для yolo_head: вариант весов (по умолчанию из config)")
    parser.add_argument("--conf", type=float, default=None,
                        help="порог score/conf (по умолчанию из config)")
    parser.add_argument("--radius", type=int, default=40,
                        help="радиус гауссова пятна на точку, px (по умолчанию 40)")
    parser.add_argument("--decay", type=float, default=0.98,
                        help="затухание аккумулятора за кадр, 0..1 (по умолчанию 0.98)")
    parser.add_argument("--alpha", type=float, default=0.55,
                        help="непрозрачность оверлея в горячих зонах, 0..1 (по умолчанию 0.55)")
    args = parser.parse_args()

    if args.variant is not None:
        settings.count_head_variant = args.variant
    if args.conf is not None:
        settings.count_score_thresh = args.conf

    path = Path(args.path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Загружаю модель '{args.model}' на {device} ...")
    backend = _BACKENDS[args.model](device, settings)
    backend.load()

    run_video(backend, path, args.radius, args.decay, args.alpha)


if __name__ == "__main__":
    main()
