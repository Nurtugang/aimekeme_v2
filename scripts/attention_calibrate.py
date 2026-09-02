"""Калибровка направления взгляда под свою камеру. НЕ часть сервиса.

Зачем. «Вовлечён» = смотрит в сторону доски. Какие это углы — зависит от того,
где висит камера относительно доски. Ставить их на глаз нельзя: при камере
сбоку вся аудитория окажется distracted. Здесь углы ЗАМЕРЯЮТСЯ, а скрипт
печатает готовые строки для .env.

Заодно проверяются знаки: вы смотрите влево и вправо, а скрипт говорит, растёт
ли yaw в ту сторону, в которую вы повернулись. Если знак перевёрнут, это видно
сразу, а не через неделю по странной статистике.

Запуск (из корня репозитория):

    python scripts/attention_calibrate.py
    python scripts/attention_calibrate.py --camera 1

        Смотрите на доску (как внимательный слушатель)  -> `c`
        Посмотрите на дальний ЛЕВЫЙ край аудитории      -> `l`
        Посмотрите на дальний ПРАВЫЙ край              -> `r`
        Выход и печать настроек                        -> `q`
        Каждую позу можно снимать несколько раз — берётся среднее.

Что считается:
    ATTENTION_YAW_CENTER / ATTENTION_PITCH_CENTER — средний взгляд «на доску»;
    ATTENTION_YAW_TOLERANCE — половина разброса между краями аудитории, чтобы
        крайние ряды не считались отвернувшимися.

Позы «влево/вправо» можно не снимать — тогда печатаются только центры,
а допуски останутся дефолтными.
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.attention.detector import AttentionDetector  # noqa: E402
from app.config import settings                       # noqa: E402

_COLOR = (0, 200, 255)
_KEYS = {ord("c"): "center", ord("l"): "left", ord("r"): "right"}


def mean_of(samples):
    n = len(samples)
    return sum(y for y, _ in samples) / n, sum(p for _, p in samples) / n


def angles_in(people):
    """Люди -> [(yaw, pitch), ...] только по тем, у кого видно лицо."""
    return [(p["gaze_yaw"], p["gaze_pitch"]) for p in people if p["gaze_yaw"] is not None]


def draw(frame, people, counts):
    """Углы крупно у каждого лица + счётчики снятых поз."""
    for p in people:
        x1, y1, x2, y2 = (int(v) for v in p["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), _COLOR, 2)
        if p["gaze_yaw"] is None:
            cv2.putText(frame, "лицо не видно", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 2, cv2.LINE_AA)
            continue
        cv2.putText(frame, f"yaw {p['gaze_yaw']:+.1f}", (x1, max(20, y1 - 32)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, f"pitch {p['gaze_pitch']:+.1f}", (x1, max(40, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR, 2, cv2.LINE_AA)

    hint = ("[c] на доску: {center}   [l] влево: {left}   "
            "[r] вправо: {right}   [q] выход").format(**{k: len(v) for k, v in counts.items()})
    cv2.putText(frame, hint, (15, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def report(samples):
    """Снятые позы -> строки для .env + проверка знаков."""
    center = samples["center"]
    if not center:
        print("\nПоза «на доску» не снята — калибровать нечего.")
        return

    yaw_c, pitch_c = mean_of(center)
    print("\n--- Замеры ---")
    print(f"На доску: образцов {len(center)}, yaw={yaw_c:+.2f}, pitch={pitch_c:+.2f}")

    lines = [f"ATTENTION_YAW_CENTER={yaw_c:.1f}",
             f"ATTENTION_PITCH_CENTER={pitch_c:.1f}"]

    left, right = samples["left"], samples["right"]
    if left and right:
        yaw_l, _ = mean_of(left)
        yaw_r, _ = mean_of(right)
        print(f"Влево:    образцов {len(left)}, yaw={yaw_l:+.2f}")
        print(f"Вправо:   образцов {len(right)}, yaw={yaw_r:+.2f}")

        # Проверка знака: поворот влево и вправо обязан разводить yaw в РАЗНЫЕ
        # стороны от центра. Если нет — модель видит не то, что мы думаем.
        if (yaw_l - yaw_c) * (yaw_r - yaw_c) >= 0:
            print("\nВНИМАНИЕ: влево и вправо дали yaw по одну сторону от центра.")
            print("Либо позы сняты неверно, либо лицо не отслеживается — переснимите.")
        else:
            half = (abs(yaw_l - yaw_c) + abs(yaw_r - yaw_c)) / 2.0
            print(f"\nЗнаки в порядке: разворот на края даёт +-{half:.1f} градусов.")
            # Берём чуть больше половины: крайний ряд должен оставаться вовлечённым.
            lines.append(f"ATTENTION_YAW_TOLERANCE={max(15.0, half * 0.75):.1f}")
    else:
        print("\nКрая аудитории не сняты: допуск по yaw останется дефолтным,")
        print("а знак угла не проверен.")

    print("\n--- Впишите в .env ---")
    for line in lines:
        print(line)
    print()


def run_camera(detector, index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Не могу открыть камеру с индексом {index}")

    samples = {"center": [], "left": [], "right": []}
    print("\nОкно открыто. c — на доску, l — влево, r — вправо, q — выход.\n")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Кадр не прочитан — камера отключилась?")
                break

            people = detector.analyze(frame, "calibrate")
            cv2.imshow("attention calibrate", draw(frame, people, samples))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in _KEYS:
                found = angles_in(people)
                if not found:
                    print("  Лица в кадре не видно — образец не записан.")
                    continue
                name = _KEYS[key]
                samples[name].extend(found)
                yaw_m, pitch_m = mean_of(found)
                print(f"  [{name}] +{len(found)}: yaw={yaw_m:+.2f}, pitch={pitch_m:+.2f}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    report(samples)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="индекс камеры (по умолчанию 0)")
    parser.add_argument("--pose-variant", choices=list("nsmlx"), default="s",
                        help="для калибровки хватает мелкой модели (по умолчанию s)")
    parser.add_argument("--pose-imgsz", type=int, default=640,
                        help="вход детектора поз; для вебкамеры 640 достаточно")
    args = parser.parse_args()

    # Для калибровки предметы не нужны, а окно ставим минимальным: нас
    # интересуют мгновенные углы, а не вовлечённость за полминуты.
    run_settings = settings.model_copy(update={
        "pose_variant": args.pose_variant,
        "pose_imgsz": args.pose_imgsz,
        "object_enabled": False,
        "track_min_hits": 1,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Загружаю модели на {device} ...")
    detector = AttentionDetector(run_settings, device)
    detector.load()
    run_camera(detector, args.camera)


if __name__ == "__main__":
    main()
