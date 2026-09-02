"""Проверка живого эндпоинта POST /detect/attention. НЕ часть сервиса.

В отличие от attention_infer.py (тот грузит модели себе в процесс), этот скрипт
бьёт по УЖЕ ЗАПУЩЕННОМУ сервису по HTTP: проверяется реальный контракт —
кодирование кадра, схема ответа, коды ошибок, задержка под сетью.

Заодно показывает, как должен опрашивать брокер: кадры идут серией с паузой,
на один и тот же camera_id, потому что треки и временные окна сервис ведёт
на камеру. Если слать каждый кадр с новым camera_id, окно никогда не наберётся
и все останутся warming_up.

Запуск (сервис уже поднят: uvicorn app:app):
    python scripts/attention_request.py 0 --frames 60 --interval 0.5
    python scripts/attention_request.py lecture_row3.jpg --json
    python scripts/attention_request.py 0 --url http://10.0.0.5:8000
"""

import argparse
import base64
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import httpx


def encode(frame) -> str:
    """BGR-кадр -> base64-JPEG, как его шлёт брокер."""
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Не удалось закодировать кадр в JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def post_frame(client: httpx.Client, url: str, frame, camera_id: str) -> dict:
    """Один кадр -> распарсенный ответ сервиса. Ошибки показываем как есть."""
    response = client.post(url, json={"frame": encode(frame), "camera_id": camera_id})
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json()


def print_result(idx: int, result: dict) -> None:
    rate = result["engagement_rate"]
    rate_str = "n/a" if rate is None else f"{rate:.3f}"
    print(f"  [{idx:3d}] людей={result['count']:3d}  вовлечены={result['engaged_count']:3d}  "
          f"доля={rate_str:>5}  server={result['processing_ms']:6.1f} мс")
    for p in result["people"]:
        warm = "  (набор окна)" if p["warming_up"] else ""
        held = f"  предметы={'+'.join(p['held_objects'])}" if p["held_objects"] else ""
        print(f"          #{p['track_id']:<3} {p['state']:<11} "
              f"вовлечённость={p['engagement']:.2f}  "
              f"взгляд {p['gaze_hold_s']:.1f}с{held}{warm}")


def summarize(rates, wall_ms) -> None:
    measured = [r for r in rates if r is not None]
    print()
    print("--- Сводка ---")
    print(f"Запросов:                 {len(rates)}")
    if measured:
        print(f"Кадров с людьми:          {len(measured)}")
        print(f"Медиана доли вовлечённых: {statistics.median(measured):.3f}")
        print(f"Мин / макс:               {min(measured):.3f} / {max(measured):.3f}")
    else:
        print("Людей не найдено ни на одном кадре — медиану считать не из чего.")
    if wall_ms:
        print(f"Задержка round-trip:      среднее {statistics.mean(wall_ms):.1f} мс, "
              f"макс {max(wall_ms):.1f} мс")


def frames_from_camera(index: int, count: int, interval: float):
    """Генератор кадров с камеры с паузой между ними (как опрос брокера)."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Не могу открыть камеру с индексом {index}")
    try:
        for i in range(count):
            if i:
                time.sleep(interval)
            ret, frame = cap.read()
            if not ret:
                print("Кадр не прочитан — камера отключилась?")
                return
            yield frame
    finally:
        cap.release()


def frames_from_photo(path: Path):
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Не могу открыть фото: {path}")
    yield frame


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="индекс камеры (0, 1, ...) или путь к фото")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="базовый адрес сервиса (по умолчанию http://localhost:8000)")
    parser.add_argument("--frames", type=int, default=60,
                        help="сколько кадров снять с камеры (по умолчанию 60)")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="пауза между кадрами, секунды (по умолчанию 0.5)")
    parser.add_argument("--camera-id", default="probe",
                        help="id камеры: на него сервис ведёт треки и окна")
    parser.add_argument("--json", action="store_true",
                        help="печатать сырой JSON ответа целиком")
    args = parser.parse_args()

    endpoint = f"{args.url.rstrip('/')}/detect/attention"
    source = (frames_from_camera(int(args.source), args.frames, args.interval)
              if args.source.isdigit() else frames_from_photo(Path(args.source)))

    print()
    print(f"Шлю кадры на {endpoint} (camera_id={args.camera_id})")
    print()
    rates = []
    wall_ms = []

    with httpx.Client(timeout=60.0) as client:
        for idx, frame in enumerate(source, 1):
            t0 = time.perf_counter()
            try:
                result = post_frame(client, endpoint, frame, args.camera_id)
            except httpx.ConnectError:
                sys.exit(f"Сервис не отвечает на {args.url}. Он запущен? "
                         f"(uvicorn app:app --host 0.0.0.0 --port 8000)")
            wall_ms.append((time.perf_counter() - t0) * 1000.0)

            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print_result(idx, result)
            rates.append(result["engagement_rate"])

    summarize(rates, wall_ms)


if __name__ == "__main__":
    main()
