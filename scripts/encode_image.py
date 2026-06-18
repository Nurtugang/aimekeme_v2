"""Кодирует одно изображение в тело запроса для POST /detect/face.

JSON сохраняется РЯДОМ с фото и с тем же именем (меняется только расширение):
    known_faces/nurtugan.jpg  ->  known_faces/nurtugan.json   {"frame": "<base64>"}

Инструмент для разработчика/тестов.

Пример:
    python scripts/encode_image.py known_faces/nurtugan.jpg
    curl -X POST http://localhost:8000/detect/face \\
         -H "Content-Type: application/json" \\
         -d @known_faces/nurtugan.json
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", help="Path to an image (jpg/png).")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        sys.exit(f"File not found: {image_path}")

    # Проверяем, что это вообще декодируемое изображение (как это сделает API).
    if cv2.imread(str(image_path)) is None:
        sys.exit(f"Not a decodable image: {image_path}")

    # Кодируем исходные байты файла как есть (jpg/png и так читаются cv2.imdecode).
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")

    out_path = image_path.with_suffix(".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"frame": encoded}, f)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
