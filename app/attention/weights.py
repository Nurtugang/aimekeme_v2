"""Кеш весов YOLO: не даём ultralytics ронять .pt в рабочую директорию.

`YOLO("yolo11x.pt")` качает файл в ТЕКУЩУЮ директорию — то есть в корень
репозитория, откуда запускают сервис. Здесь скачанный файл переносится в
~/.cache/aimekeme/, как это уже делает app/counting/model_yolo_head.py.

Именно перенос, а не model.save(): save() пересериализует чекпоинт, а перенос
сохраняет ровно те байты, что отдал релиз.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger("surveillance.attention.weights")

CACHE_ROOT = Path.home() / ".cache" / "aimekeme"


def cached_yolo(name: str, subdir: str):
    """Загружает YOLO по имени веса, держа сам файл в кеше."""
    from ultralytics import YOLO

    cache_dir = CACHE_ROOT / subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / name

    if target.exists():
        return YOLO(str(target))

    model = YOLO(name)          # качает в текущую директорию
    downloaded = Path(name)
    if downloaded.is_file():
        try:
            # shutil.move, а не Path.replace: репозиторий и кеш нередко лежат
            # на разных дисках, а os.replace через границу тома не работает
            # (на Windows это WinError 17).
            shutil.move(str(downloaded), str(target))
            logger.info("Веса перенесены в кеш: %s", target)
        except OSError as exc:  # права, занятый файл — не критично
            logger.warning("Не удалось перенести %s в %s: %s", downloaded, target, exc)
    return model
