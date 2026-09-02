"""Общие фикстуры тестов.

Тесты сознательно НЕ трогают модели: проверяется логика (геометрия, трекинг,
временное окно), которая целиком лежит в чистых модулях и не требует ни GPU,
ни весов. Именно в ней живут пороги и правила, которые легко сломать правкой.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def cfg():
    """Настройки по умолчанию — те же значения, что в app/config.py."""
    from app.config import Settings
    return Settings()


@pytest.fixture
def tune():
    """Собрать настройки с изменёнными полями."""
    from app.config import Settings

    def _tune(**over):
        return Settings().model_copy(update=over)
    return _tune
