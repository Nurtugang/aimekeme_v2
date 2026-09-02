"""Violence Detection API package.

Exposing ``app`` here lets the service be launched either as
``uvicorn app:app`` (per the spec) or ``uvicorn app.main:app``.

Импорт ЛЕНИВЫЙ (PEP 562). Раньше здесь был обычный ``from app.main import app``,
из-за чего ЛЮБОЙ ``import app.<модуль>`` тянул app.main, а с ним fastapi и все
пять детекторов сразу. Офлайн-скриптам из scripts/ нужен ровно один модуль,
поэтому app.main грузится только когда реально просят атрибут ``app``.
"""

__all__ = ["app"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "app":
        from app.main import app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
