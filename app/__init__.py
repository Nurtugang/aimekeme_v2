"""Violence Detection API package.

Exposing ``app`` here lets the service be launched either as
``uvicorn app:app`` (per the spec) or ``uvicorn app.main:app``.
"""

from app.main import app

__all__ = ["app"]
__version__ = "0.1.0"
