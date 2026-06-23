"""Распознавание лиц: ArcFace (buffalo_l, insightface) + мини-база с N эталонами.

База лиц — в папке known_faces/:
- faces.db   — SQLite с метаданными: faces(id, name, created_at). ID автоинкрементны
               и не переиспользуются (на них ссылается журнал событий платформы);
- <id>_<k>.jpg — фото человека, по одному файлу на эталон (k = 1..N);
- эмбеддинги — в памяти, считаются на старте из <id>_<k>.jpg.

Поддержка нескольких эталонов на человека: при матче берём МАКСИМУМ косинусной
близости по всем эталонам человека (не среднее). ArcFace разводит разных людей
сильно (~0.0-0.3), своего узнаёт высоко (~0.5-0.9), поэтому порог 0.42 надёжен.

CRUD по базе идёт через API (router.py) и обновляет её вживую, без рестарта.
predict на 1 кадр: insightface находит лица -> эмбеддинг каждого -> сравнение с базой.
"""

from __future__ import annotations

import base64
import binascii
import logging
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.face.model import load_face_model
from app.config import Settings

logger = logging.getLogger("surveillance.face")

_DATA_URI_MARKER = "base64,"
_DB_FILE = "faces.db"
_UNKNOWN = "unknown"


class InvalidImageError(ValueError):
    """Кадр/файл не удалось декодировать как изображение."""


class EnrollmentError(ValueError):
    """Фото не годятся для записи (ни на одном — ровно одного чёткого лица)."""


class DuplicateNameError(ValueError):
    """Человек с таким именем уже есть в базе."""


class FaceNotFoundError(ValueError):
    """Лицо с таким id не найдено в базе."""


class FaceDetector:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._app = None
        self._device_str = "cpu"
        # хот-патч распознавания в памяти; БД + jpg — персистентность:
        self._gallery: dict[int, np.ndarray] = {}  # id -> (k, 512) normed embeddings
        self._names: dict[int, str] = {}           # id -> имя (для ответа)
        self._lock = threading.Lock()

    # --- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        providers, ctx_id = self._resolve_device()
        logger.info("Loading face model (buffalo_l), ctx_id=%d ...", ctx_id)
        self._app = load_face_model(
            providers, ctx_id, self._settings.face_det_size, self._settings.face_det_thresh)
        # фактически активный провайдер (CUDA мог тихо откатиться на CPU)
        rec = self._app.models.get("recognition")
        active = rec.session.get_providers() if rec else []
        self._device_str = "cuda" if "CUDAExecutionProvider" in active else "cpu"
        self._load_gallery()
        logger.info("Face detector ready on %s. Known faces: %d %s",
                    self._device_str, len(self._names), list(self._names.values()))

    def _resolve_device(self) -> tuple[list[str], int]:
        dev = self._settings.device
        if dev == "auto":
            import torch
            use_cuda = torch.cuda.is_available()
            ctx_id = 0
        elif dev.startswith("cuda"):
            use_cuda, ctx_id = True, (int(dev.split(":")[1]) if ":" in dev else 0)
        else:
            use_cuda, ctx_id = False, -1
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if use_cuda else ["CPUExecutionProvider"])
        return providers, ctx_id

    @property
    def is_ready(self) -> bool:
        return self._app is not None

    @property
    def device(self) -> str:
        return self._device_str

    # --- storage (SQLite + <id>_<k>.jpg) ----------------------------------

    def _connect(self) -> sqlite3.Connection:
        path = Path(self._settings.known_faces_dir)
        path.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path / _DB_FILE)
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE IF NOT EXISTS faces ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "name TEXT NOT NULL UNIQUE, "
            "created_at TEXT NOT NULL)"
        )
        return conn

    def _image_paths(self, face_id: int) -> list[Path]:
        """Все фото-эталоны человека: <id>_<k>.jpg, отсортированы по k."""
        directory = Path(self._settings.known_faces_dir)
        return sorted(directory.glob(f"{face_id}_*.jpg"),
                      key=lambda p: int(p.stem.split("_")[1]))

    def _next_image_path(self, face_id: int) -> Path:
        n = len(self._image_paths(face_id)) + 1
        return Path(self._settings.known_faces_dir) / f"{face_id}_{n}.jpg"

    # --- gallery ----------------------------------------------------------

    def _load_gallery(self) -> None:
        self._gallery = {}
        self._names = {}
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT id, name FROM faces ORDER BY id").fetchall()

        for row in rows:
            face_id, name = row["id"], row["name"]
            self._names[face_id] = name
            embs = []
            for path in self._image_paths(face_id):
                bgr = cv2.imread(str(path))
                if bgr is None:
                    logger.warning("Cannot read %s — пропускаю эталон", path)
                    continue
                emb = self._embed_single(bgr)
                if emb is None:
                    logger.warning("No single face in %s — пропускаю эталон", path)
                    continue
                embs.append(emb)
            if embs:
                self._gallery[face_id] = np.stack(embs)
            else:
                logger.warning("Face id=%d (%s): ни одного валидного эталона", face_id, name)

    def _get_faces(self, bgr: np.ndarray) -> list:
        """Все лица на кадре (с фильтром по det_thresh), под локом GPU."""
        with self._lock:
            faces = self._app.get(bgr)
        thr = self._settings.face_det_thresh
        return [f for f in faces if float(f.det_score) >= thr]

    def _embed_single(self, bgr: np.ndarray) -> np.ndarray | None:
        """Ровно одно лицо на картинке -> normed_embedding (512,) float32, иначе None."""
        faces = self._get_faces(bgr)
        if len(faces) != 1:
            return None
        return faces[0].normed_embedding.astype(np.float32)

    # --- CRUD по базе (через API) -----------------------------------------

    def add_face(self, name: str, images: list[bytes]) -> dict:
        """Создаёт человека из 1..N фото. Сохраняет только годные кадры (ровно одно лицо)."""
        name = name.strip()
        if not name:
            raise EnrollmentError("name is required")

        prepared = self._prepare_embeddings(images)  # [(pil, emb), ...]
        if not prepared:
            raise EnrollmentError("no usable face found (need exactly one clear face per photo)")

        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                with closing(self._connect()) as conn:
                    cur = conn.execute(
                        "INSERT INTO faces(name, created_at) VALUES(?, ?)", (name, created_at))
                    conn.commit()
                    face_id = cur.lastrowid
            except sqlite3.IntegrityError as exc:
                raise DuplicateNameError(f"name '{name}' already exists") from exc

            embs = self._persist(face_id, prepared)
            self._names[face_id] = name
            self._gallery[face_id] = embs

        logger.info("Enrolled face id=%d name=%s with %d photo(s)", face_id, name, len(embs))
        return {"id": face_id, "name": name, "created_at": created_at, "photos": len(embs)}

    def add_images(self, face_id: int, images: list[bytes]) -> dict:
        """Догружает фото существующему человеку."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT name, created_at FROM faces WHERE id = ?", (face_id,)).fetchone()
        if row is None:
            raise FaceNotFoundError(f"face id={face_id} not found")

        prepared = self._prepare_embeddings(images)
        if not prepared:
            raise EnrollmentError("no usable face found (need exactly one clear face per photo)")

        with self._lock:
            new_embs = self._persist(face_id, prepared)
            existing = self._gallery.get(face_id)
            self._gallery[face_id] = (
                new_embs if existing is None else np.concatenate([existing, new_embs]))
            total = len(self._gallery[face_id])

        logger.info("Added %d photo(s) to face id=%d (total %d)", len(new_embs), face_id, total)
        return {"id": face_id, "name": row["name"],
                "created_at": row["created_at"], "photos": total}

    def delete_face(self, face_id: int) -> None:
        with self._lock:
            with closing(self._connect()) as conn:
                cur = conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))
                conn.commit()
            if cur.rowcount == 0:
                raise FaceNotFoundError(f"face id={face_id} not found")

            name = self._names.pop(face_id, None)
            self._gallery.pop(face_id, None)
            for path in self._image_paths(face_id):
                path.unlink(missing_ok=True)

        logger.info("Deleted face id=%d name=%s", face_id, name)

    def list_faces(self) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, name, created_at FROM faces ORDER BY id").fetchall()
        return [{**dict(r), "photos": len(self._image_paths(r["id"]))} for r in rows]

    def get_face_image(self, face_id: int) -> Path:
        paths = self._image_paths(face_id)
        if not paths:
            raise FaceNotFoundError(f"face id={face_id} not found")
        return paths[0]

    # --- helpers для enroll ------------------------------------------------

    def _prepare_embeddings(self, images: list[bytes]) -> list[tuple[Image.Image, np.ndarray]]:
        """Для каждого файла: препроцесс -> эмбеддинг (если ровно одно лицо). Битые/пустые пропускаем."""
        out = []
        for data in images:
            try:
                pil = self._preprocess(data)
            except InvalidImageError:
                continue
            emb = self._embed_single(self._to_bgr(pil))
            if emb is not None:
                out.append((pil, emb))
        return out

    def _persist(self, face_id: int, prepared: list[tuple[Image.Image, np.ndarray]]) -> np.ndarray:
        """Сохраняет фото как <id>_<k>.jpg и возвращает их эмбеддинги (m, 512)."""
        directory = Path(self._settings.known_faces_dir)
        directory.mkdir(parents=True, exist_ok=True)
        embs = []
        for pil, emb in prepared:
            pil.save(self._next_image_path(face_id), "JPEG", quality=95)
            embs.append(emb)
        return np.stack(embs)

    # --- inference ---------------------------------------------------------

    def predict(self, frame: str) -> dict:
        if self._app is None:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()
        bgr = self._decode(frame)
        faces = self._get_faces(bgr)

        faces_out: list[dict] = []
        for f in faces:
            identity, identity_id, similarity = self._match(f.normed_embedding)
            faces_out.append({
                "box": [float(v) for v in f.bbox.tolist()],
                "det_confidence": round(float(f.det_score), 4),
                "identity": identity,
                "identity_id": identity_id,
                "similarity": round(similarity, 4),
            })

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        recognized = [f["identity"] for f in faces_out if f["identity"] != _UNKNOWN]
        if recognized:
            logger.info("Recognized %s (%.1f ms)", recognized, elapsed_ms)
        else:
            logger.debug("faces=%d, none recognized (%.1f ms)", len(faces_out), elapsed_ms)

        return {
            "faces": faces_out,
            "count": len(faces_out),
            "processing_ms": round(elapsed_ms, 2),
        }

    def _match(self, embedding: np.ndarray) -> tuple[str, int | None, float]:
        """Максимум косинусной близости по всем эталонам каждого человека."""
        e = np.asarray(embedding, dtype=np.float32)
        best_id, best = None, -1.0
        for face_id, embs in list(self._gallery.items()):  # снимок: безопасно при конкурентном enroll
            score = float(np.max(embs @ e))
            if score > best:
                best_id, best = face_id, score
        if best_id is not None and best >= self._settings.face_match_threshold:
            return self._names[best_id], best_id, best
        return _UNKNOWN, None, (best if best_id is not None else 0.0)

    # --- helpers -----------------------------------------------------------

    def _preprocess(self, data: bytes) -> Image.Image:
        """Байты файла -> RGB-картинка: EXIF-поворот, даунскейл, без EXIF-мусора."""
        try:
            image = Image.open(BytesIO(data))
            image = ImageOps.exif_transpose(image)  # учитываем поворот с телефона
            image = image.convert("RGB")
        except Exception as exc:  # noqa: BLE001 — любая ошибка чтения = битый файл
            raise InvalidImageError() from exc

        max_side = self._settings.face_max_image_side
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side))  # сохраняет пропорции, in-place
        return image

    @staticmethod
    def _to_bgr(pil: Image.Image) -> np.ndarray:
        """PIL RGB -> BGR uint8 ndarray (формат, который ждёт insightface)."""
        return np.asarray(pil)[:, :, ::-1].copy()

    @staticmethod
    def _decode(raw: str) -> np.ndarray:
        if _DATA_URI_MARKER in raw:
            raw = raw.split(_DATA_URI_MARKER, 1)[1]
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidImageError() from exc
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError()
        return image
