"""Модель распознавания лиц: ArcFace (buffalo_l) через insightface/onnxruntime.

Один объект FaceAnalysis делает всё: детекция + landmarks + выравнивание + эмбеддинг.
`app.get(bgr)` -> список лиц, у каждого `bbox`, `det_score`, `normed_embedding`
(512, уже L2-нормирован, поэтому косинус = скалярное произведение).

ВАЖНО про GPU на Blackwell (sm_120): `import torch` стоит ДО insightface намеренно.
torch при импорте подгружает в процесс CUDA-библиотеки (libcublasLt/cudnn), которые
нужны CUDAExecutionProvider onnxruntime. Без этого onnxruntime молча падает на CPU
(ошибка `libcublasLt.so.12: cannot open shared object file`).

Веса buffalo_l качаются один раз в ~/.insightface/models/.
"""

import logging

import torch  # noqa: F401 -- грузит CUDA-либы для onnxruntime, импорт обязателен ПЕРВЫМ
from insightface.app import FaceAnalysis

logger = logging.getLogger("surveillance.face.model")


def load_face_model(providers: list[str], ctx_id: int, det_size: int,
                    det_thresh: float) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size), det_thresh=det_thresh)
    logger.info("Face model (buffalo_l) ready: ctx_id=%d providers=%s", ctx_id, providers)
    return app
