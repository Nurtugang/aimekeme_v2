"""Модель распознавания лиц: ArcFace (buffalo_l) через insightface/onnxruntime.

detect_faces(app, bgr) -> список лиц: `bbox`, `det_score`, `normed_embedding`
(512, L2-нормирован, поэтому косинус = скалярное произведение), `sex`, `age`.

ВАЖНО про GPU на Blackwell (sm_120): `import torch` стоит ДО insightface намеренно.
torch при импорте подгружает в процесс CUDA-библиотеки (libcublasLt/cudnn), которые
нужны CUDAExecutionProvider onnxruntime. Без этого onnxruntime молча падает на CPU
(ошибка `libcublasLt.so.12: cannot open shared object file`).

Веса buffalo_l качаются один раз в ~/.insightface/models/.
"""

import logging

import cv2
import numpy as np
import torch  # noqa: F401 -- грузит CUDA-либы для onnxruntime, импорт обязателен ПЕРВЫМ
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align

logger = logging.getLogger("surveillance.face.model")

# Без allowed_modules FaceAnalysis грузит все 5 моделей пака и гоняет каждую
# на каждом лице. Две landmark-модели нам не нужны и стоят 2.7 мс на лицо.
_MODULES = ["detection", "recognition", "genderage"]


def load_face_model(providers: list[str], ctx_id: int, det_size: int,
                    det_thresh: float) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=providers,
                       allowed_modules=_MODULES)
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size), det_thresh=det_thresh)
    logger.info("Face model (buffalo_l) ready: ctx_id=%d modules=%s providers=%s",
                ctx_id, _MODULES, providers)
    return app


def detect_faces(app: FaceAnalysis, bgr: np.ndarray) -> list[Face]:
    """Все лица кадра: детекция, затем каждая модель одним батчем на все лица.

    Замена app.get(), который гоняет recognition и genderage в цикле по одному
    лицу — на GPU это в 3-4 раза дороже. Объекты Face возвращаются те же.
    """
    bboxes, kpss = app.det_model.detect(bgr, max_num=0, metric="default")
    if bboxes.shape[0] == 0:
        return []

    faces = [
        Face(bbox=bboxes[i, 0:4], kps=(kpss[i] if kpss is not None else None),
             det_score=bboxes[i, 4])
        for i in range(bboxes.shape[0])
    ]

    _embed_batch(app.models["recognition"], bgr, faces)
    genderage = app.models.get("genderage")
    if genderage is not None:
        _genderage_batch(genderage, bgr, faces)
    return faces


def _embed_batch(rec, bgr: np.ndarray, faces: list[Face]) -> None:
    """ArcFace на все лица разом (выравнивание по 5 точкам, как в ArcFaceONNX.get)."""
    size = rec.input_size[0]
    crops = [face_align.norm_crop(bgr, landmark=f.kps, image_size=size) for f in faces]
    for face, feat in zip(faces, rec.get_feat(crops)):
        face.embedding = feat.flatten()


def _genderage_batch(model, bgr: np.ndarray, faces: list[Face]) -> None:
    """Пол/возраст на все лица разом (кроп по bbox с запасом 1.5, как в Attribute.get)."""
    size = model.input_size[0]
    crops = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        center = ((x2 + x1) / 2, (y2 + y1) / 2)
        scale = size / (max(x2 - x1, y2 - y1) * 1.5)
        crops.append(face_align.transform(bgr, center, size, scale, 0)[0])

    blob = cv2.dnn.blobFromImages(
        crops, 1.0 / model.input_std, model.input_size,
        (model.input_mean,) * 3, swapRB=True)
    preds = model.session.run(model.output_names, {model.input_name: blob})[0]
    for face, pred in zip(faces, preds):
        face.gender = int(np.argmax(pred[:2]))
        face.age = int(np.round(pred[2] * 100))
