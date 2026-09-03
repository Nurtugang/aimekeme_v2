"""Модель распознавания лиц: ArcFace (buffalo_l) через insightface/onnxruntime.

Один объект FaceAnalysis делает всё: детекция + выравнивание + эмбеддинг + пол/возраст.
`app.get(bgr)` -> список лиц, у каждого `bbox`, `det_score`, `normed_embedding`
(512, уже L2-нормирован, поэтому косинус = скалярное произведение), `sex`, `age`.

ВАЖНО про allowed_modules: без него FaceAnalysis грузит ВСЕ пять моделей пака
buffalo_l и прогоняет каждую на КАЖДОМ лице. Две landmark-модели
(1k3d68.onnx 137 МБ, 2d106det.onnx) нам не нужны, а стоят 2.66 мс на лицо —
больше, чем само распознавание. Замер на RTX 5070 Ti, мс на одно лицо:
    все 5 моделей     5.74
    det+rec           2.39
    det+rec+genderage 3.08   <- берём это (genderage = 0.69 мс, нужен для аналитики)
На кадре с 20 лицами это 120 мс против 66 мс.

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

# Модули пака buffalo_l, которые реально используем (см. докстринг выше).
_MODULES = ["detection", "recognition", "genderage"]


def load_face_model(providers: list[str], ctx_id: int, det_size: int,
                    det_thresh: float) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=providers,
                       allowed_modules=_MODULES)
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size), det_thresh=det_thresh)
    logger.info("Face model (buffalo_l) ready: ctx_id=%d modules=%s providers=%s",
                ctx_id, _MODULES, providers)
    return app
