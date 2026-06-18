"""Модели для распознавания лиц (facenet-pytorch).

- MTCNN              — детектор лиц (находит и выравнивает лица в кадре);
- InceptionResnetV1  — эмбеддер (лицо -> вектор 512, L2-нормированный).

Веса InceptionResnetV1 (vggface2) качаются один раз в кеш torch hub.
"""

import logging

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

logger = logging.getLogger("surveillance.face.model")


def load_face_models(device: torch.device) -> tuple[MTCNN, InceptionResnetV1]:
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    logger.info("Face models loaded on device=%s", device)
    return mtcnn, resnet
