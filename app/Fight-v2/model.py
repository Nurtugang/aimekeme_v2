"""Модель X3D-M: загрузка весов и препроцессинг кадров."""

import logging

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

logger = logging.getLogger("surveillance.fight.model")

# Вход X3D-M: 224×224, нормализация из pytorchvideo.
_INPUT_SIZE = (224, 224)
_MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
_STD = np.array([0.225, 0.225, 0.225], dtype=np.float32)


def load_model(device: torch.device) -> torch.nn.Module:
    """Скачивает веса с Hugging Face и собирает готовую к инференсу модель."""
    import torch.nn as nn
    from pytorchvideo.models.hub import x3d_m

    ckpt_path = hf_hub_download(
        repo_id="visionlab-ai/school-violence-detection-models",
        filename="final/final_x3d_realtime.pt",
    )
    logger.info("Weights: %s", ckpt_path)

    model = x3d_m(pretrained=False)

    # Чекпоинт использует Sequential(Dropout, Linear), а не просто Linear.
    model.blocks[5].proj = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(2048, 2),
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))

    # Срезаем префикс 'backbone.', который добавила обёртка при сохранении.
    state_dict = {k.replace("backbone.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def preprocess_frames(frames: list[np.ndarray]) -> torch.Tensor:
    """Список из 16 BGR-кадров (как отдаёт OpenCV) -> тензор (1, 3, 16, 224, 224)."""
    processed = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, _INPUT_SIZE)
        frame_float = frame_resized.astype(np.float32) / 255.0
        frame_norm = (frame_float - _MEAN) / _STD
        processed.append(frame_norm)

    # (T, H, W, C) -> (C, T, H, W) -> (1, C, T, H, W)
    video_np = np.stack(processed, axis=0)
    video_t = torch.from_numpy(video_np).permute(3, 0, 1, 2)
    return video_t.unsqueeze(0)
