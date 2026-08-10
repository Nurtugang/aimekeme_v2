import cv2
import torch
import numpy as np
from collections import deque
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


def load_model(device):
    print("Скачиваю веса модели...")
    ckpt_path = hf_hub_download(
        repo_id="visionlab-ai/school-violence-detection-models",
        filename="final/final_x3d_realtime.pt"
    )
    print(f"Веса: {ckpt_path}")

    from pytorchvideo.models.hub import x3d_m
    import torch.nn as nn

    model = x3d_m(pretrained=False)

    # Чекпоинт использует Sequential(Dropout, Linear), а не просто Linear
    model.blocks[5].proj = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(2048, 2)
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))

    # Срезаем префикс 'backbone.' который добавила обёртка при сохранении
    state_dict = {k.replace("backbone.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"Модель загружена на {device}")
    return model

def preprocess_frames(frames):
    """
    Принимает список из 16 кадров (BGR numpy arrays от OpenCV).
    Возвращает тензор (1, C=3, T=16, H=224, W=224) для X3D-M.
    """
    processed = []
    for frame in frames:
        # BGR (OpenCV) → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize до 224×224 — входной размер X3D-M
        frame_resized = cv2.resize(frame_rgb, (224, 224))

        # [0, 255] → [0.0, 1.0]
        frame_float = frame_resized.astype(np.float32) / 255.0

        # Нормализация от pytorchvideo: mean=0.45, std=0.225
        mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        std  = np.array([0.225, 0.225, 0.225], dtype=np.float32)
        frame_norm = (frame_float - mean) / std

        processed.append(frame_norm)

    # (T, H, W, C) → (C, T, H, W) → (1, C, T, H, W)
    video_np = np.stack(processed, axis=0)           # (16, 224, 224, 3)
    video_t  = torch.from_numpy(video_np)
    video_t  = video_t.permute(3, 0, 1, 2)           # (3, 16, 224, 224)
    return video_t.unsqueeze(0)                       # (1, 3, 16, 224, 224)


def draw_overlay(frame, is_fight, prob_fight):
    """
    Рисует результат в верхнем левом углу кадра.
    FIGHT — красный, NORMAL — зелёный.
    """
    label     = "FIGHT"  if is_fight else "NORMAL"
    color_box = (0, 0, 220)  if is_fight else (0, 180, 0)
    pct       = int(prob_fight * 100)

    # Полупрозрачный чёрный фон
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (260, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Цветная рамка
    cv2.rectangle(frame, (10, 10), (260, 70), color_box, 2)

    # Метка и уверенность
    cv2.putText(frame, label, (22, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_box, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{pct}%", (155, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_box, 2, cv2.LINE_AA)

    # Прогресс-бар вероятности
    bar_w = int(230 * prob_fight)
    cv2.rectangle(frame, (15, 57), (245, 65), (50, 50, 50), -1)
    cv2.rectangle(frame, (15, 57), (15 + bar_w, 65), color_box, -1)

    return frame


def run(input_path: str, output_path: str, threshold: float = 0.4):
    """
    Читает input_path, запускает X3D-M на каждые 16 кадров,
    рисует оверлей, сохраняет в output_path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не могу открыть видео: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    # Буфер последних 16 кадров (deque автоматически выбрасывает старые)
    BUFFER_SIZE = 16
    STRIDE      = 8     # Inference каждые 8 кадров (~0.25 сек при 30fps)

    buffer        = deque(maxlen=BUFFER_SIZE)
    is_fight      = False
    prob_fight    = 0.0
    frame_idx     = 0

    print(f"\nОбрабатываю: {input_path}  ({total} кадров, {fps:.1f} fps)")
    print(f"Устройство:  {device}")
    print(f"Порог:       {threshold}\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        buffer.append(frame.copy())
        frame_idx += 1

        # Запускаем inference когда буфер заполнен и каждые STRIDE кадров
        if len(buffer) == BUFFER_SIZE and frame_idx % STRIDE == 0:
            tensor = preprocess_frames(list(buffer)).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs  = F.softmax(logits, dim=1)[0]

            prob_fight = probs[1].item()   # вероятность "violent"
            is_fight   = prob_fight >= threshold

            status = "🔴 FIGHT" if is_fight else "🟢 OK"
            print(f"  Кадр {frame_idx:5d}/{total}  |  {status}  |  {prob_fight:.3f}")

        frame = draw_overlay(frame, is_fight, prob_fight)
        out.write(frame)

    cap.release()
    out.release()
    print(f"\n✅ Готово! Сохранено в: {output_path}")


if __name__ == "__main__":
    run(
        input_path="/home/nurtugan/aimekeme_v2/test/fight.avi",
        output_path="output.mp4",
        threshold=0.4
    )