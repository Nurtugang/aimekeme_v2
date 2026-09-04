"""Настройки приложения."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_version: str = "0.2.0"
    device: str = "auto"          # auto | cuda | cpu | cuda:0  (общий для всех моделей)

    # fight
    expected_frames: int = 16     # Модель для классификации драки X3D-M требует 16 кадров
    fight_threshold: float = 0.4

    # fire: выбор модели
    # siglip2 (классификатор кадра) | yolo_dfire (боксы, AGPL)
    fire_model: str = "siglip2"
    fire_threshold: float = 0.4   # fire/smoke ниже порога уверенности => normal
    use_tiling: bool = True       # Включить/выключить нарезку 2x2 для поиска мелких очагов

    # counting: выбор модели
    count_model: str = "yolo_head"        # frcnn (torchvision, класс person) | yolo_head (YOLOv8 SCUT-HEAD)
    count_score_thresh: float = 0.4   # порог score(frcnn)/conf(yolo_head) — боксы ниже в подсчёт не идут
    count_head_variant: str = "medium"  # только для yolo_head: medium (точнее) | nano (быстрее)

    # --- face (ArcFace / buffalo_l) ---
    known_faces_dir: str = "known_faces"
    face_match_threshold: float = 0.42
    face_det_thresh: float = 0.5
    face_det_size: int = 640
    face_max_upload_bytes: int = 5 * 1024 * 1024
    face_max_image_side: int = 1600

    # --- persons (YOLOv8-pose + BoT-SORT)
    persons_weights: str = "yolov8l-pose.pt"
    persons_conf_thresh: float = 0.4


settings = Settings()
