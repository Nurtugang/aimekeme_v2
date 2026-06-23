"""Настройки приложения."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_version: str = "0.2.0"
    device: str = "auto"          # auto | cuda | cpu | cuda:0  (общий для всех моделей)

    # fight
    expected_frames: int = 16     # Модель для классификации драки X3D-M требует 16 кадров
    fight_threshold: float = 0.5

    # --- face (ArcFace / buffalo_l) ---
    known_faces_dir: str = "known_faces"
    face_match_threshold: float = 0.42
    face_det_thresh: float = 0.5
    face_det_size: int = 640
    face_max_upload_bytes: int = 5 * 1024 * 1024
    face_max_image_side: int = 1600


settings = Settings()
