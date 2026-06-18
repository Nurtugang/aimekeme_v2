"""Настройки приложения."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_version: str = "0.2.0"
    device: str = "auto"          # auto | cuda | cpu | cuda:0  (общий для всех моделей)

    # --- fight ---
    expected_frames: int = 16     # X3D-M требует ровно столько кадров
    fight_threshold: float = 0.5  # P(fight) >= threshold  =>  label "fight"

    # --- face ---
    known_faces_dir: str = "known_faces"  # папка с фото известных людей (<имя>.jpg)
    face_match_threshold: float = 0.6     # косинусная близость >= порог => узнан


settings = Settings()
