"""Настройки приложения."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    expected_frames: int = 16     # X3D-M требует ровно столько кадров
    api_version: str = "0.1.0"
    fight_threshold: float = 0.5  # P(fight) >= threshold  =>  label "fight"
    device: str = "auto"          # auto | cuda | cpu | cuda:0


settings = Settings()
