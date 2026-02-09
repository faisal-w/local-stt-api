"""Configuration settings for the STT API."""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model settings
    MODEL_NAME: Literal["faster-whisper", "sensevoice", "moonshine"] = "faster-whisper"
    MODEL_SIZE: str = "base"

    # Device settings
    DEVICE: Literal["cpu", "cuda", "auto"] = "auto"
    COMPUTE_TYPE: Literal["int8", "float16", "float32", "auto"] = "auto"

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Logging
    LOG_LEVEL: str = "INFO"

    def get_device(self) -> str:
        """Get the actual device to use."""
        if self.DEVICE == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.DEVICE

    def get_compute_type(self) -> str:
        """Get the compute type based on device."""
        if self.COMPUTE_TYPE == "auto":
            device = self.get_device()
            return "float16" if device == "cuda" else "int8"
        return self.COMPUTE_TYPE


settings = Settings()
