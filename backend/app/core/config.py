from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "License Plate Recognition API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    UPLOAD_DIR: Path = Path("uploads")
    ANNOTATED_DIR: Path = Path("uploads/annotated")
    PLATE_CROP_DIR: Path = Path("uploads/plates")
    ENHANCED_DIR: Path = Path("uploads/enhanced")
    YOLO_MODEL_PATH: Path = Path("models/license_plate_detector.pt")
    YOLO_CONFIDENCE_THRESHOLD: float = 0.25
    YOLO_ALLOW_GENERIC_FALLBACK: bool = False
    REAL_ESRGAN_MODEL_PATH: Path = Path("models/RealESRGAN_x4plus.pth")
    REAL_ESRGAN_SCALE: int = 4
    REAL_ESRGAN_TILE: int = 0
    REAL_ESRGAN_HALF: bool = False
    OCR_LANGUAGES: str = "en"
    OCR_GPU: bool = False
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
