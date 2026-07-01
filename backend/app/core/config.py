from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_ENV: str = "development"
    PROJECT_NAME: str = "License Plate Recognition API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    API_PORT: int = 8000
    FRONTEND_PORT: int = 3000
    UPLOAD_DIR: Path = Path("uploads")
    ANNOTATED_DIR: Path = Path("uploads/annotated")
    PLATE_CROP_DIR: Path = Path("uploads/plates")
    ENHANCED_DIR: Path = Path("uploads/enhanced")
    DATABASE_URL: str | None = None
    REDIS_URL: str | None = None
    S3_ENDPOINT: str | None = None
    S3_BUCKET: str | None = None
    S3_ACCESS_KEY: str | None = None
    S3_SECRET_KEY: str | None = None
    AI_SERVICE_URL: str | None = None
    DETECTION_CONF_THRESHOLD: float | None = None
    YOLO_MODEL_PATH: Path = Path("models/license_plate_detector.pt")
    YOLO_CONFIDENCE_THRESHOLD: float = 0.25
    YOLO_NMS_IOU_THRESHOLD: float = 0.45
    YOLO_IMAGE_SIZE: int = 960
    YOLO_MAX_DETECTIONS: int = 20
    YOLO_CLASSES: str = ""
    YOLO_CLASS_NAMES: str = "license_plate,plate,number_plate,licence_plate"
    YOLO_AGNOSTIC_NMS: bool = False
    YOLO_MIN_BOX_WIDTH: int = 12
    YOLO_MIN_BOX_HEIGHT: int = 6
    YOLO_MIN_BOX_AREA_RATIO: float = 0.00005
    YOLO_MAX_BOX_AREA_RATIO: float = 0.20
    YOLO_MIN_ASPECT_RATIO: float = 1.8
    YOLO_MAX_ASPECT_RATIO: float = 5.5
    YOLO_CROP_PADDING_X: float = 0.06
    YOLO_CROP_PADDING_Y: float = 0.16
    YOLO_ALLOW_GENERIC_FALLBACK: bool = False
    FALLBACK_PREPROCESSING_VARIANTS: str = "original,contrast_clahe,low_light_gamma,sharpened"
    FALLBACK_MAX_IMAGE_SIDE: int = 1280
    FALLBACK_MAX_CONTOURS: int = 40
    FALLBACK_MAX_OCR_CANDIDATES: int = 4
    FALLBACK_MAX_DETECTIONS: int = 2
    FALLBACK_ENABLE_UPSCALE: bool = False
    FALLBACK_ENABLE_WHOLE_IMAGE_PLATE_CANDIDATE: bool = True
    REAL_ESRGAN_MODEL_PATH: Path = Path("weights/RealESRGAN_x4plus.pth")
    REAL_ESRGAN_SCALE: int = 4
    REAL_ESRGAN_TILE: int = 128
    REAL_ESRGAN_HALF: bool = False
    OCR_LANGUAGES: str = "en"
    OCR_GPU: bool = False
    OCR_PREPROCESSING_STRATEGIES: str = "original_resized,clahe,adaptive_threshold"
    OCR_MAX_VARIANTS: int = 3
    OCR_MIN_VARIANTS_BEFORE_EARLY_EXIT: int = 1
    OCR_EARLY_EXIT_CONFIDENCE: float = 0.82
    OCR_TARGET_HEIGHT: int = 96
    OCR_MAX_UPSCALE: int = 4
    OCR_MAX_IMAGE_WIDTH: int = 768
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024

    @model_validator(mode="after")
    def validate_startup_environment(self):
        errors = []

        self.APP_ENV = self.APP_ENV.strip().lower()
        if self.APP_ENV not in {"development", "test", "production"}:
            errors.append("APP_ENV must be one of: development, test, production.")

        required_non_empty = {
            "PROJECT_NAME": self.PROJECT_NAME,
            "API_VERSION": self.API_VERSION,
            "API_PREFIX": self.API_PREFIX,
            "OCR_LANGUAGES": self.OCR_LANGUAGES,
            "CORS_ORIGINS": self.CORS_ORIGINS,
        }
        for name, value in required_non_empty.items():
            if not value or not str(value).strip():
                errors.append(f"{name} is required and cannot be empty.")

        if not str(self.API_PREFIX).startswith("/"):
            errors.append("API_PREFIX must start with '/'.")

        if self.API_PORT <= 0 or self.API_PORT > 65535:
            errors.append("API_PORT must be between 1 and 65535.")

        if self.FRONTEND_PORT <= 0 or self.FRONTEND_PORT > 65535:
            errors.append("FRONTEND_PORT must be between 1 and 65535.")

        if self.MAX_UPLOAD_SIZE_BYTES <= 0:
            errors.append("MAX_UPLOAD_SIZE_BYTES must be greater than 0.")

        if self.DETECTION_CONF_THRESHOLD is not None:
            self.YOLO_CONFIDENCE_THRESHOLD = self.DETECTION_CONF_THRESHOLD

        if self.YOLO_CONFIDENCE_THRESHOLD < 0 or self.YOLO_CONFIDENCE_THRESHOLD > 1:
            errors.append("YOLO_CONFIDENCE_THRESHOLD must be between 0 and 1.")

        if self.YOLO_NMS_IOU_THRESHOLD < 0 or self.YOLO_NMS_IOU_THRESHOLD > 1:
            errors.append("YOLO_NMS_IOU_THRESHOLD must be between 0 and 1.")

        if self.YOLO_IMAGE_SIZE <= 0:
            errors.append("YOLO_IMAGE_SIZE must be greater than 0.")

        if self.YOLO_MAX_DETECTIONS <= 0:
            errors.append("YOLO_MAX_DETECTIONS must be greater than 0.")

        if self.YOLO_MIN_BOX_WIDTH <= 0:
            errors.append("YOLO_MIN_BOX_WIDTH must be greater than 0.")

        if self.YOLO_MIN_BOX_HEIGHT <= 0:
            errors.append("YOLO_MIN_BOX_HEIGHT must be greater than 0.")

        if self.YOLO_MIN_BOX_AREA_RATIO < 0 or self.YOLO_MAX_BOX_AREA_RATIO <= 0:
            errors.append("YOLO box area ratios must be greater than or equal to 0.")

        if self.YOLO_MIN_BOX_AREA_RATIO >= self.YOLO_MAX_BOX_AREA_RATIO:
            errors.append("YOLO_MIN_BOX_AREA_RATIO must be lower than YOLO_MAX_BOX_AREA_RATIO.")

        if self.YOLO_MIN_ASPECT_RATIO <= 0 or self.YOLO_MAX_ASPECT_RATIO <= 0:
            errors.append("YOLO aspect ratios must be greater than 0.")

        if self.YOLO_MIN_ASPECT_RATIO >= self.YOLO_MAX_ASPECT_RATIO:
            errors.append("YOLO_MIN_ASPECT_RATIO must be lower than YOLO_MAX_ASPECT_RATIO.")

        if self.YOLO_CROP_PADDING_X < 0 or self.YOLO_CROP_PADDING_Y < 0:
            errors.append("YOLO crop padding values must be 0 or greater.")

        if self.FALLBACK_MAX_IMAGE_SIDE <= 0:
            errors.append("FALLBACK_MAX_IMAGE_SIDE must be greater than 0.")

        if self.FALLBACK_MAX_CONTOURS <= 0:
            errors.append("FALLBACK_MAX_CONTOURS must be greater than 0.")

        if self.FALLBACK_MAX_OCR_CANDIDATES <= 0:
            errors.append("FALLBACK_MAX_OCR_CANDIDATES must be greater than 0.")

        if self.FALLBACK_MAX_DETECTIONS <= 0:
            errors.append("FALLBACK_MAX_DETECTIONS must be greater than 0.")

        if self.REAL_ESRGAN_SCALE <= 0:
            errors.append("REAL_ESRGAN_SCALE must be greater than 0.")

        if self.REAL_ESRGAN_TILE < 0:
            errors.append("REAL_ESRGAN_TILE must be 0 or greater.")

        if self.OCR_MAX_VARIANTS <= 0:
            errors.append("OCR_MAX_VARIANTS must be greater than 0.")

        if self.OCR_MIN_VARIANTS_BEFORE_EARLY_EXIT <= 0:
            errors.append("OCR_MIN_VARIANTS_BEFORE_EARLY_EXIT must be greater than 0.")

        if self.OCR_EARLY_EXIT_CONFIDENCE < 0 or self.OCR_EARLY_EXIT_CONFIDENCE > 1:
            errors.append("OCR_EARLY_EXIT_CONFIDENCE must be between 0 and 1.")

        if self.OCR_TARGET_HEIGHT <= 0:
            errors.append("OCR_TARGET_HEIGHT must be greater than 0.")

        if self.OCR_MAX_UPSCALE <= 0:
            errors.append("OCR_MAX_UPSCALE must be greater than 0.")

        if self.OCR_MAX_IMAGE_WIDTH <= 0:
            errors.append("OCR_MAX_IMAGE_WIDTH must be greater than 0.")

        if self.APP_ENV == "production":
            production_required = {
                "DATABASE_URL": self.DATABASE_URL,
                "REDIS_URL": self.REDIS_URL,
                "S3_ENDPOINT": self.S3_ENDPOINT,
                "S3_BUCKET": self.S3_BUCKET,
                "S3_ACCESS_KEY": self.S3_ACCESS_KEY,
                "S3_SECRET_KEY": self.S3_SECRET_KEY,
                "AI_SERVICE_URL": self.AI_SERVICE_URL,
            }
            missing = [name for name, value in production_required.items() if not value or not value.strip()]
            if missing:
                errors.append(
                    "Missing required production environment variables: "
                    + ", ".join(sorted(missing))
                    + "."
                )

        if errors:
            raise ValueError("Invalid environment configuration: " + " ".join(errors))

        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
