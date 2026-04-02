"""
Application Configuration

All settings are loaded from environment variables (or a .env file) via
Pydantic Settings, which provides type validation and default values.

Usage:
    from backend.config import settings
    print(settings.host, settings.port)

To override any value create a .env file in the project root or export
the corresponding environment variable (case-insensitive).
"""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralised configuration for the Face Recognition Attendance System."""

    # ── Application ─────────────────────────────────────────────────────────
    app_name: str = Field(default="Face Recognition Attendance System", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=True, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = Field(default="127.0.0.1", env="HOST")
    port: int = Field(default=8000, env="PORT")

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///./data/database/attendance.db",
        env="DATABASE_URL",
    )

    # ── Security ─────────────────────────────────────────────────────────────
    secret_key: str = Field(
        default="your-secret-key-change-this-in-production",
        env="SECRET_KEY",
    )
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # ── Face Recognition Thresholds ──────────────────────────────────────────
    face_detection_confidence: float = Field(
        default=0.5,
        env="FACE_DETECTION_CONFIDENCE",
        description="Minimum confidence for face detection (0–1)",
    )
    face_recognition_threshold: float = Field(
        default=0.6,
        env="FACE_RECOGNITION_THRESHOLD",
        description="Minimum cosine similarity for a face match (higher = stricter)",
    )
    anti_spoofing_threshold: float = Field(
        default=0.7,
        env="ANTI_SPOOFING_THRESHOLD",
        description="Minimum texture score to pass liveness check (0–1)",
    )

    # ── File Paths ───────────────────────────────────────────────────────────
    student_images_path: str = Field(default="./data/student_images", env="STUDENT_IMAGES_PATH")
    embeddings_path: str = Field(default="./data/embeddings", env="EMBEDDINGS_PATH")
    logs_path: str = Field(default="./data/logs", env="LOGS_PATH")
    models_path: str = Field(default="./models", env="MODELS_PATH")

    # ── Model Paths ──────────────────────────────────────────────────────────
    facenet_model_path: str = Field(default="./models/facenet_model", env="FACENET_MODEL_PATH")
    anti_spoofing_model_path: str = Field(
        default="./models/anti_spoofing_model",
        env="ANTI_SPOOFING_MODEL_PATH",
    )

    # ── Attendance Window ────────────────────────────────────────────────────
    # Times are in HH:MM format (24-hour clock).
    attendance_start_time: str = Field(default="08:00", env="ATTENDANCE_START_TIME")
    attendance_end_time: str = Field(default="18:00", env="ATTENDANCE_END_TIME")
    # Arrivals after this time (HH:MM) are recorded as "late" instead of "present".
    late_threshold_time: str = Field(default="09:30", env="LATE_THRESHOLD_TIME")

    duplicate_attendance_threshold_hours: int = Field(
        default=1,
        env="DUPLICATE_ATTENDANCE_THRESHOLD_HOURS",
    )

    # ── Camera (optional webcam scanning) ───────────────────────────────────
    camera_index: int = Field(default=0, env="CAMERA_INDEX")
    camera_width: int = Field(default=640, env="CAMERA_WIDTH")
    camera_height: int = Field(default=480, env="CAMERA_HEIGHT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

    def create_directories(self) -> None:
        """
        Ensure all required data directories exist.

        Called explicitly during application startup (see main.py lifespan),
        *not* as a module-level import side-effect.
        """
        dirs = [
            self.student_images_path,
            self.embeddings_path,
            self.logs_path,
            self.models_path,
            self.facenet_model_path,
            self.anti_spoofing_model_path,
            os.path.dirname(self.database_url.replace("sqlite:///", "")),
        ]
        for d in dirs:
            if d:  # skip empty strings
                Path(d).mkdir(parents=True, exist_ok=True)


# Single global instance — import this throughout the project
settings = Settings()


# ---------------------------------------------------------------------------
# CLI / manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    for field_name, value in settings.model_dump().items():
        print(f"  {field_name}: {value}")
    print("\nCreating directories …")
    settings.create_directories()
    print("✓ Done.")
