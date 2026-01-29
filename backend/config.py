"""
Configuration Module for Face Recognition Attendance System

This module manages all application settings using Pydantic Settings.
It loads configuration from environment variables and provides type-safe access.

Author: Face Recognition Team
Date: January 2026
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """
    Application Settings Class
    
    All settings are loaded from environment variables or .env file.
    Provides type validation and default values.
    """
    
    # Application Settings
    app_name: str = Field(default="Face Recognition Attendance System", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=True, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server Configuration
    host: str = Field(default="127.0.0.1", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./data/database/attendance.db",
        env="DATABASE_URL"
    )
    
    # Security Settings
    secret_key: str = Field(
        default="your-secret-key-change-this-in-production",
        env="SECRET_KEY"
    )
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=30,
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    
    # Face Recognition Thresholds
    face_detection_confidence: float = Field(
        default=0.5,
        env="FACE_DETECTION_CONFIDENCE",
        description="Minimum confidence for face detection (0-1)"
    )
    face_recognition_threshold: float = Field(
        default=0.6,
        env="FACE_RECOGNITION_THRESHOLD",
        description="Maximum distance for face match (lower = stricter)"
    )
    anti_spoofing_threshold: float = Field(
        default=0.7,
        env="ANTI_SPOOFING_THRESHOLD",
        description="Minimum confidence for liveness detection (0-1)"
    )
    
    # File Paths
    student_images_path: str = Field(
        default="./data/student_images",
        env="STUDENT_IMAGES_PATH"
    )
    embeddings_path: str = Field(
        default="./data/embeddings",
        env="EMBEDDINGS_PATH"
    )
    logs_path: str = Field(default="./data/logs", env="LOGS_PATH")
    models_path: str = Field(default="./models", env="MODELS_PATH")
    
    # Model Paths
    facenet_model_path: str = Field(
        default="./models/facenet_model",
        env="FACENET_MODEL_PATH"
    )
    anti_spoofing_model_path: str = Field(
        default="./models/anti_spoofing_model",
        env="ANTI_SPOOFING_MODEL_PATH"
    )
    
    # Attendance Settings
    attendance_window_start: str = Field(
        default="08:00",
        env="ATTENDANCE_WINDOW_START"
    )
    attendance_window_end: str = Field(
        default="18:00",
        env="ATTENDANCE_WINDOW_END"
    )
    # Aliases for backward compatibility
    attendance_start_time: str = Field(
        default="08:00",
        env="ATTENDANCE_START_TIME"
    )
    attendance_end_time: str = Field(
        default="18:00",
        env="ATTENDANCE_END_TIME"
    )
    duplicate_attendance_threshold_hours: int = Field(
        default=1,
        env="DUPLICATE_ATTENDANCE_THRESHOLD_HOURS"
    )
    
    # Camera Settings
    camera_index: int = Field(default=0, env="CAMERA_INDEX")
    camera_width: int = Field(default=640, env="CAMERA_WIDTH")
    camera_height: int = Field(default=480, env="CAMERA_HEIGHT")
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def create_directories(self):
        """
        Create necessary directories if they don't exist
        
        This ensures all required folders are present before the app runs.
        """
        directories = [
            self.student_images_path,
            self.embeddings_path,
            self.logs_path,
            self.models_path,
            self.facenet_model_path,
            self.anti_spoofing_model_path,
            os.path.dirname(self.database_url.replace("sqlite:///", ""))
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✓ Directory ensured: {directory}")


# Global settings instance
settings = Settings()


# Create directories on module import
if __name__ != "__main__":
    settings.create_directories()


if __name__ == "__main__":
    """Test configuration loading"""
    print("=" * 60)
    print("CONFIGURATION TEST")
    print("=" * 60)
    print(f"App Name: {settings.app_name}")
    print(f"Version: {settings.app_version}")
    print(f"Environment: {settings.environment}")
    print(f"Debug Mode: {settings.debug}")
    print(f"Host: {settings.host}:{settings.port}")
    print(f"Database: {settings.database_url}")
    print(f"Face Detection Confidence: {settings.face_detection_confidence}")
    print(f"Face Recognition Threshold: {settings.face_recognition_threshold}")
    print(f"Anti-Spoofing Threshold: {settings.anti_spoofing_threshold}")
    print("=" * 60)
    print("\nCreating directories...")
    settings.create_directories()
    print("\n✓ Configuration loaded successfully!")
