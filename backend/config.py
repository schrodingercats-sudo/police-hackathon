"""Application Configuration & Settings for Indian Police Stolen Vehicle AI System."""

import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core application settings with environment variable overrides."""

    # Project metadata
    PROJECT_NAME: str = "Indian Police Stolen Vehicle AI Command Center"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    DEBUG: bool = False

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Database configuration
    # Default to local SQLite database; can be overridden via DATABASE_URL env var
    DATABASE_URL: str = os.environ.get(
        "DATABASE_URL",
        "sqlite:////tmp/stolen_vehicle_ai.db" if os.environ.get("VERCEL") else "sqlite:///./stolen_vehicle_ai.db",
    )
    DB_ECHO: bool = False

    # Directory Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = Path("/tmp/data") if os.environ.get("VERCEL") else BASE_DIR / "data"
    EVIDENCE_DIR: Path = DATA_DIR / "evidence"
    CAMERA_DATA_DIR: Path = DATA_DIR / "cameras"
    SYNTHETIC_FEEDS_DIR: Path = DATA_DIR / "synthetic_feeds"

    # Spatio-Temporal & Matching Engine Parameters
    MAX_SPEED_KMH: float = 120.0  # Max realistic speed for spatio-temporal feasibility
    MIN_PLATE_CONFIDENCE: float = 0.50  # Minimum confidence threshold for OCR
    DEFAULT_WEIGHT_PLATE: float = 0.50
    DEFAULT_WEIGHT_REID: float = 0.35
    DEFAULT_WEIGHT_ATTR: float = 0.15

    # OCR Confusion Penalties
    OCR_SUBSTITUTION_PENALTY: float = 0.30  # Reduced penalty for known OCR confusions (e.g. 0/O, 1/I)
    OCR_DEFAULT_MISMATCH_PENALTY: float = 1.0
    CONFUSION_PENALTY_WEIGHT: float = 0.10

    # Human-in-the-Loop Thresholds
    CONFIRMATION_THRESHOLD: float = 0.85
    MANUAL_REVIEW_THRESHOLD: float = 0.60

    # Evidence & Forensic Compliance
    SECTION_65B_HASH_ALGORITHM: str = "sha256"
    SYSTEM_SALT: str = os.environ.get("SYSTEM_SALT", "delhi-police-cctns-sighting-v1")

    # Authentication & Security
    API_KEY: Optional[str] = os.environ.get("POLICE_API_KEY", None)
    REQUIRE_AUTH: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )


settings = Settings()

# Ensure required storage directories exist
for directory in [settings.DATA_DIR, settings.EVIDENCE_DIR, settings.CAMERA_DATA_DIR, settings.SYNTHETIC_FEEDS_DIR]:
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        pass
