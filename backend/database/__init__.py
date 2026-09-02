"""Database Package Initialization."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.database.models import (
    Base,
    Camera,
    Vehicle,
    Sighting,
    Case,
    CaseMatch,
    CameraTopology,
    AuditLog,
    VerificationStatus,
    CaseStatus,
    CameraStatus,
)
from backend.database.session import (
    engine,
    SessionLocal,
    get_db,
    init_db,
    reset_db,
    create_db_engine,
)

__all__ = [
    "Base",
    "Camera",
    "Vehicle",
    "Sighting",
    "Case",
    "CaseMatch",
    "CameraTopology",
    "AuditLog",
    "VerificationStatus",
    "CaseStatus",
    "CameraStatus",
    "engine",
    "SessionLocal",
    "get_db",
    "init_db",
    "reset_db",
    "create_db_engine",
]
