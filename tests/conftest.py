"""
PyTest Configuration and Shared Fixtures for Indian Police Stolen Vehicle AI System.
Author: E2E Test Suite Architect (M0)
Standard: ISO/IEC/IEEE 29119 & Section 65B Electronic Evidence Standard
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import math
import hashlib
import json
import pytest
import numpy as np

# Ensure project and package root are in sys.path
TEST_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent

for p in [str(PACKAGE_ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Try importing backend models and app
try:
    from backend.database.models import (
        Base, Camera, Vehicle, Sighting, Case, CaseMatch, CameraTopology, 
        AuditLog, VerificationStatus, CaseStatus
    )
    from backend.database.session import get_db
    MODELS_AVAILABLE = True
except ImportError:
    try:
        from stolen_vehicle_ai.backend.database.models import (
            Base, Camera, Vehicle, Sighting, Case, CaseMatch, CameraTopology,
            AuditLog, VerificationStatus, CaseStatus
        )
        from stolen_vehicle_ai.backend.database.session import get_db
        MODELS_AVAILABLE = True
    except ImportError:
        MODELS_AVAILABLE = False


# ============================================================================
# 1. DATABASE & ORM FIXTURES
# ============================================================================

@pytest.fixture(scope="function")
def db_engine():
    """In-memory SQLite engine with StaticPool for test isolation."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    if MODELS_AVAILABLE:
        Base.metadata.create_all(bind=engine)
    yield engine
    if MODELS_AVAILABLE:
        Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Isolated database session per test function."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


# ============================================================================
# 2. CAMERA NETWORK FIXTURES (6 DELHI NCR NODES)
# ============================================================================

DELHI_NCR_CAMERAS = [
    {
        "id": "CAM_DEL_001",
        "name": "Connaught Place Outer Circle Camera #1",
        "latitude": 28.6315,
        "longitude": 77.2167,
        "road_name": "Connaught Circus, New Delhi",
        "direction_bearing": 90.0,
        "camera_type": "urban",
        "status": "active"
    },
    {
        "id": "CAM_DEL_002",
        "name": "ITO Junction Ring Road Camera #2",
        "latitude": 28.6289,
        "longitude": 77.2410,
        "road_name": "Vikas Marg / Ring Road, New Delhi",
        "direction_bearing": 110.0,
        "camera_type": "junction",
        "status": "active"
    },
    {
        "id": "CAM_DEL_003",
        "name": "Akshardham Flyover NH-24 Camera #3",
        "latitude": 28.6180,
        "longitude": 77.2790,
        "road_name": "NH-24 / Akshardham Setu, East Delhi",
        "direction_bearing": 125.0,
        "camera_type": "highway",
        "status": "active"
    },
    {
        "id": "CAM_DEL_004",
        "name": "DND Toll Plaza Camera #4",
        "latitude": 28.5820,
        "longitude": 77.3100,
        "road_name": "DND Flyway Expressway, Noida Entry",
        "direction_bearing": 135.0,
        "camera_type": "toll",
        "status": "active"
    },
    {
        "id": "CAM_DEL_005",
        "name": "Noida Expressway Sector 128 Camera #5",
        "latitude": 28.5020,
        "longitude": 77.4000,
        "road_name": "Noida-Greater Noida Expressway",
        "direction_bearing": 140.0,
        "camera_type": "highway",
        "status": "active"
    },
    {
        "id": "CAM_DEL_006",
        "name": "Pari Chowk Junction Camera #6",
        "latitude": 28.4650,
        "longitude": 77.5100,
        "road_name": "Pari Chowk Circle, Greater Noida",
        "direction_bearing": 150.0,
        "camera_type": "junction",
        "status": "active"
    },
]


@pytest.fixture(scope="function")
def camera_nodes_data():
    """Raw dictionary definitions of the 6 Delhi NCR CCTV surveillance cameras."""
    return list(DELHI_NCR_CAMERAS)


@pytest.fixture(scope="function")
def populated_db(db_session, camera_nodes_data):
    """Database session pre-populated with the 6 Delhi NCR cameras and network topology."""
    if not MODELS_AVAILABLE:
        pytest.skip("SQLAlchemy models not yet imported/available")
    
    cameras = []
    for cdata in camera_nodes_data:
        cam = Camera(**cdata)
        db_session.add(cam)
        cameras.append(cam)
    db_session.commit()

    # Create topology links between consecutive cameras
    for i in range(len(cameras) - 1):
        c1, c2 = cameras[i], cameras[i+1]
        # Calculate Haversine
        R = 6371.0
        dlat = math.radians(c2.latitude - c1.latitude)
        dlon = math.radians(c2.longitude - c1.longitude)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(c1.latitude)) * math.cos(math.radians(c2.latitude)) * math.sin(dlon/2)**2
        dist_km = R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

        top = CameraTopology(
            from_camera_id=c1.id,
            to_camera_id=c2.id,
            distance_km=round(dist_km, 3),
            min_travel_time_sec=round((dist_km / 120.0) * 3600.0, 1),
            max_travel_time_sec=round((dist_km / 15.0) * 3600.0, 1),
            typical_speed_kmh=50.0,
            is_connected=True
        )
        db_session.add(top)
    db_session.commit()
    return db_session


# ============================================================================
# 3. SYNTHETIC VECTORS, IMAGES & STOLEN CASE FIXTURES
# ============================================================================

def make_unit_embedding(seed: int = 42, dim: int = 512) -> list:
    """Generate a reproducible, mathematically exact L2 unit-norm 512-D embedding."""
    rng = np.random.RandomState(seed)
    raw = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(raw)
    normalized = raw / (norm + 1e-9)
    return normalized.tolist()


@pytest.fixture(scope="session")
def target_embedding():
    """512-D unit embedding for target stolen White Creta."""
    return make_unit_embedding(seed=101)


@pytest.fixture(scope="session")
def distractor_embeddings():
    """List of 5 distinct unit embeddings for distractor vehicles."""
    return [make_unit_embedding(seed=200 + i) for i in range(5)]


@pytest.fixture(scope="function")
def target_stolen_case():
    """Canonical test FIR case payload for target stolen vehicle."""
    return {
        "fir_number": "FIR-2026-DEL-0941",
        "reported_plate": "MH12AB1234",
        "theft_datetime": datetime(2026, 9, 1, 8, 30, 0),
        "theft_latitude": 28.6315,
        "theft_longitude": 77.2167,
        "theft_location_name": "Connaught Place Inner Circle, New Delhi",
        "vehicle_type": "car",
        "vehicle_color": "white",
        "make": "Hyundai",
        "model": "Creta",
        "distinctive_features": "Black roof wrap, left rear door scratch",
        "investigating_officer": "Inspector Rajesh Kumar (Badge #DL-4821)",
        "police_station": "Parliament Street Police Station",
        "status": "open"
    }


@pytest.fixture(scope="function")
def synthetic_bgr_frame():
    """Generates a synthetic 720x1280 3-channel BGR frame."""
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    # Add simple road lanes
    frame[400:720, :, :] = 50
    return frame


# ============================================================================
# 4. FASTAPI TEST CLIENT FIXTURE
# ============================================================================

@pytest.fixture(scope="function")
def test_client(db_session):
    """FastAPI TestClient with database session override."""
    try:
        from fastapi.testclient import TestClient
        try:
            from backend.main import app
            from backend.database.session import get_db
        except ImportError:
            from stolen_vehicle_ai.backend.main import app
            from stolen_vehicle_ai.backend.database.session import get_db

        def override_get_db():
            try:
                yield db_session
            finally:
                pass

        app.dependency_overrides[get_db] = override_get_db
        with TestClient(app) as client:
            yield client
        app.dependency_overrides.clear()
    except Exception as e:
        pytest.skip(f"FastAPI app or TestClient not available: {e}")
