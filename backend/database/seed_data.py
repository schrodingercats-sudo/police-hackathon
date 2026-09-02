"""Seed Data Generator for Indian Police Stolen Vehicle AI Command Center.

Populates initial Delhi NCR surveillance camera network (6 connected corridor nodes),
camera topology graph edges, seed test vehicles, multi-camera target vehicle sightings,
distractor traffic, FIR stolen vehicle cases, case matches, and forensic audit logs.
"""

import hashlib
import json
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sqlalchemy.orm import Session

from backend.database.models import (
    Camera,
    CameraStatus,
    CameraTopology,
    Vehicle,
    Sighting,
    Case,
    CaseStatus,
    CaseMatch,
    VerificationStatus,
    AuditLog,
)
from backend.database.session import SessionLocal, init_db, reset_db

logger = logging.getLogger(__name__)


def generate_mock_embedding(seed_int: int, dim: int = 512) -> List[float]:
    """Generates a deterministic L2-normalized synthetic 512-D embedding."""
    raw = [math.sin(seed_int * 0.13 + i * 0.07) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    if norm == 0.0:
        return [0.0] * dim
    return [round(x / norm, 6) for x in raw]


def compute_sha256(payload: str) -> str:
    """Computes SHA-256 cryptographic digest for evidence integrity."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# 1. Camera Network: Delhi NCR Major Traffic Corridor (6 Monitored Nodes)
# ----------------------------------------------------------------------
DELHI_NCR_CAMERAS = [
    {
        "id": "CAM_DEL_001",
        "name": "Connaught Place Radial Road 1 (Outer Circle)",
        "latitude": 28.6315,
        "longitude": 77.2167,
        "road_name": "Radial Road 1 / Connaught Circus, New Delhi",
        "direction_bearing": 45.0,
        "camera_type": "urban",
        "stream_url": "rtsp://cctv.delhipolice.gov.in/live/cam_del_001",
        "status": CameraStatus.ACTIVE.value,
    },
    {
        "id": "CAM_DEL_002",
        "name": "ITO Junction (Vikas Minar Crossing)",
        "latitude": 28.6289,
        "longitude": 77.2410,
        "road_name": "Vikas Marg / Ring Road, New Delhi",
        "direction_bearing": 95.0,
        "camera_type": "junction",
        "stream_url": "rtsp://cctv.delhipolice.gov.in/live/cam_del_002",
        "status": CameraStatus.ACTIVE.value,
    },
    {
        "id": "CAM_DEL_003",
        "name": "Akshardham Setu / Delhi-Meerut Expressway",
        "latitude": 28.6180,
        "longitude": 77.2790,
        "road_name": "NH-24 / Delhi-Meerut Expressway, East Delhi",
        "direction_bearing": 110.0,
        "camera_type": "highway",
        "stream_url": "rtsp://cctv.delhipolice.gov.in/live/cam_del_003",
        "status": CameraStatus.ACTIVE.value,
    },
    {
        "id": "CAM_NOIDA_001",
        "name": "DND Flyway Toll Plaza (Delhi-Noida Direct)",
        "latitude": 28.5820,
        "longitude": 77.3100,
        "road_name": "Delhi-Noida Direct (DND) Flyway, Toll Plaza",
        "direction_bearing": 140.0,
        "camera_type": "toll",
        "stream_url": "rtsp://cctv.delhipolice.gov.in/live/cam_noida_001",
        "status": CameraStatus.ACTIVE.value,
    },
    {
        "id": "CAM_NOIDA_002",
        "name": "Noida-Greater Noida Expressway (Sector 128)",
        "latitude": 28.5020,
        "longitude": 77.4000,
        "road_name": "Noida-Greater Noida Expressway, Sector 128",
        "direction_bearing": 145.0,
        "camera_type": "highway",
        "stream_url": "rtsp://cctv.delhipolice.gov.in/live/cam_noida_002",
        "status": CameraStatus.ACTIVE.value,
    },
    {
        "id": "CAM_GRNOIDA_001",
        "name": "Pari Chowk Roundabout & Expressway Exit",
        "latitude": 28.4650,
        "longitude": 77.5100,
        "road_name": "Pari Chowk Roundabout, Greater Noida",
        "direction_bearing": 160.0,
        "camera_type": "junction",
        "stream_url": "rtsp://cctv.delhipolice.gov.in/live/cam_grnoida_001",
        "status": CameraStatus.ACTIVE.value,
    },
]

# ----------------------------------------------------------------------
# 2. Camera Topology Edges (Corridor Distance & Travel Time Constraints)
# ----------------------------------------------------------------------
DELHI_NCR_TOPOLOGY = [
    {
        "from_camera_id": "CAM_DEL_001",
        "to_camera_id": "CAM_DEL_002",
        "distance_km": 2.45,
        "min_travel_time_sec": 75.0,
        "max_travel_time_sec": 900.0,
        "typical_speed_kmh": 45.0,
        "is_connected": True,
    },
    {
        "from_camera_id": "CAM_DEL_002",
        "to_camera_id": "CAM_DEL_003",
        "distance_km": 3.90,
        "min_travel_time_sec": 120.0,
        "max_travel_time_sec": 1200.0,
        "typical_speed_kmh": 55.0,
        "is_connected": True,
    },
    {
        "from_camera_id": "CAM_DEL_003",
        "to_camera_id": "CAM_NOIDA_001",
        "distance_km": 5.10,
        "min_travel_time_sec": 155.0,
        "max_travel_time_sec": 1500.0,
        "typical_speed_kmh": 65.0,
        "is_connected": True,
    },
    {
        "from_camera_id": "CAM_NOIDA_001",
        "to_camera_id": "CAM_NOIDA_002",
        "distance_km": 12.50,
        "min_travel_time_sec": 380.0,
        "max_travel_time_sec": 2400.0,
        "typical_speed_kmh": 80.0,
        "is_connected": True,
    },
    {
        "from_camera_id": "CAM_NOIDA_002",
        "to_camera_id": "CAM_GRNOIDA_001",
        "distance_km": 11.80,
        "min_travel_time_sec": 360.0,
        "max_travel_time_sec": 2400.0,
        "typical_speed_kmh": 80.0,
        "is_connected": True,
    },
]


def seed_cameras(db: Session) -> List[Camera]:
    """Seeds the standard 6 Delhi NCR surveillance cameras."""
    cameras = []
    for item in DELHI_NCR_CAMERAS:
        cam = db.query(Camera).filter(Camera.id == item["id"]).first()
        if not cam:
            cam = Camera(**item)
            db.add(cam)
            cameras.append(cam)
        else:
            cameras.append(cam)
    db.commit()
    logger.info(f"Seeded {len(cameras)} surveillance cameras.")
    return cameras


def seed_topology(db: Session) -> List[CameraTopology]:
    """Seeds camera connectivity topology edges with distances and travel bounds."""
    topology_edges = []
    for edge in DELHI_NCR_TOPOLOGY:
        existing = (
            db.query(CameraTopology)
            .filter(
                CameraTopology.from_camera_id == edge["from_camera_id"],
                CameraTopology.to_camera_id == edge["to_camera_id"],
            )
            .first()
        )
        if not existing:
            top = CameraTopology(**edge)
            db.add(top)
            topology_edges.append(top)
        else:
            topology_edges.append(existing)
    db.commit()
    logger.info(f"Seeded {len(topology_edges)} camera topology edges.")
    return topology_edges


def seed_vehicles_and_sightings(db: Session) -> None:
    """
    Seeds vehicles and multi-camera sightings including:
    - Target stolen vehicle (MH12AB1234: White Hyundai Creta) traversing 6 cameras.
    - Multiple distractor vehicles to test specificity and filtering.
    """
    # 1. Target Stolen Vehicle
    target_plate = "MH12AB1234"
    target_vehicle = db.query(Vehicle).filter(Vehicle.plate_number == target_plate).first()
    if not target_vehicle:
        target_vehicle = Vehicle(
            plate_number=target_plate,
            vehicle_type="car",
            color="white",
            make="Hyundai",
            model="Creta",
            first_seen=datetime(2026, 9, 1, 8, 35, 0),
            last_seen=datetime(2026, 9, 1, 9, 24, 0),
        )
        db.add(target_vehicle)

    # 2. Distractor Vehicles
    distractors = [
        {"plate_number": "DL01CA4321", "vehicle_type": "car", "color": "red", "make": "Maruti", "model": "Swift"},
        {"plate_number": "HR26DQ9988", "vehicle_type": "car", "color": "black", "make": "Mahindra", "model": "Scorpio"},
        {"plate_number": "DL1RAB5566", "vehicle_type": "auto_rickshaw", "color": "yellow_green", "make": "Bajaj", "model": "RE Compact"},
        {"plate_number": "UP16BT7711", "vehicle_type": "truck", "color": "blue", "make": "Tata", "model": "Signa"},
        {"plate_number": "DL3SCA1234", "vehicle_type": "car", "color": "white", "make": "Maruti", "model": "WagonR"},
        {"plate_number": "MH12AB1235", "vehicle_type": "car", "color": "silver", "make": "Honda", "model": "City"},
    ]

    for d in distractors:
        if not db.query(Vehicle).filter(Vehicle.plate_number == d["plate_number"]).first():
            db.add(Vehicle(**d))

    db.commit()

    # Target embedding base vector (representative of White Hyundai Creta)
    target_embedding = generate_mock_embedding(seed_int=101)

    # 3. Target Vehicle Sighting Journey across 6 Delhi NCR Cameras
    target_sighting_data = [
        {
            "camera_id": "CAM_DEL_001",
            "timestamp": datetime(2026, 9, 1, 8, 35, 0),
            "track_id": 101,
            "plate_text": "MH12AB1234",
            "plate_confidence": 0.98,
            "plate_box": [320, 480, 450, 520],
            "vehicle_box": [220, 280, 680, 600],
            "vehicle_type": "car",
            "vehicle_color": "white",
            "make": "Hyundai",
            "model": "Creta",
            "direction_travel": "South-East",
            "speed_estimate_kmh": 42.5,
            "image_path": "/data/evidence/frames/CAM_DEL_001_20260901_083500.jpg",
            "crop_path": "/data/evidence/crops/crop_CAM_DEL_001_101.jpg",
            "plate_crop_path": "/data/evidence/crops/plate_CAM_DEL_001_101.jpg",
        },
        {
            "camera_id": "CAM_DEL_002",
            "timestamp": datetime(2026, 9, 1, 8, 42, 0),
            "track_id": 204,
            "plate_text": "MH12AB1234",
            "plate_confidence": 0.94,
            "plate_box": [310, 470, 440, 510],
            "vehicle_box": [210, 270, 670, 590],
            "vehicle_type": "car",
            "vehicle_color": "white",
            "make": "Hyundai",
            "model": "Creta",
            "direction_travel": "East",
            "speed_estimate_kmh": 48.0,
            "image_path": "/data/evidence/frames/CAM_DEL_002_20260901_084200.jpg",
            "crop_path": "/data/evidence/crops/crop_CAM_DEL_002_204.jpg",
            "plate_crop_path": "/data/evidence/crops/plate_CAM_DEL_002_204.jpg",
        },
        {
            "camera_id": "CAM_DEL_003",
            "timestamp": datetime(2026, 9, 1, 8, 51, 0),
            "track_id": 312,
            "plate_text": "MH12AB1234",
            "plate_confidence": 0.38,
            "plate_box": [305, 465, 435, 505],
            "vehicle_box": [200, 260, 660, 580],
            "vehicle_type": "car",
            "vehicle_color": "white",
            "make": "Hyundai",
            "model": "Creta",
            "direction_travel": "South-East",
            "speed_estimate_kmh": 58.0,
            "image_path": "/data/evidence/frames/CAM_DEL_003_20260901_085100.jpg",
            "crop_path": "/data/evidence/crops/crop_CAM_DEL_003_312.jpg",
            "plate_crop_path": "/data/evidence/crops/plate_CAM_DEL_003_312.jpg",
        },
        {
            "camera_id": "CAM_NOIDA_001",
            "timestamp": datetime(2026, 9, 1, 8, 58, 0),
            "track_id": 408,
            "plate_text": "MH12AB1234",
            "plate_confidence": 0.96,
            "plate_box": [330, 490, 460, 530],
            "vehicle_box": [225, 285, 685, 605],
            "vehicle_type": "car",
            "vehicle_color": "white",
            "make": "Hyundai",
            "model": "Creta",
            "direction_travel": "South-East",
            "speed_estimate_kmh": 68.5,
            "image_path": "/data/evidence/frames/CAM_NOIDA_001_20260901_085800.jpg",
            "crop_path": "/data/evidence/crops/crop_CAM_NOIDA_001_408.jpg",
            "plate_crop_path": "/data/evidence/crops/plate_CAM_NOIDA_001_408.jpg",
        },
        {
            "camera_id": "CAM_NOIDA_002",
            "timestamp": datetime(2026, 9, 1, 9, 12, 0),
            "track_id": 515,
            "plate_text": "MH12AB1234",
            "plate_confidence": 0.91,
            "plate_box": [315, 475, 445, 515],
            "vehicle_box": [215, 275, 675, 595],
            "vehicle_type": "car",
            "vehicle_color": "white",
            "make": "Hyundai",
            "model": "Creta",
            "direction_travel": "South-East",
            "speed_estimate_kmh": 76.0,
            "image_path": "/data/evidence/frames/CAM_NOIDA_002_20260901_091200.jpg",
            "crop_path": "/data/evidence/crops/crop_CAM_NOIDA_002_515.jpg",
            "plate_crop_path": "/data/evidence/crops/plate_CAM_NOIDA_002_515.jpg",
        },
        {
            "camera_id": "CAM_GRNOIDA_001",
            "timestamp": datetime(2026, 9, 1, 9, 24, 0),
            "track_id": 622,
            "plate_text": "MH12AB1234",
            "plate_confidence": 0.95,
            "plate_box": [325, 485, 455, 525],
            "vehicle_box": [220, 280, 680, 600],
            "vehicle_type": "car",
            "vehicle_color": "white",
            "make": "Hyundai",
            "model": "Creta",
            "direction_travel": "South",
            "speed_estimate_kmh": 52.0,
            "image_path": "/data/evidence/frames/CAM_GRNOIDA_001_20260901_092400.jpg",
            "crop_path": "/data/evidence/crops/crop_CAM_GRNOIDA_001_622.jpg",
            "plate_crop_path": "/data/evidence/crops/plate_CAM_GRNOIDA_001_622.jpg",
        },
    ]

    target_veh = db.query(Vehicle).filter(Vehicle.plate_number == target_plate).first()
    target_veh_id = target_veh.id if target_veh else None

    for s_idx, item in enumerate(target_sighting_data):
        existing = (
            db.query(Sighting)
            .filter(
                Sighting.camera_id == item["camera_id"],
                Sighting.timestamp == item["timestamp"],
                Sighting.plate_text == item["plate_text"],
            )
            .first()
        )
        if not existing:
            # Slightly perturb embedding to simulate realistic multi-camera variation
            varied_emb = [
                round(val + math.sin(s_idx * 0.5 + idx) * 0.02, 6)
                for idx, val in enumerate(target_embedding)
            ]
            norm_v = math.sqrt(sum(x * x for x in varied_emb))
            unit_emb = [round(x / norm_v, 6) for x in varied_emb]

            hash_digest = compute_sha256(f"{item['camera_id']}_{item['timestamp'].isoformat()}_{item['plate_text']}")

            sighting = Sighting(
                vehicle_id=target_veh_id,
                embedding=unit_emb,
                sha256_hash=hash_digest,
                **item,
            )
            db.add(sighting)

    # 4. Distractor Sightings (Background Urban Traffic)
    distractor_sightings = [
        {
            "camera_id": "CAM_DEL_001",
            "timestamp": datetime(2026, 9, 1, 8, 32, 0),
            "track_id": 99,
            "plate_text": "DL01CA4321",
            "plate_confidence": 0.96,
            "plate_box": [280, 420, 390, 460],
            "vehicle_box": [190, 240, 580, 520],
            "vehicle_type": "car",
            "vehicle_color": "red",
            "make": "Maruti",
            "model": "Swift",
            "direction_travel": "North-West",
            "speed_estimate_kmh": 38.0,
            "image_path": "/data/evidence/frames/CAM_DEL_001_20260901_083200.jpg",
            "crop_path": "/data/evidence/crops/crop_CAM_DEL_001_99.jpg",
            "plate_crop_path": "/data/evidence/crops/plate_CAM_DEL_001_99.jpg",
        },
        {
            "camera_id": "CAM_DEL_002",
            "timestamp": datetime(2026, 9, 1, 8, 40, 0),
            "track_id": 201,
            "plate_text": "HR26DQ9988",
            "plate_confidence": 0.92,
            "plate_box": [300, 450, 420, 490],
            "vehicle_box": [200, 250, 620, 550],
            "vehicle_type": "car",
            "vehicle_color": "black",
            "make": "Mahindra",
            "model": "Scorpio",
            "direction_travel": "South",
            "speed_estimate_kmh": 50.0,
            "image_path": "/data/evidence/frames/CAM_DEL_002_20260901_084000.jpg",
            "crop_path": "/data/evidence/crops/crop_CAM_DEL_002_201.jpg",
            "plate_crop_path": "/data/evidence/crops/plate_CAM_DEL_002_201.jpg",
        },
        {
            "camera_id": "CAM_DEL_003",
            "timestamp": datetime(2026, 9, 1, 8, 48, 0),
            "track_id": 308,
            "plate_text": "DL3SCA1234",  # Similar plate distractor (WagonR)
            "plate_confidence": 0.89,
            "plate_box": [290, 440, 410, 480],
            "vehicle_box": [195, 245, 600, 530],
            "vehicle_type": "car",
            "vehicle_color": "white",
            "make": "Maruti",
            "model": "WagonR",
            "direction_travel": "East",
            "speed_estimate_kmh": 54.0,
            "image_path": "/data/evidence/frames/CAM_DEL_003_20260901_084800.jpg",
            "crop_path": "/data/evidence/crops/crop_CAM_DEL_003_308.jpg",
            "plate_crop_path": "/data/evidence/crops/plate_CAM_DEL_003_308.jpg",
        },
    ]

    for d_idx, item in enumerate(distractor_sightings):
        existing = (
            db.query(Sighting)
            .filter(
                Sighting.camera_id == item["camera_id"],
                Sighting.timestamp == item["timestamp"],
                Sighting.plate_text == item["plate_text"],
            )
            .first()
        )
        if not existing:
            dist_emb = generate_mock_embedding(seed_int=500 + d_idx)
            hash_digest = compute_sha256(f"{item['camera_id']}_{item['timestamp'].isoformat()}_{item['plate_text']}")
            sighting = Sighting(
                embedding=dist_emb,
                sha256_hash=hash_digest,
                **item,
            )
            db.add(sighting)

    db.commit()
    logger.info("Seeded vehicles and multi-camera sightings.")


def seed_cases_and_matches(db: Session) -> Case:
    """Seeds a primary stolen vehicle FIR case and links candidate matches."""
    fir_num = "FIR-2026-DEL-0941"
    existing_case = db.query(Case).filter(Case.fir_number == fir_num).first()

    ref_emb = generate_mock_embedding(seed_int=101)

    if not existing_case:
        existing_case = Case(
            fir_number=fir_num,
            reported_plate="MH12AB1234",
            theft_datetime=datetime(2026, 9, 1, 8, 30, 0),
            theft_latitude=28.6315,
            theft_longitude=77.2167,
            theft_location_name="Radial Road 1, Connaught Place Inner Circle, New Delhi",
            vehicle_type="car",
            vehicle_color="white",
            make="Hyundai",
            model="Creta",
            distinctive_features="Black roof wrap, dented left rear bumper, aftermarket alloy wheels",
            reference_image_path="/data/evidence/ref_MH12AB1234.jpg",
            reference_embedding=ref_emb,
            status=CaseStatus.TRACKING,
            investigating_officer="Inspector Rajesh Kumar (Badge #DL-4821)",
            police_station="Parliament Street Police Station, New Delhi",
        )
        db.add(existing_case)
        db.commit()

    # Link candidate sightings with CaseMatch records
    target_sightings = (
        db.query(Sighting)
        .filter(Sighting.plate_text == "MH12AB1234")
        .order_by(Sighting.timestamp.asc())
        .all()
    )

    for seq, sighting in enumerate(target_sightings, start=1):
        match = (
            db.query(CaseMatch)
            .filter(CaseMatch.case_id == existing_case.id, CaseMatch.sighting_id == sighting.id)
            .first()
        )
        if not match:
            # Verified for first 3 sightings, pending for remaining
            v_status = VerificationStatus.VERIFIED if seq <= 3 else VerificationStatus.PENDING
            reviewed_by = "DL-4821" if seq <= 3 else None
            reviewed_at = (
                (sighting.timestamp + timedelta(minutes=5))
                if seq <= 3
                else None
            )

            match = CaseMatch(
                case_id=existing_case.id,
                sighting_id=sighting.id,
                plate_score=round(sighting.plate_confidence, 4),
                visual_score=0.94,
                attribute_score=1.0,
                spatio_temporal_score=1.0,
                composite_score=round(0.50 * sighting.plate_confidence + 0.35 * 0.94 + 0.15 * 1.0, 4),
                sequence_order=seq,
                verification_status=v_status,
                reviewed_by=reviewed_by,
                reviewed_at=reviewed_at,
                review_notes="Verified visual match on Creta white body and rear dent profile." if seq <= 3 else None,
            )
            db.add(match)

    db.commit()
    logger.info(f"Seeded stolen vehicle case {fir_num} with candidate matches.")
    return existing_case


def seed_audit_logs(db: Session) -> None:
    """Seeds baseline forensic audit logs for DPDP Act & Section 65B compliance."""
    logs = [
        {
            "user_badge_id": "DL-4821",
            "user_name": "Inspector Rajesh Kumar",
            "action": "CASE_REGISTER",
            "resource_type": "case",
            "resource_id": "FIR-2026-DEL-0941",
            "endpoint": "/api/report",
            "ip_address": "10.20.4.15",
            "details": {"reported_plate": "MH12AB1234", "station": "Parliament Street"},
            "timestamp": datetime(2026, 9, 1, 8, 31, 0),
        },
        {
            "user_badge_id": "DL-4821",
            "user_name": "Inspector Rajesh Kumar",
            "action": "SEARCH_SIGHTINGS",
            "resource_type": "sighting",
            "resource_id": None,
            "endpoint": "/api/search",
            "ip_address": "10.20.4.15",
            "details": {"plate": "MH12AB1234", "color": "white", "type": "car"},
            "timestamp": datetime(2026, 9, 1, 8, 45, 0),
        },
        {
            "user_badge_id": "DL-4821",
            "user_name": "Inspector Rajesh Kumar",
            "action": "VERIFY_SIGHTING",
            "resource_type": "case_match",
            "resource_id": "1",
            "endpoint": "/api/verify",
            "ip_address": "10.20.4.15",
            "details": {"status": "verified", "case": "FIR-2026-DEL-0941", "sighting_id": 1},
            "timestamp": datetime(2026, 9, 1, 8, 47, 0),
        },
    ]

    for entry in logs:
        existing = (
            db.query(AuditLog)
            .filter(
                AuditLog.user_badge_id == entry["user_badge_id"],
                AuditLog.action == entry["action"],
                AuditLog.timestamp == entry["timestamp"],
            )
            .first()
        )
        if not existing:
            db.add(AuditLog(**entry))

    db.commit()
    logger.info("Seeded forensic audit logs.")


def seed_all(db: Optional[Session] = None, reset: bool = False) -> None:
    """
    Master function to seed the entire database with cameras, topology,
    vehicles, sightings, cases, matches, and audit logs.
    """
    should_close = False
    if db is None:
        init_db()
        db = SessionLocal()
        should_close = True

    try:
        if reset:
            reset_db()

        logger.info("Starting database seeding process...")
        seed_cameras(db)
        seed_topology(db)
        seed_vehicles_and_sightings(db)
        seed_cases_and_matches(db)
        seed_audit_logs(db)
        logger.info("Database seeding completed successfully.")
    finally:
        if should_close:
            db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    seed_all(reset=True)
