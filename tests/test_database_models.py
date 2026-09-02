"""
Test Suite: Relational Database Models, Relationships, Constraints & JSON Storage.
Standard: ISO/IEC/IEEE 29119 & ACID Database Compliance
Author: E2E Test Suite Architect (M0)
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError

try:
    from backend.database.models import (
        Camera, Vehicle, Sighting, Case, CaseMatch, CameraTopology, 
        AuditLog, VerificationStatus, CaseStatus
    )
except ImportError:
    from stolen_vehicle_ai.backend.database.models import (
        Camera, Vehicle, Sighting, Case, CaseMatch, CameraTopology, 
        AuditLog, VerificationStatus, CaseStatus
    )


class TestCameraModel:
    """Tier 1: Camera node model tests."""

    def test_create_camera_node(self, db_session):
        """Verify camera node creation with GPS coordinates and bearing."""
        cam = Camera(
            id="CAM_TEST_001",
            name="Test Junction Camera",
            latitude=28.6139,
            longitude=77.2090,
            road_name="Janpath Road",
            direction_bearing=180.0,
            camera_type="junction",
            status="active"
        )
        db_session.add(cam)
        db_session.commit()

        retrieved = db_session.query(Camera).filter(Camera.id == "CAM_TEST_001").first()
        assert retrieved is not None
        assert retrieved.name == "Test Junction Camera"
        assert abs(retrieved.latitude - 28.6139) < 1e-5
        assert abs(retrieved.longitude - 77.2090) < 1e-5
        assert retrieved.direction_bearing == 180.0
        assert retrieved.status == "active"
        assert isinstance(retrieved.created_at, datetime)

    def test_camera_primary_key_uniqueness(self, db_session):
        """Verify camera ID uniqueness is strictly enforced."""
        cam1 = Camera(id="CAM_DUP_001", name="Camera 1", latitude=28.1, longitude=77.1, road_name="Road 1")
        db_session.add(cam1)
        db_session.commit()

        db_session.expunge_all()
        cam2 = Camera(id="CAM_DUP_001", name="Camera 2", latitude=28.2, longitude=77.2, road_name="Road 2")
        db_session.add(cam2)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()


class TestVehicleModel:
    """Tier 1: Unique vehicle registry tests."""

    def test_create_vehicle_record(self, db_session):
        """Verify vehicle creation with attributes."""
        veh = Vehicle(
            plate_number="MH12AB1234",
            vehicle_type="car",
            color="white",
            make="Hyundai",
            model="Creta"
        )
        db_session.add(veh)
        db_session.commit()

        retrieved = db_session.query(Vehicle).filter(Vehicle.plate_number == "MH12AB1234").first()
        assert retrieved is not None
        assert retrieved.plate_number == "MH12AB1234"
        assert retrieved.color == "white"
        assert retrieved.make == "Hyundai"

    def test_vehicle_plate_uniqueness(self, db_session):
        """Verify vehicle plate number is unique in registry."""
        v1 = Vehicle(plate_number="DL01CA1000", vehicle_type="car", color="black")
        v2 = Vehicle(plate_number="DL01CA1000", vehicle_type="car", color="red")
        db_session.add(v1)
        db_session.commit()

        db_session.add(v2)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()


class TestSightingModel:
    """Tier 1: Sighting ingestion, 512-D vector JSON storage & relationships."""

    def test_create_sighting_with_512d_vector_and_boxes(self, db_session, camera_nodes_data, target_embedding):
        """Verify sighting stores 512-D embedding, bounding boxes, and metadata."""
        cam = Camera(**camera_nodes_data[0])
        db_session.add(cam)
        db_session.commit()

        sighting = Sighting(
            camera_id=cam.id,
            timestamp=datetime(2026, 9, 1, 8, 35, 0),
            track_id=14,
            plate_text="MH12AB1234",
            plate_confidence=0.96,
            plate_box=[240, 180, 310, 210],
            vehicle_box=[150, 80, 520, 410],
            vehicle_type="car",
            vehicle_color="white",
            make="Hyundai",
            model="Creta",
            embedding=target_embedding,
            image_path="evidence/cam1_frame100.jpg",
            crop_path="evidence/cam1_crop100.jpg",
            plate_crop_path="evidence/cam1_plate100.jpg",
            direction_travel="Eastbound",
            speed_estimate_kmh=45.5
        )
        db_session.add(sighting)
        db_session.commit()

        retrieved = db_session.query(Sighting).filter(Sighting.track_id == 14).first()
        assert retrieved is not None
        assert retrieved.camera_id == cam.id
        assert retrieved.plate_text == "MH12AB1234"
        assert retrieved.plate_confidence == 0.96
        assert len(retrieved.embedding) == 512
        assert retrieved.vehicle_box == [150, 80, 520, 410]
        assert retrieved.plate_box == [240, 180, 310, 210]
        assert retrieved.camera.name == cam.name

    def test_camera_sightings_cascade_deletion(self, db_session, camera_nodes_data):
        """Verify deleting a camera cascades and removes associated sightings."""
        cam = Camera(**camera_nodes_data[0])
        db_session.add(cam)
        db_session.commit()

        sighting = Sighting(
            camera_id=cam.id,
            timestamp=datetime(2026, 9, 1, 9, 0, 0),
            vehicle_box=[100, 100, 300, 300],
            image_path="img.jpg",
            crop_path="crop.jpg"
        )
        db_session.add(sighting)
        db_session.commit()

        # Delete camera
        db_session.delete(cam)
        db_session.commit()

        remaining_sightings = db_session.query(Sighting).filter(Sighting.camera_id == cam.id).all()
        assert len(remaining_sightings) == 0


class TestCaseAndMatchModels:
    """Tier 1: FIR Stolen Case, Case Matches, and Enums."""

    def test_create_fir_case_and_matches(self, db_session, target_stolen_case, camera_nodes_data, target_embedding):
        """Verify Case creation and linking to Sighting via CaseMatch."""
        cam = Camera(**camera_nodes_data[0])
        db_session.add(cam)
        db_session.commit()

        sighting = Sighting(
            camera_id=cam.id,
            timestamp=datetime(2026, 9, 1, 8, 35, 0),
            plate_text="MH12AB1234",
            plate_confidence=0.95,
            vehicle_box=[100, 100, 300, 300],
            image_path="img.jpg",
            crop_path="crop.jpg",
            embedding=target_embedding
        )
        db_session.add(sighting)
        db_session.commit()

        # Create Case
        case_data = dict(target_stolen_case)
        case_data["reference_embedding"] = target_embedding
        case_obj = Case(**case_data)
        db_session.add(case_obj)
        db_session.commit()

        # Create Match
        match = CaseMatch(
            case_id=case_obj.id,
            sighting_id=sighting.id,
            plate_score=0.95,
            visual_score=0.98,
            attribute_score=1.0,
            spatio_temporal_score=1.0,
            composite_score=0.965,
            sequence_order=1,
            verification_status=VerificationStatus.PENDING
        )
        db_session.add(match)
        db_session.commit()

        retrieved_case = db_session.query(Case).filter(Case.fir_number == target_stolen_case["fir_number"]).first()
        assert retrieved_case is not None
        assert len(retrieved_case.matches) == 1
        assert retrieved_case.matches[0].composite_score == 0.965
        assert retrieved_case.matches[0].verification_status == VerificationStatus.PENDING

    def test_fir_number_uniqueness(self, db_session, target_stolen_case):
        """Verify FIR number duplicate raises IntegrityError."""
        c1 = Case(**target_stolen_case)
        db_session.add(c1)
        db_session.commit()

        c2 = Case(**target_stolen_case)
        db_session.add(c2)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_case_match_cascade_on_case_deletion(self, db_session, target_stolen_case, camera_nodes_data):
        """Verify deleting a case removes all associated CaseMatch records."""
        cam = Camera(**camera_nodes_data[0])
        db_session.add(cam)
        db_session.commit()

        sighting = Sighting(
            camera_id=cam.id,
            timestamp=datetime(2026, 9, 1, 8, 35, 0),
            vehicle_box=[10, 10, 200, 200],
            image_path="img.jpg",
            crop_path="crop.jpg"
        )
        db_session.add(sighting)
        db_session.commit()

        case_obj = Case(**target_stolen_case)
        db_session.add(case_obj)
        db_session.commit()

        match = CaseMatch(
            case_id=case_obj.id,
            sighting_id=sighting.id,
            composite_score=0.9
        )
        db_session.add(match)
        db_session.commit()

        db_session.delete(case_obj)
        db_session.commit()

        remaining_matches = db_session.query(CaseMatch).filter(CaseMatch.case_id == case_obj.id).all()
        assert len(remaining_matches) == 0


class TestCameraTopologyModel:
    """Tier 1: Graph network edge topology between cameras."""

    def test_create_topology_edge(self, db_session, camera_nodes_data):
        """Verify directed camera-to-camera distance and speed bounds."""
        cam1 = Camera(**camera_nodes_data[0])
        cam2 = Camera(**camera_nodes_data[1])
        db_session.add_all([cam1, cam2])
        db_session.commit()

        topo = CameraTopology(
            from_camera_id=cam1.id,
            to_camera_id=cam2.id,
            distance_km=2.85,
            min_travel_time_sec=85.5,
            max_travel_time_sec=684.0,
            typical_speed_kmh=45.0,
            is_connected=True
        )
        db_session.add(topo)
        db_session.commit()

        retrieved = db_session.query(CameraTopology).filter(
            CameraTopology.from_camera_id == cam1.id,
            CameraTopology.to_camera_id == cam2.id
        ).first()
        assert retrieved is not None
        assert retrieved.distance_km == 2.85
        assert retrieved.min_travel_time_sec == 85.5
        assert retrieved.is_connected is True


class TestAuditLogModel:
    """Tier 1: Immutable forensic audit trail models."""

    def test_create_audit_log_entry(self, db_session):
        """Verify audit log stores user badge, action, endpoint, and JSON payload."""
        audit = AuditLog(
            user_badge_id="DL-4821",
            user_name="Inspector Rajesh Kumar",
            action="VERIFY_SIGHTING",
            resource_type="sighting",
            resource_id="101",
            endpoint="/api/verify",
            ip_address="10.20.4.15",
            details={"verdict": "verified", "reason": "Dented left door verified"}
        )
        db_session.add(audit)
        db_session.commit()

        retrieved = db_session.query(AuditLog).filter(AuditLog.user_badge_id == "DL-4821").first()
        assert retrieved is not None
        assert retrieved.action == "VERIFY_SIGHTING"
        assert retrieved.details["verdict"] == "verified"
        assert isinstance(retrieved.timestamp, datetime)


class TestDatabaseEngineAndSession:
    """Tier 1: Engine creation, table initialization, and reset routines."""

    def test_sqlite_engine_creation(self):
        from backend.database.session import create_db_engine, init_db, reset_db
        engine = create_db_engine("sqlite:///:memory:")
        assert engine is not None
        assert "sqlite" in engine.dialect.name

        init_db(engine_override=engine)
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        assert {"cameras", "vehicles", "vehicle_sightings", "cases", "case_matches", "camera_topology", "audit_logs"}.issubset(tables)

        reset_db(engine_override=engine)
        inspector = inspect(engine)
        assert len(inspector.get_table_names()) == 7


class TestModelSerialization:
    """Tier 1: Model to_dict serialization for API and audit logging."""

    def test_model_to_dict_methods(self, db_session):
        cam = Camera(id="CAM_SER_01", name="Serial Cam", latitude=28.6, longitude=77.2, road_name="Road")
        veh = Vehicle(plate_number="MH12XY9999", vehicle_type="car", color="blue")
        db_session.add_all([cam, veh])
        db_session.commit()

        sighting = Sighting(
            camera_id=cam.id,
            vehicle_id=veh.id,
            timestamp=datetime(2026, 9, 1, 8, 0, 0),
            plate_text="MH12XY9999",
            plate_confidence=0.99,
            vehicle_box=[10, 10, 100, 100],
            embedding=[0.1] * 512,
            image_path="/path/frame.jpg",
            crop_path="/path/crop.jpg",
        )
        case = Case(
            fir_number="FIR-SER-001",
            reported_plate="MH12XY9999",
            theft_datetime=datetime(2026, 9, 1, 7, 0, 0),
            theft_latitude=28.6,
            theft_longitude=77.2,
            theft_location_name="Test Loc",
            vehicle_type="car",
            vehicle_color="blue",
            investigating_officer="Officer Test",
            police_station="Test PS",
        )
        db_session.add_all([sighting, case])
        db_session.commit()

        match = CaseMatch(case_id=case.id, sighting_id=sighting.id, composite_score=0.95)
        top = CameraTopology(from_camera_id=cam.id, to_camera_id=cam.id, distance_km=0.0, min_travel_time_sec=0.0, max_travel_time_sec=0.0)
        audit = AuditLog(user_badge_id="B001", user_name="Officer A", action="LOGIN", resource_type="user", endpoint="/login", ip_address="127.0.0.1")
        db_session.add_all([match, top, audit])
        db_session.commit()

        assert cam.to_dict()["id"] == "CAM_SER_01"
        assert veh.to_dict()["plate_number"] == "MH12XY9999"
        assert sighting.to_dict()["embedding_dim"] == 512
        assert case.to_dict()["fir_number"] == "FIR-SER-001"
        assert match.to_dict()["composite_score"] == 0.95
        assert top.to_dict()["from_camera_id"] == "CAM_SER_01"
        assert audit.to_dict()["user_badge_id"] == "B001"


class TestSeedDataModule:
    """Tier 1: Full database seed data generator test."""

    def test_seed_all_populates_full_dataset(self, db_session):
        from backend.database.seed_data import seed_all
        seed_all(db=db_session, reset=False)

        cameras = db_session.query(Camera).all()
        assert len(cameras) == 6
        cam_ids = {c.id for c in cameras}
        assert {"CAM_DEL_001", "CAM_DEL_002", "CAM_DEL_003", "CAM_NOIDA_001", "CAM_NOIDA_002", "CAM_GRNOIDA_001"}.issubset(cam_ids)

        topologies = db_session.query(CameraTopology).all()
        assert len(topologies) == 5

        vehicles = db_session.query(Vehicle).all()
        assert len(vehicles) >= 7

        target_sightings = (
            db_session.query(Sighting)
            .filter(Sighting.plate_text == "MH12AB1234")
            .order_by(Sighting.timestamp.asc())
            .all()
        )
        assert len(target_sightings) == 6

        case = db_session.query(Case).filter(Case.fir_number == "FIR-2026-DEL-0941").first()
        assert case is not None
        assert len(case.matches) == 6

        audit_logs = db_session.query(AuditLog).all()
        assert len(audit_logs) >= 3

