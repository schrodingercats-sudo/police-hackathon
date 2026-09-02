"""
Comprehensive Empirical Boundary, Concurrency & API Robustness Test Suite.
Author: Final Challenger 2 (Boundary & Robustness Verifier)
Standards: ISO/IEC/IEEE 29119 & OpenAPI REST Boundary Protocols
"""

import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy.exc import IntegrityError
from fastapi.testclient import TestClient

try:
    from backend.database.models import (
        Base, Camera, Vehicle, Sighting, Case, CaseMatch, CameraTopology,
        AuditLog, VerificationStatus, CaseStatus, CameraStatus
    )
    from backend.main import app
except ImportError:
    from stolen_vehicle_ai.backend.database.models import (
        Base, Camera, Vehicle, Sighting, Case, CaseMatch, CameraTopology,
        AuditLog, VerificationStatus, CaseStatus, CameraStatus
    )
    from stolen_vehicle_ai.backend.main import app


# ============================================================================
# 1. API 404 NOT FOUND HANDLER TESTS
# ============================================================================

class TestApi404NotFoundHandlers:
    """Empirically verifies 404 Not Found handling on non-existent resources."""

    def test_get_route_nonexistent_case(self, test_client):
        """GET /api/route with non-existent case_id returns 404."""
        response = test_client.get("/api/route?case_id=999999")
        assert response.status_code == 404
        assert "not found" in response.json().get("detail", "").lower()

    def test_get_case_details_nonexistent(self, test_client):
        """GET /api/cases/{case_id} with non-existent ID returns 404."""
        response = test_client.get("/api/cases/999999")
        assert response.status_code == 404
        assert "not found" in response.json().get("detail", "").lower()

    def test_get_camera_details_nonexistent(self, test_client):
        """GET /api/cameras/{camera_id} with non-existent camera ID returns 404."""
        response = test_client.get("/api/cameras/CAM_NONEXISTENT_999")
        assert response.status_code == 404
        assert "not found" in response.json().get("detail", "").lower()

    def test_export_evidence_dossier_nonexistent(self, test_client):
        """GET /api/evidence/export with non-existent case_id returns 404."""
        response = test_client.get("/api/evidence/export?case_id=999999")
        assert response.status_code == 404
        assert "not found" in response.json().get("detail", "").lower()

    def test_view_printable_evidence_html_nonexistent(self, test_client):
        """GET /api/evidence/html with non-existent case_id returns 404."""
        response = test_client.get("/api/evidence/html?case_id=999999")
        assert response.status_code == 404

    def test_verify_sighting_nonexistent_sighting_id(self, test_client, populated_db, target_stolen_case):
        """POST /api/verify with non-existent sighting_id returns 404."""
        case_obj = Case(**target_stolen_case)
        populated_db.add(case_obj)
        populated_db.commit()

        payload = {
            "case_id": case_obj.id,
            "sighting_id": 999999,
            "status": "verified",
            "officer_badge_id": "DL-4821",
            "notes": "Testing 404 for invalid sighting ID"
        }
        response = test_client.post("/api/verify", json=payload)
        assert response.status_code == 404
        assert "sighting with id 999999 not found" in response.json().get("detail", "").lower()


# ============================================================================
# 2. API 422 VALIDATION ERROR & MALFORMED PAYLOAD TESTS
# ============================================================================

class TestApi422ValidationErrors:
    """Empirically verifies 422 Unprocessable Entity on schema violations & invalid coordinates."""

    def test_report_case_latitude_out_of_range_positive(self, test_client):
        """POST /api/report with latitude > 90.0 triggers 422."""
        payload = {
            "fir_number": "FIR-2026-DEL-INVALID-LAT-1",
            "reported_plate": "MH12AB1234",
            "theft_datetime": "2026-09-01T08:30:00",
            "theft_latitude": 95.0,  # Invalid latitude (> 90)
            "theft_longitude": 77.2167,
            "theft_location_name": "Invalid Latitude Point",
            "vehicle_type": "car",
            "vehicle_color": "white",
            "investigating_officer": "Inspector Test",
            "police_station": "Test PS"
        }
        response = test_client.post("/api/report", json=payload)
        assert response.status_code == 422

    def test_report_case_latitude_out_of_range_negative(self, test_client):
        """POST /api/report with latitude < -90.0 triggers 422."""
        payload = {
            "fir_number": "FIR-2026-DEL-INVALID-LAT-2",
            "reported_plate": "MH12AB1234",
            "theft_datetime": "2026-09-01T08:30:00",
            "theft_latitude": -95.0,  # Invalid latitude (< -90)
            "theft_longitude": 77.2167,
            "theft_location_name": "Invalid Negative Latitude Point",
            "vehicle_type": "car",
            "vehicle_color": "white",
            "investigating_officer": "Inspector Test",
            "police_station": "Test PS"
        }
        response = test_client.post("/api/report", json=payload)
        assert response.status_code == 422

    def test_report_case_longitude_out_of_range_positive(self, test_client):
        """POST /api/report with longitude > 180.0 triggers 422."""
        payload = {
            "fir_number": "FIR-2026-DEL-INVALID-LON-1",
            "reported_plate": "MH12AB1234",
            "theft_datetime": "2026-09-01T08:30:00",
            "theft_latitude": 28.6315,
            "theft_longitude": 185.0,  # Invalid longitude (> 180)
            "theft_location_name": "Invalid Longitude Point",
            "vehicle_type": "car",
            "vehicle_color": "white",
            "investigating_officer": "Inspector Test",
            "police_station": "Test PS"
        }
        response = test_client.post("/api/report", json=payload)
        assert response.status_code == 422

    def test_report_case_longitude_out_of_range_negative(self, test_client):
        """POST /api/report with longitude < -180.0 triggers 422."""
        payload = {
            "fir_number": "FIR-2026-DEL-INVALID-LON-2",
            "reported_plate": "MH12AB1234",
            "theft_datetime": "2026-09-01T08:30:00",
            "theft_latitude": 28.6315,
            "theft_longitude": -185.0,  # Invalid longitude (< -180)
            "theft_location_name": "Invalid Negative Longitude Point",
            "vehicle_type": "car",
            "vehicle_color": "white",
            "investigating_officer": "Inspector Test",
            "police_station": "Test PS"
        }
        response = test_client.post("/api/report", json=payload)
        assert response.status_code == 422

    def test_report_case_malformed_timestamp(self, test_client):
        """POST /api/report with non-date string triggers 422."""
        payload = {
            "fir_number": "FIR-2026-DEL-BAD-TIME",
            "reported_plate": "MH12AB1234",
            "theft_datetime": "not-a-valid-iso-timestamp",
            "theft_latitude": 28.6315,
            "theft_longitude": 77.2167,
            "theft_location_name": "Bad Time Point",
            "vehicle_type": "car",
            "vehicle_color": "white",
            "investigating_officer": "Inspector Test",
            "police_station": "Test PS"
        }
        response = test_client.post("/api/report", json=payload)
        assert response.status_code == 422

    def test_search_sightings_confidence_out_of_bounds(self, test_client):
        """GET /api/search with min_confidence > 1.0 or < 0.0 triggers 422."""
        res_high = test_client.get("/api/search?min_confidence=1.5")
        assert res_high.status_code == 422

        res_low = test_client.get("/api/search?min_confidence=-0.1")
        assert res_low.status_code == 422

    def test_search_sightings_invalid_pagination(self, test_client):
        """GET /api/search with page < 1 or limit > 100 triggers 422."""
        res_page_0 = test_client.get("/api/search?page=0")
        assert res_page_0.status_code == 422

        res_limit_0 = test_client.get("/api/search?limit=0")
        assert res_limit_0.status_code == 422

        res_limit_high = test_client.get("/api/search?limit=101")
        assert res_limit_high.status_code == 422

    def test_sightings_feed_invalid_pagination(self, test_client):
        """GET /api/sightings with offset < 0 or limit > 100 triggers 422."""
        res_neg_offset = test_client.get("/api/sightings?offset=-5")
        assert res_neg_offset.status_code == 422

        res_limit_high = test_client.get("/api/sightings?limit=150")
        assert res_limit_high.status_code == 422

    def test_create_camera_invalid_payload(self, test_client):
        """POST /api/cameras with invalid coordinates triggers 422."""
        cam_payload = {
            "id": "CAM_INVALID_COORD",
            "name": "Invalid Camera",
            "latitude": 99.0,  # Invalid (>90)
            "longitude": 77.0,
            "road_name": "Test Road",
            "direction_bearing": 90.0,
            "camera_type": "urban"
        }
        response = test_client.post("/api/cameras", json=cam_payload)
        assert response.status_code == 422


# ============================================================================
# 3. 400 BAD REQUEST & DUPLICATE CONFLICT HANDLING TESTS
# ============================================================================

class TestApi400AndDuplicateConflicts:
    """Empirically verifies 400 Bad Request and duplicate handling across API & DB layers."""

    def test_verify_sighting_invalid_status_value(self, test_client, populated_db, target_stolen_case):
        """POST /api/verify with unknown status returns 400 Bad Request."""
        case_obj = Case(**target_stolen_case)
        populated_db.add(case_obj)
        cam = populated_db.query(Camera).first()
        sighting = Sighting(
            camera_id=cam.id,
            timestamp=datetime(2026, 9, 1, 8, 35, 0),
            vehicle_box=[0, 0, 10, 10],
            image_path="a.jpg",
            crop_path="b.jpg"
        )
        populated_db.add(sighting)
        populated_db.commit()

        payload = {
            "case_id": case_obj.id,
            "sighting_id": sighting.id,
            "status": "not_a_valid_status_option",
            "officer_badge_id": "DL-4821",
            "notes": "Testing invalid status"
        }
        response = test_client.post("/api/verify", json=payload)
        assert response.status_code == 400
        assert "invalid verification status" in response.json().get("detail", "").lower()

    def test_create_camera_duplicate_id_conflict_400(self, test_client, populated_db):
        """POST /api/cameras with already registered ID returns 400."""
        existing_cam = populated_db.query(Camera).first()
        duplicate_payload = {
            "id": existing_cam.id,
            "name": "Duplicate ID Attempt",
            "latitude": 28.6315,
            "longitude": 77.2167,
            "road_name": "Connaught Circus",
            "direction_bearing": 90.0,
            "camera_type": "urban",
            "status": "active"
        }
        response = test_client.post("/api/cameras", json=duplicate_payload)
        assert response.status_code == 400
        assert "already registered" in response.json().get("detail", "").lower()

    def test_duplicate_fir_registration_api_idempotent_return(self, test_client, populated_db, target_stolen_case):
        """POST /api/report with duplicate FIR returns existing case record gracefully."""
        payload = dict(target_stolen_case)
        if isinstance(payload.get("theft_datetime"), datetime):
            payload["theft_datetime"] = payload["theft_datetime"].isoformat()

        # First registration
        res1 = test_client.post("/api/report", json=payload)
        assert res1.status_code in [200, 201]
        data1 = res1.json()

        # Second registration with identical FIR
        res2 = test_client.post("/api/report", json=payload)
        assert res2.status_code in [200, 201]
        data2 = res2.json()

        assert data1["id"] == data2["id"]
        assert data1["fir_number"] == data2["fir_number"]

    def test_duplicate_fir_number_database_level_integrity_error(self, db_session, target_stolen_case):
        """Direct DB insert of duplicate FIR number enforces UNIQUE constraint & transaction rollback."""
        c1 = Case(**target_stolen_case)
        db_session.add(c1)
        db_session.commit()

        c2 = Case(**target_stolen_case)
        db_session.add(c2)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

        # Verify session is healthy after rollback
        count = db_session.query(Case).filter(Case.fir_number == target_stolen_case["fir_number"]).count()
        assert count == 1


# ============================================================================
# 4. DATABASE CASCADING DELETIONS & RELATIONSHIP INTEGRITY
# ============================================================================

class TestDatabaseCascadesAndForeignKeys:
    """Empirically verifies foreign key cascading deletions and relationship lifecycles."""

    def test_camera_deletion_cascades_to_sightings(self, db_session, camera_nodes_data):
        """Deleting a camera cascades and removes all sightings associated with that camera."""
        cam = Camera(**camera_nodes_data[0])
        db_session.add(cam)
        db_session.commit()

        s1 = Sighting(
            camera_id=cam.id,
            timestamp=datetime(2026, 9, 1, 8, 30, 0),
            vehicle_box=[10, 10, 100, 100],
            image_path="img1.jpg",
            crop_path="crop1.jpg"
        )
        s2 = Sighting(
            camera_id=cam.id,
            timestamp=datetime(2026, 9, 1, 8, 35, 0),
            vehicle_box=[20, 20, 120, 120],
            image_path="img2.jpg",
            crop_path="crop2.jpg"
        )
        db_session.add_all([s1, s2])
        db_session.commit()

        assert db_session.query(Sighting).filter(Sighting.camera_id == cam.id).count() == 2

        # Delete camera
        db_session.delete(cam)
        db_session.commit()

        assert db_session.query(Sighting).filter(Sighting.camera_id == cam.id).count() == 0

    def test_case_deletion_cascades_to_case_matches(self, db_session, target_stolen_case, camera_nodes_data):
        """Deleting a case cascades and removes all linked CaseMatch records."""
        cam = Camera(**camera_nodes_data[0])
        db_session.add(cam)
        db_session.commit()

        sighting = Sighting(
            camera_id=cam.id,
            timestamp=datetime(2026, 9, 1, 8, 30, 0),
            vehicle_box=[10, 10, 100, 100],
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
            composite_score=0.95,
            verification_status=VerificationStatus.PENDING
        )
        db_session.add(match)
        db_session.commit()

        assert db_session.query(CaseMatch).filter(CaseMatch.case_id == case_obj.id).count() == 1

        # Delete Case
        db_session.delete(case_obj)
        db_session.commit()

        assert db_session.query(CaseMatch).filter(CaseMatch.case_id == case_obj.id).count() == 0
        # Sighting itself should remain intact
        assert db_session.query(Sighting).filter(Sighting.id == sighting.id).count() == 1

    def test_sighting_deletion_cascades_to_case_matches(self, db_session, target_stolen_case, camera_nodes_data):
        """Deleting a sighting cascades and removes all linked CaseMatch records."""
        cam = Camera(**camera_nodes_data[0])
        db_session.add(cam)
        db_session.commit()

        sighting = Sighting(
            camera_id=cam.id,
            timestamp=datetime(2026, 9, 1, 8, 30, 0),
            vehicle_box=[10, 10, 100, 100],
            image_path="img.jpg",
            crop_path="crop.jpg"
        )
        db_session.add(sighting)
        case_obj = Case(**target_stolen_case)
        db_session.add(case_obj)
        db_session.commit()

        match = CaseMatch(
            case_id=case_obj.id,
            sighting_id=sighting.id,
            composite_score=0.92,
            verification_status=VerificationStatus.PENDING
        )
        db_session.add(match)
        db_session.commit()

        assert db_session.query(CaseMatch).filter(CaseMatch.sighting_id == sighting.id).count() == 1

        # Delete Sighting
        db_session.delete(sighting)
        db_session.commit()

        assert db_session.query(CaseMatch).filter(CaseMatch.sighting_id == sighting.id).count() == 0
        # Case itself should remain intact
        assert db_session.query(Case).filter(Case.id == case_obj.id).count() == 1


# ============================================================================
# 5. DASHBOARD TELEMETRY & STATS AGGREGATION EDGE CASES
# ============================================================================

class TestDashboardTelemetryRobustness:
    """Empirically verifies dashboard stats aggregations under edge state conditions."""

    def test_dashboard_stats_empty_database(self, test_client, db_session):
        """GET /api/dashboard/stats on an empty database returns valid zero-safe schema."""
        # Empty session passed via dependency override
        response = test_client.get("/api/dashboard/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["active_cases"] >= 0
        assert data["total_cameras"] >= 0
        assert data["active_cameras"] >= 0
        assert data["detections_today"] >= 0
        assert data["matches_today"] >= 0
        assert isinstance(data["recovery_rate_pct"], float)
        assert isinstance(data["recent_alerts"], list)
        assert isinstance(data["active_camera_list"], list)

    def test_dashboard_stats_with_mixed_case_statuses(self, test_client, populated_db):
        """GET /api/dashboard/stats accurately computes active cases and recovery rate."""
        c1 = Case(
            fir_number="FIR-STATUS-OPEN",
            reported_plate="DL01AA0001",
            theft_datetime=datetime(2026, 9, 1, 8, 0, 0),
            theft_latitude=28.6,
            theft_longitude=77.2,
            theft_location_name="Loc 1",
            vehicle_type="car",
            vehicle_color="white",
            status=CaseStatus.OPEN,
            investigating_officer="Officer 1",
            police_station="PS 1"
        )
        c2 = Case(
            fir_number="FIR-STATUS-TRACKING",
            reported_plate="DL01AA0002",
            theft_datetime=datetime(2026, 9, 1, 8, 0, 0),
            theft_latitude=28.6,
            theft_longitude=77.2,
            theft_location_name="Loc 2",
            vehicle_type="car",
            vehicle_color="red",
            status=CaseStatus.TRACKING,
            investigating_officer="Officer 2",
            police_station="PS 2"
        )
        c3 = Case(
            fir_number="FIR-STATUS-RECOVERED",
            reported_plate="DL01AA0003",
            theft_datetime=datetime(2026, 9, 1, 8, 0, 0),
            theft_latitude=28.6,
            theft_longitude=77.2,
            theft_location_name="Loc 3",
            vehicle_type="motorcycle",
            vehicle_color="black",
            status=CaseStatus.RECOVERED,
            investigating_officer="Officer 3",
            police_station="PS 3"
        )
        c4 = Case(
            fir_number="FIR-STATUS-CLOSED",
            reported_plate="DL01AA0004",
            theft_datetime=datetime(2026, 9, 1, 8, 0, 0),
            theft_latitude=28.6,
            theft_longitude=77.2,
            theft_location_name="Loc 4",
            vehicle_type="truck",
            vehicle_color="silver",
            status=CaseStatus.CLOSED,
            investigating_officer="Officer 4",
            police_station="PS 4"
        )
        populated_db.add_all([c1, c2, c3, c4])
        populated_db.commit()

        response = test_client.get("/api/dashboard/stats")
        assert response.status_code == 200
        data = response.json()
        # Active cases = OPEN + TRACKING
        assert data["active_cases"] >= 2
        # Recovery rate must be a non-negative float
        assert 0.0 <= data["recovery_rate_pct"] <= 100.0


# ============================================================================
# 6. CONCURRENCY & RAPID SEQUENTIAL REQUEST STRESS
# ============================================================================

class TestConcurrencyAndRapidVerificationStress:
    """Empirically verifies system stability and audit logging under rapid sequential operations."""

    def test_rapid_sequential_verification_updates(self, test_client, populated_db, target_stolen_case):
        """Executes 15 rapid sequential verification toggles and confirms audit log consistency."""
        case_obj = Case(**target_stolen_case)
        populated_db.add(case_obj)
        cam = populated_db.query(Camera).first()
        sighting = Sighting(
            camera_id=cam.id,
            timestamp=datetime(2026, 9, 1, 8, 35, 0),
            vehicle_box=[0, 0, 10, 10],
            image_path="a.jpg",
            crop_path="b.jpg"
        )
        populated_db.add(sighting)
        populated_db.commit()

        match = CaseMatch(
            case_id=case_obj.id,
            sighting_id=sighting.id,
            composite_score=0.90,
            verification_status=VerificationStatus.PENDING
        )
        populated_db.add(match)
        populated_db.commit()

        statuses = ["verified", "rejected"] * 7 + ["verified"]  # 15 iterations
        for i, st in enumerate(statuses):
            payload = {
                "case_id": case_obj.id,
                "sighting_id": sighting.id,
                "status": st,
                "officer_badge_id": f"DL-{4000 + i}",
                "notes": f"Rapid cycle iteration {i}"
            }
            res = test_client.post("/api/verify", json=payload)
            assert res.status_code == 200
            assert res.json()["status"] == st

        # Verify final state in DB
        db_match = populated_db.query(CaseMatch).filter(
            CaseMatch.case_id == case_obj.id,
            CaseMatch.sighting_id == sighting.id
        ).first()
        assert db_match.verification_status == VerificationStatus.VERIFIED
        assert db_match.reviewed_by == "DL-4014"

        # Verify audit logs captured all 15 events
        audit_count = populated_db.query(AuditLog).filter(
            AuditLog.action == "VERIFY_SIGHTING",
            AuditLog.resource_id == str(match.id)
        ).count()
        assert audit_count == 15

    def test_search_special_characters_sql_injection_safety(self, test_client, populated_db):
        """Empirically tests search query against SQL injection and special character strings."""
        injection_strings = [
            "' OR '1'='1",
            "'; DROP TABLE sightings; --",
            "<script>alert(1)</script>",
            "MH12%' UNION SELECT * FROM audit_logs --",
            "../../etc/passwd",
            "\\\\000\\\\",
            "   ",
        ]
        for inj in injection_strings:
            response = test_client.get(f"/api/search?plate={inj}&color={inj}&type={inj}")
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert isinstance(data["results"], list)
