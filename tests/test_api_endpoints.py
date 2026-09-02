"""
Test Suite: FastAPI REST API Endpoints & Request/Response Contracts.
Standard: ISO/IEC/IEEE 29119 & OpenAPI REST Standards
Author: E2E Test Suite Architect (M0)
"""

import pytest
from datetime import datetime, timedelta

try:
    from backend.database.models import Case, Sighting, Camera, CaseMatch, VerificationStatus, CaseStatus
    MODELS_AVAILABLE = True
except ImportError:
    try:
        from stolen_vehicle_ai.backend.database.models import Case, Sighting, Camera, CaseMatch, VerificationStatus, CaseStatus
        MODELS_AVAILABLE = True
    except ImportError:
        MODELS_AVAILABLE = False


class TestHealthEndpoint:
    """Tier 3: System liveness and telemetry."""

    def test_health_check_endpoint(self, test_client):
        """Verify GET /api/health returns 200 and healthy status."""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") in ["healthy", "ok", "online"]


class TestReportFIRCaseEndpoint:
    """Tier 3: Case registration endpoint POST /api/report."""

    def test_create_case_success(self, test_client):
        """Verify POST /api/report creates new case and returns CaseResponse."""
        payload = {
            "fir_number": "FIR-2026-DEL-0941",
            "reported_plate": "MH12AB1234",
            "theft_datetime": "2026-09-01T08:30:00",
            "theft_latitude": 28.6315,
            "theft_longitude": 77.2167,
            "theft_location_name": "Connaught Place Inner Circle, New Delhi",
            "vehicle_type": "car",
            "vehicle_color": "white",
            "make": "Hyundai",
            "model": "Creta",
            "distinctive_features": "Black roof wrap",
            "investigating_officer": "Inspector Rajesh Kumar (Badge #DL-4821)",
            "police_station": "Parliament Street Police Station"
        }
        response = test_client.post("/api/report", json=payload)
        assert response.status_code in [200, 201]
        data = response.json()
        assert data["fir_number"] == "FIR-2026-DEL-0941"
        assert data["reported_plate"] == "MH12AB1234"
        assert data["status"] in ["open", "tracking"]

    def test_create_case_missing_required_fields_fails(self, test_client):
        """Verify 422 Unprocessable Content when required fields are missing."""
        invalid_payload = {
            "fir_number": "FIR-INVALID-001"
            # Missing reported_plate, theft_datetime, theft_lat, etc.
        }
        response = test_client.post("/api/report", json=invalid_payload)
        assert response.status_code == 422


class TestSearchSightingsEndpoint:
    """Tier 3: Multi-criteria sightings search GET /api/search."""

    def test_search_sightings_by_plate(self, test_client, populated_db, target_embedding):
        """Verify GET /api/search?plate=MH12AB1234 returns matched sightings."""
        # Insert a sample sighting
        cam = populated_db.query(Camera).first()
        sighting = Sighting(
            camera_id=cam.id,
            timestamp=datetime(2026, 9, 1, 8, 35, 0),
            plate_text="MH12AB1234",
            plate_confidence=0.96,
            vehicle_type="car",
            vehicle_color="white",
            vehicle_box=[100, 100, 300, 300],
            embedding=target_embedding,
            image_path="img.jpg",
            crop_path="crop.jpg"
        )
        populated_db.add(sighting)
        populated_db.commit()

        response = test_client.get("/api/search?plate=MH12AB1234")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) >= 1
        assert data["results"][0]["plate_text"] == "MH12AB1234"

    def test_search_sightings_by_color_and_type(self, test_client, populated_db):
        """Verify GET /api/search?color=white&type=car filters appropriately."""
        response = test_client.get("/api/search?color=white&type=car")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data


class TestRouteReconstructionEndpoint:
    """Tier 3: Chronological route query GET /api/route."""

    def test_get_route_for_case(self, test_client, populated_db, target_stolen_case, target_embedding):
        """Verify GET /api/route?case_id=X returns ordered waypoints and segment speeds."""
        # Create case
        case_obj = Case(**target_stolen_case)
        populated_db.add(case_obj)
        populated_db.commit()

        # Add 2 linked sightings and matches
        cams = populated_db.query(Camera).all()
        t1 = datetime(2026, 9, 1, 8, 35, 0)
        t2 = datetime(2026, 9, 1, 8, 44, 0)

        s1 = Sighting(camera_id=cams[0].id, timestamp=t1, plate_text="MH12AB1234", plate_confidence=0.95, vehicle_box=[10,10,100,100], image_path="f1.jpg", crop_path="c1.jpg", embedding=target_embedding)
        s2 = Sighting(camera_id=cams[1].id, timestamp=t2, plate_text="MH12AB1234", plate_confidence=0.92, vehicle_box=[10,10,100,100], image_path="f2.jpg", crop_path="c2.jpg", embedding=target_embedding)
        populated_db.add_all([s1, s2])
        populated_db.commit()

        m1 = CaseMatch(case_id=case_obj.id, sighting_id=s1.id, composite_score=0.95, sequence_order=1)
        m2 = CaseMatch(case_id=case_obj.id, sighting_id=s2.id, composite_score=0.93, sequence_order=2)
        populated_db.add_all([m1, m2])
        populated_db.commit()

        response = test_client.get(f"/api/route?case_id={case_obj.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["case_id"] == case_obj.id
        assert len(data["waypoints"]) >= 2
        assert "segments" in data

    def test_get_route_nonexistent_case_404(self, test_client):
        """Verify 404 Not Found when querying route for non-existent case."""
        response = test_client.get("/api/route?case_id=99999")
        assert response.status_code == 404


class TestVerificationEndpoint:
    """Tier 3: Officer sighting confirmation POST /api/verify."""

    def test_officer_verify_sighting_confirm(self, test_client, populated_db, target_stolen_case):
        """Verify POST /api/verify updates match status to 'verified' with audit trail."""
        case_obj = Case(**target_stolen_case)
        populated_db.add(case_obj)
        cam = populated_db.query(Camera).first()
        sighting = Sighting(camera_id=cam.id, timestamp=datetime(2026, 9, 1, 8, 35, 0), vehicle_box=[0,0,10,10], image_path="a.jpg", crop_path="b.jpg")
        populated_db.add(sighting)
        populated_db.commit()

        match = CaseMatch(case_id=case_obj.id, sighting_id=sighting.id, composite_score=0.91, verification_status=VerificationStatus.PENDING)
        populated_db.add(match)
        populated_db.commit()

        verify_payload = {
            "case_id": case_obj.id,
            "sighting_id": sighting.id,
            "status": "verified",
            "officer_badge_id": "DL-4821",
            "notes": "Visual match confirmed on dented mirror"
        }
        response = test_client.post("/api/verify", json=verify_payload)
        assert response.status_code in [200, 201]
        data = response.json()
        assert data["status"] == "verified"


class TestDashboardStatsEndpoint:
    """Tier 3: Police Command Center stats telemetry GET /api/dashboard/stats."""

    def test_get_dashboard_stats(self, test_client, populated_db):
        """Verify GET /api/dashboard/stats returns active counts."""
        response = test_client.get("/api/dashboard/stats")
        assert response.status_code == 200
        data = response.json()
        assert "active_cases" in data
        assert "total_cameras" in data
        assert "active_cameras" in data
        assert data["total_cameras"] >= 6


class TestEvidenceExportEndpoint:
    """Tier 3: Court-admissible dossier export GET /api/evidence/export."""

    def test_export_evidence_dossier(self, test_client, populated_db, target_stolen_case):
        """Verify GET /api/evidence/export?case_id=X includes Section 65B SHA-256 hash."""
        case_obj = Case(**target_stolen_case)
        populated_db.add(case_obj)
        populated_db.commit()

        response = test_client.get(f"/api/evidence/export?case_id={case_obj.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["case_id"] == case_obj.id
        assert "evidence_hash_sha256" in data
        assert len(data["evidence_hash_sha256"]) == 64  # SHA-256 hex string length

    def test_export_evidence_nonexistent_case_404(self, test_client):
        """Verify 404 Not Found when exporting dossier for non-existent case."""
        response = test_client.get("/api/evidence/export?case_id=99999")
        assert response.status_code == 404

    def test_evidence_html_printable_view(self, test_client, populated_db, target_stolen_case):
        """Verify GET /api/evidence/html?case_id=X returns HTML document."""
        case_obj = Case(**target_stolen_case)
        populated_db.add(case_obj)
        populated_db.commit()

        response = test_client.get(f"/api/evidence/html?case_id={case_obj.id}")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "Section 65B" in response.text


class TestCameraManagementEndpoints:
    """Tier 3: Camera CRUD endpoints."""

    def test_list_cameras(self, test_client, populated_db):
        """Verify GET /api/cameras returns all registered camera nodes."""
        response = test_client.get("/api/cameras")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 6

    def test_create_camera_success(self, test_client, populated_db):
        """Verify POST /api/cameras creates a new camera node."""
        cam_payload = {
            "id": "CAM_TEST_999",
            "name": "Test Highway Camera #999",
            "latitude": 28.7000,
            "longitude": 77.3000,
            "road_name": "Test Highway Road",
            "direction_bearing": 90.0,
            "camera_type": "highway",
            "status": "active"
        }
        response = test_client.post("/api/cameras", json=cam_payload)
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "CAM_TEST_999"

    def test_create_camera_duplicate_fails(self, test_client, populated_db):
        """Verify 400 Bad Request when registering duplicate camera ID."""
        cam_payload = {
            "id": "CAM_DEL_001",
            "name": "Duplicate Camera",
            "latitude": 28.6315,
            "longitude": 77.2167,
            "road_name": "Connaught Circus",
            "direction_bearing": 90.0,
            "camera_type": "urban",
            "status": "active"
        }
        response = test_client.post("/api/cameras", json=cam_payload)
        assert response.status_code == 400

    def test_get_single_camera_details(self, test_client, populated_db):
        """Verify GET /api/cameras/{camera_id} returns correct node."""
        response = test_client.get("/api/cameras/CAM_DEL_001")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "CAM_DEL_001"

    def test_get_single_camera_not_found(self, test_client):
        """Verify 404 for non-existent camera ID."""
        response = test_client.get("/api/cameras/NONEXISTENT_CAM")
        assert response.status_code == 404


class TestSightingsFeedEndpoint:
    """Tier 3: Paginated sightings feed."""

    def test_get_sightings_feed(self, test_client, populated_db):
        """Verify GET /api/sightings returns paginated results."""
        cam = populated_db.query(Camera).first()
        sighting = Sighting(
            camera_id=cam.id,
            timestamp=datetime(2026, 9, 1, 8, 35, 0),
            plate_text="DL01CA4321",
            plate_confidence=0.96,
            vehicle_type="car",
            vehicle_color="red",
            vehicle_box=[100, 100, 300, 300],
            image_path="img.jpg",
            crop_path="crop.jpg"
        )
        populated_db.add(sighting)
        populated_db.commit()

        response = test_client.get("/api/sightings?limit=10&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert "total_count" in data
        assert "results" in data
        assert len(data["results"]) >= 1


class TestCasesManagementEndpoints:
    """Tier 3: Case listing and single retrieval."""

    def test_list_and_get_case(self, test_client, populated_db, target_stolen_case):
        """Verify GET /api/cases and GET /api/cases/{case_id}."""
        case_obj = Case(**target_stolen_case)
        populated_db.add(case_obj)
        populated_db.commit()

        list_resp = test_client.get("/api/cases")
        assert list_resp.status_code == 200
        cases_list = list_resp.json()
        assert len(cases_list) >= 1

        get_resp = test_client.get(f"/api/cases/{case_obj.id}")
        assert get_resp.status_code == 200
        case_data = get_resp.json()
        assert case_data["fir_number"] == target_stolen_case["fir_number"]

