"""
Test Suite: Full End-to-End Multi-Camera Police Pursuit & Court Dossier Scenario.
Scenario: Delhi NCR Corridor Stolen Vehicle Hunt (Connaught Place -> Pari Chowk)
Standard: ISO/IEC/IEEE 29119 & Section 65B Court Evidence Standards
Author: E2E Test Suite Architect (M0)
"""

import pytest
from datetime import datetime, timedelta, timezone
import numpy as np

try:
    from backend.database.models import (
        Camera, Vehicle, Sighting, Case, CaseMatch, CameraTopology,
        AuditLog, VerificationStatus, CaseStatus
    )
    from backend.services.matching_service import MatchingEngine
    from backend.services.spatio_temporal import (
        haversine_distance_km, validate_spatio_temporal_transition,
        compute_composite_matching_score, reconstruct_route_dag
    )
    from backend.services.evidence_service import generate_evidence_dossier
    E2E_MODULES_AVAILABLE = True
except ImportError:
    try:
        from stolen_vehicle_ai.backend.database.models import (
            Camera, Vehicle, Sighting, Case, CaseMatch, CameraTopology,
            AuditLog, VerificationStatus, CaseStatus
        )
        from stolen_vehicle_ai.backend.services.matching_service import MatchingEngine
        from stolen_vehicle_ai.backend.services.spatio_temporal import (
            haversine_distance_km, validate_spatio_temporal_transition,
            compute_composite_matching_score, reconstruct_route_dag
        )
        from stolen_vehicle_ai.backend.services.evidence_service import generate_evidence_dossier
        E2E_MODULES_AVAILABLE = True
    except ImportError:
        E2E_MODULES_AVAILABLE = False


class TestEndToEndPolicePursuitScenario:
    """
    Tier 4: Comprehensive Multi-Camera Stolen Vehicle Hunt Simulation.
    Simulates real-world operational pursuit from theft report to court dossier export.
    """

    def test_full_6_camera_pursuit_and_court_dossier(
        self,
        populated_db,
        target_stolen_case,
        target_embedding,
        distractor_embeddings
    ):
        """
        Full E2E Scenario Execution:
        1. Register FIR: White Hyundai Creta (MH12AB1234) stolen at Connaught Place at 08:30 IST.
        2. Ingest sightings across 6 cameras (Cam 1 -> Cam 6) + inject distractors.
        3. Camera 3 tests Mud-on-Plate OCR degradation -> Visual Re-ID fallback.
        4. Cross-camera matching & spatio-temporal validation.
        5. Chronological route DAG reconstruction.
        6. Officer review & verification.
        7. Section 65B court evidence dossier generation.
        """
        if not E2E_MODULES_AVAILABLE:
            pytest.skip("Backend and matching services not yet implemented")

        # -------------------------------------------------------------------------
        # Step 1: Register FIR Case in Database
        # -------------------------------------------------------------------------
        case_data = dict(target_stolen_case)
        case_data["reference_embedding"] = target_embedding
        case_obj = Case(**case_data)
        populated_db.add(case_obj)
        populated_db.commit()

        assert case_obj.id is not None
        assert case_obj.status == CaseStatus.OPEN

        # Fetch 6 Delhi NCR Cameras in geographical corridor sequence
        corridor_cam_ids = [
            "CAM_DEL_001", "CAM_DEL_002", "CAM_DEL_003",
            "CAM_NOIDA_001", "CAM_NOIDA_002", "CAM_GRNOIDA_001",
        ]
        cam_map = {c.id: c for c in populated_db.query(Camera).all()}
        cameras = [cam_map[cid] for cid in corridor_cam_ids if cid in cam_map]
        assert len(cameras) == 6

        # -------------------------------------------------------------------------
        # Step 2: Ingest Sightings for Target Stolen Vehicle Across 6 Cameras
        # -------------------------------------------------------------------------
        base_time = datetime(2026, 9, 1, 8, 30, 0)
        target_timestamps = [
            base_time + timedelta(minutes=5),   # Cam 1 (08:35) - CP Outer Circle
            base_time + timedelta(minutes=14),  # Cam 2 (08:44) - ITO Junction
            base_time + timedelta(minutes=22),  # Cam 3 (08:52) - Akshardham (Mud on plate)
            base_time + timedelta(minutes=35),  # Cam 4 (09:05) - DND Toll Plaza
            base_time + timedelta(minutes=48),  # Cam 5 (09:18) - Noida Expressway
            base_time + timedelta(minutes=62),  # Cam 6 (09:32) - Pari Chowk
        ]

        target_sightings = []
        for i, (cam, ts) in enumerate(zip(cameras, target_timestamps)):
            # Cam 3 has mud on plate: degraded OCR text and confidence
            if i == 2:
                p_text = "MH12AB" # Partial / degraded
                p_conf = 0.35
            else:
                p_text = "MH12AB1234"
                p_conf = 0.95

            # Slight perturbation on embedding to simulate real-world sensor variation
            vec = np.array(target_embedding) + (np.random.RandomState(i).randn(512) * 0.02).astype(np.float32)
            vec = (vec / np.linalg.norm(vec)).tolist()

            s = Sighting(
                camera_id=cam.id,
                timestamp=ts,
                track_id=100 + i,
                plate_text=p_text,
                plate_confidence=p_conf,
                vehicle_box=[120, 80, 480, 360],
                vehicle_type="car",
                vehicle_color="white",
                make="Hyundai",
                model="Creta",
                embedding=vec,
                image_path=f"evidence/pursuit_cam_{cam.id}_frame.jpg",
                crop_path=f"evidence/pursuit_cam_{cam.id}_crop.jpg",
                plate_crop_path=f"evidence/pursuit_cam_{cam.id}_plate.jpg",
                direction_travel="Southeast",
                speed_estimate_kmh=50.0 + i * 5
            )
            populated_db.add(s)
            target_sightings.append(s)

        # -------------------------------------------------------------------------
        # Step 3: Inject Distractor Vehicles (Noise Traffic)
        # -------------------------------------------------------------------------
        distractor_configs = [
            ("DL01XY9999", "car", "black", "Mahindra", "Scorpio", distractor_embeddings[0]),
            ("HR26AB5555", "car", "red", "Maruti", "Swift", distractor_embeddings[1]),
            ("DL1R1234", "auto_rickshaw", "yellow", "Bajaj", "Compact", distractor_embeddings[2]),
            ("UP32T8888", "truck", "blue", "Tata", "Prima", distractor_embeddings[3]),
        ]

        for d_idx, (d_plate, d_type, d_color, d_make, d_model, d_emb) in enumerate(distractor_configs):
            for c_idx, cam in enumerate(cameras[:3]):
                d_s = Sighting(
                    camera_id=cam.id,
                    timestamp=base_time + timedelta(minutes=7 + c_idx * 10 + d_idx),
                    track_id=500 + d_idx * 10 + c_idx,
                    plate_text=d_plate,
                    plate_confidence=0.92,
                    vehicle_box=[50, 50, 250, 250],
                    vehicle_type=d_type,
                    vehicle_color=d_color,
                    make=d_make,
                    model=d_model,
                    embedding=d_emb,
                    image_path=f"evidence/distractor_{d_idx}_cam_{cam.id}.jpg",
                    crop_path=f"evidence/distractor_{d_idx}_crop.jpg"
                )
                populated_db.add(d_s)

        populated_db.commit()

        # -------------------------------------------------------------------------
        # Step 4: Run Matching Engine & Evaluate Target vs Distractors
        # -------------------------------------------------------------------------
        all_sightings = populated_db.query(Sighting).all()
        assert len(all_sightings) >= 6 + 12 # 6 target + 12 distractor events

        matches = []
        for s in all_sightings:
            score, breakdown = compute_composite_matching_score(
                reported_plate=case_obj.reported_plate,
                sighting_plate=s.plate_text,
                plate_confidence=s.plate_confidence,
                reference_embedding=case_obj.reference_embedding,
                sighting_embedding=s.embedding,
                reported_type=case_obj.vehicle_type,
                sighting_type=s.vehicle_type,
                reported_color=case_obj.vehicle_color,
                sighting_color=s.vehicle_color
            )

            # Threshold for candidate association
            if score >= 0.65:
                cm = CaseMatch(
                    case_id=case_obj.id,
                    sighting_id=s.id,
                    plate_score=breakdown["plate_score"],
                    visual_score=breakdown["visual_score"],
                    attribute_score=breakdown["attribute_score"],
                    composite_score=score,
                    verification_status=VerificationStatus.PENDING
                )
                populated_db.add(cm)
                matches.append(cm)

        populated_db.commit()

        # Target sightings must be matched (all 6, including Cam 3 with mud on plate)
        matched_sighting_ids = [m.sighting_id for m in matches]
        for ts in target_sightings:
            assert ts.id in matched_sighting_ids, f"Target sighting at Cam {ts.camera_id} must be matched"

        # Distractor sightings must NOT be matched (all score < 0.65)
        assert len(matches) == 6, f"Expected exactly 6 matches for target, got {len(matches)}"

        # -------------------------------------------------------------------------
        # Step 5: Verify Chronological Route DAG Reconstruction & Feasibility
        # -------------------------------------------------------------------------
        sighting_dicts = []
        for ts in target_sightings:
            cam = populated_db.query(Camera).filter(Camera.id == ts.camera_id).first()
            sighting_dicts.append({
                "sighting_id": ts.id,
                "camera_id": cam.id,
                "camera_name": cam.name,
                "latitude": cam.latitude,
                "longitude": cam.longitude,
                "timestamp": ts.timestamp,
                "composite_score": 0.95
            })

        route = reconstruct_route_dag(sighting_dicts)
        assert len(route["waypoints"]) == 6
        assert len(route["segments"]) == 5
        assert all(seg["feasible"] is True for seg in route["segments"])
        assert 30.0 <= route["total_distance_km"] <= 45.0

        # -------------------------------------------------------------------------
        # Step 6: Officer Reviews & Verifies All 6 Sightings
        # -------------------------------------------------------------------------
        for match in matches:
            match.verification_status = VerificationStatus.VERIFIED
            match.reviewed_by = "Inspector Rajesh Kumar (Badge #DL-4821)"
            match.reviewed_at = datetime.now(timezone.utc)
            match.review_notes = "Confirmed target vehicle trajectory"

        case_obj.status = CaseStatus.TRACKING
        populated_db.commit()

        verified_count = populated_db.query(CaseMatch).filter(
            CaseMatch.case_id == case_obj.id,
            CaseMatch.verification_status == VerificationStatus.VERIFIED
        ).count()
        assert verified_count == 6

        # -------------------------------------------------------------------------
        # Step 7: Export Court-Admissible Section 65B Dossier
        # -------------------------------------------------------------------------
        dossier = generate_evidence_dossier(populated_db, case_obj.id)
        assert dossier is not None
        assert dossier["fir_number"] == "FIR-2026-DEL-0941"
        assert len(dossier["evidence_hash_sha256"]) == 64
        assert len(dossier["verified_timeline"]) == 6
        assert "chain_of_custody" in dossier
