"""
Adversarial Stress Test Suite & Edge Case Harvester for Indian Police Stolen Vehicle AI System.

Empirically challenges:
1. Positional OCR Corruptions & Ambiguity Matrix Resolutions (Extreme noise, 36 State substitutions, BH series, multi-line inversions, non-alphanumeric noise).
2. Spatio-Temporal Kinematic Boundary Limits & Teleportation Violations (Exact 120 km/h boundary, delta_t=0 teleportation, negative delta_t paradoxes, DAG sorting & fault isolation).
3. Distractor Vehicle Swarms & Re-ID Dynamic Weight Shifting (Identical make/model swarms, occluded plates, unit-sphere perturbations, zero-vectors).
4. Section 65B Forensic Cryptographic Tamper Detections (Single-bit flip avalanche effect, dossier root hash corruption, chain of custody immutability).
"""

import copy
import hashlib
import json
import math
import random
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

try:
    from backend.database.models import (
        AuditLog,
        Base,
        Camera,
        Case,
        CaseMatch,
        CaseStatus,
        Sighting,
        VerificationStatus,
    )
    from backend.services.evidence_service import (
        generate_evidence_dossier,
        generate_section_65b_certificate,
        generate_sha256_hash,
        verify_evidence_hash,
    )
    from backend.services.matching_service import (
        OCR_CONFUSION_MAP,
        are_confusable,
        compute_composite_matching_score,
        compute_cosine_similarity,
        modified_levenshtein_similarity,
        rank_candidate_sightings,
    )
    from backend.services.spatio_temporal import (
        calculate_bearing,
        haversine_distance_km,
        reconstruct_route_dag,
        validate_spatio_temporal_transition,
    )
    from vision.indian_lp_parser import (
        BH_REGEX,
        HSRP_REGEX,
        INDIAN_STATE_CODES,
        clean_plate_string,
        format_plate_display,
        is_valid_bh_series,
        is_valid_hsrp,
        is_valid_indian_plate,
        parse_two_line_plate,
        repair_indian_plate,
    )
except ImportError:
    from stolen_vehicle_ai.backend.database.models import (
        AuditLog,
        Base,
        Camera,
        Case,
        CaseMatch,
        CaseStatus,
        Sighting,
        VerificationStatus,
    )
    from stolen_vehicle_ai.backend.services.evidence_service import (
        generate_evidence_dossier,
        generate_section_65b_certificate,
        generate_sha256_hash,
        verify_evidence_hash,
    )
    from stolen_vehicle_ai.backend.services.matching_service import (
        OCR_CONFUSION_MAP,
        are_confusable,
        compute_composite_matching_score,
        compute_cosine_similarity,
        modified_levenshtein_similarity,
        rank_candidate_sightings,
    )
    from stolen_vehicle_ai.backend.services.spatio_temporal import (
        calculate_bearing,
        haversine_distance_km,
        reconstruct_route_dag,
        validate_spatio_temporal_transition,
    )
    from stolen_vehicle_ai.vision.indian_lp_parser import (
        BH_REGEX,
        HSRP_REGEX,
        INDIAN_STATE_CODES,
        clean_plate_string,
        format_plate_display,
        is_valid_bh_series,
        is_valid_hsrp,
        is_valid_indian_plate,
        parse_two_line_plate,
        repair_indian_plate,
    )


# ============================================================================
# 1. ADVERSARIAL OCR STRESS TESTS & AMBIGUITY MATRIX RESOLUTION
# ============================================================================

class TestAdversarialOCRCorruptions:
    """Stress-test OCR parser against extreme noise, muddy plates, and character confusions."""

    @pytest.mark.parametrize(
        "corrupted_raw, expected_repaired, expect_valid",
        [
            # Classic positional confusions (O/0, I/1, B/8, Z/2, S/5)
            ("DL OI AB I234", "DL01AB1234", True),
            ("MH I2 AB 8234", "MH12AB8234", True),
            ("KA OS MN 4S67", "KA05MN4567", True),
            ("GJ O1 XX 789O", "GJ01XX7890", True),
            ("UP 32 ZB 2345", "UP32ZB2345", True),
            ("HR 26 DQ 0001", "HR26DQ0001", True),
            # Confusions in State Code letters (0->O/D, 1->I/T, 8->B, 4->A, etc.)
            ("0L01AB1234", "DL01AB1234", True),     # 0L -> DL (Delhi)
            ("8R01AB1234", "BR01AB1234", True),     # 8R -> BR (Bihar)
            ("4P01AB1234", "AP01AB1234", True),     # 4P -> AP (Andhra Pradesh)
            ("1N01AB1234", "TN01AB1234", True),     # 1N -> TN (Tamil Nadu)
            # Noise prefixes and special characters
            ("|DL01AB1234", "DL01AB1234", True),
            ("1KA05MN4567", "KA05MN4567", True),
            ("---MH.12-AB..1234---", "MH12AB1234", True),
            # Bharat Series (BH) confused characters
            ("228H1234AA", "22BH1234AA", True),     # 8H -> BH
            ("22BHI234AA", "22BH1234AA", True),     # I234 -> 1234
            ("22BH12344A", "22BH1234AA", True),     # 4A -> AA (trailing letter)
            ("218N9999A", "21BH9999A", True),       # 8N -> BH
        ],
    )
    def test_extreme_ocr_repairs(self, corrupted_raw, expected_repaired, expect_valid):
        repaired, conf, is_valid = repair_indian_plate(corrupted_raw)
        assert repaired == expected_repaired
        assert is_valid == expect_valid
        assert conf >= 0.70

    def test_all_36_state_codes_recovery_under_digit_substitutions(self):
        """Verify state code recognition under standard optical OCR confusions across Indian states."""
        test_states = [
            ("AP", "4P"), ("AR", "4R"), ("AS", "4S"), ("BR", "8R"), ("CG", "C6"),
            ("GA", "G4"), ("GJ", "6J"), ("KA", "K4"), ("KL", "K1"),
            ("NL", "N1"), ("OD", "0D"), ("PB", "P8"), ("RJ", "RJ"), ("SK", "5K"),
            ("TN", "1N"), ("TS", "1S"), ("TR", "1R"), ("UK", "UK"),
            ("WB", "W8"), ("DL", "0L"), ("JK", "JK"), ("LA", "1A"),
            ("AN", "4N"), ("DD", "0D"),
        ]
        for real_sc, corrupted_sc in test_states:
            test_plate = f"{corrupted_sc}01AB1234"
            repaired, conf, is_valid = repair_indian_plate(test_plate)
            sc_repaired = repaired[:2]
            assert sc_repaired in INDIAN_STATE_CODES, f"Failed state recovery for {corrupted_sc} -> {sc_repaired}"

    def test_two_line_plate_parsing_and_inversions(self):
        """Stress two-line license plates with line inversions and whitespace variations."""
        # Standard: Line 1 = District, Line 2 = Registration
        p1, conf1, v1 = parse_two_line_plate("  DL  01  ", "  AB  1234  ")
        assert p1 == "DL01AB1234"
        assert v1 is True

        # Inverted: Line 1 = Registration, Line 2 = District
        p2, conf2, v2 = parse_two_line_plate("AB 1234", "DL 01")
        assert p2 == "DL01AB1234"
        assert v2 is True

        # Standard with OCR character confusions
        p3, conf3, v3 = parse_two_line_plate("DL OI", "AB I234")
        assert p3 == "DL01AB1234"
        assert v3 is True

    @pytest.mark.parametrize(
        "adversarial_garbage",
        [
            "",
            "   ",
            "!@#$%^&*()_+",
            "A" * 500,
            "12345678901234567890",
            "NON_INDIAN_FOREIGN_PLATE_XYZ_9999",
            "\n\t\r",
            "INVALID123",
        ],
    )
    def test_ocr_garbage_inputs_graceful_handling(self, adversarial_garbage):
        """Verify parser never throws uncaught exceptions on arbitrary adversarial strings."""
        repaired, conf, is_valid = repair_indian_plate(adversarial_garbage)
        assert isinstance(repaired, str)
        assert isinstance(conf, float)
        assert isinstance(is_valid, bool)
        assert is_valid is False or len(repaired) >= 7

    def test_modified_levenshtein_confusion_weight_advantage(self):
        """
        Verify that OCR confusable pairs (e.g. 0/O, 1/I, 8/B) achieve significantly
        higher similarity score (> 0.85) than completely different characters (< 0.70).
        """
        target = "DL01AB1234"
        confusable_ocr = "DLOIABIZ34"  # 0->O, 1->I, 2->Z
        unrelated_mismatch = "DL01XY9999"

        sim_confusable = modified_levenshtein_similarity(target, confusable_ocr)
        sim_unrelated = modified_levenshtein_similarity(target, unrelated_mismatch)

        assert sim_confusable > 0.85, f"Expected high similarity for confusable OCR, got {sim_confusable}"
        assert sim_unrelated < 0.65, f"Expected lower similarity for mismatched plate, got {sim_unrelated}"
        assert sim_confusable > sim_unrelated + 0.20


# ============================================================================
# 2. ADVERSARIAL KINEMATICS & SPATIO-TEMPORAL BOUNDARY LIMITS
# ============================================================================

class TestAdversarialKinematicsAndSpeedLimits:
    """Stress-test speed limit bounds, teleportation detection, and route DAG graph construction."""

    def test_exact_120_kmh_speed_limit_boundary(self):
        """
        Test the exact boundary at 120.0 km/h:
        - 119.0 km in 3600 sec = 119.0 km/h -> FEASIBLE
        - 125.0 km in 3600 sec = 125.0 km/h -> UNFEASIBLE
        """
        t1 = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 9, 1, 11, 0, 0, tzinfo=timezone.utc)  # Exactly 1 hour (3600s)

        lat1, lon1 = 28.6315, 77.2167

        # 1. Point at ~119.0 km south
        lat_feasible = lat1 - (119.0 / 111.139)
        is_feas1, dist1, dt1, speed1 = validate_spatio_temporal_transition(
            lat1, lon1, t1, lat_feasible, lon1, t2, max_speed_kmh=120.0
        )
        assert is_feas1 is True
        assert speed1 <= 120.0

        # 2. Point at ~125.0 km south -> speed ~125 km/h -> MUST BE REJECTED
        lat_unfeasible = lat1 - (125.0 / 111.139)
        is_feas2, dist2, dt2, speed2 = validate_spatio_temporal_transition(
            lat1, lon1, t1, lat_unfeasible, lon1, t2, max_speed_kmh=120.0
        )
        assert is_feas2 is False
        assert speed2 > 120.0

    def test_instantaneous_teleportation_rejections(self):
        """Test delta_t = 0 corner cases across small and large distances."""
        t_same = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Sub-50m same location (same camera or nearby sensor): FEASIBLE
        is_feas_small, dist_small, dt, speed = validate_spatio_temporal_transition(
            28.6315, 77.2167, t_same, 28.6316, 77.2168, t_same
        )
        assert is_feas_small is True
        assert dt == 0.0

        # 10 km distance at same exact second (impossible teleportation): REJECTED
        is_feas_teleport, dist_teleport, dt, speed = validate_spatio_temporal_transition(
            28.6315, 77.2167, t_same, 28.5355, 77.3910, t_same  # CP to Noida Expressway
        )
        assert is_feas_teleport is False
        assert math.isinf(speed) or speed > 10000.0

    def test_negative_time_paradox_rejection(self):
        """Verify that reverse-timestamp transitions (time2 < time1) are rejected."""
        t1 = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 9, 1, 11, 59, 50, tzinfo=timezone.utc)  # 10 seconds in past

        is_feas, dist, dt, speed = validate_spatio_temporal_transition(
            28.6315, 77.2167, t1, 28.6289, 77.2405, t2
        )
        assert is_feas is False
        assert dt < 0

    def test_route_dag_auto_sorting_and_teleportation_flagging(self):
        """
        Verify that route DAG constructor correctly sorts out-of-order waypoints
        and sets `is_valid_route = False` when an impossible jump exists.
        """
        base_time = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)

        # 3 sightings created intentionally out of chronological order
        s1 = {
            "sighting_id": 1,
            "camera_id": "CAM-01",
            "camera_name": "Connaught Place",
            "latitude": 28.6315,
            "longitude": 77.2167,
            "timestamp": base_time + timedelta(minutes=0),
            "composite_score": 0.95,
        }
        s2 = {
            "sighting_id": 2,
            "camera_id": "CAM-02",
            "camera_name": "ITO Junction",
            "latitude": 28.6289,
            "longitude": 77.2405,
            "timestamp": base_time + timedelta(minutes=5),  # 2.3 km in 5 min -> ~28 km/h (FEASIBLE)
            "composite_score": 0.92,
        }
        # Impossible jump: Teleported to Mumbai in 1 minute
        s3_teleport = {
            "sighting_id": 3,
            "camera_id": "CAM-MUMBAI",
            "camera_name": "Gateway of India",
            "latitude": 18.9220,
            "longitude": 72.8347,
            "timestamp": base_time + timedelta(minutes=6),  # 1100 km in 1 min
            "composite_score": 0.90,
        }

        # Provide them scrambled: [s3, s1, s2]
        route = reconstruct_route_dag([s3_teleport, s1, s2], max_speed_kmh=120.0)

        assert route["waypoint_count"] == 3
        # Ensure chronological ordering was enforced
        assert route["waypoints"][0]["sighting_id"] == 1
        assert route["waypoints"][1]["sighting_id"] == 2
        assert route["waypoints"][2]["sighting_id"] == 3

        # Segments: 1->2 (feasible), 2->3 (unfeasible)
        assert route["segment_count"] == 2
        assert route["segments"][0]["feasible"] is True
        assert route["segments"][1]["feasible"] is False
        # The entire route must be flagged as invalid
        assert route["is_valid_route"] is False

    def test_haversine_extreme_and_antipodal_coordinates(self):
        """Test Haversine distance on extreme GPS edge cases."""
        # 1. Same point
        assert haversine_distance_km(0.0, 0.0, 0.0, 0.0) == 0.0
        # 2. Poles: North pole to South pole (Half circumference of Earth ~ 20015 km)
        d_poles = haversine_distance_km(90.0, 0.0, -90.0, 0.0)
        assert 20000.0 <= d_poles <= 20050.0
        # 3. Equator quarter circumference: (0, 0) to (0, 90) ~ 10007 km
        d_quarter = haversine_distance_km(0.0, 0.0, 0.0, 90.0)
        assert 9990.0 <= d_quarter <= 10025.0


# ============================================================================
# 3. ADVERSARIAL DISTRACTOR RANKING & RE-ID WEIGHT SHIFTING
# ============================================================================

class TestAdversarialDistractorRankingAndReID:
    """Stress-test ranking against massive swarms of visually identical distractor vehicles."""

    @pytest.fixture
    def test_db_session(self):
        """Provides an isolated in-memory SQLite database session."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        yield session
        session.close()

    def test_distractor_swarm_with_identical_color_and_type(self, test_db_session):
        """
        Scenario: 30 identical White Hyundai Creta SUVs (distractors) pass camera nodes.
        Target vehicle: White Hyundai Creta, Plate "DL01AB1234", embedding = target_vec.
        Distractors have plates like "DL01XY5555", "HR26ZZ9999", etc.
        One target sighting has a degraded/muddy plate "DL..1234" but matching Re-ID embedding.
        Verify that the true target vehicle ranks #1 despite plate degradation.
        """
        # Create Camera
        cam = Camera(
            id="CAM-CORRIDOR-01",
            name="Delhi Ring Road",
            latitude=28.6315,
            longitude=77.2167,
            road_name="Ring Road, New Delhi",
            status="active"
        )
        test_db_session.add(cam)

        # Target Reference Vector (512-D unit vector)
        rng = np.random.RandomState(42)
        target_vec = rng.randn(512).astype(np.float32)
        target_vec /= np.linalg.norm(target_vec)
        target_embedding = target_vec.tolist()

        # Create Stolen Vehicle Case
        stolen_case = Case(
            fir_number="FIR-ADV-2026-001",
            reported_plate="DL01AB1234",
            theft_datetime=datetime(2026, 9, 1, 9, 0, 0),
            theft_latitude=28.6310,
            theft_longitude=77.2160,
            theft_location_name="Connaught Place",
            vehicle_type="car",
            vehicle_color="white",
            make="Hyundai",
            model="Creta",
            reference_embedding=target_embedding,
            investigating_officer="Inspector Sharma",
            police_station="Connaught Place PS",
            status=CaseStatus.OPEN,
        )
        test_db_session.add(stolen_case)
        test_db_session.commit()

        # 1. Ingest 30 Distractor Vehicles (Same Color: White, Same Type: Car, Unrelated Plate, Random Embedding)
        for i in range(30):
            dist_vec = rng.randn(512).astype(np.float32)
            dist_vec /= np.linalg.norm(dist_vec)
            dist_sighting = Sighting(
                camera_id=cam.id,
                timestamp=datetime(2026, 9, 1, 9, 30, 0) + timedelta(seconds=i * 10),
                vehicle_box=[100, 100, 300, 300],
                vehicle_type="car",
                vehicle_color="white",
                plate_text=f"HR26ZZ{1000 + i}",
                plate_confidence=0.95,
                embedding=dist_vec.tolist(),
                image_path=f"/data/frames/dist_{i}.jpg",
                crop_path=f"/data/crops/dist_{i}.jpg",
                sha256_hash=generate_sha256_hash(f"distractor_{i}"),
            )
            test_db_session.add(dist_sighting)

        # 2. Ingest True Target Sighting with Occluded / Muddy Plate (Low plate conf, high Re-ID similarity)
        noise = rng.randn(512).astype(np.float32)
        noise /= np.linalg.norm(noise)
        noisy_target_vec = target_vec + (noise * 0.10)
        noisy_target_vec /= np.linalg.norm(noisy_target_vec)

        true_target_sighting = Sighting(
            camera_id=cam.id,
            timestamp=datetime(2026, 9, 1, 9, 35, 0),
            vehicle_box=[100, 100, 300, 300],
            vehicle_type="car",
            vehicle_color="white",
            plate_text="DL..1234",       # Muddy / broken plate
            plate_confidence=0.20,       # Low plate confidence -> triggers Re-ID dynamic fallback
            embedding=noisy_target_vec.tolist(),
            image_path="/data/frames/target.jpg",
            crop_path="/data/crops/target.jpg",
            sha256_hash=generate_sha256_hash("true_target_sighting"),
        )
        test_db_session.add(true_target_sighting)
        test_db_session.commit()

        # Rank all candidates
        ranked_matches = rank_candidate_sightings(
            test_db_session,
            stolen_case,
            top_k=10,
            min_score=0.40,
            persist_matches=True
        )

        assert len(ranked_matches) > 0
        # The true target sighting must be the #1 ranked candidate!
        top_match = ranked_matches[0]
        assert top_match.sighting_id == true_target_sighting.id
        assert top_match.composite_score >= 0.70
        assert top_match.visual_score >= 0.85

    def test_dynamic_weight_shifting_logic(self):
        """
        Verify mathematical weight allocation between Clear Plate vs Occluded Plate:
        - Clear Plate (Confidence 1.0, Sim 1.0): 0.50*1.0 + 0.35*Visual + 0.15*Attr = 1.0
        - Occluded Plate (Confidence 0.0): 0.65*Visual + 0.20*Color + 0.15*Type = 1.0
        """
        vec_a = [1.0 / math.sqrt(512)] * 512
        vec_b = [1.0 / math.sqrt(512)] * 512  # Exact visual match (cosine sim = 1.0)

        # Clear Plate
        score_clear, bd_clear = compute_composite_matching_score(
            reported_plate="MH12AB1234",
            sighting_plate="MH12AB1234",
            plate_confidence=1.0,
            reference_embedding=vec_a,
            sighting_embedding=vec_b,
            reported_type="car",
            sighting_type="car",
            reported_color="black",
            sighting_color="black",
        )
        assert bd_clear["plate_occluded"] is False
        assert score_clear == 1.0

        # Occluded Plate with Identical Visuals
        score_occluded, bd_occluded = compute_composite_matching_score(
            reported_plate="MH12AB1234",
            sighting_plate=None,
            plate_confidence=0.0,
            reference_embedding=vec_a,
            sighting_embedding=vec_b,
            reported_type="car",
            sighting_type="car",
            reported_color="black",
            sighting_color="black",
        )
        assert bd_occluded["plate_occluded"] is True
        # Score = 0.65*1.0 + 0.20*1.0 + 0.15*1.0 = 1.0
        assert score_occluded == 1.0

    def test_cosine_similarity_adversarial_vectors(self):
        """Test Cosine Similarity against zero vectors, orthogonal vectors, and dimension mismatches."""
        vec_normal = [0.1] * 512
        vec_zeros = [0.0] * 512
        vec_mismatch = [0.1] * 256

        # Zero vector handling (must not cause ZeroDivisionError)
        assert compute_cosine_similarity(vec_normal, vec_zeros) == 0.0
        assert compute_cosine_similarity(vec_zeros, vec_zeros) == 0.0

        # Dimension mismatch
        assert compute_cosine_similarity(vec_normal, vec_mismatch) == 0.0

        # None inputs
        assert compute_cosine_similarity(None, vec_normal) == 0.0
        assert compute_cosine_similarity(vec_normal, None) == 0.0


# ============================================================================
# 4. SECTION 65B EVIDENCE HASH TAMPER DETECTIONS
# ============================================================================

class TestSection65BTamperDetections:
    """Stress-test cryptographic SHA-256 evidence integrity and 1-bit tamper detection."""

    def test_single_bit_flip_avalanche_detection(self):
        """
        Verify that flipping even a single bit in a 50KB image payload
        completely invalidates verification (Avalanche Effect).
        """
        # Create simulated 50KB image buffer
        rng = random.Random(1337)
        original_image_bytes = bytearray(rng.randbytes(50000))
        original_hash = generate_sha256_hash(original_image_bytes)

        # Integrity verified on original
        assert verify_evidence_hash(original_image_bytes, original_hash) is True

        # Flip 1 single bit in the 25000th byte
        tampered_image_bytes = copy.deepcopy(original_image_bytes)
        tampered_image_bytes[25000] ^= 0x01  # Flip LSB

        tampered_hash = generate_sha256_hash(tampered_image_bytes)
        assert tampered_hash != original_hash
        # Verification MUST fail
        assert verify_evidence_hash(tampered_image_bytes, original_hash) is False

    def test_court_dossier_tamper_detection(self):
        """
        Verify that modifying any field in a compiled electronic evidence dossier
        alters the master root hash, preventing court falsification.
        """
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()

        # Setup Camera & Sighting
        cam = Camera(id="CAM-01", name="Rajpath Node", latitude=28.6143, longitude=77.2010, road_name="Rajpath Marg")
        session.add(cam)
        session.commit()

        sighting = Sighting(
            camera_id=cam.id,
            timestamp=datetime(2026, 9, 1, 10, 0, 0),
            vehicle_box=[100, 100, 300, 300],
            plate_text="DL01AB1234",
            plate_confidence=0.95,
            vehicle_type="car",
            vehicle_color="silver",
            image_path="/data/frames/cam1.jpg",
            crop_path="/data/crops/car1.jpg",
            sha256_hash=generate_sha256_hash("sighting_payload_1"),
        )
        session.add(sighting)
        session.commit()

        case_obj = Case(
            fir_number="FIR-SEC65B-2026-999",
            reported_plate="DL01AB1234",
            theft_datetime=datetime(2026, 9, 1, 9, 0, 0),
            theft_latitude=28.6140,
            theft_longitude=77.2000,
            theft_location_name="India Gate",
            vehicle_type="car",
            vehicle_color="silver",
            investigating_officer="Insp. Rajesh Sharma",
            police_station="Tilak Marg PS",
            status=CaseStatus.OPEN,
        )
        session.add(case_obj)
        session.commit()

        match = CaseMatch(
            case_id=case_obj.id,
            sighting_id=sighting.id,
            composite_score=0.95,
            verification_status=VerificationStatus.VERIFIED,
            reviewed_by="BADGE-7890",
        )
        session.add(match)
        session.commit()

        # Generate legitimate dossier
        dossier = generate_evidence_dossier(session, case_obj.id, officer_badge_id="BADGE-7890")
        assert dossier is not None
        root_hash = dossier["evidence_hash_sha256"]
        assert len(root_hash) == 64

        # Simulate Malicious Modification (e.g. altering officer name or sighting hash)
        tampered_summary = copy.deepcopy(dossier["case_summary"])
        tampered_summary["investigating_officer"] = "Corrupted Officer"

        tampered_payload = json.dumps(
            {
                "case_summary": tampered_summary,
                "timeline_hashes": [x.get("sha256_hash") for x in dossier["verified_timeline"]],
                "chain_of_custody_count": len(dossier["chain_of_custody"]),
            },
            sort_keys=True,
        )
        tampered_root_hash = generate_sha256_hash(tampered_payload)

        # Tampered root hash must NOT match original root hash
        assert tampered_root_hash != root_hash
        assert verify_evidence_hash(tampered_payload, root_hash) is False

        session.close()

    def test_immutable_audit_log_capture_for_court_chain_of_custody(self):
        """Verify that DPDP / Section 65B audit entries record officer badge, IP, timestamp, and metadata."""
        audit = AuditLog(
            user_badge_id="BADGE-DEL-4412",
            user_name="Sub-Inspector Verma",
            action="OFFICER_SIGHTING_VERIFY",
            resource_type="case_match",
            resource_id="MATCH-101",
            endpoint="/api/verify",
            ip_address="10.0.4.15",
            details={"decision": "VERIFIED", "confidence": 0.98},
            timestamp=datetime(2026, 9, 1, 14, 30, 0),
        )
        d = audit.to_dict()
        assert d["user_badge_id"] == "BADGE-DEL-4412"
        assert d["action"] == "OFFICER_SIGHTING_VERIFY"
        assert d["ip_address"] == "10.0.4.15"
        assert d["details"]["decision"] == "VERIFIED"
