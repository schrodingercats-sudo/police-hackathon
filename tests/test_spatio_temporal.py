"""
Test Suite: Spatio-Temporal Association, Haversine Kinematics & Route Reconstruction.
Standard: ISO/IEC/IEEE 29119 & Kinematic Feasibility Constraints
Author: E2E Test Suite Architect (M0)
"""

import math
from datetime import datetime, timedelta
import pytest

try:
    from backend.services.spatio_temporal import (
        haversine_distance_km,
        modified_levenshtein_similarity,
        compute_cosine_similarity,
        compute_composite_matching_score,
        validate_spatio_temporal_transition,
        reconstruct_route_dag,
        OCR_CONFUSION_MAP
    )
    SPATIO_AVAILABLE = True
except ImportError:
    try:
        from stolen_vehicle_ai.backend.services.spatio_temporal import (
            haversine_distance_km,
            modified_levenshtein_similarity,
            compute_cosine_similarity,
            compute_composite_matching_score,
            validate_spatio_temporal_transition,
            reconstruct_route_dag,
            OCR_CONFUSION_MAP
        )
        SPATIO_AVAILABLE = True
    except ImportError:
        SPATIO_AVAILABLE = False


class TestHaversineDistance:
    """Tier 1: Geodesic great-circle distance math."""

    def test_zero_distance_same_point(self):
        """Verify identical coordinates return 0.0 km."""
        if not SPATIO_AVAILABLE:
            pytest.skip("Spatio-temporal service not yet available")
        d = haversine_distance_km(28.6139, 77.2090, 28.6139, 77.2090)
        assert abs(d - 0.0) < 1e-6

    def test_known_delhi_ncr_distance_benchmark(self):
        """Verify distance between Connaught Place and ITO Junction (~2.38 km)."""
        if not SPATIO_AVAILABLE:
            pytest.skip("Spatio-temporal service not yet available")
        # Connaught Place: 28.6315, 77.2167 -> ITO Junction: 28.6289, 77.2410
        d = haversine_distance_km(28.6315, 77.2167, 28.6289, 77.2410)
        assert 2.0 <= d <= 3.0, f"Expected ~2.38 km, got {d}"

    def test_delhi_to_noida_pari_chowk_distance(self):
        """Verify distance between Connaught Place and Pari Chowk Greater Noida (~34-38 km)."""
        if not SPATIO_AVAILABLE:
            pytest.skip("Spatio-temporal service not yet available")
        d = haversine_distance_km(28.6315, 77.2167, 28.4650, 77.5100)
        assert 30.0 <= d <= 45.0, f"Expected ~35 km, got {d}"


class TestSpatioTemporalFeasibility:
    """Tier 1 & 2: Velocity thresholding (v <= 120 km/h) and impossible transitions."""

    def test_valid_urban_commute_feasible(self):
        """5 km in 10 minutes = 30 km/h -> Feasible."""
        if not SPATIO_AVAILABLE:
            pytest.skip("Spatio-temporal service not yet available")
        t1 = datetime(2026, 9, 1, 8, 30, 0)
        t2 = datetime(2026, 9, 1, 8, 40, 0)
        feasible, dist, dt, speed = validate_spatio_temporal_transition(
            28.6315, 77.2167, t1,
            28.6289, 77.2410, t2,
            max_speed_kmh=120.0
        )
        assert feasible is True
        assert speed < 120.0
        assert dt == 600.0

    def test_super_speed_rejection(self):
        """30 km in 2 minutes = 900 km/h -> Infeasible (Rejection)."""
        if not SPATIO_AVAILABLE:
            pytest.skip("Spatio-temporal service not yet available")
        t1 = datetime(2026, 9, 1, 8, 30, 0)
        t2 = datetime(2026, 9, 1, 8, 32, 0)
        # Connaught Place to Pari Chowk (~35 km) in 2 mins
        feasible, dist, dt, speed = validate_spatio_temporal_transition(
            28.6315, 77.2167, t1,
            28.4650, 77.5100, t2,
            max_speed_kmh=120.0
        )
        assert feasible is False
        assert speed > 120.0

    def test_instantaneous_displacement_teleportation_rejected(self):
        """Distance > 1 km at delta_t = 0 -> Infeasible."""
        if not SPATIO_AVAILABLE:
            pytest.skip("Spatio-temporal service not yet available")
        t = datetime(2026, 9, 1, 8, 30, 0)
        feasible, dist, dt, speed = validate_spatio_temporal_transition(
            28.6315, 77.2167, t,
            28.6289, 77.2410, t,
            max_speed_kmh=120.0
        )
        assert feasible is False


class TestModifiedLevenshteinSimilarity:
    """Tier 1 & 2: String matching with OCR confusion discount."""

    def test_exact_match(self):
        """Identical plates return similarity 1.0."""
        if not SPATIO_AVAILABLE:
            pytest.skip("Spatio-temporal service not yet available")
        sim = modified_levenshtein_similarity("MH12AB1234", "MH12AB1234")
        assert sim == 1.0

    def test_ocr_confusion_discounted_penalty(self):
        """Substitution of O for 0 incurs less penalty than random letter."""
        if not SPATIO_AVAILABLE:
            pytest.skip("Spatio-temporal service not yet available")
        # Target: MH12AB1230 vs OCR: MH12AB123O (O instead of 0)
        sim_confusion = modified_levenshtein_similarity("MH12AB1230", "MH12AB123O")
        # Target: MH12AB1230 vs OCR: MH12AB123X (X instead of 0)
        sim_random = modified_levenshtein_similarity("MH12AB1230", "MH12AB123X")

        assert sim_confusion > sim_random, "OCR confusion pair should score higher than arbitrary substitution"
        assert sim_confusion >= 0.90


class TestCompositeMatchingScore:
    """Tier 1 & 2: Multi-modal composite scoring with dynamic weights."""

    def test_composite_score_clear_plate_and_visual_match(self, target_embedding):
        """High plate confidence + matching visual Re-ID -> Composite > 0.90."""
        if not SPATIO_AVAILABLE:
            pytest.skip("Spatio-temporal service not yet available")

        score, breakdown = compute_composite_matching_score(
            reported_plate="MH12AB1234",
            sighting_plate="MH12AB1234",
            plate_confidence=0.98,
            reference_embedding=target_embedding,
            sighting_embedding=target_embedding,
            reported_type="car",
            sighting_type="car",
            reported_color="white",
            sighting_color="white"
        )
        assert 0.90 <= score <= 1.0
        assert breakdown["plate_score"] >= 0.95
        assert breakdown["visual_score"] >= 0.95
        assert breakdown["attribute_score"] == 1.0

    def test_composite_score_occluded_plate_reid_fallback(self, target_embedding):
        """Mud on plate (no plate) but matching Re-ID + Color/Type -> Composite retained > 0.70."""
        if not SPATIO_AVAILABLE:
            pytest.skip("Spatio-temporal service not yet available")

        score, breakdown = compute_composite_matching_score(
            reported_plate="MH12AB1234",
            sighting_plate=None,
            plate_confidence=0.0,
            reference_embedding=target_embedding,
            sighting_embedding=target_embedding,
            reported_type="car",
            sighting_type="car",
            reported_color="white",
            sighting_color="white"
        )
        assert score >= 0.70, "Visual Re-ID + attributes must sustain match when plate is missing"
        assert breakdown["visual_score"] >= 0.95

    def test_composite_score_distractor_vehicle_low_score(self, target_embedding, distractor_embeddings):
        """Different plate, distinct embedding, different color -> Composite < 0.40."""
        if not SPATIO_AVAILABLE:
            pytest.skip("Spatio-temporal service not yet available")

        score, breakdown = compute_composite_matching_score(
            reported_plate="MH12AB1234",
            sighting_plate="DL01XY9999",
            plate_confidence=0.95,
            reference_embedding=target_embedding,
            sighting_embedding=distractor_embeddings[0],
            reported_type="car",
            sighting_type="truck",
            reported_color="white",
            sighting_color="black"
        )
        assert score < 0.40, f"Distractor score must be low, got {score}"


class TestRouteDAGReconstruction:
    """Tier 1: Chronological route trajectory building."""

    def test_reconstruct_ordered_route_segments(self, camera_nodes_data):
        """Verify reconstruct_route_dag builds ordered waypoints and transitions."""
        if not SPATIO_AVAILABLE:
            pytest.skip("Spatio-temporal service not yet available")

        base_time = datetime(2026, 9, 1, 8, 30, 0)
        sightings = [
            {
                "sighting_id": 1,
                "camera_id": camera_nodes_data[0]["id"],
                "camera_name": camera_nodes_data[0]["name"],
                "latitude": camera_nodes_data[0]["latitude"],
                "longitude": camera_nodes_data[0]["longitude"],
                "timestamp": base_time + timedelta(minutes=5),
                "composite_score": 0.96
            },
            {
                "sighting_id": 2,
                "camera_id": camera_nodes_data[1]["id"],
                "camera_name": camera_nodes_data[1]["name"],
                "latitude": camera_nodes_data[1]["latitude"],
                "longitude": camera_nodes_data[1]["longitude"],
                "timestamp": base_time + timedelta(minutes=14),
                "composite_score": 0.94
            },
            {
                "sighting_id": 3,
                "camera_id": camera_nodes_data[2]["id"],
                "camera_name": camera_nodes_data[2]["name"],
                "latitude": camera_nodes_data[2]["latitude"],
                "longitude": camera_nodes_data[2]["longitude"],
                "timestamp": base_time + timedelta(minutes=22),
                "composite_score": 0.91
            },
        ]

        route = reconstruct_route_dag(sightings)
        assert route is not None
        assert len(route["waypoints"]) == 3
        assert len(route["segments"]) == 2
        assert route["total_distance_km"] > 0
        assert all(seg["feasible"] is True for seg in route["segments"])
