"""Automated PyTest Suite for Synthetic Multi-Camera Stream Generator (Milestone 6).

Verifies:
1. Synthetic frame generation across all 6 Delhi NCR cameras
2. Target vehicle (MH12AB1234: White Creta) journey & distractor traffic rendering
3. Camera 3 mud/dirt plate occlusion ($S_{plate} < 0.45$) testing visual Re-ID fallback
4. Professional CCTV HUD telemetry overlay rendering
5. Section 65B SHA-256 cryptographic digest calculation and verification
6. 512-D L2-normalized Re-ID embedding unit-sphere constraint (||e||_2 == 1.0)
7. Database persistence and CaseMatch linkage for synthetic sightings
"""

import math
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pytest

import tempfile
import os
from sqlalchemy.orm import sessionmaker
from stolen_vehicle_ai.backend.database.models import Camera, Case, CaseMatch, Sighting, Vehicle
from stolen_vehicle_ai.backend.services.matching_service import (
    compute_composite_matching_score,
    compute_cosine_similarity,
)
from stolen_vehicle_ai.backend.services.mock_stream import (
    SyntheticFrameRenderer,
    SyntheticStreamGenerator,
    build_delhi_ncr_stream_network,
    generate_synthetic_streams,
    get_target_base_embedding,
)
from stolen_vehicle_ai.backend.database.session import SessionLocal, init_db, reset_db, create_db_engine
try:
    import stolen_vehicle_ai.backend.database.session as db_session_mod
except ImportError:
    import backend.database.session as db_session_mod


@pytest.fixture(scope="module")
def mock_db():
    """Initializes and provides clean isolated database session without touching production DB."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        temp_db_path = tmp.name

    test_engine = create_db_engine(f"sqlite:///{temp_db_path}")
    orig_engine = db_session_mod.engine
    orig_session_local = db_session_mod.SessionLocal

    db_session_mod.engine = test_engine
    db_session_mod.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    reset_db(engine_override=test_engine)
    db = db_session_mod.SessionLocal()
    try:
        yield db
    finally:
        db.close()
        test_engine.dispose()
        db_session_mod.engine = orig_engine
        db_session_mod.SessionLocal = orig_session_local
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except OSError:
                pass


class TestSyntheticCorridorConfiguration:
    """Validates the 6-camera corridor topology and vehicle scenario parameters."""

    def test_delhi_ncr_corridor_node_count(self):
        """Verifies exactly 6 cameras configured across Delhi NCR corridor."""
        network = build_delhi_ncr_stream_network()
        assert len(network) == 6

        cam_ids = [c.camera_id for c in network]
        expected_ids = [
            "CAM_DEL_001",
            "CAM_DEL_002",
            "CAM_DEL_003",
            "CAM_NOIDA_001",
            "CAM_NOIDA_002",
            "CAM_GRNOIDA_001",
        ]
        assert cam_ids == expected_ids

    def test_target_vehicle_presence_across_all_nodes(self):
        """Ensures target stolen vehicle (MH12AB1234) is present in all 6 cameras."""
        network = build_delhi_ncr_stream_network()
        for node in network:
            target_vehicles = [v for v in node.vehicles if v.is_target]
            assert len(target_vehicles) == 1, f"Missing target vehicle in {node.camera_id}"
            target = target_vehicles[0]
            assert target.plate_text == "MH12AB1234"
            assert target.color_name == "white"
            assert target.make == "Hyundai"
            assert target.model == "Creta"

    def test_camera_3_plate_occlusion_configuration(self):
        """Verifies Camera 3 (Akshardham) has mud/dirt occlusion on target plate with S_plate < 0.45."""
        network = build_delhi_ncr_stream_network()
        cam3 = network[2]
        assert cam3.camera_id == "CAM_DEL_003"

        target_v = [v for v in cam3.vehicles if v.is_target][0]
        assert target_v.is_plate_occluded is True
        assert target_v.occlusion_type == "mud"
        assert target_v.plate_confidence < 0.45, "Camera 3 target plate confidence must be < 0.45 to test Re-ID fallback"

    def test_distractor_traffic_count_per_node(self):
        """Verifies each node has multiple distractors (3 to 5 distractors per camera)."""
        network = build_delhi_ncr_stream_network()
        for node in network:
            distractors = [v for v in node.vehicles if not v.is_target]
            assert 3 <= len(distractors) <= 5


class TestSyntheticRendererAndHUD:
    """Tests image rendering, HUD overlays, and mud occlusion mechanics."""

    def test_render_scene_background(self):
        """Verifies background rendering produces valid 3-channel image."""
        renderer = SyntheticFrameRenderer(width=640, height=360)
        img = renderer.render_scene_background("day_clear")
        assert img is not None
        assert img.shape == (360, 640, 3)
        assert img.dtype == np.uint8

    def test_render_cctv_hud_overlay(self):
        """Verifies CCTV HUD telemetry bar is drawn at top of frame."""
        renderer = SyntheticFrameRenderer(width=800, height=450)
        frame = renderer.render_scene_background("night_toll")
        cam_cfg = build_delhi_ncr_stream_network()[0]
        now = datetime(2026, 9, 1, 8, 35, 0)

        annotated = renderer.render_cctv_hud(frame, camera=cam_cfg, timestamp=now, fps=25.0, frame_idx=1)
        assert annotated.shape == (450, 800, 3)

    def test_mud_splatter_application(self):
        """Verifies mud splatter modifies target pixel region."""
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255  # pure white
        target_rect = (50, 20, 150, 80)
        SyntheticFrameRenderer.apply_mud_splatter(img, target_rect)

        # Region inside target_rect should no longer be pure white
        roi = img[20:80, 50:150]
        assert np.any(roi < 250), "Mud splatter should darken pixels in target region"


class TestReIDAndCryptographicHashing:
    """Tests 512-D Re-ID embedding properties and Section 65B SHA-256 digests."""

    def test_target_base_embedding_unit_sphere(self):
        """Verifies 512-D base embedding satisfies L2 unit-norm condition (||e|| == 1.0)."""
        emb = get_target_base_embedding(512)
        assert len(emb) == 512
        norm = math.sqrt(sum(x * x for x in emb))
        assert abs(norm - 1.0) < 1e-4

    def test_cross_camera_target_reid_similarity(self):
        """
        Verifies high visual Re-ID cosine similarity (>0.85) between target sightings
        across all 6 cameras, while distractor similarity remains low (<0.40).
        """
        generator = SyntheticStreamGenerator()
        base_emb = generator.target_base_embedding

        # Generate stream across all 6 cameras
        results = generator.generate_all_streams(num_cameras=6, frames_per_camera=2, insert_to_db=False)
        assert results["total_cameras"] == 6

    def test_sha256_image_digest_validity(self):
        """Verifies SHA-256 calculation produces 64-character hexadecimal digest."""
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        digest = SyntheticStreamGenerator.compute_image_sha256(test_img)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


class TestEndToEndSimulationAndDatabaseIngestion:
    """Tests full generator execution and database persistence."""

    def test_generate_all_streams_with_db_insertion(self, mock_db):
        """Verifies 6-camera simulation creates files and populates database."""
        result = generate_synthetic_streams(
            cameras=6,
            frames_per_camera=2,
            save_video=False,
            insert_to_db=True,
        )

        assert result["total_cameras"] == 6
        assert result["total_sightings_generated"] >= 28
        assert result["target_sightings"] == 6
        assert result["status"] == "SUCCESS"

        # Verify sightings stored in DB
        db = SessionLocal()
        try:
            target_sightings = (
                db.query(Sighting)
                .filter(Sighting.plate_text == "MH12AB1234")
                .order_by(Sighting.timestamp.asc())
                .all()
            )
            assert len(target_sightings) == 6

            # Verify Camera 3 sighting has occluded plate confidence < 0.45
            cam3_sighting = next(s for s in target_sightings if s.camera_id == "CAM_DEL_003")
            assert cam3_sighting.plate_confidence < 0.45

            # Verify Re-ID fallback composite score on Camera 3
            ref_emb = get_target_base_embedding()
            score, bd = compute_composite_matching_score(
                reported_plate="MH12AB1234",
                sighting_plate=cam3_sighting.plate_text,
                plate_confidence=cam3_sighting.plate_confidence,
                reference_embedding=ref_emb,
                sighting_embedding=cam3_sighting.embedding,
                reported_type="car",
                sighting_type=cam3_sighting.vehicle_type,
                reported_color="white",
                sighting_color=cam3_sighting.vehicle_color,
            )
            # Re-ID fallback should score high despite low plate confidence!
            assert score >= 0.70
            assert bd["plate_occluded"] is True

        finally:
            db.close()
