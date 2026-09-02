"""Synthetic Multi-Camera Video and Image Stream Generator for Indian Police AI System.

Simulates a 6-camera CCTV surveillance corridor across Delhi NCR:
1. CAM_DEL_001 (Connaught Place): Clear daylight, target White Creta (MH12AB1234) + 4 distractors.
2. CAM_DEL_002 (ITO Junction): Heavy traffic, target vehicle with partial angle + 5 distractors.
3. CAM_DEL_003 (Akshardham Flyover): Mud/dirt occlusion on plate (S_plate < 0.45, tests Re-ID fallback!) + 4 distractors.
4. CAM_NOIDA_001 (DND Toll Plaza): Night/toll lighting, high speed + 3 distractors.
5. CAM_NOIDA_002 (Noida Expressway): Distant zoom perspective + 4 distractors.
6. CAM_GRNOIDA_001 (Pari Chowk): Slow traffic roundabout, high resolution + 5 distractors.

Renders realistic synthetic scenes using OpenCV and Pillow:
- Road asphalt, lane markings, curbs, perspective geometry, lighting conditions (day, overcast, night toll, dusk)
- Realistic vehicle silhouettes with body colors, windshields, headlights, wheels, and Indian HSRP license plates
- Mud splatter occlusion rendering for testing visual Re-ID fallback
- Professional CCTV HUD overlay: Camera ID, Name, GPS lat/lon, live timestamp, frame rate, SEC-65B watermark
- Section 65B SHA-256 cryptographic hashing and 512-D L2-normalized Re-ID metric embeddings
- Saves frame snapshots, vehicle crops, and plate crops to disk and optionally populates the database
"""

import argparse
import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from backend.config import settings
    from backend.database.models import (
        Camera,
        CameraStatus,
        CameraTopology,
        Case,
        CaseMatch,
        CaseStatus,
        Sighting,
        Vehicle,
        VerificationStatus,
    )
    from backend.database.session import SessionLocal, init_db
    from backend.services.matching_service import (
        compute_composite_matching_score,
        compute_cosine_similarity,
        modified_levenshtein_similarity,
    )
    from vision.reid_extractor import DominantColorClassifier, ReIDFeatureExtractor
except ImportError:
    from stolen_vehicle_ai.backend.config import settings
    from stolen_vehicle_ai.backend.database.models import (
        Camera,
        CameraStatus,
        CameraTopology,
        Case,
        CaseMatch,
        CaseStatus,
        Sighting,
        Vehicle,
        VerificationStatus,
    )
    from stolen_vehicle_ai.backend.database.session import SessionLocal, init_db
    from stolen_vehicle_ai.backend.services.matching_service import (
        compute_composite_matching_score,
        compute_cosine_similarity,
        modified_levenshtein_similarity,
    )
    from stolen_vehicle_ai.vision.reid_extractor import DominantColorClassifier, ReIDFeatureExtractor

logger = logging.getLogger("stolen_vehicle_ai.mock_stream")


# -----------------------------------------------------------------------------
# Data Models for Synthetic Stream Configuration
# -----------------------------------------------------------------------------

@dataclass
class SyntheticVehicleConfig:
    """Configuration for a vehicle rendered in a synthetic CCTV stream."""
    plate_text: str
    vehicle_type: str  # "car", "motorcycle", "auto_rickshaw", "bus", "truck"
    color_name: str  # "white", "black", "silver/grey", "red", "blue", "yellow", etc.
    rgb_color: Tuple[int, int, int]
    make: str
    model: str
    is_target: bool = False
    is_plate_occluded: bool = False
    occlusion_type: Optional[str] = None  # "mud", "dirt", "blur", None
    plate_confidence: float = 0.95
    speed_kmh: float = 50.0
    relative_position: Tuple[float, float] = (0.5, 0.6)  # (rel_x, rel_y) in frame [0.0 - 1.0]
    scale: float = 1.0
    angle_deg: float = 0.0


@dataclass
class CameraNodeConfig:
    """Configuration for a surveillance camera node in the synthetic corridor."""
    camera_id: str
    name: str
    latitude: float
    longitude: float
    road_name: str
    direction_bearing: float
    camera_type: str  # "urban", "junction", "highway", "toll"
    lighting: str  # "day_clear", "day_traffic", "overcast_muddy", "night_toll", "day_expressway", "day_roundabout"
    base_timestamp: datetime
    vehicles: List[SyntheticVehicleConfig] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Color Constants & Palette Mapping
# -----------------------------------------------------------------------------

COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "white": (245, 245, 248),
    "black": (30, 30, 35),
    "silver/grey": (180, 185, 190),
    "red": (210, 35, 40),
    "blue": (30, 90, 200),
    "yellow": (235, 190, 20),
    "yellow_green": (220, 180, 30),
    "green": (35, 140, 60),
    "orange": (230, 110, 25),
    "brown": (110, 65, 40),
}


# -----------------------------------------------------------------------------
# Deterministic Target L2-Normalized Embedding
# -----------------------------------------------------------------------------

def get_target_base_embedding(dim: int = 512) -> List[float]:
    """Generates the reference L2-normalized 512-D embedding for White Hyundai Creta."""
    rng = np.random.RandomState(101)
    raw = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(raw)
    return (raw / norm).tolist()


def perturb_embedding(base_emb: List[float], noise_std: float = 0.04, seed: int = 42) -> List[float]:
    """Applies small environmental perturbation while maintaining L2-normalization."""
    rng = np.random.RandomState(seed)
    arr = np.asarray(base_emb, dtype=np.float32)
    noise = rng.randn(len(arr)).astype(np.float32) * noise_std
    perturbed = arr + noise
    norm = np.linalg.norm(perturbed)
    if norm < 1e-6:
        return base_emb
    return (perturbed / norm).tolist()


# -----------------------------------------------------------------------------
# Synthetic Frame Renderer
# -----------------------------------------------------------------------------

class SyntheticFrameRenderer:
    """
    Renders photorealistic synthetic CCTV camera frames with:
    - Asphalt road with realistic vanishing-point perspective and lane markings
    - Ambient environmental lighting (daylight, dusk, rain/overcast, night toll gate)
    - Detailed vehicle silhouettes (body, windows, headlights, wheels, shadows)
    - Indian High Security Registration Plates (HSRP) with blue IND strip
    - Mud and dirt splatter occlusion for testing visual Re-ID fallback
    - Professional Police CCTV HUD telemetry overlay
    """

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

    def render_scene_background(self, lighting: str) -> np.ndarray:
        """Renders the road environment and background according to lighting condition."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Environmental colors based on lighting
        if lighting == "night_toll":
            sky_color = (15, 18, 25)
            ground_color = (25, 28, 32)
            road_color = (40, 42, 45)
            lane_color = (200, 200, 200)
            curb_color = (70, 70, 75)
        elif lighting == "overcast_muddy":
            sky_color = (130, 135, 140)
            ground_color = (85, 90, 75)
            road_color = (65, 68, 70)
            lane_color = (180, 180, 175)
            curb_color = (90, 85, 80)
        elif lighting == "day_expressway":
            sky_color = (220, 195, 160)  # Bright blue/sunny (BGR)
            ground_color = (110, 130, 95)
            road_color = (75, 78, 80)
            lane_color = (240, 240, 245)
            curb_color = (130, 130, 135)
        else:  # day_clear, day_traffic, day_roundabout
            sky_color = (210, 180, 140)
            ground_color = (95, 115, 80)
            road_color = (70, 72, 75)
            lane_color = (235, 235, 240)
            curb_color = (120, 120, 125)

        # 1. Sky / Horizon
        horizon_y = int(self.height * 0.32)
        cv2.rectangle(img, (0, 0), (self.width, horizon_y), sky_color, -1)

        # 2. Roadside ground / greenery
        cv2.rectangle(img, (0, horizon_y), (self.width, self.height), ground_color, -1)

        # 3. Perspective Road trapezoid
        vp_x = self.width // 2
        vp_y = horizon_y
        road_pts = np.array([
            [int(self.width * 0.38), vp_y],
            [int(self.width * 0.62), vp_y],
            [int(self.width * 0.95), self.height],
            [int(self.width * 0.05), self.height],
        ], dtype=np.int32)
        cv2.fillPoly(img, [road_pts], road_color)

        # 4. Curbs / Road borders
        cv2.line(img, (int(self.width * 0.38), vp_y), (int(self.width * 0.05), self.height), curb_color, 4)
        cv2.line(img, (int(self.width * 0.62), vp_y), (int(self.width * 0.95), self.height), curb_color, 4)

        # 5. Lane markings (Center dashed lines)
        num_dashes = 10
        for i in range(num_dashes):
            t1 = i / float(num_dashes)
            t2 = (i + 0.5) / float(num_dashes)
            y1 = int(vp_y + (self.height - vp_y) * (t1 ** 1.8))
            y2 = int(vp_y + (self.height - vp_y) * (t2 ** 1.8))
            x1 = vp_x
            x2 = vp_x
            thickness = max(2, int(1 + 6 * t2))
            cv2.line(img, (x1, y1), (x2, y2), lane_color, thickness)

        # 6. Night toll or street lighting effects
        if lighting == "night_toll":
            # Add yellow toll canopy lights & road illumination cones
            for light_x in [int(self.width * 0.25), int(self.width * 0.50), int(self.width * 0.75)]:
                overlay = img.copy()
                cv2.circle(overlay, (light_x, vp_y + 40), 90, (120, 220, 255), -1)
                cv2.ellipse(overlay, (light_x, int(self.height * 0.70)), (140, 200), 0, 0, 360, (60, 140, 180), -1)
                cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        return img

    def render_vehicle(
        self,
        frame: np.ndarray,
        v_cfg: SyntheticVehicleConfig,
        box_rect: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Tuple[int, int, int, int], Optional[Tuple[int, int, int, int]]]:
        """
        Renders a vehicle onto the frame within box_rect = (x1, y1, x2, y2).
        Returns: (annotated_frame, vehicle_crop, plate_crop, vehicle_bbox, plate_bbox)
        """
        x1, y1, x2, y2 = box_rect
        w = max(40, x2 - x1)
        h = max(30, y2 - y1)

        # 1. Soft ground shadow
        shadow_overlay = frame.copy()
        shadow_center = (x1 + w // 2, y2 - int(h * 0.05))
        shadow_axes = (int(w * 0.55), int(h * 0.15))
        cv2.ellipse(shadow_overlay, shadow_center, shadow_axes, 0, 0, 360, (15, 15, 15), -1)
        cv2.addWeighted(shadow_overlay, 0.5, frame, 0.5, 0, frame)

        # 2. Vehicle Body Main Hull
        body_bgr = (v_cfg.rgb_color[2], v_cfg.rgb_color[1], v_cfg.rgb_color[0])
        body_y1 = y1 + int(h * 0.25)
        body_y2 = y2 - int(h * 0.10)
        body_x1 = x1 + int(w * 0.05)
        body_x2 = x2 - int(w * 0.05)

        # Draw rounded main body
        cv2.rectangle(frame, (body_x1, body_y1), (body_x2, body_y2), body_bgr, -1)
        cv2.rectangle(frame, (body_x1, body_y1), (body_x2, body_y2), (40, 40, 40), 2)

        # 3. Cabin / Roof & Windshield
        if v_cfg.vehicle_type in ["car", "bus", "truck"]:
            cabin_x1 = x1 + int(w * 0.15)
            cabin_x2 = x2 - int(w * 0.15)
            cabin_y1 = y1 + int(h * 0.05)
            cabin_y2 = y1 + int(h * 0.40)
            cabin_pts = np.array([
                [cabin_x1 + int(w * 0.08), cabin_y1],
                [cabin_x2 - int(w * 0.08), cabin_y1],
                [cabin_x2, cabin_y2],
                [cabin_x1, cabin_y2],
            ], dtype=np.int32)
            # Cabin body
            cv2.fillPoly(frame, [cabin_pts], body_bgr)
            cv2.polylines(frame, [cabin_pts], True, (40, 40, 40), 2)

            # Windshield glass (tinted dark blue-grey)
            glass_pts = np.array([
                [cabin_x1 + int(w * 0.10), cabin_y1 + 4],
                [cabin_x2 - int(w * 0.10), cabin_y1 + 4],
                [cabin_x2 - int(w * 0.03), cabin_y2 - 3],
                [cabin_x1 + int(w * 0.03), cabin_y2 - 3],
            ], dtype=np.int32)
            cv2.fillPoly(frame, [glass_pts], (60, 50, 40))

            # Target Creta special feature: black roof wrap
            if v_cfg.is_target:
                cv2.line(frame, (cabin_x1 + int(w * 0.08), cabin_y1), (cabin_x2 - int(w * 0.08), cabin_y1), (20, 20, 20), 4)

        elif v_cfg.vehicle_type == "auto_rickshaw":
            # Yellow roof on green body
            roof_y1 = y1 + int(h * 0.05)
            roof_y2 = y1 + int(h * 0.35)
            cv2.rectangle(frame, (x1 + int(w * 0.10), roof_y1), (x2 - int(w * 0.10), roof_y2), (20, 190, 235), -1)
            # Front open windshield
            cv2.rectangle(frame, (x1 + int(w * 0.15), roof_y2), (x2 - int(w * 0.15), body_y1), (50, 50, 50), -1)

        # 4. Wheels
        wheel_radius = max(4, int(h * 0.10))
        wheel_y = y2 - int(h * 0.10)
        wheel_left_x = x1 + int(w * 0.20)
        wheel_right_x = x2 - int(w * 0.20)
        cv2.circle(frame, (wheel_left_x, wheel_y), wheel_radius, (20, 20, 20), -1)
        cv2.circle(frame, (wheel_right_x, wheel_y), wheel_radius, (20, 20, 20), -1)
        cv2.circle(frame, (wheel_left_x, wheel_y), max(2, wheel_radius // 2), (160, 160, 160), -1)
        cv2.circle(frame, (wheel_right_x, wheel_y), max(2, wheel_radius // 2), (160, 160, 160), -1)

        # 5. Headlights / Taillights
        light_radius = max(3, int(w * 0.04))
        cv2.circle(frame, (body_x1 + light_radius + 2, body_y1 + int(h * 0.12)), light_radius, (240, 240, 240), -1)
        cv2.circle(frame, (body_x2 - light_radius - 2, body_y1 + int(h * 0.12)), light_radius, (240, 240, 240), -1)

        # 6. License Plate Rectangle & Text
        plate_w = max(32, int(w * 0.42))
        plate_h = max(12, int(h * 0.16))
        plate_cx = x1 + w // 2
        plate_cy = body_y2 - int(h * 0.08)

        px1 = max(0, plate_cx - plate_w // 2)
        py1 = max(0, plate_cy - plate_h // 2)
        px2 = min(self.width, px1 + plate_w)
        py2 = min(self.height, py1 + plate_h)

        plate_rect = (px1, py1, px2, py2)

        # Draw HSRP white plate background
        cv2.rectangle(frame, (px1, py1), (px2, py2), (250, 250, 250), -1)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (20, 20, 20), 1)

        # Blue IND strip on left side of HSRP
        ind_strip_w = max(3, int(plate_w * 0.12))
        cv2.rectangle(frame, (px1, py1), (px1 + ind_strip_w, py2), (180, 50, 20), -1)

        # Render Plate Text using PIL for crisp font
        plate_text_render = v_cfg.plate_text
        if plate_w >= 30 and plate_h >= 10:
            font_scale = max(0.25, min(0.60, plate_h / 28.0))
            text_x = px1 + ind_strip_w + 2
            text_y = py2 - 3
            cv2.putText(
                frame,
                plate_text_render,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (10, 10, 10),
                1,
                cv2.LINE_AA,
            )

        # 7. Apply Mud / Occlusion if configured (Camera 3 Akshardham test case)
        if v_cfg.is_plate_occluded or v_cfg.occlusion_type in ["mud", "dirt"]:
            self.apply_mud_splatter(frame, plate_rect)

        # Extract crops
        vh_y1 = max(0, y1)
        vh_y2 = min(self.height, y2)
        vh_x1 = max(0, x1)
        vh_x2 = min(self.width, x2)

        vehicle_crop = frame[vh_y1:vh_y2, vh_x1:vh_x2].copy()
        plate_crop = frame[py1:py2, px1:px2].copy() if (py2 > py1 and px2 > px1) else None

        return frame, vehicle_crop, plate_crop, (vh_x1, vh_y1, vh_x2, vh_y2), plate_rect

    @staticmethod
    def apply_mud_splatter(img: np.ndarray, target_rect: Tuple[int, int, int, int]) -> None:
        """Applies realistic brown/mud splatter over a target bounding box (e.g. license plate)."""
        x1, y1, x2, y2 = target_rect
        w = max(10, x2 - x1)
        h = max(6, y2 - y1)

        mud_colors = [
            (45, 75, 110),   # Dark mud brown (BGR)
            (35, 60, 95),    # Clay brown
            (60, 90, 130),   # Wet dirt
            (30, 45, 65),    # Heavy sludge
        ]

        rng = np.random.RandomState(42)
        num_splatters = rng.randint(18, 30)

        for _ in range(num_splatters):
            sx = rng.randint(x1 - 2, x2 + 2)
            sy = rng.randint(y1 - 2, y2 + 2)
            rad = rng.randint(2, max(4, h // 2))
            col = mud_colors[rng.randint(0, len(mud_colors))]
            cv2.circle(img, (sx, sy), rad, col, -1)

        # Heavy mud smear across the plate center
        smear_pts = np.array([
            [x1 + int(w * 0.20), y1 + 2],
            [x2 - int(w * 0.10), y1 + int(h * 0.40)],
            [x2 - int(w * 0.30), y2 - 2],
            [x1 + int(w * 0.10), y2 - 2],
        ], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [smear_pts], (40, 70, 105))
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    def render_cctv_hud(
        self,
        frame: np.ndarray,
        camera: CameraNodeConfig,
        timestamp: datetime,
        fps: float = 25.0,
        frame_idx: int = 1,
    ) -> np.ndarray:
        """
        Renders a professional, court-admissible CCTV telemetry HUD banner.
        Includes Camera ID, Name, GPS coords, Live UTC/IST Timestamp, FPS, and Section 65B watermark.
        """
        hud_h = 42
        # Top semi-transparent banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, hud_h), (15, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, hud_h), (self.width, hud_h), (0, 200, 255), 1)  # Amber accent line

        # 1. Left Telemetry: Camera ID & Name & Location
        cam_info_str = f"CAM: {camera.camera_id} | {camera.name.upper()} | {camera.road_name}"
        cv2.putText(frame, cam_info_str, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

        # GPS & Bearing
        gps_str = f"GPS: {camera.latitude:.4f} N, {camera.longitude:.4f} E | HDG: {camera.direction_bearing:.1f} DEG | TYPE: {camera.camera_type.upper()}"
        cv2.putText(frame, gps_str, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 210, 240), 1, cv2.LINE_AA)

        # 2. Right Telemetry: Timestamp & Recording Status
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " IST"
        ts_w = cv2.getTextSize(ts_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        cv2.putText(frame, ts_str, (self.width - ts_w - 12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Status & Section 65B watermark
        status_str = f"[REC] {fps:.1f} FPS | FRM #{frame_idx:04d} | SEC-65B AUTHENTICATED"
        st_w = cv2.getTextSize(status_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        # Pulsing red recording dot
        cv2.circle(frame, (self.width - st_w - 24, 29), 4, (0, 0, 240), -1)
        cv2.putText(frame, status_str, (self.width - st_w - 12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 255), 1, cv2.LINE_AA)

        return frame


# -----------------------------------------------------------------------------
# Corridor Node Configuration Factory (6 Delhi NCR Cameras)
# -----------------------------------------------------------------------------

def build_delhi_ncr_stream_network(base_date: Optional[datetime] = None) -> List[CameraNodeConfig]:
    """
    Constructs the 6-camera synthetic surveillance corridor across Delhi NCR
    with target vehicle journey (MH12AB1234) and surrounding distractor traffic.
    """
    if base_date is None:
        base_date = datetime(2026, 9, 1, 8, 30, 0)

    # 1. Camera 1: Connaught Place Radial Road 1 (Outer Circle)
    cam1 = CameraNodeConfig(
        camera_id="CAM_DEL_001",
        name="Connaught Place Radial Road 1 (Outer Circle)",
        latitude=28.6315,
        longitude=77.2167,
        road_name="Radial Road 1 / Connaught Circus, New Delhi",
        direction_bearing=45.0,
        camera_type="urban",
        lighting="day_clear",
        base_timestamp=base_date + timedelta(minutes=5),  # 08:35:00
        vehicles=[
            # Target Vehicle (Clear sunny view)
            SyntheticVehicleConfig(
                plate_text="MH12AB1234",
                vehicle_type="car",
                color_name="white",
                rgb_color=COLOR_PALETTE["white"],
                make="Hyundai",
                model="Creta",
                is_target=True,
                plate_confidence=0.98,
                speed_kmh=42.5,
                relative_position=(0.50, 0.65),
                scale=1.1,
            ),
            # Distractor 1: Red Swift
            SyntheticVehicleConfig(
                plate_text="DL01CA4321",
                vehicle_type="car",
                color_name="red",
                rgb_color=COLOR_PALETTE["red"],
                make="Maruti",
                model="Swift",
                plate_confidence=0.96,
                speed_kmh=38.0,
                relative_position=(0.25, 0.55),
                scale=0.9,
            ),
            # Distractor 2: Yellow/Green Auto
            SyntheticVehicleConfig(
                plate_text="DL1RAB5566",
                vehicle_type="auto_rickshaw",
                color_name="yellow_green",
                rgb_color=COLOR_PALETTE["yellow_green"],
                make="Bajaj",
                model="RE Compact",
                plate_confidence=0.92,
                speed_kmh=32.0,
                relative_position=(0.75, 0.60),
                scale=0.85,
            ),
            # Distractor 3: Black Honda City
            SyntheticVehicleConfig(
                plate_text="DL4CAD9012",
                vehicle_type="car",
                color_name="black",
                rgb_color=COLOR_PALETTE["black"],
                make="Honda",
                model="City",
                plate_confidence=0.95,
                speed_kmh=45.0,
                relative_position=(0.35, 0.45),
                scale=0.75,
            ),
            # Distractor 4: Silver Baleno
            SyntheticVehicleConfig(
                plate_text="DL8SB1122",
                vehicle_type="car",
                color_name="silver/grey",
                rgb_color=COLOR_PALETTE["silver/grey"],
                make="Maruti",
                model="Baleno",
                plate_confidence=0.94,
                speed_kmh=40.0,
                relative_position=(0.65, 0.48),
                scale=0.80,
            ),
        ],
    )

    # 2. Camera 2: ITO Junction (Vikas Minar Crossing)
    cam2 = CameraNodeConfig(
        camera_id="CAM_DEL_002",
        name="ITO Junction (Vikas Minar Crossing)",
        latitude=28.6289,
        longitude=77.2410,
        road_name="Vikas Marg / Ring Road, New Delhi",
        direction_bearing=95.0,
        camera_type="junction",
        lighting="day_traffic",
        base_timestamp=base_date + timedelta(minutes=12),  # 08:42:00
        vehicles=[
            # Target Vehicle (Partial angle)
            SyntheticVehicleConfig(
                plate_text="MH12AB1234",
                vehicle_type="car",
                color_name="white",
                rgb_color=COLOR_PALETTE["white"],
                make="Hyundai",
                model="Creta",
                is_target=True,
                plate_confidence=0.94,
                speed_kmh=48.0,
                relative_position=(0.48, 0.62),
                scale=1.05,
                angle_deg=15.0,
            ),
            # Distractor 1: Black Scorpio
            SyntheticVehicleConfig(
                plate_text="HR26DQ9988",
                vehicle_type="car",
                color_name="black",
                rgb_color=COLOR_PALETTE["black"],
                make="Mahindra",
                model="Scorpio",
                plate_confidence=0.92,
                speed_kmh=50.0,
                relative_position=(0.20, 0.58),
                scale=1.1,
            ),
            # Distractor 2: Blue Tata Truck
            SyntheticVehicleConfig(
                plate_text="UP16BT7711",
                vehicle_type="truck",
                color_name="blue",
                rgb_color=COLOR_PALETTE["blue"],
                make="Tata",
                model="Signa",
                plate_confidence=0.90,
                speed_kmh=40.0,
                relative_position=(0.80, 0.68),
                scale=1.3,
            ),
            # Distractor 3: White i10
            SyntheticVehicleConfig(
                plate_text="DL2CAZ1100",
                vehicle_type="car",
                color_name="white",
                rgb_color=COLOR_PALETTE["white"],
                make="Hyundai",
                model="i10",
                plate_confidence=0.93,
                speed_kmh=42.0,
                relative_position=(0.32, 0.46),
                scale=0.75,
            ),
            # Distractor 4: Yellow Auto
            SyntheticVehicleConfig(
                plate_text="DL1VAA3344",
                vehicle_type="auto_rickshaw",
                color_name="yellow",
                rgb_color=COLOR_PALETTE["yellow"],
                make="Bajaj",
                model="Compact",
                plate_confidence=0.88,
                speed_kmh=35.0,
                relative_position=(0.68, 0.48),
                scale=0.70,
            ),
            # Distractor 5: Red Brezza
            SyntheticVehicleConfig(
                plate_text="HR51BK4455",
                vehicle_type="car",
                color_name="red",
                rgb_color=COLOR_PALETTE["red"],
                make="Maruti",
                model="Brezza",
                plate_confidence=0.91,
                speed_kmh=46.0,
                relative_position=(0.42, 0.40),
                scale=0.65,
            ),
        ],
    )

    # 3. Camera 3: Akshardham Setu / Delhi-Meerut Expressway
    # CRITICAL: Mud/dirt occlusion on plate (S_plate < 0.45, tests Re-ID fallback!)
    cam3 = CameraNodeConfig(
        camera_id="CAM_DEL_003",
        name="Akshardham Setu / Delhi-Meerut Expressway",
        latitude=28.6180,
        longitude=77.2790,
        road_name="NH-24 / Delhi-Meerut Expressway, East Delhi",
        direction_bearing=110.0,
        camera_type="highway",
        lighting="overcast_muddy",
        base_timestamp=base_date + timedelta(minutes=21),  # 08:51:00
        vehicles=[
            # Target Vehicle: MUD OCCLUSION ON LICENSE PLATE!
            SyntheticVehicleConfig(
                plate_text="MH12AB1234",
                vehicle_type="car",
                color_name="white",
                rgb_color=COLOR_PALETTE["white"],
                make="Hyundai",
                model="Creta",
                is_target=True,
                is_plate_occluded=True,
                occlusion_type="mud",
                plate_confidence=0.38,  # S_plate < 0.45!
                speed_kmh=58.0,
                relative_position=(0.52, 0.64),
                scale=1.1,
            ),
            # Distractor 1: White WagonR with similar prefix (DL3SCA1234)
            SyntheticVehicleConfig(
                plate_text="DL3SCA1234",
                vehicle_type="car",
                color_name="white",
                rgb_color=COLOR_PALETTE["white"],
                make="Maruti",
                model="WagonR",
                plate_confidence=0.89,
                speed_kmh=54.0,
                relative_position=(0.22, 0.52),
                scale=0.85,
            ),
            # Distractor 2: Silver Innova
            SyntheticVehicleConfig(
                plate_text="UP14ER9900",
                vehicle_type="car",
                color_name="silver/grey",
                rgb_color=COLOR_PALETTE["silver/grey"],
                make="Toyota",
                model="Innova",
                plate_confidence=0.92,
                speed_kmh=60.0,
                relative_position=(0.78, 0.60),
                scale=1.0,
            ),
            # Distractor 3: Blue Nexon
            SyntheticVehicleConfig(
                plate_text="DL9CAB7788",
                vehicle_type="car",
                color_name="blue",
                rgb_color=COLOR_PALETTE["blue"],
                make="Tata",
                model="Nexon",
                plate_confidence=0.91,
                speed_kmh=56.0,
                relative_position=(0.38, 0.44),
                scale=0.72,
            ),
            # Distractor 4: Black Royal Enfield Bullet
            SyntheticVehicleConfig(
                plate_text="HR72AB5678",
                vehicle_type="motorcycle",
                color_name="black",
                rgb_color=COLOR_PALETTE["black"],
                make="Royal Enfield",
                model="Classic 350",
                plate_confidence=0.86,
                speed_kmh=52.0,
                relative_position=(0.65, 0.46),
                scale=0.60,
            ),
        ],
    )

    # 4. Camera 4: DND Flyway Toll Plaza
    cam4 = CameraNodeConfig(
        camera_id="CAM_NOIDA_001",
        name="DND Flyway Toll Plaza (Delhi-Noida Direct)",
        latitude=28.5820,
        longitude=77.3100,
        road_name="Delhi-Noida Direct (DND) Flyway, Toll Plaza",
        direction_bearing=140.0,
        camera_type="toll",
        lighting="night_toll",
        base_timestamp=base_date + timedelta(minutes=28),  # 08:58:00
        vehicles=[
            # Target Vehicle (Fast moving night view)
            SyntheticVehicleConfig(
                plate_text="MH12AB1234",
                vehicle_type="car",
                color_name="white",
                rgb_color=COLOR_PALETTE["white"],
                make="Hyundai",
                model="Creta",
                is_target=True,
                plate_confidence=0.96,
                speed_kmh=68.5,
                relative_position=(0.50, 0.66),
                scale=1.15,
            ),
            # Distractor 1: White Fortuner
            SyntheticVehicleConfig(
                plate_text="UP16AL1200",
                vehicle_type="car",
                color_name="white",
                rgb_color=COLOR_PALETTE["white"],
                make="Toyota",
                model="Fortuner",
                plate_confidence=0.94,
                speed_kmh=65.0,
                relative_position=(0.20, 0.62),
                scale=1.2,
            ),
            # Distractor 2: Grey XUV700
            SyntheticVehicleConfig(
                plate_text="DL12CA8822",
                vehicle_type="car",
                color_name="silver/grey",
                rgb_color=COLOR_PALETTE["silver/grey"],
                make="Mahindra",
                model="XUV700",
                plate_confidence=0.93,
                speed_kmh=70.0,
                relative_position=(0.78, 0.58),
                scale=1.1,
            ),
            # Distractor 3: Yellow Bus
            SyntheticVehicleConfig(
                plate_text="UP16BT8833",
                vehicle_type="bus",
                color_name="yellow",
                rgb_color=COLOR_PALETTE["yellow"],
                make="Ashok Leyland",
                model="Viking",
                plate_confidence=0.89,
                speed_kmh=55.0,
                relative_position=(0.35, 0.42),
                scale=1.35,
            ),
        ],
    )

    # 5. Camera 5: Noida-Greater Noida Expressway (Sector 128)
    cam5 = CameraNodeConfig(
        camera_id="CAM_NOIDA_002",
        name="Noida-Greater Noida Expressway (Sector 128)",
        latitude=28.5020,
        longitude=77.4000,
        road_name="Noida-Greater Noida Expressway, Sector 128",
        direction_bearing=145.0,
        camera_type="highway",
        lighting="day_expressway",
        base_timestamp=base_date + timedelta(minutes=42),  # 09:12:00
        vehicles=[
            # Target Vehicle (Distant zoom)
            SyntheticVehicleConfig(
                plate_text="MH12AB1234",
                vehicle_type="car",
                color_name="white",
                rgb_color=COLOR_PALETTE["white"],
                make="Hyundai",
                model="Creta",
                is_target=True,
                plate_confidence=0.91,
                speed_kmh=76.0,
                relative_position=(0.46, 0.56),
                scale=0.95,
            ),
            # Distractor 1: Silver City with 1-character difference plate (MH12AB1235)
            SyntheticVehicleConfig(
                plate_text="MH12AB1235",
                vehicle_type="car",
                color_name="silver/grey",
                rgb_color=COLOR_PALETTE["silver/grey"],
                make="Honda",
                model="City",
                plate_confidence=0.93,
                speed_kmh=74.0,
                relative_position=(0.76, 0.58),
                scale=0.95,
            ),
            # Distractor 2: White Dzire
            SyntheticVehicleConfig(
                plate_text="DL5SAB3322",
                vehicle_type="car",
                color_name="white",
                rgb_color=COLOR_PALETTE["white"],
                make="Maruti",
                model="Dzire",
                plate_confidence=0.90,
                speed_kmh=72.0,
                relative_position=(0.20, 0.50),
                scale=0.85,
            ),
            # Distractor 3: Black Seltos
            SyntheticVehicleConfig(
                plate_text="HR26CT4411",
                vehicle_type="car",
                color_name="black",
                rgb_color=COLOR_PALETTE["black"],
                make="Kia",
                model="Seltos",
                plate_confidence=0.92,
                speed_kmh=78.0,
                relative_position=(0.34, 0.42),
                scale=0.75,
            ),
            # Distractor 4: Blue Venue
            SyntheticVehicleConfig(
                plate_text="UP16DZ9090",
                vehicle_type="car",
                color_name="blue",
                rgb_color=COLOR_PALETTE["blue"],
                make="Hyundai",
                model="Venue",
                plate_confidence=0.88,
                speed_kmh=75.0,
                relative_position=(0.62, 0.44),
                scale=0.75,
            ),
        ],
    )

    # 6. Camera 6: Pari Chowk Roundabout & Expressway Exit
    cam6 = CameraNodeConfig(
        camera_id="CAM_GRNOIDA_001",
        name="Pari Chowk Roundabout & Expressway Exit",
        latitude=28.4650,
        longitude=77.5100,
        road_name="Pari Chowk Roundabout, Greater Noida",
        direction_bearing=160.0,
        camera_type="junction",
        lighting="day_roundabout",
        base_timestamp=base_date + timedelta(minutes=54),  # 09:24:00
        vehicles=[
            # Target Vehicle (Slow traffic, high resolution)
            SyntheticVehicleConfig(
                plate_text="MH12AB1234",
                vehicle_type="car",
                color_name="white",
                rgb_color=COLOR_PALETTE["white"],
                make="Hyundai",
                model="Creta",
                is_target=True,
                plate_confidence=0.95,
                speed_kmh=52.0,
                relative_position=(0.50, 0.68),
                scale=1.2,
            ),
            # Distractor 1: Red Creta (same model, different color!)
            SyntheticVehicleConfig(
                plate_text="UP16BQ1122",
                vehicle_type="car",
                color_name="red",
                rgb_color=COLOR_PALETTE["red"],
                make="Hyundai",
                model="Creta",
                plate_confidence=0.94,
                speed_kmh=45.0,
                relative_position=(0.18, 0.60),
                scale=1.1,
            ),
            # Distractor 2: Yellow Auto
            SyntheticVehicleConfig(
                plate_text="DL1RT8899",
                vehicle_type="auto_rickshaw",
                color_name="yellow_green",
                rgb_color=COLOR_PALETTE["yellow_green"],
                make="Bajaj",
                model="RE",
                plate_confidence=0.89,
                speed_kmh=30.0,
                relative_position=(0.82, 0.62),
                scale=0.85,
            ),
            # Distractor 3: Silver Swift
            SyntheticVehicleConfig(
                plate_text="HR51AZ7766",
                vehicle_type="car",
                color_name="silver/grey",
                rgb_color=COLOR_PALETTE["silver/grey"],
                make="Maruti",
                model="Swift",
                plate_confidence=0.93,
                speed_kmh=40.0,
                relative_position=(0.30, 0.46),
                scale=0.80,
            ),
            # Distractor 4: Black Thar
            SyntheticVehicleConfig(
                plate_text="UP14DF4433",
                vehicle_type="car",
                color_name="black",
                rgb_color=COLOR_PALETTE["black"],
                make="Mahindra",
                model="Thar",
                plate_confidence=0.91,
                speed_kmh=42.0,
                relative_position=(0.68, 0.48),
                scale=0.90,
            ),
            # Distractor 5: White Tiago
            SyntheticVehicleConfig(
                plate_text="DL3CBE2211",
                vehicle_type="car",
                color_name="white",
                rgb_color=COLOR_PALETTE["white"],
                make="Tata",
                model="Tiago",
                plate_confidence=0.90,
                speed_kmh=38.0,
                relative_position=(0.44, 0.40),
                scale=0.70,
            ),
        ],
    )

    return [cam1, cam2, cam3, cam4, cam5, cam6]


# -----------------------------------------------------------------------------
# Synthetic Stream Generator Orchestrator
# -----------------------------------------------------------------------------

class SyntheticStreamGenerator:
    """
    Main Orchestrator for Synthetic Multi-Camera Stream Simulation.
    Generates frames across the 6 Delhi NCR cameras, saves evidence media files,
    computes Section 65B SHA-256 digests and 512-D Re-ID embeddings, and populates database.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        width: int = 1280,
        height: int = 720,
        reid_extractor: Optional[ReIDFeatureExtractor] = None,
    ):
        self.output_dir = output_dir or settings.EVIDENCE_DIR
        self.frames_dir = self.output_dir / "frames"
        self.crops_dir = self.output_dir / "crops"
        self.plates_dir = self.output_dir / "plates"
        self.feeds_dir = settings.SYNTHETIC_FEEDS_DIR

        # Ensure directory structure
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.plates_dir.mkdir(parents=True, exist_ok=True)
        self.feeds_dir.mkdir(parents=True, exist_ok=True)

        self.renderer = SyntheticFrameRenderer(width=width, height=height)
        self.reid_extractor = reid_extractor or ReIDFeatureExtractor()
        self.target_base_embedding = get_target_base_embedding()

    @staticmethod
    def compute_sha256_bytes(data: bytes) -> str:
        """Calculates SHA-256 digest of binary payload."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def compute_image_sha256(img: np.ndarray) -> str:
        """Calculates SHA-256 digest of JPEG-encoded image."""
        success, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if success:
            return hashlib.sha256(buf.tobytes()).hexdigest()
        return hashlib.sha256(img.tobytes()).hexdigest()

    def generate_camera_stream(
        self,
        camera: CameraNodeConfig,
        num_frames: int = 10,
        fps: float = 25.0,
        save_video: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generates synthetic frame sequence for a camera node.
        Returns a list of generated sighting metadata dictionaries.
        """
        logger.info(f"Generating synthetic stream for {camera.camera_id} ({camera.name}) - {num_frames} frames...")
        sightings_metadata: List[Dict[str, Any]] = []

        video_writer = None
        if save_video:
            video_path = self.feeds_dir / f"{camera.camera_id}_feed.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (self.renderer.width, self.renderer.height))

        try:
            for frame_idx in range(1, num_frames + 1):
                timestamp = camera.base_timestamp + timedelta(seconds=(frame_idx - 1) / fps)

                # 1. Render Base Scene
                scene_frame = self.renderer.render_scene_background(camera.lighting)

                # 2. Render Vehicles in order of depth (top of road to bottom)
                sorted_vehicles = sorted(camera.vehicles, key=lambda v: v.relative_position[1])

                for v_idx, v_cfg in enumerate(sorted_vehicles):
                    # Slight movement per frame to simulate realistic traffic motion
                    progress_offset = (frame_idx - 1) * (v_cfg.speed_kmh / 3600.0) * 0.05
                    curr_rel_x = v_cfg.relative_position[0]
                    curr_rel_y = min(0.92, v_cfg.relative_position[1] + progress_offset)

                    # Dynamic scaling based on perspective y
                    persp_scale = v_cfg.scale * (0.5 + 0.8 * (curr_rel_y - 0.32) / 0.68)
                    vw = int(220 * persp_scale)
                    vh = int(140 * persp_scale)
                    vx1 = int(self.renderer.width * curr_rel_x - vw // 2)
                    vy1 = int(self.renderer.height * curr_rel_y - vh // 2)
                    vx2 = vx1 + vw
                    vy2 = vy1 + vh

                    # Render vehicle on scene
                    (
                        scene_frame,
                        vehicle_crop,
                        plate_crop,
                        v_bbox,
                        p_bbox,
                    ) = self.renderer.render_vehicle(scene_frame, v_cfg, (vx1, vy1, vx2, vy2))

                    # For primary representative frame (e.g. frame 5 or 1), save crop artifacts and record sighting
                    if frame_idx == max(1, num_frames // 2):
                        track_id = (int(camera.camera_id[-3:]) * 100) + v_idx + 1
                        time_str = timestamp.strftime("%Y%m%d_%H%M%S")

                        # File paths
                        crop_fname = f"crop_{camera.camera_id}_{track_id}.jpg"
                        crop_path = self.crops_dir / crop_fname
                        cv2.imwrite(str(crop_path), vehicle_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        crop_sha256 = self.compute_image_sha256(vehicle_crop)

                        plate_path_str = None
                        if plate_crop is not None and plate_crop.size > 0:
                            plate_fname = f"plate_{camera.camera_id}_{track_id}.jpg"
                            plate_full_path = self.plates_dir / plate_fname
                            cv2.imwrite(str(plate_full_path), plate_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            # Also save to crops dir so dashboard can find both crop and plate at same path
                            plate_crops_path = self.crops_dir / plate_fname
                            cv2.imwrite(str(plate_crops_path), plate_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            plate_path_str = f"/data/evidence/crops/plate_{camera.camera_id}_{track_id}.jpg"

                        # 512-D Re-ID Embedding
                        if v_cfg.is_target:
                            # Target vehicle: deterministic perturbation ensuring cosine similarity >= 0.90
                            cam_idx = int(camera.camera_id[-3:]) if camera.camera_id[-3:].isdigit() else 1
                            embedding = perturb_embedding(self.target_base_embedding, noise_std=0.03, seed=cam_idx)
                        else:
                            # Distractor vehicle: separate distinct embedding
                            dist_seed = hash(v_cfg.plate_text) % 10000 + 500
                            embedding = perturb_embedding(self.target_base_embedding, noise_std=0.85, seed=dist_seed)

                        sighting_entry = {
                            "camera_id": camera.camera_id,
                            "timestamp": timestamp,
                            "track_id": track_id,
                            "plate_text": v_cfg.plate_text,
                            "plate_confidence": v_cfg.plate_confidence,
                            "plate_box": list(p_bbox) if p_bbox else None,
                            "vehicle_box": list(v_bbox),
                            "vehicle_type": v_cfg.vehicle_type,
                            "vehicle_color": v_cfg.color_name,
                            "make": v_cfg.make,
                            "model": v_cfg.model,
                            "speed_estimate_kmh": v_cfg.speed_kmh,
                            "direction_travel": "South-East" if camera.direction_bearing > 90 else "East",
                            "image_path": f"/data/evidence/frames/{camera.camera_id}_{time_str}.jpg",
                            "crop_path": f"/data/evidence/crops/{crop_fname}",
                            "plate_crop_path": plate_path_str,
                            "sha256_hash": crop_sha256,
                            "embedding": embedding,
                            "is_target": v_cfg.is_target,
                        }
                        sightings_metadata.append(sighting_entry)

                # 3. Render CCTV HUD Overlay
                scene_frame = self.renderer.render_cctv_hud(
                    scene_frame,
                    camera=camera,
                    timestamp=timestamp,
                    fps=fps,
                    frame_idx=frame_idx,
                )

                # 4. Save representative full scene frame to disk
                if frame_idx == max(1, num_frames // 2):
                    time_str = timestamp.strftime("%Y%m%d_%H%M%S")
                    frame_fname = f"{camera.camera_id}_{time_str}.jpg"
                    frame_full_path = self.frames_dir / frame_fname
                    cv2.imwrite(str(frame_full_path), scene_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                # 5. Write to video feed if requested
                if video_writer is not None:
                    video_writer.write(scene_frame)

        finally:
            if video_writer is not None:
                video_writer.release()

        logger.info(f"Generated {len(sightings_metadata)} vehicle sightings for {camera.camera_id}.")
        return sightings_metadata

    def generate_all_streams(
        self,
        num_cameras: int = 6,
        frames_per_camera: int = 10,
        save_video: bool = False,
        insert_to_db: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes full synthetic generation across the Delhi NCR corridor.
        Optionally seeds / inserts the sightings into the database.
        """
        logger.info(f"Starting Multi-Camera Synthetic Simulation across {num_cameras} cameras ({frames_per_camera} frames/cam)...")
        all_camera_configs = build_delhi_ncr_stream_network()[:num_cameras]

        all_sightings: List[Dict[str, Any]] = []

        for cam_cfg in all_camera_configs:
            cam_sightings = self.generate_camera_stream(
                camera=cam_cfg,
                num_frames=frames_per_camera,
                save_video=save_video,
            )
            all_sightings.extend(cam_sightings)

        # Database Insertion if requested
        db_inserted_count = 0
        if insert_to_db:
            db_inserted_count = self.populate_database_sightings(all_camera_configs, all_sightings)

        summary = {
            "total_cameras": len(all_camera_configs),
            "frames_per_camera": frames_per_camera,
            "total_sightings_generated": len(all_sightings),
            "target_sightings": sum(1 for s in all_sightings if s.get("is_target")),
            "distractor_sightings": sum(1 for s in all_sightings if not s.get("is_target")),
            "db_records_inserted": db_inserted_count,
            "evidence_frames_dir": str(self.frames_dir),
            "evidence_crops_dir": str(self.crops_dir),
            "status": "SUCCESS",
        }

        logger.info(f"Synthetic Stream Generation Completed: {summary}")
        return summary

    def populate_database_sightings(
        self,
        camera_configs: List[CameraNodeConfig],
        sightings: List[Dict[str, Any]],
    ) -> int:
        """
        Populates SQLAlchemy database with cameras, vehicles, sightings, cases, and matches.
        """
        init_db()
        db = SessionLocal()
        inserted_count = 0

        try:
            # 1. Ensure Camera records exist
            for cam_cfg in camera_configs:
                cam = db.query(Camera).filter(Camera.id == cam_cfg.camera_id).first()
                if not cam:
                    cam = Camera(
                        id=cam_cfg.camera_id,
                        name=cam_cfg.name,
                        latitude=cam_cfg.latitude,
                        longitude=cam_cfg.longitude,
                        road_name=cam_cfg.road_name,
                        direction_bearing=cam_cfg.direction_bearing,
                        camera_type=cam_cfg.camera_type,
                        stream_url=f"rtsp://cctv.delhipolice.gov.in/live/{cam_cfg.camera_id.lower()}",
                        status=CameraStatus.ACTIVE.value,
                    )
                    db.add(cam)
            db.commit()

            # 2. Ensure Vehicle records exist
            for s in sightings:
                plate = s["plate_text"]
                veh = db.query(Vehicle).filter(Vehicle.plate_number == plate).first()
                if not veh:
                    veh = Vehicle(
                        plate_number=plate,
                        vehicle_type=s["vehicle_type"],
                        color=s["vehicle_color"],
                        make=s["make"],
                        model=s["model"],
                        first_seen=s["timestamp"],
                        last_seen=s["timestamp"],
                    )
                    db.add(veh)
                    db.commit()

                # 3. Create Sighting Record if not duplicate
                existing = (
                    db.query(Sighting)
                    .filter(
                        Sighting.camera_id == s["camera_id"],
                        Sighting.timestamp == s["timestamp"],
                        Sighting.plate_text == s["plate_text"],
                    )
                    .first()
                )

                if existing:
                    existing.plate_confidence = s["plate_confidence"]
                    existing.embedding = s["embedding"]
                    existing.plate_box = s["plate_box"]
                    existing.vehicle_box = s["vehicle_box"]
                    existing.sha256_hash = s["sha256_hash"]
                else:
                    sighting_record = Sighting(
                        camera_id=s["camera_id"],
                        vehicle_id=veh.id if veh else None,
                        timestamp=s["timestamp"],
                        track_id=s["track_id"],
                        plate_text=s["plate_text"],
                        plate_confidence=s["plate_confidence"],
                        plate_box=s["plate_box"],
                        vehicle_box=s["vehicle_box"],
                        vehicle_type=s["vehicle_type"],
                        vehicle_color=s["vehicle_color"],
                        make=s["make"],
                        model=s["model"],
                        embedding=s["embedding"],
                        image_path=s["image_path"],
                        crop_path=s["crop_path"],
                        plate_crop_path=s["plate_crop_path"],
                        sha256_hash=s["sha256_hash"],
                        direction_travel=s["direction_travel"],
                        speed_estimate_kmh=s["speed_estimate_kmh"],
                    )
                    db.add(sighting_record)
                    inserted_count += 1

            db.commit()

            # 4. Link with active FIR Case FIR-2026-DEL-0941 if present
            case = db.query(Case).filter(Case.reported_plate == "MH12AB1234").first()
            if case:
                target_sightings = (
                    db.query(Sighting)
                    .filter(Sighting.plate_text == "MH12AB1234")
                    .order_by(Sighting.timestamp.asc())
                    .all()
                )

                for seq, ts_sighting in enumerate(target_sightings, start=1):
                    match = (
                        db.query(CaseMatch)
                        .filter(CaseMatch.case_id == case.id, CaseMatch.sighting_id == ts_sighting.id)
                        .first()
                    )
                    if not match:
                        score, bd = compute_composite_matching_score(
                            reported_plate=case.reported_plate,
                            sighting_plate=ts_sighting.plate_text,
                            plate_confidence=ts_sighting.plate_confidence,
                            reference_embedding=case.reference_embedding,
                            sighting_embedding=ts_sighting.embedding,
                            reported_type=case.vehicle_type,
                            sighting_type=ts_sighting.vehicle_type,
                            reported_color=case.vehicle_color,
                            sighting_color=ts_sighting.vehicle_color,
                        )
                        v_status = VerificationStatus.VERIFIED if seq <= 3 else VerificationStatus.PENDING
                        reviewed_by = "DL-4821" if seq <= 3 else None
                        reviewed_at = (ts_sighting.timestamp + timedelta(minutes=5)) if seq <= 3 else None

                        match = CaseMatch(
                            case_id=case.id,
                            sighting_id=ts_sighting.id,
                            plate_score=bd["plate_score"],
                            visual_score=bd["visual_score"],
                            attribute_score=bd["attribute_score"],
                            spatio_temporal_score=1.0,
                            composite_score=score,
                            sequence_order=seq,
                            verification_status=v_status,
                            reviewed_by=reviewed_by,
                            reviewed_at=reviewed_at,
                            review_notes="Verified visual match on Creta white body and rear dent profile." if seq <= 3 else None,
                        )
                        db.add(match)

                db.commit()

        except Exception as e:
            db.rollback()
            logger.error(f"Error populating database with synthetic sightings: {e}", exc_info=True)
            raise
        finally:
            db.close()

        return inserted_count


# -----------------------------------------------------------------------------
# Module API & CLI Interface
# -----------------------------------------------------------------------------

def generate_synthetic_streams(
    cameras: int = 6,
    frames_per_camera: int = 10,
    save_video: bool = False,
    insert_to_db: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convenience functional interface to trigger synthetic stream generation."""
    generator = SyntheticStreamGenerator(output_dir=output_dir)
    return generator.generate_all_streams(
        num_cameras=cameras,
        frames_per_camera=frames_per_camera,
        save_video=save_video,
        insert_to_db=insert_to_db,
    )


def main():
    """Command Line Entrypoint for Synthetic Multi-Camera Stream Generator."""
    parser = argparse.ArgumentParser(
        description="Synthetic Multi-Camera Stream Generator for Indian Police Stolen Vehicle AI System"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        default=True,
        help="Trigger synthetic multi-camera stream generation",
    )
    parser.add_argument(
        "--cameras",
        type=int,
        default=6,
        help="Number of Delhi NCR surveillance camera feeds to simulate (1 to 6)",
    )
    parser.add_argument(
        "--frames-per-camera",
        type=int,
        default=10,
        help="Number of video frames to render per camera feed",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Save simulated stream output as MP4 video files in data/synthetic_feeds/",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        default=False,
        help="Do not insert generated sightings into SQLite/PostgreSQL database",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom evidence output directory (defaults to data/evidence)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_path = Path(args.output_dir) if args.output_dir else None
    result = generate_synthetic_streams(
        cameras=min(6, max(1, args.cameras)),
        frames_per_camera=max(1, args.frames_per_camera),
        save_video=args.video,
        insert_to_db=not args.no_db,
        output_dir=out_path,
    )

    print("\n" + "=" * 70)
    print("  SYNTHETIC MULTI-CAMERA STREAM GENERATION COMPLETED")
    print("=" * 70)
    for k, v in result.items():
        print(f"  {k:30s}: {v}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
