"""
Unit and Integration Tests for Computer Vision & Indian ANPR Pipeline (Milestone 2).
"""

import os
import shutil
import tempfile
from datetime import datetime, timezone
import numpy as np
import cv2
import pytest

from stolen_vehicle_ai.vision.detector import (
    VehicleDetection,
    BaseVehicleDetector,
    YOLOVehicleDetector,
    MockVehicleDetector,
)
from stolen_vehicle_ai.vision.preprocessor import (
    clahe_enhance,
    bilateral_denoise,
    laplacian_variance,
    is_blurry,
    gamma_correction,
    auto_gamma,
    binarize_plate,
    preprocess_for_ocr,
)
from stolen_vehicle_ai.vision.tracker import (
    KalmanBoxTracker,
    Tracklet,
    ByteTrackTracker,
    compute_snapshot_quality,
    compute_iou,
)
from stolen_vehicle_ai.vision.plate_detector import (
    LicensePlateDetector,
    four_point_transform,
    order_points,
)
from stolen_vehicle_ai.vision.indian_lp_parser import (
    INDIAN_STATE_CODES,
    HSRP_REGEX,
    BH_REGEX,
    clean_plate_string,
    is_valid_hsrp,
    is_valid_bh_series,
    is_valid_indian_plate,
    repair_indian_plate,
    parse_two_line_plate,
    format_plate_display,
)
from stolen_vehicle_ai.vision.ocr_engine import OCREngine
from stolen_vehicle_ai.vision.reid_extractor import (
    ReIDFeatureExtractor,
    DominantColorClassifier,
)
from stolen_vehicle_ai.pipeline.snapshot_manager import SnapshotManager
from stolen_vehicle_ai.pipeline.video_processor import VideoProcessor, SightingResult


# =====================================================================
# 1. Vehicle Detector Tests
# =====================================================================

def test_mock_vehicle_detector_detection():
    detector = MockVehicleDetector(confidence_threshold=0.30)
    # Create a synthetic frame with a bright rectangle (vehicle)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 150), (350, 350), (200, 200, 200), -1)

    detections = detector.detect(frame)
    assert len(detections) >= 1
    det = detections[0]
    assert isinstance(det, VehicleDetection)
    assert det.width > 50
    assert det.height > 50
    assert det.confidence >= 0.30
    assert det.class_name in detector.SUPPORTED_CLASSES
    assert det.crop.shape[0] > 0 and det.crop.shape[1] > 0


def test_yolo_vehicle_detector_fallback():
    # Model path that triggers fallback to mock/cv detector
    detector = YOLOVehicleDetector(model_path="non_existent_model.pt")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (150, 150), (450, 350), (220, 220, 220), -1)

    detections = detector.detect(frame)
    assert len(detections) >= 1
    assert detections[0].class_name in ["car", "motorcycle", "auto_rickshaw", "bus", "truck"]


# =====================================================================
# 2. Image Preprocessor Tests
# =====================================================================

def test_clahe_enhance():
    # Test BGR image
    bgr_img = np.random.randint(50, 150, (100, 100, 3), dtype=np.uint8)
    enhanced = clahe_enhance(bgr_img, clip_limit=2.5)
    assert enhanced.shape == bgr_img.shape
    assert enhanced.dtype == np.uint8

    # Test Grayscale image
    gray_img = np.random.randint(50, 150, (100, 100), dtype=np.uint8)
    enhanced_gray = clahe_enhance(gray_img)
    assert enhanced_gray.shape == gray_img.shape


def test_laplacian_variance_and_blur():
    # Sharp image with high-frequency pattern
    sharp_img = np.zeros((100, 100), dtype=np.uint8)
    sharp_img[::2, ::2] = 255
    sharp_var = laplacian_variance(sharp_img)

    # Blurry image
    blurry_img = cv2.GaussianBlur(sharp_img, (21, 21), 0)
    blurry_var = laplacian_variance(blurry_img)

    assert sharp_var > blurry_var
    is_blur, var_val = is_blurry(blurry_img, threshold=50.0)
    assert is_blur is True
    assert var_val == blurry_var


def test_gamma_correction_and_auto_gamma():
    dark_img = np.full((100, 100, 3), 30, dtype=np.uint8)
    brightened = gamma_correction(dark_img, gamma=0.5)
    assert np.mean(brightened) > np.mean(dark_img)

    auto_bright = auto_gamma(dark_img, target_mean=128.0)
    assert np.mean(auto_bright) > np.mean(dark_img)


def test_binarize_plate():
    plate_crop = np.full((60, 240, 3), 220, dtype=np.uint8)
    # Add dark characters
    cv2.putText(plate_crop, "MH12AB1234", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    binary = binarize_plate(plate_crop, method="otsu")
    assert binary.shape == (60, 240)
    # Binary should have both 0 and 255 values
    assert np.count_nonzero(binary == 255) > 0
    assert np.count_nonzero(binary == 0) > 0


# =====================================================================
# 3. Multi-Object Tracker & Quality Scoring Tests
# =====================================================================

def test_compute_iou():
    box1 = (100, 100, 200, 200)
    box2 = (100, 100, 200, 200)
    assert compute_iou(box1, box2) == pytest.approx(1.0, 1e-4)

    box3 = (300, 300, 400, 400)
    assert compute_iou(box1, box3) == 0.0

    box4 = (150, 100, 250, 200)  # 50% overlap horizontally
    iou_val = compute_iou(box1, box4)
    assert 0.30 <= iou_val <= 0.40


def test_compute_snapshot_quality():
    crop = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    bbox = (100, 100, 300, 300)
    frame_shape = (720, 1280, 3)

    q = compute_snapshot_quality(crop, bbox, frame_shape)
    assert 0.0 <= q <= 1.0
    assert isinstance(q, float)


def test_bytetrack_continuity():
    tracker = ByteTrackTracker(high_thresh=0.40, match_thresh=0.70)
    frame_shape = (480, 640, 3)

    # Simulate a vehicle moving across 5 consecutive frames
    track_ids_seen = []
    for frame_idx in range(5):
        x1 = 100 + frame_idx * 15
        y1 = 150 + frame_idx * 5
        bbox = (x1, y1, x1 + 120, y1 + 100)
        crop = np.full((100, 120, 3), 180, dtype=np.uint8)
        det = VehicleDetection(
            bbox=bbox,
            confidence=0.90,
            class_id=2,
            class_name="car",
            crop=crop,
        )
        active_tracklets = tracker.update([det], frame_shape)
        if active_tracklets:
            track_ids_seen.append(active_tracklets[0].track_id)

    # Track ID should remain consistent across frames (no ID switches)
    assert len(track_ids_seen) >= 3
    assert len(set(track_ids_seen)) == 1, f"Track ID switched unexpectedly: {track_ids_seen}"


# =====================================================================
# 4. License Plate Detector & 4-Point Transform Tests
# =====================================================================

def test_order_points():
    pts = np.array([[200, 100], [50, 50], [210, 220], [60, 210]], dtype=np.float32)
    ordered = order_points(pts)
    assert ordered.shape == (4, 2)
    # Top-left is [50, 50]
    assert np.allclose(ordered[0], [50, 50])


def test_four_point_transform():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 150), (255, 255, 255), -1)
    pts = np.array([[50, 50], [250, 50], [250, 150], [50, 150]], dtype=np.float32)
    warped = four_point_transform(img, pts, target_w=240, target_h=60)
    assert warped.shape == (60, 240, 3)


def test_license_plate_detector():
    detector = LicensePlateDetector()
    vehicle_crop = np.full((200, 200, 3), 100, dtype=np.uint8)
    # Draw a plate in the lower half
    cv2.rectangle(vehicle_crop, (40, 130), (160, 170), (240, 240, 240), -1)
    cv2.putText(vehicle_crop, "DL01C1234", (45, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    plate_crop, plate_bbox, conf = detector.detect_plate(vehicle_crop)
    assert plate_crop.shape == (60, 240, 3)
    assert conf > 0.0
    assert plate_bbox is not None


# =====================================================================
# 5. Indian License Plate Parser & 36 States Tests
# =====================================================================

def test_indian_state_codes_completeness():
    assert len(INDIAN_STATE_CODES) >= 36
    major_states = ["MH", "DL", "KA", "GJ", "TN", "UP", "HR", "PB", "RJ", "TS", "AP", "WB", "KL", "MP", "JK"]
    for st in major_states:
        assert st in INDIAN_STATE_CODES, f"Missing state code: {st}"


def test_hsrp_and_bh_regex_validation():
    valid_plates = [
        "MH12AB1234",
        "DL01C9999",
        "DL1C9999",
        "KA05MN4567",
        "GJ01XX7890",
        "UP32AZ1111",
        "TS09UB4321",
        "22BH1234AA",
        "21BH9999A",
    ]
    for plate in valid_plates:
        cleaned = clean_plate_string(plate)
        assert is_valid_indian_plate(cleaned), f"Failed valid plate check: {plate}"

    invalid_plates = [
        "ZZ99XX1111",  # Invalid state code ZZ
        "INVALID123",
        "12345",
        "",
    ]
    for inv in invalid_plates:
        cleaned = clean_plate_string(inv)
        assert not is_valid_hsrp(cleaned), f"Incorrectly marked valid: {inv}"


def test_positional_character_confusion_repair():
    # 1. Last 4 digits: letters to numbers
    # 'O' -> '0'
    rep, conf, valid = repair_indian_plate("MH12AB123O")
    assert rep == "MH12AB1230"
    assert valid is True

    # 'I' -> '1'
    rep, conf, valid = repair_indian_plate("DL01AB123I")
    assert rep == "DL01AB1231"
    assert valid is True

    # 'B' -> '8'
    rep, conf, valid = repair_indian_plate("KA05CD567B")
    assert rep == "KA05CD5678"
    assert valid is True

    # 'Z' -> '2'
    rep, conf, valid = repair_indian_plate("GJ01EF890Z")
    assert rep == "GJ01EF8902"
    assert valid is True

    # 'S' -> '5'
    rep, conf, valid = repair_indian_plate("UP32GH111S")
    assert rep == "UP32GH1115"
    assert valid is True

    # 2. State Code: numbers to letters
    # '0L01C9999' -> 'DL01C9999'
    rep, conf, valid = repair_indian_plate("0L01C9999")
    assert rep == "DL01C9999"
    assert valid is True

    # 3. District Code: letters to numbers
    rep, conf, valid = repair_indian_plate("MHIOAB1234")
    assert rep == "MH10AB1234"
    assert valid is True


def test_two_line_plate_parsing():
    # Standard 2-line plate
    line1 = "DL 01"
    line2 = "AB 1234"
    repaired, conf, valid = parse_two_line_plate(line1, line2)
    assert repaired == "DL01AB1234"
    assert valid is True

    # Commercial 2-line plate
    l1 = "MH 14"
    l2 = "EU 3344"
    repaired2, conf2, valid2 = parse_two_line_plate(l1, l2)
    assert repaired2 == "MH14EU3344"
    assert valid2 is True

    # Inverted lines
    inv_l1 = "EU 3344"
    inv_l2 = "MH 14"
    repaired3, conf3, valid3 = parse_two_line_plate(inv_l1, inv_l2)
    assert repaired3 == "MH14EU3344"
    assert valid3 is True


def test_format_plate_display():
    assert format_plate_display("MH12AB1234") == "MH 12 AB 1234"
    assert format_plate_display("DL1C9999") == "DL 1 C 9999"
    assert format_plate_display("22BH1234AA") == "22 BH 1234 AA"


# =====================================================================
# 6. Re-ID Embedding and Dominant Color Tests
# =====================================================================

def test_reid_feature_extractor_embedding():
    extractor = ReIDFeatureExtractor()
    crop = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    emb = extractor.extract_embedding(crop)
    assert len(emb) == 512
    # Verify L2 Unit Normalization (sum(x_i^2) == 1.0)
    norm = np.linalg.norm(np.array(emb, dtype=np.float32))
    assert norm == pytest.approx(1.0, abs=1e-5)

    # Identical crop should have cosine similarity == 1.0
    sim_self = extractor.compute_cosine_similarity(emb, emb)
    assert sim_self == pytest.approx(1.0, abs=1e-4)

    # Different crop
    crop2 = np.zeros((128, 128, 3), dtype=np.uint8)
    emb2 = extractor.extract_embedding(crop2)
    assert len(emb2) == 512
    sim_diff = extractor.compute_cosine_similarity(emb, emb2)
    assert -1.0 <= sim_diff <= 1.0


def test_dominant_color_classifier():
    classifier = DominantColorClassifier()

    # Pure Red Swatch
    red_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    red_crop[:, :] = (0, 0, 230)  # BGR Red
    color, conf = classifier.classify_color(red_crop)
    assert color == "red"
    assert conf >= 0.60

    # Pure Blue Swatch
    blue_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    blue_crop[:, :] = (230, 0, 0)  # BGR Blue
    color_b, conf_b = classifier.classify_color(blue_crop)
    assert color_b == "blue"
    assert conf_b >= 0.60

    # Pure White Swatch
    white_crop = np.full((100, 100, 3), 245, dtype=np.uint8)
    color_w, conf_w = classifier.classify_color(white_crop)
    assert color_w == "white"
    assert conf_w >= 0.60

def test_all_ten_dominant_color_classes():
    classifier = DominantColorClassifier()

    # 1. White
    white = np.full((100, 100, 3), 250, dtype=np.uint8)
    assert classifier.classify_color(white)[0] == "white"

    # 2. Black
    black = np.full((100, 100, 3), 10, dtype=np.uint8)
    assert classifier.classify_color(black)[0] == "black"

    # 3. Silver/Grey
    silver = np.full((100, 100, 3), 100, dtype=np.uint8)
    assert classifier.classify_color(silver)[0] == "silver/grey"

    # 4. Red (BGR: 0, 0, 240)
    red = np.zeros((100, 100, 3), dtype=np.uint8)
    red[:, :] = (0, 0, 240)
    assert classifier.classify_color(red)[0] == "red"

    # 5. Blue (BGR: 240, 0, 0)
    blue = np.zeros((100, 100, 3), dtype=np.uint8)
    blue[:, :] = (240, 0, 0)
    assert classifier.classify_color(blue)[0] == "blue"

    # 6. Yellow (BGR: 0, 240, 240)
    yellow = np.zeros((100, 100, 3), dtype=np.uint8)
    yellow[:, :] = (0, 240, 240)
    assert classifier.classify_color(yellow)[0] == "yellow"

    # 7. Green (BGR: 0, 200, 0)
    green = np.zeros((100, 100, 3), dtype=np.uint8)
    green[:, :] = (0, 200, 0)
    assert classifier.classify_color(green)[0] == "green"

    # 8. Orange (BGR: 0, 140, 240)
    orange = np.zeros((100, 100, 3), dtype=np.uint8)
    orange[:, :] = (0, 140, 240)
    assert classifier.classify_color(orange)[0] == "orange"

    # 9. Brown (BGR: 30, 60, 120)
    brown = np.zeros((100, 100, 3), dtype=np.uint8)
    brown[:, :] = (30, 60, 120)
    assert classifier.classify_color(brown)[0] == "brown"


def test_all_36_state_codes_validation():
    for state_code in sorted(INDIAN_STATE_CODES):
        plate_str = f"{state_code}01AB1234"
        assert is_valid_hsrp(plate_str) is True
        rep, conf, valid = repair_indian_plate(plate_str)
        assert valid is True
        assert rep == plate_str
        assert conf >= 0.90


def test_ocr_engine_with_synthetic_plate():
    engine = OCREngine(preferred_engine="auto")
    assert engine.engine_name in ["paddleocr", "easyocr", "pytesseract", "cv_structural"]

    # Synthetic plate image with text
    plate = np.full((60, 240, 3), 240, dtype=np.uint8)
    cv2.putText(plate, "MH12AB1234", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)

    text, conf, valid, meta = engine.recognize_plate(plate)
    assert isinstance(text, str)
    assert isinstance(conf, float)
    assert isinstance(valid, bool)
    assert isinstance(meta, dict)


def test_bytetrack_multi_vehicle_tracking():
    tracker = ByteTrackTracker(high_thresh=0.40, match_thresh=0.70)
    frame_shape = (480, 640, 3)

    # Frame 1: Two vehicles
    det1 = VehicleDetection(bbox=(50, 100, 150, 200), confidence=0.9, class_id=2, class_name="car", crop=np.zeros((100, 100, 3), dtype=np.uint8))
    det2 = VehicleDetection(bbox=(300, 100, 400, 200), confidence=0.85, class_id=7, class_name="truck", crop=np.zeros((100, 100, 3), dtype=np.uint8))
    t1 = tracker.update([det1, det2], frame_shape)
    assert len(t1) == 2

    # Frame 2: Both moved slightly
    det1_moved = VehicleDetection(bbox=(60, 105, 160, 205), confidence=0.9, class_id=2, class_name="car", crop=np.zeros((100, 100, 3), dtype=np.uint8))
    det2_moved = VehicleDetection(bbox=(310, 105, 410, 205), confidence=0.85, class_id=7, class_name="truck", crop=np.zeros((100, 100, 3), dtype=np.uint8))
    t2 = tracker.update([det1_moved, det2_moved], frame_shape)
    assert len(t2) == 2

    # Track IDs should remain identical
    ids_t1 = {tr.track_id for tr in t1}
    ids_t2 = {tr.track_id for tr in t2}
    assert ids_t1 == ids_t2


# =====================================================================
# 7. Snapshot Manager & SHA-256 Tests
# =====================================================================

def test_snapshot_manager_hashing_and_storage():
    temp_dir = tempfile.mkdtemp()
    try:
        mgr = SnapshotManager(base_dir=temp_dir)
        crop = np.full((80, 80, 3), 200, dtype=np.uint8)
        plate = np.full((30, 120, 3), 240, dtype=np.uint8)
        frame = np.full((240, 320, 3), 100, dtype=np.uint8)

        now = datetime.now(timezone.utc)
        pkg = mgr.save_evidence_package(
            camera_id="TEST_CAM_01",
            track_id=101,
            timestamp=now,
            vehicle_crop=crop,
            plate_crop=plate,
            full_frame=frame,
        )

        assert os.path.exists(pkg["crop_image_path"])
        assert os.path.exists(pkg["plate_crop_path"])
        assert os.path.exists(pkg["frame_image_path"])

        # Check SHA-256 format (64-char hex)
        sha = pkg["sha256_hash"]
        assert len(sha) == 64
        assert all(c in "0123456789abcdefABCDEF" for c in sha)

        # Verify integrity check
        assert mgr.verify_integrity(pkg["crop_image_path"], sha) is True
        assert mgr.verify_integrity(pkg["crop_image_path"], "bad_hash" * 8) is False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =====================================================================
# 8. Video Processor End-to-End Tests
# =====================================================================

def test_video_processor_end_to_end():
    temp_dir = tempfile.mkdtemp()
    try:
        snapshot_mgr = SnapshotManager(base_dir=temp_dir)
        processor = VideoProcessor(snapshot_manager=snapshot_mgr, save_evidence=True)

        # Create a synthetic frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Vehicle rectangle
        cv2.rectangle(frame, (100, 100), (350, 350), (220, 220, 220), -1)
        # License plate on vehicle
        cv2.rectangle(frame, (150, 280), (300, 320), (250, 250, 250), -1)
        cv2.putText(frame, "MH12AB1234", (160, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        sightings = processor.process_frame(frame, camera_id="CAM_TEST_100")
        assert len(sightings) >= 1
        s = sightings[0]

        assert isinstance(s, SightingResult)
        assert s.camera_id == "CAM_TEST_100"
        assert s.vehicle_type in ["car", "motorcycle", "auto_rickshaw", "bus", "truck"]
        assert isinstance(s.vehicle_color, str)
        assert len(s.reid_embedding) == 512
        assert np.linalg.norm(np.array(s.reid_embedding)) == pytest.approx(1.0, abs=1e-5)
        assert len(s.sha256_hash) == 64
        assert os.path.exists(s.crop_image_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


