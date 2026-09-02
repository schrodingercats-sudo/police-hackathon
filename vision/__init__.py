"""
Vision Module for Indian Police Stolen Vehicle AI System.
"""

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

__all__ = [
    "VehicleDetection",
    "BaseVehicleDetector",
    "YOLOVehicleDetector",
    "MockVehicleDetector",
    "clahe_enhance",
    "bilateral_denoise",
    "laplacian_variance",
    "is_blurry",
    "gamma_correction",
    "auto_gamma",
    "binarize_plate",
    "preprocess_for_ocr",
    "KalmanBoxTracker",
    "Tracklet",
    "ByteTrackTracker",
    "compute_snapshot_quality",
    "compute_iou",
    "LicensePlateDetector",
    "four_point_transform",
    "order_points",
    "INDIAN_STATE_CODES",
    "HSRP_REGEX",
    "BH_REGEX",
    "clean_plate_string",
    "is_valid_hsrp",
    "is_valid_bh_series",
    "is_valid_indian_plate",
    "repair_indian_plate",
    "parse_two_line_plate",
    "format_plate_display",
    "OCREngine",
    "ReIDFeatureExtractor",
    "DominantColorClassifier",
]
