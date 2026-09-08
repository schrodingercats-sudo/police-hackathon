"""
End-to-End Multi-Camera Video and Image Stream Processing Pipeline.

Coordinates:
1. Vehicle Detection & Classification (YOLO / BaseVehicleDetector)
2. Multi-Object Tracking & Peak Quality Snapshot Selection (ByteTrack)
3. Indian License Plate Localization & 4-Point Perspective Rectification
4. Multi-Stage OCR with Positional Confusion Matrix Parsing
5. 512-D Metric Visual Re-ID Embedding Extraction
6. 10-Class HSV Dominant Color Classification
7. Section 65B SHA-256 Snapshot Evidence Archival
8. Output formatted SightingResult objects matching System Interface Contract
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

try:
    from vision.detector import (
        BaseVehicleDetector,
        YOLOVehicleDetector,
        MockVehicleDetector,
        VehicleDetection,
    )
    from vision.tracker import ByteTrackTracker, Tracklet, compute_snapshot_quality
    from vision.plate_detector import LicensePlateDetector
    from vision.ocr_engine import OCREngine
    from vision.reid_extractor import (
        ReIDFeatureExtractor,
        DominantColorClassifier,
    )
    from pipeline.snapshot_manager import SnapshotManager
except ImportError:
    from stolen_vehicle_ai.vision.detector import (
        BaseVehicleDetector,
        YOLOVehicleDetector,
        MockVehicleDetector,
        VehicleDetection,
    )
    from stolen_vehicle_ai.vision.tracker import ByteTrackTracker, Tracklet, compute_snapshot_quality
    from stolen_vehicle_ai.vision.plate_detector import LicensePlateDetector
    from stolen_vehicle_ai.vision.ocr_engine import OCREngine
    from stolen_vehicle_ai.vision.reid_extractor import (
        ReIDFeatureExtractor,
        DominantColorClassifier,
    )
    from stolen_vehicle_ai.pipeline.snapshot_manager import SnapshotManager


@dataclass
class SightingResult:
    """
    Standardized atomic sighting result produced by the computer vision pipeline.
    Matches interface contract in PROJECT.md.
    """
    camera_id: str
    timestamp: datetime
    track_id: int
    vehicle_type: str                  # "car", "motorcycle", "auto_rickshaw", "bus", "truck"
    vehicle_color: str                 # "white", "black", "silver/grey", "red", "blue", ...
    color_confidence: float
    vehicle_bbox: List[int]            # [x1, y1, x2, y2]
    plate_text: Optional[str]          # Cleaned/repaired e.g. "MH12AB1234"
    plate_confidence: float            # 0.0 - 1.0
    plate_is_valid_format: bool
    plate_bbox: Optional[List[int]]
    reid_embedding: List[float]        # 512-D L2-normalized float list
    quality_score: float
    frame_image_path: Optional[str] = None
    crop_image_path: Optional[str] = None
    plate_crop_path: Optional[str] = None
    sha256_hash: str = ""
    metadata: Optional[Dict[str, Any]] = None


class VideoProcessor:
    """
    Unified Computer Vision & ANPR Processing Pipeline for Video Streams and Frame Feeds.
    """

    def __init__(
        self,
        detector: Optional[BaseVehicleDetector] = None,
        tracker: Optional[ByteTrackTracker] = None,
        plate_detector: Optional[LicensePlateDetector] = None,
        ocr_engine: Optional[OCREngine] = None,
        reid_extractor: Optional[ReIDFeatureExtractor] = None,
        color_classifier: Optional[DominantColorClassifier] = None,
        snapshot_manager: Optional[SnapshotManager] = None,
        save_evidence: bool = True,
    ):
        self.detector = detector or YOLOVehicleDetector()
        self.tracker = tracker or ByteTrackTracker()
        self.plate_detector = plate_detector or LicensePlateDetector()
        self.ocr_engine = ocr_engine or OCREngine()
        self.reid_extractor = reid_extractor or ReIDFeatureExtractor()
        self.color_classifier = color_classifier or DominantColorClassifier()
        self.snapshot_manager = snapshot_manager or SnapshotManager()
        self.save_evidence = save_evidence

    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
        timestamp: Optional[datetime] = None,
    ) -> List[SightingResult]:
        """
        Process a single video frame through the full computer vision & ANPR pipeline.

        Args:
            frame: BGR image frame (H, W, 3).
            camera_id: Identifier of the source camera (e.g. 'DEL_CAM_001').
            timestamp: Observation timestamp (defaults to current UTC).

        Returns:
            List[SightingResult]: List of confirmed vehicle sightings in this frame.
        """
        if frame is None or frame.size == 0:
            return []

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # 1. Vehicle Detection
        detections = self.detector.detect(frame)
        if not detections:
            return []

        # 2. Multi-Object Tracking & Snapshot Quality Association
        active_tracklets = self.tracker.update(detections, frame.shape)

        results: List[SightingResult] = []

        for tracklet in active_tracklets:
            # Use the peak-quality vehicle crop
            crop = tracklet.best_crop
            bbox = list(tracklet.latest_bbox)
            v_type = tracklet.class_name

            # 3. Dominant Color Classification
            color, color_conf = self.color_classifier.classify_color(crop)

            # 4. 512-D Visual Re-ID Feature Extraction
            embedding = self.reid_extractor.extract_embedding(crop)

            # 5. License Plate Localization & Rectification
            plate_crop, plate_bbox_rel, plate_det_conf = self.plate_detector.detect_plate(crop)

            # 6. Multi-Engine OCR & Positional Confusion Repair
            plate_text, plate_ocr_conf, is_valid_fmt, ocr_meta = self.ocr_engine.recognize_plate(
                plate_crop
            )

            # Composite plate confidence
            combined_plate_conf = float(plate_det_conf * plate_ocr_conf) if plate_text else 0.0

            # Absolute coordinates for plate bbox
            abs_plate_bbox = None
            if plate_bbox_rel is not None:
                abs_plate_bbox = [
                    bbox[0] + plate_bbox_rel[0],
                    bbox[1] + plate_bbox_rel[1],
                    bbox[0] + plate_bbox_rel[2],
                    bbox[1] + plate_bbox_rel[3],
                ]

            # 7. Evidence Snapshot Archival & Section 65B SHA-256 Hashing
            if self.save_evidence:
                ev_pkg = self.snapshot_manager.save_evidence_package(
                    camera_id=camera_id,
                    track_id=tracklet.track_id,
                    timestamp=timestamp,
                    vehicle_crop=crop,
                    plate_crop=plate_crop,
                    full_frame=frame,
                )
                crop_path = ev_pkg["crop_image_path"]
                plate_path = ev_pkg["plate_crop_path"]
                frame_path = ev_pkg["frame_image_path"]
                sha256_digest = ev_pkg["sha256_hash"]
            else:
                crop_path = None
                plate_path = None
                frame_path = None
                sha256_digest = SnapshotManager.compute_sha256(crop)

            sighting = SightingResult(
                camera_id=camera_id,
                timestamp=timestamp,
                track_id=tracklet.track_id,
                vehicle_type=v_type,
                vehicle_color=color,
                color_confidence=color_conf,
                vehicle_bbox=bbox,
                plate_text=plate_text if plate_text else None,
                plate_confidence=combined_plate_conf,
                plate_is_valid_format=is_valid_fmt,
                plate_bbox=abs_plate_bbox,
                reid_embedding=embedding,
                quality_score=tracklet.best_quality,
                frame_image_path=frame_path,
                crop_image_path=crop_path,
                plate_crop_path=plate_path,
                sha256_hash=sha256_digest,
                metadata={"ocr_meta": ocr_meta, "total_frames": tracklet.total_frames},
            )
            results.append(sighting)

        return results

    def process_video(
        self,
        video_path: str,
        camera_id: str,
        start_time: Optional[datetime] = None,
        sample_rate: int = 1,
        max_frames: Optional[int] = None,
    ) -> List[SightingResult]:
        """
        Process a video file stream and extract all vehicle sightings.

        Args:
            video_path: Path to video file.
            camera_id: Camera identifier.
            start_time: Start timestamp of the video.
            sample_rate: Process every Nth frame (e.g. 1 = every frame, 5 = every 5th frame).
            max_frames: Maximum number of frames to process.

        Returns:
            List[SightingResult]: Chronological list of vehicle sightings.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        if start_time is None:
            start_time = datetime.now(timezone.utc)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0
        all_sightings: List[SightingResult] = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    delta_seconds = frame_idx / fps
                    frame_time = datetime.fromtimestamp(
                        start_time.timestamp() + delta_seconds, tz=timezone.utc
                    )
                    sightings = self.process_frame(frame, camera_id, frame_time)
                    all_sightings.extend(sightings)

                frame_idx += 1
                if max_frames is not None and frame_idx >= max_frames:
                    break
        finally:
            cap.release()

        return all_sightings
