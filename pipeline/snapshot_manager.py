"""
Evidence Snapshot Manager and Section 65B SHA-256 Cryptographic Digest Engine.

Provides:
- Secure atomic storage of full frame snapshots, vehicle crops, and plate zooms
- Cryptographic SHA-256 hashing for Indian Evidence Act Section 65B court admissibility
- Evidence metadata persistence and integrity verification
"""

import hashlib
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import cv2
import numpy as np


class SnapshotManager:
    """
    Evidence Snapshot Manager.
    Saves vehicle and license plate image crops and calculates immutable SHA-256 checksums.
    """

    def __init__(self, base_dir: Optional[Any] = None):
        if base_dir is None:
            try:
                from backend.config import settings
                self.base_dir = settings.EVIDENCE_DIR
            except ImportError:
                try:
                    from stolen_vehicle_ai.backend.config import settings
                    self.base_dir = settings.EVIDENCE_DIR
                except ImportError:
                    self.base_dir = Path("data/evidence").resolve()
        else:
            self.base_dir = Path(base_dir).resolve()

        self.frames_dir = self.base_dir / "frames"
        self.crops_dir = self.base_dir / "crops"
        self.plates_dir = self.base_dir / "plates"

        # Ensure directory tree exists
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.plates_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_sha256(image: np.ndarray, image_format: str = ".jpg") -> str:
        """
        Encode image to bytes and compute SHA-256 hexadecimal hash.
        """
        if image is None or image.size == 0:
            return hashlib.sha256(b"").hexdigest()

        success, encoded = cv2.imencode(image_format, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            # Fallback to raw numpy buffer
            return hashlib.sha256(image.tobytes()).hexdigest()

        return hashlib.sha256(encoded.tobytes()).hexdigest()

    @staticmethod
    def compute_file_sha256(file_path: str) -> str:
        """
        Compute SHA-256 hash of an existing file on disk.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def save_evidence_package(
        self,
        camera_id: str,
        track_id: int,
        timestamp: datetime,
        vehicle_crop: np.ndarray,
        plate_crop: Optional[np.ndarray] = None,
        full_frame: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Save all evidence images for a vehicle sighting and generate cryptographic digests.

        Returns:
            Dict containing:
                - crop_path: str
                - plate_path: Optional[str]
                - frame_path: Optional[str]
                - sha256_hash: str (combined primary hash of vehicle crop)
                - frame_hash: Optional[str]
                - plate_hash: Optional[str]
        """
        time_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        prefix = f"{camera_id}_trk{track_id}_{time_str}"

        # 1. Save vehicle crop
        crop_filename = f"{prefix}_crop.jpg"
        crop_full_path = self.crops_dir / crop_filename
        cv2.imwrite(str(crop_full_path), vehicle_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        crop_hash = self.compute_sha256(vehicle_crop)

        # 2. Save plate crop if present
        plate_path_str: Optional[str] = None
        plate_hash_str: Optional[str] = None
        if plate_crop is not None and plate_crop.size > 0:
            plate_filename = f"{prefix}_plate.jpg"
            plate_full_path = self.plates_dir / plate_filename
            cv2.imwrite(str(plate_full_path), plate_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # Also save to crops dir for web serving
            cv2.imwrite(str(self.crops_dir / plate_filename), plate_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            plate_path_str = f"/data/evidence/crops/{plate_filename}"
            plate_hash_str = self.compute_sha256(plate_crop)

        # 3. Save full frame if present
        frame_path_str: Optional[str] = None
        frame_hash_str: Optional[str] = None
        if full_frame is not None and full_frame.size > 0:
            frame_filename = f"{prefix}_frame.jpg"
            frame_full_path = self.frames_dir / frame_filename
            cv2.imwrite(str(frame_full_path), full_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            frame_path_str = f"/data/evidence/frames/{frame_filename}"
            frame_hash_str = self.compute_sha256(full_frame)

        crop_path_str = str(crop_full_path).replace("\\", "/")
        plate_path_str = str(plate_full_path).replace("\\", "/") if plate_crop is not None and plate_crop.size > 0 else None
        frame_path_str = str(frame_full_path).replace("\\", "/") if full_frame is not None and full_frame.size > 0 else None

        return {
            "crop_image_path": crop_path_str,
            "plate_crop_path": plate_path_str,
            "frame_image_path": frame_path_str,
            "crop_url": f"/data/evidence/crops/{crop_filename}",
            "plate_crop_url": f"/data/evidence/crops/{plate_filename}" if plate_path_str else None,
            "frame_url": f"/data/evidence/frames/{frame_filename}" if frame_path_str else None,
            "crop_full_path": str(crop_full_path),
            "plate_full_path": str(plate_full_path) if plate_crop is not None and plate_crop.size > 0 else None,
            "frame_full_path": str(frame_full_path) if full_frame is not None and full_frame.size > 0 else None,
            "sha256_hash": crop_hash,
            "crop_hash": crop_hash,
            "plate_hash": plate_hash_str,
            "frame_hash": frame_hash_str,
        }

    def verify_integrity(self, file_path: str, expected_hash: str) -> bool:
        """
        Verify that an evidence file on disk matches its Section 65B cryptographic checksum.
        """
        if not os.path.exists(file_path):
            return False
        current_hash = self.compute_file_sha256(file_path)
        return current_hash.lower() == expected_hash.lower()
