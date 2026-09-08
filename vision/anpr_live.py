"""Real-Time ANPR Processor using EasyOCR.

Processes camera frames/images to detect and read actual license plates
using EasyOCR (no mock/dummy data). Supports Indian plate formats.
"""

import logging
import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("stolen_vehicle_ai.anpr_live")

# Indian plate regex patterns
INDIAN_PLATE_PATTERNS = [
    # Standard: XX 00 XX 0000 (e.g., MH 12 AB 1234)
    re.compile(r'[A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4}', re.IGNORECASE),
    # BH series: 00 BH 0000 XX
    re.compile(r'\d{2}\s*BH\s*\d{4}\s*[A-Z]{1,2}', re.IGNORECASE),
    # Generic alphanumeric plate-like pattern
    re.compile(r'[A-Z0-9]{2,4}\s*[A-Z0-9]{2,4}\s*[A-Z0-9]{2,4}', re.IGNORECASE),
]


class LiveANPRProcessor:
    """Real-time Automatic Number Plate Recognition using EasyOCR."""

    def __init__(self, languages=None, gpu=False):
        self.reader = None
        self.languages = languages or ['en']
        self.gpu = gpu
        self._init_reader()

    def _init_reader(self):
        """Initialize EasyOCR reader."""
        try:
            import easyocr
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
            logger.info("EasyOCR reader initialized successfully.")
        except ImportError:
            logger.warning("EasyOCR not installed. Using OpenCV fallback OCR.")
            self.reader = None
        except Exception as e:
            logger.warning(f"EasyOCR init failed: {e}. Using fallback.")
            self.reader = None

    def detect_plates_in_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect and read license plates from a camera frame.

        Returns list of detections with:
        - plate_text: OCR'd text
        - confidence: OCR confidence score
        - bbox: [x1, y1, x2, y2] bounding box
        - plate_crop: cropped plate image (numpy array)
        """
        if frame is None or frame.size == 0:
            return []

        detections = []

        if self.reader is not None:
            detections = self._detect_with_easyocr(frame)
        else:
            detections = self._detect_with_opencv(frame)

        # Filter for Indian plate patterns only (exclude HUD overlay text)
        # HUD text exclusion list
        hud_words = {'rec', 'fps', 'ist', 'live', 'cam', 'online', 'connaught',
                     'place', 'radial', 'road', 'outer', 'circle', 'junction',
                     'minar', 'crossing', 'akshardham', 'setu', 'flyway',
                     'toll', 'plaza', 'noida', 'expressway', 'sector', 'pari',
                     'chowk', 'roundabout', 'greater', 'delhi', 'vikas', 'circl'}

        filtered = []
        for det in detections:
            cleaned = self._clean_plate_text(det['plate_text'])
            if not cleaned or len(cleaned) < 6:
                continue
            # Must have both letters and digits
            has_alpha = any(c.isalpha() for c in cleaned)
            has_digit = any(c.isdigit() for c in cleaned)
            if not (has_alpha and has_digit):
                continue
            # Exclude if any HUD word found
            lower = cleaned.lower()
            if any(w in lower for w in hud_words):
                continue
            # Must match Indian plate-like pattern (2 letters + digits)
            no_space = cleaned.replace(' ', '')
            if len(no_space) >= 6 and no_space[:2].isalpha() and any(c.isdigit() for c in no_space[2:5]):
                det['plate_text'] = cleaned
                filtered.append(det)

        return filtered

    def _detect_with_easyocr(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Use EasyOCR for text detection and recognition."""
        results = []

        try:
            # Run OCR on full frame
            ocr_results = self.reader.readtext(frame, detail=1, paragraph=False)

            for (bbox_pts, text, conf) in ocr_results:
                if conf < 0.3 or len(text.strip()) < 3:
                    continue

                # Check if text matches plate pattern
                cleaned = text.strip().upper().replace(' ', '')
                is_plate = any(p.search(cleaned) for p in INDIAN_PLATE_PATTERNS)

                if is_plate or (conf > 0.5 and len(cleaned) >= 6 and any(c.isdigit() for c in cleaned) and any(c.isalpha() for c in cleaned)):
                    # Convert bbox points to x1,y1,x2,y2
                    pts = np.array(bbox_pts, dtype=np.int32)
                    x1, y1 = pts.min(axis=0)
                    x2, y2 = pts.max(axis=0)

                    # Crop plate region
                    h, w = frame.shape[:2]
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(w, x2), min(h, y2)
                    plate_crop = frame[y1c:y2c, x1c:x2c].copy() if y2c > y1c and x2c > x1c else None

                    results.append({
                        'plate_text': text.strip().upper(),
                        'confidence': round(conf, 4),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'plate_crop': plate_crop,
                    })
        except Exception as e:
            logger.error(f"EasyOCR detection error: {e}")

        return results

    def _detect_with_opencv(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback: use OpenCV contour-based plate detection."""
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Bilateral filter + edge detection for plate regions
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h) if h > 0 else 0

                # Indian plates are roughly 3:1 to 5:1 aspect ratio
                if 2.0 <= aspect_ratio <= 6.0 and w > 60 and h > 15:
                    plate_crop = frame[y:y+h, x:x+w].copy()
                    results.append({
                        'plate_text': 'UNREADABLE',
                        'confidence': 0.3,
                        'bbox': [x, y, x+w, y+h],
                        'plate_crop': plate_crop,
                    })

        return results

    @staticmethod
    def _clean_plate_text(text: str) -> str:
        """Clean and normalize detected plate text."""
        if not text:
            return ''

        # Remove special characters except spaces
        cleaned = re.sub(r'[^A-Za-z0-9\s]', '', text.upper())
        # Collapse multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # Remove spaces for comparison
        no_space = cleaned.replace(' ', '')

        if len(no_space) < 4:
            return ''

        return cleaned

    def process_image_file(self, image_path: str) -> List[Dict[str, Any]]:
        """Process a single image file for plate detection."""
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Could not read image: {image_path}")
            return []
        return self.detect_plates_in_frame(frame)

    def process_video_frame(self, video_path: str, frame_number: int = 0) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        """Extract a frame from video and detect plates."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None, []

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None, []

        detections = self.detect_plates_in_frame(frame)
        return frame, detections


def compute_frame_hash(frame: np.ndarray) -> str:
    """Compute SHA-256 hash for Section 65B compliance."""
    success, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if success:
        return hashlib.sha256(buf.tobytes()).hexdigest()
    return hashlib.sha256(frame.tobytes()).hexdigest()
