"""
License Plate Detection and Perspective Rectification Module.

Provides:
- Localization of Indian single-line and double-line license plates on vehicle crops
- 4-point perspective warp and deskewing to standardized planar 240x60 pixel crops
- High-contrast candidate filtering based on aspect ratio, morphology, and edge density
"""

from typing import Tuple, Optional, List
import numpy as np
import cv2

from stolen_vehicle_ai.vision.preprocessor import clahe_enhance


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 polygon points in order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right has smallest diff (x - y)
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest diff (x - y)

    return rect


def four_point_transform(
    image: np.ndarray, pts: np.ndarray, target_w: int = 240, target_h: int = 60
) -> np.ndarray:
    """
    Applies 4-point perspective transform to deskew quadrilateral plate ROI
    into a planar target_w x target_h output.
    """
    rect = order_points(pts)
    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (target_w, target_h), flags=cv2.INTER_CUBIC)
    return warped


class LicensePlateDetector:
    """
    Indian License Plate Detector and Perspective Normalizer.
    Locates plate region, deskews angle/tilt, and produces normalized 240x60 crops.
    """

    TARGET_WIDTH = 240
    TARGET_HEIGHT = 60

    def __init__(self, confidence_threshold: float = 0.40):
        self.confidence_threshold = confidence_threshold

    def detect_plate(
        self, vehicle_crop: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], float]:
        """
        Detect and extract normalized license plate from vehicle crop.

        Args:
            vehicle_crop: BGR vehicle image crop.

        Returns:
            Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], float]:
                - plate_crop: 240x60 BGR image
                - plate_bbox: (x1, y1, x2, y2) relative to vehicle_crop, or None
                - confidence: 0.0 - 1.0 confidence score
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            blank = np.zeros((self.TARGET_HEIGHT, self.TARGET_WIDTH, 3), dtype=np.uint8)
            return blank, None, 0.0

        vh, vw = vehicle_crop.shape[:2]
        if vh < 20 or vw < 20:
            resized = cv2.resize(vehicle_crop, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
            return resized, (0, 0, vw, vh), 0.3

        # 1. Morphological BlackHat / TopHat filtering for text on light/colored plate background
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        # Rectangular structuring element to connect horizontal plate characters
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        tophat = cv2.morphologyEx(gray_enhanced, cv2.MORPH_TOPHAT, rect_kernel)
        blackhat = cv2.morphologyEx(gray_enhanced, cv2.MORPH_BLACKHAT, rect_kernel)
        grad_plate = cv2.add(tophat, blackhat)

        # Sobel vertical edges
        sobel_x = cv2.Sobel(grad_plate, cv2.CV_32F, 1, 0, ksize=3)
        sobel_x = np.absolute(sobel_x)
        min_v, max_v = np.min(sobel_x), np.max(sobel_x)
        if max_v > min_v:
            sobel_norm = (255 * (sobel_x - min_v) / (max_v - min_v)).astype(np.uint8)
        else:
            sobel_norm = np.zeros_like(gray, dtype=np.uint8)

        # Blur and threshold
        blurred = cv2.GaussianBlur(sobel_norm, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Close gaps between characters
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        # 2. Find plate contours
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        best_score = -1.0
        best_crop: Optional[np.ndarray] = None
        best_bbox: Optional[Tuple[int, int, int, int]] = None
        best_conf = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (vw * vh * 0.005) or area > (vw * vh * 0.40):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(1.0, float(bh))

            # Indian plates aspect: single-line ~3.0 - 5.5, two-line ~1.2 - 2.5
            is_single_line = (2.6 <= aspect <= 5.8)
            is_two_line = (1.1 <= aspect <= 2.5)

            if not (is_single_line or is_two_line):
                continue

            # Position check: plates are rarely at the very top of a vehicle
            rel_y = (y + bh / 2.0) / float(vh)
            y_weight = 1.0 if rel_y >= 0.35 else 0.5

            # Attempt 4-point polygon approximation for perspective warp
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

            score = area * y_weight * (1.5 if is_single_line else 1.0)
            if score > best_score:
                best_score = score
                best_bbox = (x, y, x + bw, y + bh)
                best_conf = float(np.clip(0.60 + 0.35 * (area / (vw * vh * 0.15)), 0.50, 0.95))

                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    best_crop = four_point_transform(
                        vehicle_crop, pts, self.TARGET_WIDTH, self.TARGET_HEIGHT
                    )
                else:
                    # Straight crop with padding
                    pad_x = int(bw * 0.05)
                    pad_y = int(bh * 0.05)
                    cx1 = max(0, x - pad_x)
                    cy1 = max(0, y - pad_y)
                    cx2 = min(vw, x + bw + pad_x)
                    cy2 = min(vh, y + bh + pad_y)
                    raw_crop = vehicle_crop[cy1:cy2, cx1:cx2]
                    if raw_crop.size > 0:
                        best_crop = cv2.resize(
                            raw_crop, (self.TARGET_WIDTH, self.TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC
                        )

        # 3. Fallback: Lower-middle bumper ROI if no clean plate contour detected
        if best_crop is None:
            # Typical Indian vehicle plate location: bottom 40%, middle 60%
            x1 = int(vw * 0.20)
            y1 = int(vh * 0.55)
            x2 = int(vw * 0.80)
            y2 = int(vh * 0.95)
            fallback_roi = vehicle_crop[y1:y2, x1:x2]
            if fallback_roi.size > 0:
                best_crop = cv2.resize(
                    fallback_roi, (self.TARGET_WIDTH, self.TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC
                )
                best_bbox = (x1, y1, x2, y2)
                best_conf = 0.40
            else:
                best_crop = cv2.resize(
                    vehicle_crop, (self.TARGET_WIDTH, self.TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC
                )
                best_bbox = (0, 0, vw, vh)
                best_conf = 0.30

        return best_crop, best_bbox, float(best_conf)
