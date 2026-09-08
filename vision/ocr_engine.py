"""
Multi-Engine OCR and Indian License Plate Character Recognition.

Provides:
- Multi-engine OCR support (PaddleOCR / EasyOCR / PyTesseract / Computer Vision Fallback)
- Character whitelisting: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
- Multi-line character reading and sorting
- End-to-end integration with Indian LP grammar and positional confusion parser
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import cv2

try:
    from vision.preprocessor import preprocess_for_ocr, binarize_plate
    from vision.indian_lp_parser import repair_indian_plate, clean_plate_string
except ImportError:
    from stolen_vehicle_ai.vision.preprocessor import preprocess_for_ocr, binarize_plate
    from stolen_vehicle_ai.vision.indian_lp_parser import repair_indian_plate, clean_plate_string


class OCREngine:
    """
    Unified OCR Engine for Indian License Plate Recognition.
    Supports EasyOCR, PaddleOCR, PyTesseract, and built-in CV character segmentation.
    """

    ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    def __init__(
        self,
        preferred_engine: str = "auto",  # 'auto', 'paddle', 'easyocr', 'tesseract', 'cv'
        gpu: bool = False,
    ):
        self.preferred_engine = preferred_engine
        self.gpu = gpu
        self.engine_name = "none"
        self._easyocr_reader = None
        self._paddle_ocr = None
        self._tesseract_available = False

        self._init_engine()

    def _init_engine(self) -> None:
        if self.preferred_engine in ("auto", "paddle"):
            try:
                from paddleocr import PaddleOCR
                self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                self.engine_name = "paddleocr"
                return
            except Exception:
                pass

        if self.preferred_engine in ("auto", "easyocr"):
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(["en"], gpu=self.gpu, verbose=False)
                self.engine_name = "easyocr"
                return
            except Exception:
                pass

        if self.preferred_engine in ("auto", "tesseract"):
            try:
                import pytesseract
                # Test call
                pytesseract.get_tesseract_version()
                self._tesseract_available = True
                self.engine_name = "pytesseract"
                return
            except Exception:
                pass

        self.engine_name = "cv_structural"

    def recognize_plate(
        self, plate_crop: np.ndarray
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Recognize license plate text from 240x60 crop.

        Args:
            plate_crop: BGR or grayscale plate image.

        Returns:
            Tuple[str, float, bool, Dict[str, Any]]:
                - plate_text: Cleaned and repaired plate string (e.g. 'MH12AB1234')
                - confidence: Float confidence (0.0 to 1.0)
                - is_valid_format: True if valid Indian plate format
                - metadata: Additional diagnostic details
        """
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0, False, {"engine": self.engine_name, "raw_text": ""}

        preprocessed = preprocess_for_ocr(plate_crop)
        raw_text = ""
        ocr_conf = 0.0

        # Try active OCR backend
        if self.engine_name == "paddleocr" and self._paddle_ocr is not None:
            try:
                paddle_input = (
                    cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
                    if len(preprocessed.shape) == 2
                    else preprocessed
                )
                results = self._paddle_ocr.ocr(paddle_input, cls=True)
                lines = []
                conf_list = []
                if results and results[0]:
                    for line in results[0]:
                        text, conf = line[1]
                        lines.append(text)
                        conf_list.append(conf)
                raw_text = " ".join(lines)
                ocr_conf = float(np.mean(conf_list)) if conf_list else 0.5
            except Exception:
                raw_text = ""

        elif self.engine_name == "easyocr" and self._easyocr_reader is not None:
            try:
                results = self._easyocr_reader.readtext(
                    preprocessed,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
                    detail=1,
                )
                lines = []
                conf_list = []
                for _, text, conf in results:
                    lines.append(text)
                    conf_list.append(conf)
                raw_text = " ".join(lines)
                ocr_conf = float(np.mean(conf_list)) if conf_list else 0.5
            except Exception:
                raw_text = ""

        elif self.engine_name == "pytesseract" and self._tesseract_available:
            try:
                import pytesseract
                config = (
                    r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )
                raw_text = pytesseract.image_to_string(preprocessed, config=config).strip()
                ocr_conf = 0.80 if len(raw_text) >= 8 else 0.40
            except Exception:
                raw_text = ""

        # If external OCR engine produced empty or not available, use structural CV reader
        if not raw_text:
            raw_text, ocr_conf = self._recognize_cv_structural(preprocessed)

        # Parse and repair using Indian positional confusion grammar
        repaired_plate, parser_conf, is_valid = repair_indian_plate(raw_text)

        # Weighted final confidence
        final_conf = float(0.40 * ocr_conf + 0.60 * parser_conf) if repaired_plate else 0.0
        final_conf = min(1.0, max(0.0, final_conf))

        metadata = {
            "engine": self.engine_name,
            "raw_text": raw_text,
            "ocr_confidence": ocr_conf,
            "parser_confidence": parser_conf,
            "is_valid_format": is_valid,
        }

        return repaired_plate, final_conf, is_valid, metadata

    def _recognize_cv_structural(self, preprocessed_gray: np.ndarray) -> Tuple[str, float]:
        """
        Computer vision character segmentation and structural feature reader.
        Isolates character bounding boxes from left-to-right, filters noise,
        and analyzes character topology.
        """
        h, w = preprocessed_gray.shape[:2]
        binary = binarize_plate(preprocessed_gray, method="adaptive")

        # Find connected character blobs
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        char_candidates = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Character criteria: height between 25% and 90% of plate height, width > 4px
            if (0.25 * h <= bh <= 0.95 * h) and (bw >= 4) and (bw <= 0.40 * w):
                aspect = bw / float(bh)
                if 0.15 <= aspect <= 1.2:
                    char_candidates.append((x, y, bw, bh, cnt))

        if not char_candidates:
            return "", 0.0

        # Sort characters horizontally from left to right
        char_candidates.sort(key=lambda item: item[0])

        recognized_chars: List[str] = []
        for x, y, bw, bh, cnt in char_candidates:
            char_roi = binary[y : y + bh, x : x + bw]
            ch = self._classify_single_char(char_roi, bw, bh)
            if ch:
                recognized_chars.append(ch)

        raw_str = "".join(recognized_chars)
        conf = float(np.clip(len(recognized_chars) / 10.0, 0.3, 0.85))
        return raw_str, conf

    def _classify_single_char(self, char_roi: np.ndarray, bw: int, bh: int) -> str:
        """
        Topological character classification based on Euler number, aspect ratio,
        and quadrant density distributions.
        """
        if char_roi.size == 0 or bh == 0 or bw == 0:
            return ""

        # Normalize character to 24x24
        resized = cv2.resize(char_roi, (24, 24), interpolation=cv2.INTER_AREA)
        _, norm_bin = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY)

        aspect = bw / float(bh)
        # Narrow characters (e.g. '1', 'I')
        if aspect < 0.30:
            return "1"

        # Count internal holes (Euler characteristic)
        # Invert binary so holes become background components
        pad = np.pad(norm_bin, 1, mode="constant", constant_values=0)
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(pad))
        # External background is label 0, each internal hole is additional label
        num_holes = max(0, num_labels - 2)

        # Quadrant masses
        q_tl = np.mean(norm_bin[0:12, 0:12]) / 255.0
        q_tr = np.mean(norm_bin[0:12, 12:24]) / 255.0
        q_bl = np.mean(norm_bin[12:24, 0:12]) / 255.0
        q_br = np.mean(norm_bin[12:24, 12:24]) / 255.0
        q_mid = np.mean(norm_bin[6:18, 6:18]) / 255.0

        if num_holes >= 2:
            return "8" if (q_tl + q_tr) > 0.6 else "B"
        elif num_holes == 1:
            if q_tl > 0.4 and q_tr > 0.4 and q_bl > 0.4 and q_br > 0.4:
                return "0"
            elif q_bl > 0.4 and q_br < 0.2:
                return "P"
            elif q_tl > 0.4 and q_tr > 0.4 and q_bl < 0.3:
                return "A"
            else:
                return "D"
        else:
            # No holes
            if q_mid < 0.15:
                return "H" if (q_tl > 0.3 and q_tr > 0.3) else "C"
            if (q_tl + q_tr) > (q_bl + q_br) * 1.5:
                return "T"
            if (q_bl + q_br) > (q_tl + q_tr) * 1.5:
                return "L"
            if q_tl > 0.4 and q_br > 0.4 and q_tr < 0.3 and q_bl < 0.3:
                return "Z"
            return "M" if q_tl > 0.4 and q_tr > 0.4 else "X"
