"""
Visual Re-Identification (Re-ID) and Dominant Vehicle Color Classifier.

Provides:
- ReIDFeatureExtractor: Generates 512-dimensional L2-normalized metric embeddings (||e||_2 = 1.0)
  for cross-camera vehicle re-identification and fingerprinting.
- DominantColorClassifier: Classifies vehicle body color into 10 canonical Indian vehicle colors
  using robust central body HSV spatial masking.
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import cv2


class ReIDFeatureExtractor:
    """
    512-Dimensional Visual Re-Identification Feature Extractor.
    Extracts dense multi-zone color, gradient, and spatial texture representations,
    projecting onto a 512-D unit hypersphere.
    """

    EMBEDDING_DIM = 512

    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = False):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self._deep_model = None
        self._projection_matrix: np.ndarray = self._init_projection_matrix()

    def _init_projection_matrix(self) -> np.ndarray:
        """
        Generates a deterministic orthonormal random projection matrix for metric embedding.
        """
        # Fixed seed for reproducibility across all runs and processes
        rng = np.random.RandomState(42)
        # Input raw feature dimension: 16 zones * (16 HSV + 16 LAB + 16 Grad bins) = 768
        raw_dim = 16 * 48
        rand_mat = rng.randn(raw_dim, self.EMBEDDING_DIM).astype(np.float32)
        # Orthonormalize columns using QR decomposition
        q, _ = np.linalg.qr(rand_mat)
        return q.astype(np.float32)

    def extract_embedding(self, vehicle_crop: np.ndarray) -> List[float]:
        """
        Extract 512-dimensional L2-normalized float embedding from a vehicle image crop.

        Args:
            vehicle_crop: BGR vehicle image crop.

        Returns:
            List[float]: 512-D unit vector where sum(e_i^2) == 1.0.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            vec = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
            vec[0] = 1.0
            return vec.tolist()

        # Resize to standard vehicle aspect (128x128)
        resized = cv2.resize(vehicle_crop, (128, 128), interpolation=cv2.INTER_AREA)

        # Convert color spaces
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Compute gradient magnitudes & orientations
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        zone_features: List[np.ndarray] = []

        # Divide into 4x4 spatial zones (16 zones)
        grid_size = 4
        zh = 128 // grid_size
        zw = 128 // grid_size

        for r in range(grid_size):
            for c in range(grid_size):
                y1, y2 = r * zh, (r + 1) * zh
                x1, x2 = c * zw, (c + 1) * zw

                # 1. HSV 16-bin histogram
                hsv_roi = hsv[y1:y2, x1:x2]
                h_hist = cv2.calcHist([hsv_roi], [0], None, [8], [0, 180]).flatten()
                s_hist = cv2.calcHist([hsv_roi], [1], None, [4], [0, 256]).flatten()
                v_hist = cv2.calcHist([hsv_roi], [2], None, [4], [0, 256]).flatten()
                hsv_vec = np.concatenate([h_hist, s_hist, v_hist])
                hsv_vec = hsv_vec / (np.sum(hsv_vec) + 1e-6)

                # 2. LAB 16-bin histogram
                lab_roi = lab[y1:y2, x1:x2]
                l_hist = cv2.calcHist([lab_roi], [0], None, [6], [0, 256]).flatten()
                a_hist = cv2.calcHist([lab_roi], [1], None, [5], [0, 256]).flatten()
                b_hist = cv2.calcHist([lab_roi], [2], None, [5], [0, 256]).flatten()
                lab_vec = np.concatenate([l_hist, a_hist, b_hist])
                lab_vec = lab_vec / (np.sum(lab_vec) + 1e-6)

                # 3. Gradient orientation 16-bin histogram
                ang_roi = angle[y1:y2, x1:x2]
                mag_roi = mag[y1:y2, x1:x2]
                grad_hist, _ = np.histogram(ang_roi, bins=16, range=(0, 360), weights=mag_roi)
                grad_hist = grad_hist.astype(np.float32)
                grad_hist = grad_hist / (np.sum(grad_hist) + 1e-6)

                zone_feat = np.concatenate([hsv_vec, lab_vec, grad_hist])
                zone_features.append(zone_feat)

        raw_vector = np.concatenate(zone_features).astype(np.float32)  # shape (768,)

        # Project through orthonormal metric matrix into 512 dimensions
        embedding = np.dot(raw_vector, self._projection_matrix)

        # L2-normalization onto unit sphere: ||e||_2 = 1.0
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm
        else:
            embedding = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
            embedding[0] = 1.0

        return embedding.tolist()

    @staticmethod
    def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two L2-normalized embeddings.
        Since ||vec1|| = ||vec2|| = 1.0, cos(theta) is simply the dot product.
        """
        a = np.asarray(vec1, dtype=np.float32)
        b = np.asarray(vec2, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        sim = float(np.dot(a, b) / (norm_a * norm_b))
        return float(np.clip(sim, -1.0, 1.0))


class DominantColorClassifier:
    """
    Robust 10-Class Vehicle Dominant Color Classifier.
    Classes: 'white', 'black', 'silver/grey', 'red', 'blue', 'yellow', 'green', 'orange', 'brown', 'other'
    """

    COLOR_CLASSES = [
        "white",
        "black",
        "silver/grey",
        "red",
        "blue",
        "yellow",
        "green",
        "orange",
        "brown",
        "other",
    ]

    def __init__(self):
        pass

    def classify_color(self, vehicle_crop: np.ndarray) -> Tuple[str, float]:
        """
        Extract vehicle body region and classify into one of 10 dominant colors.

        Args:
            vehicle_crop: BGR vehicle image crop.

        Returns:
            Tuple[str, float]: (color_name, confidence)
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "other", 0.0

        vh, vw = vehicle_crop.shape[:2]
        if vh < 10 or vw < 10:
            return "other", 0.0

        # Mask body region: exclude top 20% (windshield/sky) and bottom 15% (wheels/road shadow)
        # and exclude lateral 10% margins to avoid background spill
        y1 = int(vh * 0.20)
        y2 = int(vh * 0.85)
        x1 = int(vw * 0.10)
        x2 = int(vw * 0.90)

        body_roi = vehicle_crop[y1:y2, x1:x2]
        if body_roi.size == 0:
            body_roi = vehicle_crop

        hsv = cv2.cvtColor(body_roi, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        total_pixels = body_roi.shape[0] * body_roi.shape[1]
        if total_pixels == 0:
            return "other", 0.0

        # Color masks
        # White: low saturation, high value
        white_mask = (s_chan <= 38) & (v_chan >= 140)

        # Black: very low value
        black_mask = v_chan <= 45

        # Silver / Grey: low saturation, mid value
        silver_mask = (s_chan <= 45) & (v_chan > 45) & (v_chan < 140)

        # Chromatic colors (require noticeable saturation and brightness)
        chromatic = (s_chan > 45) & (v_chan > 45)

        # Red: wraps around 0 and 180 in OpenCV HSV (0-8, 165-180)
        red_mask = chromatic & ((h_chan <= 8) | (h_chan >= 165))

        # Orange: H in (8, 22], bright (V > 135)
        orange_mask = chromatic & (h_chan > 8) & (h_chan <= 22) & (v_chan > 135)

        # Brown / Bronze: H in [9, 24], moderate brightness (35 <= V <= 135)
        brown_mask = chromatic & (h_chan >= 9) & (h_chan <= 24) & (v_chan <= 135)

        # Yellow: H in (22, 38]
        yellow_mask = chromatic & (h_chan > 22) & (h_chan <= 38)

        # Green: H in (38, 85]
        green_mask = chromatic & (h_chan > 38) & (h_chan <= 85)

        # Blue: H in (85, 135]
        blue_mask = chromatic & (h_chan > 85) & (h_chan <= 135)

        counts = {
            "white": np.count_nonzero(white_mask),
            "black": np.count_nonzero(black_mask),
            "silver/grey": np.count_nonzero(silver_mask),
            "red": np.count_nonzero(red_mask),
            "orange": np.count_nonzero(orange_mask),
            "yellow": np.count_nonzero(yellow_mask),
            "green": np.count_nonzero(green_mask),
            "blue": np.count_nonzero(blue_mask),
            "brown": np.count_nonzero(brown_mask),
        }

        # Find maximum color bin
        dominant_color = max(counts, key=counts.get)
        max_count = counts[dominant_color]

        confidence = float(max_count / float(total_pixels))

        if confidence < 0.15:
            return "other", float(np.clip(1.0 - confidence, 0.40, 0.70))

        # Normalize confidence to high probability
        norm_conf = float(np.clip(0.60 + 0.38 * (confidence / 0.50), 0.50, 0.98))
        return dominant_color, norm_conf
