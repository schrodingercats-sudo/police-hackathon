"""
Multi-Object Tracking and Snapshot Quality Selection Module.

Implements:
- KalmanBoxTracker: 8-state linear Kalman Filter for bounding box state estimation
- ByteTrackTracker: Two-stage bipartite matching multi-object tracker
- Snapshot Quality Scoring: Multi-metric quality optimization (Area, Centrality, Sharpness)
- Peak Snapshot Accumulator: Automatically maintains the highest-quality vehicle crop per tracklet
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

from stolen_vehicle_ai.vision.detector import VehicleDetection
from stolen_vehicle_ai.vision.preprocessor import laplacian_variance


def compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2).
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0

    area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return float(intersection / max(1e-6, union))


def compute_iou_matrix(
    bboxes1: List[Tuple[int, int, int, int]], bboxes2: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    """
    Compute IoU matrix between list of bboxes1 (M) and bboxes2 (N) -> (M, N).
    """
    if not bboxes1 or not bboxes2:
        return np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)

    iou_mat = np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)
    for i, b1 in enumerate(bboxes1):
        for j, b2 in enumerate(bboxes2):
            iou_mat[i, j] = compute_iou(b1, b2)
    return iou_mat


def compute_snapshot_quality(
    vehicle_crop: np.ndarray,
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, ...],
) -> float:
    """
    Calculate composite snapshot quality score Q in [0.0, 1.0].

    Formula:
        Q = 0.35 * AreaScore + 0.20 * CentralityScore + 0.45 * SharpnessScore

    Args:
        vehicle_crop: BGR vehicle crop.
        bbox: Bounding box (x1, y1, x2, y2).
        frame_shape: Shape of original full frame (H, W, ...).

    Returns:
        float: Composite quality score between 0.0 and 1.0.
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return 0.0

    ch, cw = vehicle_crop.shape[:2]
    area = cw * ch

    # 1. Area Score: normalized against optimal 300x300 target
    area_score = min(1.0, area / (300.0 * 300.0))

    # 2. Centrality Score: distance of bbox center from frame center
    fh, fw = frame_shape[:2]
    if fw > 0 and fh > 0:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        norm_dx = (cx - (fw / 2.0)) / fw
        norm_dy = (cy - (fh / 2.0)) / fh
        dist_from_center = np.sqrt(norm_dx ** 2 + norm_dy ** 2)
        # Max distance from center in normalized coords is sqrt(0.5^2 + 0.5^2) = 0.707
        centrality_score = float(max(0.0, 1.0 - (dist_from_center * np.sqrt(2.0))))
    else:
        centrality_score = 0.5

    # 3. Sharpness Score via Laplacian Variance
    lap_var = laplacian_variance(vehicle_crop)
    sharpness_score = float(min(1.0, lap_var / 250.0))

    # Composite weighted formula
    total_quality = 0.35 * area_score + 0.20 * centrality_score + 0.45 * sharpness_score
    return float(np.clip(total_quality, 0.0, 1.0))


class KalmanBoxTracker:
    """
    Kalman Filter for single-object bounding box tracking.
    State: [u, v, a, h, u_dot, v_dot, a_dot, h_dot]^T
    where (u, v) is center, a is aspect ratio (w/h), h is height.
    """

    count = 0

    def __init__(self, bbox: Tuple[int, int, int, int]):
        """
        Initialize tracker with initial bounding box (x1, y1, x2, y2).
        """
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # State dimensionality
        self.dim_x = 8
        self.dim_z = 4

        # State transition matrix F
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # State vector x
        self.x = np.zeros((8, 1), dtype=np.float32)
        self._init_state(bbox)

        # Covariance P
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # High uncertainty on initial velocity

        # Process noise Q
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01

        # Measurement noise R
        self.R = np.eye(4, dtype=np.float32) * 1.0

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def _init_state(self, bbox: Tuple[int, int, int, int]) -> None:
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
        u = bbox[0] + w / 2.0
        v = bbox[1] + h / 2.0
        a = w / float(h)
        self.x[0, 0] = u
        self.x[1, 0] = v
        self.x[2, 0] = a
        self.x[3, 0] = h

    def predict(self) -> Tuple[int, int, int, int]:
        """
        Advance state vector and return predicted bounding box.
        """
        if (self.x[6, 0] + self.x[2, 0]) <= 0:
            self.x[6, 0] = 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_state_bbox()

    def update(self, bbox: Tuple[int, int, int, int]) -> None:
        """
        Update state vector with observed bounding box.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
        u = bbox[0] + w / 2.0
        v = bbox[1] + h / 2.0
        a = w / float(h)
        z = np.array([[u], [v], [a], [h]], dtype=np.float32)

        # Innovation / Residual
        y = z - np.dot(self.H, self.x)
        # Innovation covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        # Kalman Gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # State update
        self.x = self.x + np.dot(K, y)
        # Covariance update
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def get_state_bbox(self) -> Tuple[int, int, int, int]:
        """
        Convert current state vector [u, v, a, h] to (x1, y1, x2, y2).
        """
        u = self.x[0, 0]
        v = self.x[1, 0]
        a = max(1e-2, self.x[2, 0])
        h = max(1.0, self.x[3, 0])
        w = a * h

        x1 = int(round(u - w / 2.0))
        y1 = int(round(v - h / 2.0))
        x2 = int(round(u + w / 2.0))
        y2 = int(round(v + h / 2.0))

        return (x1, y1, x2, y2)


class Tracklet:
    """
    Representation of a tracked vehicle across multiple frames.
    Maintains best snapshot crop, trajectory, and classification statistics.
    """

    def __init__(self, track_id: int, detection: VehicleDetection, quality_score: float):
        self.track_id = track_id
        self.class_name = detection.class_name
        self.class_id = detection.class_id
        self.confidences: List[float] = [detection.confidence]
        self.bboxes: List[Tuple[int, int, int, int]] = [detection.bbox]
        self.quality_scores: List[float] = [quality_score]

        # Peak quality snapshot
        self.best_quality: float = quality_score
        self.best_crop: np.ndarray = detection.crop.copy()
        self.best_bbox: Tuple[int, int, int, int] = detection.bbox
        self.total_frames: int = 1

    def update(
        self, detection: VehicleDetection, quality_score: float
    ) -> None:
        self.confidences.append(detection.confidence)
        self.bboxes.append(detection.bbox)
        self.quality_scores.append(quality_score)
        self.total_frames += 1

        if quality_score > self.best_quality:
            self.best_quality = quality_score
            self.best_crop = detection.crop.copy()
            self.best_bbox = detection.bbox
            self.class_name = detection.class_name
            self.class_id = detection.class_id

    @property
    def latest_bbox(self) -> Tuple[int, int, int, int]:
        return self.bboxes[-1]

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.confidences)) if self.confidences else 0.0


class ByteTrackTracker:
    """
    ByteTrack Multi-Object Tracker.
    Associates high-confidence detections first, then low-confidence detections
    to preserve tracks through occlusions and motion blur.
    """

    def __init__(
        self,
        high_thresh: float = 0.50,
        low_thresh: float = 0.10,
        match_thresh: float = 0.70,  # Max cost (1 - IoU threshold = 0.30 IoU minimum)
        max_age: int = 30,
        min_hits: int = 2,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits

        self.trackers: List[KalmanBoxTracker] = []
        self.active_tracklets: Dict[int, Tracklet] = {}
        self.frame_count = 0

    def update(
        self,
        detections: List[VehicleDetection],
        frame_shape: Tuple[int, ...],
    ) -> List[Tracklet]:
        """
        Process new frame detections and update track states.

        Args:
            detections: List of VehicleDetection from current frame.
            frame_shape: (H, W, ...) shape of the frame.

        Returns:
            List of confirmed active Tracklets visible in current frame.
        """
        self.frame_count += 1

        # 1. Predict new positions for all existing trackers
        predicted_boxes = []
        for trk in self.trackers:
            pred_box = trk.predict()
            predicted_boxes.append(pred_box)

        # 2. Partition detections into high and low confidence sets
        high_dets: List[Tuple[int, VehicleDetection]] = []
        low_dets: List[Tuple[int, VehicleDetection]] = []

        for i, det in enumerate(detections):
            if det.confidence >= self.high_thresh:
                high_dets.append((i, det))
            elif det.confidence >= self.low_thresh:
                low_dets.append((i, det))

        # 3. First Association: High-confidence detections with existing tracks
        high_det_boxes = [det.bbox for _, det in high_dets]
        matched_tracks_1, unmatched_tracks_1, unmatched_dets_1 = self._associate(
            predicted_boxes, high_det_boxes, max_distance=self.match_thresh
        )

        # Update matched trackers from first round
        for t_idx, d_idx in matched_tracks_1:
            det = high_dets[d_idx][1]
            trk = self.trackers[t_idx]
            trk.update(det.bbox)
            q = compute_snapshot_quality(det.crop, det.bbox, frame_shape)
            if trk.id in self.active_tracklets:
                self.active_tracklets[trk.id].update(det, q)
            else:
                self.active_tracklets[trk.id] = Tracklet(trk.id, det, q)

        # 4. Second Association: Remaining unmatched tracks with low-confidence detections
        remaining_track_indices = unmatched_tracks_1
        remaining_predicted_boxes = [predicted_boxes[i] for i in remaining_track_indices]
        low_det_boxes = [det.bbox for _, det in low_dets]

        matched_tracks_2, unmatched_tracks_2, _ = self._associate(
            remaining_predicted_boxes, low_det_boxes, max_distance=0.85  # relaxed for low-conf
        )

        # Update matched trackers from second round
        for rel_t_idx, d_idx in matched_tracks_2:
            orig_t_idx = remaining_track_indices[rel_t_idx]
            det = low_dets[d_idx][1]
            trk = self.trackers[orig_t_idx]
            trk.update(det.bbox)
            q = compute_snapshot_quality(det.crop, det.bbox, frame_shape)
            if trk.id in self.active_tracklets:
                self.active_tracklets[trk.id].update(det, q)
            else:
                self.active_tracklets[trk.id] = Tracklet(trk.id, det, q)

        # 5. Initialize new trackers for unmatched high-confidence detections
        for d_idx in unmatched_dets_1:
            det = high_dets[d_idx][1]
            new_trk = KalmanBoxTracker(det.bbox)
            self.trackers.append(new_trk)
            q = compute_snapshot_quality(det.crop, det.bbox, frame_shape)
            self.active_tracklets[new_trk.id] = Tracklet(new_trk.id, det, q)

        # 6. Prune dead trackers
        surviving_trackers: List[KalmanBoxTracker] = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                surviving_trackers.append(trk)
            else:
                # Remove inactive tracklet from active pool
                self.active_tracklets.pop(trk.id, None)

        self.trackers = surviving_trackers

        # 7. Collect confirmed active tracklets for output
        active_results: List[Tracklet] = []
        for trk in self.trackers:
            if trk.time_since_update == 0 and (
                trk.hits >= self.min_hits or self.frame_count <= self.min_hits
            ):
                if trk.id in self.active_tracklets:
                    active_results.append(self.active_tracklets[trk.id])

        return active_results

    def _associate(
        self,
        track_boxes: List[Tuple[int, int, int, int]],
        det_boxes: List[Tuple[int, int, int, int]],
        max_distance: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Greedy bipartite matching based on IoU distance (1 - IoU).
        """
        if not track_boxes or not det_boxes:
            return [], list(range(len(track_boxes))), list(range(len(det_boxes)))

        iou_matrix = compute_iou_matrix(track_boxes, det_boxes)
        cost_matrix = 1.0 - iou_matrix

        matched_tracks = []
        matched_dets = set()
        matched_trks = set()

        # Sort all pairs by ascending cost
        num_trks, num_dets = cost_matrix.shape
        flat_indices = np.argsort(cost_matrix, axis=None)

        for flat_idx in flat_indices:
            t_idx = int(flat_idx // num_dets)
            d_idx = int(flat_idx % num_dets)

            if t_idx in matched_trks or d_idx in matched_dets:
                continue

            if cost_matrix[t_idx, d_idx] <= max_distance:
                matched_trks.add(t_idx)
                matched_dets.add(d_idx)
                matched_tracks.append((t_idx, d_idx))

        unmatched_tracks = [i for i in range(num_trks) if i not in matched_trks]
        unmatched_dets = [j for j in range(num_dets) if j not in matched_dets]

        return matched_tracks, unmatched_tracks, unmatched_dets

    def reset(self) -> None:
        """
        Reset tracker state.
        """
        self.trackers.clear()
        self.active_tracklets.clear()
        self.frame_count = 0
