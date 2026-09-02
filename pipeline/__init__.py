"""
Pipeline Module for Indian Police Stolen Vehicle AI System.
"""

from stolen_vehicle_ai.pipeline.snapshot_manager import SnapshotManager
from stolen_vehicle_ai.pipeline.video_processor import SightingResult, VideoProcessor

__all__ = [
    "SnapshotManager",
    "SightingResult",
    "VideoProcessor",
]
