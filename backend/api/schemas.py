"""Pydantic v2 Request & Response Schemas for Indian Police Stolen Vehicle AI System.

Standard: ISO/IEC/IEEE 29119 & Section 65B BSA Forensic Standards
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class VerificationStatusEnum(str, Enum):
    """Officer verification statuses."""
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    FLAGGED = "flagged"


class CaseStatusEnum(str, Enum):
    """Investigation case statuses."""
    OPEN = "open"
    TRACKING = "tracking"
    RECOVERED = "recovered"
    CLOSED = "closed"


class CameraStatusEnum(str, Enum):
    """Surveillance camera operational statuses."""
    ACTIVE = "active"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


# -----------------------------------------------------------------------------
# Case Schemas
# -----------------------------------------------------------------------------
class CaseCreateRequest(BaseModel):
    """Payload for registering a new First Information Report (FIR) stolen vehicle case."""
    fir_number: str = Field(..., description="Unique FIR identifier, e.g. FIR-2026-DEL-0941")
    reported_plate: str = Field(..., description="Reported registration license plate number")
    theft_datetime: datetime = Field(..., description="Timestamp of reported theft")
    theft_latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude of theft location")
    theft_longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude of theft location")
    theft_location_name: str = Field(..., description="Human readable theft location description")
    vehicle_type: str = Field(..., description="Vehicle category, e.g. car, motorcycle, truck, auto_rickshaw")
    vehicle_color: str = Field(..., description="Dominant vehicle color, e.g. white, black, red")
    make: Optional[str] = Field(None, description="Vehicle make/manufacturer, e.g. Hyundai, Maruti")
    model: Optional[str] = Field(None, description="Vehicle model name, e.g. Creta, Swift")
    distinctive_features: Optional[str] = Field(None, description="Distinctive visual markings or modifications")
    investigating_officer: str = Field(..., description="Assigned investigating officer name and badge")
    police_station: str = Field(..., description="Jurisdiction police station name")
    reference_image_base64: Optional[str] = Field(None, description="Base64-encoded reference photo of stolen vehicle")


class CaseResponse(BaseModel):
    """Response model for stolen vehicle FIR case details."""
    id: int
    fir_number: str
    reported_plate: str
    theft_datetime: datetime
    theft_latitude: float
    theft_longitude: float
    theft_location_name: str
    vehicle_type: str
    vehicle_color: str
    make: Optional[str] = None
    model: Optional[str] = None
    distinctive_features: Optional[str] = None
    status: str
    investigating_officer: str
    police_station: str
    reference_image_path: Optional[str] = None
    total_candidate_sightings: int = 0
    verified_sightings: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# -----------------------------------------------------------------------------
# Sighting & Search Schemas
# -----------------------------------------------------------------------------
class SearchRequestParams(BaseModel):
    """Query parameters for multi-criteria sighting search."""
    plate: Optional[str] = Field(None, description="Filter by license plate number")
    vehicle_type: Optional[str] = Field(None, description="Filter by vehicle body type")
    color: Optional[str] = Field(None, description="Filter by vehicle color")
    camera_id: Optional[str] = Field(None, description="Filter by surveillance camera node ID")
    start_time: Optional[datetime] = Field(None, description="Start of time window")
    end_time: Optional[datetime] = Field(None, description="End of time window")
    min_confidence: float = Field(0.50, ge=0.0, le=1.0, description="Minimum plate OCR confidence threshold")
    min_visual_similarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum visual Re-ID cosine similarity")
    page: int = Field(1, ge=1, description="Page number for pagination")
    limit: int = Field(20, ge=1, le=100, description="Items per page")


class SightingDTO(BaseModel):
    """Data Transfer Object representing a vehicle detection event."""
    id: int
    camera_id: str
    camera_name: Optional[str] = None
    camera_lat: Optional[float] = None
    camera_lon: Optional[float] = None
    road_name: Optional[str] = None
    timestamp: datetime
    track_id: Optional[int] = None
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0
    plate_box: Optional[List[int]] = None
    vehicle_box: Optional[List[int]] = None
    vehicle_type: Optional[str] = None
    vehicle_color: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    composite_score: Optional[float] = None
    image_url: Optional[str] = None
    crop_url: Optional[str] = None
    plate_crop_url: Optional[str] = None
    sha256_hash: Optional[str] = None
    direction_travel: Optional[str] = None
    speed_estimate_kmh: Optional[float] = None
    verification_status: Optional[str] = "pending"
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class SearchResponse(BaseModel):
    """Paginated search response for vehicle sightings."""
    total_count: int
    page: int
    limit: int
    results: List[SightingDTO]


# -----------------------------------------------------------------------------
# Route Reconstruction Schemas
# -----------------------------------------------------------------------------
class RouteWaypointDTO(BaseModel):
    """Chronological waypoint on the reconstructed vehicle trajectory."""
    sequence: int
    sighting_id: int
    camera_id: str
    camera_name: str
    latitude: float
    longitude: float
    road_name: str
    timestamp: datetime
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0
    composite_score: float = 0.0
    speed_from_prev_kmh: Optional[float] = None
    distance_from_prev_km: Optional[float] = None
    time_delta_sec: Optional[float] = None
    is_spatially_feasible: bool = True
    verification_status: str = "pending"
    vehicle_crop_url: Optional[str] = None
    plate_crop_url: Optional[str] = None


class RouteSegmentDTO(BaseModel):
    """Kinematic transition between consecutive camera detections."""
    from_camera_id: str
    to_camera_id: str
    from_coords: List[float] = Field(..., description="[lat, lon] of departure camera")
    to_coords: List[float] = Field(..., description="[lat, lon] of arrival camera")
    distance_km: float
    elapsed_time_sec: float
    speed_kmh: float
    feasible: bool


class RouteReconstructionResponse(BaseModel):
    """Complete chronological route trajectory and kinematic analysis."""
    case_id: int
    fir_number: str
    reported_plate: str
    theft_origin: Dict[str, Any]
    waypoints: List[RouteWaypointDTO]
    segments: List[RouteSegmentDTO]
    total_distance_km: float
    estimated_duration_min: float
    route_status: str = "Active Lead"
    last_known_camera: Optional[str] = None
    last_seen_timestamp: Optional[datetime] = None


# -----------------------------------------------------------------------------
# Verification Schemas (Human in the Loop)
# -----------------------------------------------------------------------------
class VerificationRequest(BaseModel):
    """Officer sighting verification payload."""
    case_id: int
    sighting_id: int
    status: str = Field(..., description="Verification decision: 'verified' or 'rejected'")
    officer_badge_id: str = Field(..., description="Badge ID of reviewing officer")
    notes: Optional[str] = Field(None, description="Investigation review notes and visual rationale")


class VerificationResponse(BaseModel):
    """Response confirming officer verification action."""
    match_id: Optional[int] = None
    case_id: int
    sighting_id: int
    status: str
    updated_at: datetime
    message: str


# -----------------------------------------------------------------------------
# Camera Schemas
# -----------------------------------------------------------------------------
class CameraCreateRequest(BaseModel):
    """Payload for registering a new surveillance camera node."""
    id: str = Field(..., description="Unique camera identifier, e.g. CAM_DEL_007")
    name: str = Field(..., description="Descriptive camera location name")
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    road_name: str = Field(...)
    direction_bearing: float = Field(0.0, ge=0.0, le=360.0)
    camera_type: str = Field("junction", description="urban, junction, toll, highway")
    stream_url: Optional[str] = None
    status: str = Field("active", description="active, offline, maintenance")


class CameraStatusDTO(BaseModel):
    """Surveillance camera node telemetry status."""
    id: str
    name: str
    latitude: float
    longitude: float
    road_name: str
    status: str
    total_detections_today: int = 0
    last_ping: Optional[datetime] = None
    camera_type: Optional[str] = "junction"
    direction_bearing: Optional[float] = 0.0

    model_config = ConfigDict(from_attributes=True)


# -----------------------------------------------------------------------------
# Dashboard Telemetry Schemas
# -----------------------------------------------------------------------------
class DashboardStatsResponse(BaseModel):
    """Real-time operational command center metrics."""
    active_cases: int
    total_cameras: int
    active_cameras: int
    detections_today: int
    matches_today: int
    recovery_rate_pct: float
    recent_alerts: List[Dict[str, Any]]
    active_camera_list: List[CameraStatusDTO]


# -----------------------------------------------------------------------------
# Forensic Evidence & Section 65B Export Schemas
# -----------------------------------------------------------------------------
class EvidenceExportResponse(BaseModel):
    """Court-admissible electronic evidence dossier with Section 65B checksums."""
    fir_number: str
    case_id: int
    export_timestamp: datetime
    generated_by: str
    evidence_hash_sha256: str
    case_summary: Dict[str, Any]
    chain_of_custody: List[Dict[str, Any]]
    verified_timeline: List[Dict[str, Any]]
    download_url_pdf: Optional[str] = None
    html_printable_view: Optional[str] = None
