"""SQLAlchemy ORM Models for Indian Police Stolen Vehicle AI Command Center."""

from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utc_now() -> datetime:
    """Returns current UTC timestamp without timezone offset issues."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class VerificationStatus(str, PyEnum):
    """Officer verification statuses for case sighting matches."""
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    FLAGGED = "flagged"


class CaseStatus(str, PyEnum):
    """Investigation lifecycle statuses for stolen vehicle FIR cases."""
    OPEN = "open"
    TRACKING = "tracking"
    RECOVERED = "recovered"
    CLOSED = "closed"


class CameraStatus(str, PyEnum):
    """Operational status of a surveillance camera node."""
    ACTIVE = "active"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class Camera(Base):
    """CCTV / ANPR camera node in the surveillance network."""

    __tablename__ = "cameras"

    id = Column(String(50), primary_key=True)  # e.g., "CAM_DEL_001"
    name = Column(String(150), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    road_name = Column(String(200), nullable=False)
    direction_bearing = Column(Float, default=0.0)  # 0.0 to 360.0 degrees
    camera_type = Column(String(50), default="junction")  # toll, highway, junction, urban
    stream_url = Column(String(500), nullable=True)
    status = Column(String(20), default=CameraStatus.ACTIVE.value)  # active, offline, maintenance
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Relationships
    sightings = relationship("Sighting", back_populates="camera", cascade="all, delete-orphan")
    outgoing_topology = relationship(
        "CameraTopology",
        foreign_keys="CameraTopology.from_camera_id",
        back_populates="from_camera",
        cascade="all, delete-orphan",
    )
    incoming_topology = relationship(
        "CameraTopology",
        foreign_keys="CameraTopology.to_camera_id",
        back_populates="to_camera",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_camera_coords", "latitude", "longitude"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "road_name": self.road_name,
            "direction_bearing": self.direction_bearing,
            "camera_type": self.camera_type,
            "stream_url": self.stream_url,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Vehicle(Base):
    """Distinct vehicle record indexed by license plate."""

    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plate_number = Column(String(30), unique=True, index=True, nullable=False)
    vehicle_type = Column(String(50), nullable=True)  # car, motorcycle, auto_rickshaw, bus, truck
    color = Column(String(50), nullable=True)
    make = Column(String(50), nullable=True)
    model = Column(String(50), nullable=True)
    first_seen = Column(DateTime, default=utc_now)
    last_seen = Column(DateTime, default=utc_now)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    sightings = relationship("Sighting", back_populates="vehicle")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary."""
        return {
            "id": self.id,
            "plate_number": self.plate_number,
            "vehicle_type": self.vehicle_type,
            "color": self.color,
            "make": self.make,
            "model": self.model,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Sighting(Base):
    """Individual vehicle detection event captured by a camera node."""

    __tablename__ = "vehicle_sightings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(50), ForeignKey("cameras.id"), nullable=False)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    track_id = Column(Integer, nullable=True)

    # ANPR & OCR Attributes
    plate_text = Column(String(30), index=True, nullable=True)
    plate_confidence = Column(Float, default=0.0)
    plate_box = Column(JSON, nullable=True)  # [x1, y1, x2, y2]

    # Vehicle Visual Attributes & Bounding Box
    vehicle_box = Column(JSON, nullable=False)  # [x1, y1, x2, y2]
    vehicle_type = Column(String(50), nullable=True)
    vehicle_color = Column(String(50), nullable=True)
    make = Column(String(50), nullable=True)
    model = Column(String(50), nullable=True)

    # Re-ID Embedding Vector (512-D L2-normalized float list stored as JSON)
    embedding = Column(JSON, nullable=True)

    # Evidence Media Artifacts & Forensic Checksum
    image_path = Column(String(500), nullable=False)  # Full scene frame
    crop_path = Column(String(500), nullable=False)   # Vehicle crop
    plate_crop_path = Column(String(500), nullable=True)  # License plate crop
    sha256_hash = Column(String(64), nullable=True)  # Section 65B BSA Cryptographic Digest

    # Kinematics & Tracking
    direction_travel = Column(String(50), nullable=True)
    speed_estimate_kmh = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    camera = relationship("Camera", back_populates="sightings")
    vehicle = relationship("Vehicle", back_populates="sightings")
    case_matches = relationship("CaseMatch", back_populates="sighting", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_sighting_cam_time", "camera_id", "timestamp"),
        Index("idx_sighting_plate_time", "plate_text", "timestamp"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary."""
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "vehicle_id": self.vehicle_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "track_id": self.track_id,
            "plate_text": self.plate_text,
            "plate_confidence": self.plate_confidence,
            "plate_box": self.plate_box,
            "vehicle_box": self.vehicle_box,
            "vehicle_type": self.vehicle_type,
            "vehicle_color": self.vehicle_color,
            "make": self.make,
            "model": self.model,
            "embedding_dim": len(self.embedding) if self.embedding else 0,
            "image_path": self.image_path,
            "crop_path": self.crop_path,
            "plate_crop_path": self.plate_crop_path,
            "sha256_hash": self.sha256_hash,
            "direction_travel": self.direction_travel,
            "speed_estimate_kmh": self.speed_estimate_kmh,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Case(Base):
    """Police First Information Report (FIR) stolen vehicle tracking case."""

    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fir_number = Column(String(100), unique=True, index=True, nullable=False)
    reported_plate = Column(String(30), index=True, nullable=False)
    theft_datetime = Column(DateTime, nullable=False)
    theft_latitude = Column(Float, nullable=False)
    theft_longitude = Column(Float, nullable=False)
    theft_location_name = Column(String(250), nullable=False)

    # Expected vehicle characteristics
    vehicle_type = Column(String(50), nullable=False)
    vehicle_color = Column(String(50), nullable=False)
    make = Column(String(50), nullable=True)
    model = Column(String(50), nullable=True)
    distinctive_features = Column(Text, nullable=True)

    # Reference Photo & Embedding
    reference_image_path = Column(String(500), nullable=True)
    reference_embedding = Column(JSON, nullable=True)  # 512-D float list

    # Status & Assignment
    status = Column(Enum(CaseStatus), default=CaseStatus.OPEN, nullable=False)
    investigating_officer = Column(String(150), nullable=False)
    police_station = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Relationships
    matches = relationship("CaseMatch", back_populates="case", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary."""
        return {
            "id": self.id,
            "fir_number": self.fir_number,
            "reported_plate": self.reported_plate,
            "theft_datetime": self.theft_datetime.isoformat() if self.theft_datetime else None,
            "theft_latitude": self.theft_latitude,
            "theft_longitude": self.theft_longitude,
            "theft_location_name": self.theft_location_name,
            "vehicle_type": self.vehicle_type,
            "vehicle_color": self.vehicle_color,
            "make": self.make,
            "model": self.model,
            "distinctive_features": self.distinctive_features,
            "reference_image_path": self.reference_image_path,
            "status": self.status.value if isinstance(self.status, CaseStatus) else self.status,
            "investigating_officer": self.investigating_officer,
            "police_station": self.police_station,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class CaseMatch(Base):
    """Association between a stolen vehicle case and a candidate camera sighting."""

    __tablename__ = "case_matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    sighting_id = Column(Integer, ForeignKey("vehicle_sightings.id"), nullable=False)

    # Detailed Match Breakdown
    plate_score = Column(Float, default=0.0)
    visual_score = Column(Float, default=0.0)
    attribute_score = Column(Float, default=0.0)
    spatio_temporal_score = Column(Float, default=1.0)
    composite_score = Column(Float, default=0.0)

    # Route Sequencing
    sequence_order = Column(Integer, nullable=True)

    # Officer Verification (Human in the Loop)
    verification_status = Column(Enum(VerificationStatus), default=VerificationStatus.PENDING, nullable=False)
    reviewed_by = Column(String(100), nullable=True)  # Officer Badge ID
    reviewed_at = Column(DateTime, nullable=True)
    review_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    case = relationship("Case", back_populates="matches")
    sighting = relationship("Sighting", back_populates="case_matches")

    __table_args__ = (
        Index("idx_case_match_case_score", "case_id", "composite_score"),
        Index("idx_case_match_case_sighting", "case_id", "sighting_id", unique=True),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary."""
        return {
            "id": self.id,
            "case_id": self.case_id,
            "sighting_id": self.sighting_id,
            "plate_score": self.plate_score,
            "visual_score": self.visual_score,
            "attribute_score": self.attribute_score,
            "spatio_temporal_score": self.spatio_temporal_score,
            "composite_score": self.composite_score,
            "sequence_order": self.sequence_order,
            "verification_status": (
                self.verification_status.value
                if isinstance(self.verification_status, VerificationStatus)
                else self.verification_status
            ),
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "review_notes": self.review_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CameraTopology(Base):
    """Spatio-temporal adjacency graph edge between two surveillance camera nodes."""

    __tablename__ = "camera_topology"

    id = Column(Integer, primary_key=True, autoincrement=True)
    from_camera_id = Column(String(50), ForeignKey("cameras.id"), nullable=False)
    to_camera_id = Column(String(50), ForeignKey("cameras.id"), nullable=False)
    distance_km = Column(Float, nullable=False)
    min_travel_time_sec = Column(Float, nullable=False)
    max_travel_time_sec = Column(Float, nullable=False)
    typical_speed_kmh = Column(Float, default=50.0)
    is_connected = Column(Boolean, default=True)

    # Relationships
    from_camera = relationship(
        "Camera",
        foreign_keys=[from_camera_id],
        back_populates="outgoing_topology",
    )
    to_camera = relationship(
        "Camera",
        foreign_keys=[to_camera_id],
        back_populates="incoming_topology",
    )

    __table_args__ = (
        Index("idx_topology_from_to", "from_camera_id", "to_camera_id", unique=True),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary."""
        return {
            "id": self.id,
            "from_camera_id": self.from_camera_id,
            "to_camera_id": self.to_camera_id,
            "distance_km": self.distance_km,
            "min_travel_time_sec": self.min_travel_time_sec,
            "max_travel_time_sec": self.max_travel_time_sec,
            "typical_speed_kmh": self.typical_speed_kmh,
            "is_connected": self.is_connected,
        }


class AuditLog(Base):
    """Tamper-evident forensic audit log complying with Section 65B BSA & DPDP Act."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_badge_id = Column(String(100), nullable=False)
    user_name = Column(String(150), nullable=False)
    action = Column(String(100), nullable=False)  # CASE_CREATE, SEARCH, VERIFY_SIGHTING, EVIDENCE_EXPORT
    resource_type = Column(String(50), nullable=False)  # case, sighting, camera, report
    resource_id = Column(String(100), nullable=True)
    endpoint = Column(String(200), nullable=False)
    ip_address = Column(String(50), nullable=False)
    details = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=utc_now, index=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary."""
        return {
            "id": self.id,
            "user_badge_id": self.user_badge_id,
            "user_name": self.user_name,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "endpoint": self.endpoint,
            "ip_address": self.ip_address,
            "details": self.details,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
