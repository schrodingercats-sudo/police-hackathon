"""FastAPI REST API Route Handlers for Indian Police Stolen Vehicle AI Command Center.

Handles case reporting, multi-modal search, spatio-temporal route reconstruction,
officer verification, dashboard telemetry, camera administration, and evidence exports.
"""

import base64
import logging
import math
import os
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.models import (
    AuditLog,
    Camera,
    CameraStatus,
    CameraTopology,
    Case,
    CaseMatch,
    CaseStatus,
    Sighting,
    Vehicle,
    VerificationStatus,
    utc_now,
)
from backend.database.session import get_db
from backend.api.schemas import (
    CameraCreateRequest,
    CameraStatusDTO,
    CaseCreateRequest,
    CaseResponse,
    DashboardStatsResponse,
    EvidenceExportResponse,
    RouteReconstructionResponse,
    RouteSegmentDTO,
    RouteWaypointDTO,
    SearchRequestParams,
    SearchResponse,
    SightingDTO,
    VerificationRequest,
    VerificationResponse,
)
from backend.services.audit_service import log_audit_event
from backend.services.evidence_service import (
    generate_evidence_dossier,
    generate_sha256_hash,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Police Command Services"])


# -----------------------------------------------------------------------------
# Spatio-Temporal Math Helpers (Self-Contained & Resilient)
# -----------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers using the Haversine formula."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def calculate_sighting_composite_score(
    reported_plate: str,
    sighting_plate: Optional[str],
    plate_conf: float,
    reported_type: Optional[str],
    sighting_type: Optional[str],
    reported_color: Optional[str],
    sighting_color: Optional[str],
) -> float:
    """Computes a baseline matching score for candidate association."""
    score = 0.0
    if sighting_plate and reported_plate:
        t = reported_plate.upper().replace(" ", "").replace("-", "")
        s = sighting_plate.upper().replace(" ", "").replace("-", "")
        if t == s:
            plate_sim = 1.0
        elif t in s or s in t:
            plate_sim = 0.85
        else:
            # Common characters ratio
            matches = sum(1 for a, b in zip(t, s) if a == b)
            plate_sim = matches / max(len(t), len(s)) if max(len(t), len(s)) > 0 else 0.0
        score += 0.50 * (plate_sim * max(0.5, plate_conf))
    else:
        score += 0.0

    # Type similarity
    if reported_type and sighting_type and reported_type.lower() in sighting_type.lower():
        score += 0.25

    # Color similarity
    if reported_color and sighting_color and reported_color.lower() in sighting_color.lower():
        score += 0.25

    return round(min(1.0, max(0.0, score)), 4)


# -----------------------------------------------------------------------------
# 1. Health Endpoint
# -----------------------------------------------------------------------------
@router.get("/health", summary="System Health & Telemetry Status")
def health_check():
    """Returns system status, service name, and timestamp."""
    return {
        "status": "healthy",
        "service": "stolen_vehicle_ai",
        "version": settings.VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "online",
    }


# -----------------------------------------------------------------------------
# 2. Case Registration Endpoint (POST /api/report)
# -----------------------------------------------------------------------------
@router.post(
    "/report",
    response_model=CaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register New Stolen Vehicle FIR Case",
)
def create_case_report(
    payload: CaseCreateRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Registers a new stolen vehicle FIR case, decodes and saves reference image (if provided),
    performs initial candidate sighting matching, and records an audit log.
    """
    # Check if case with same FIR already exists
    existing = db.query(Case).filter(Case.fir_number == payload.fir_number).first()
    if existing:
        logger.info(f"FIR {payload.fir_number} already registered; returning existing case record.")
        # Calculate counts
        total_candidates = db.query(CaseMatch).filter(CaseMatch.case_id == existing.id).count()
        verified = (
            db.query(CaseMatch)
            .filter(
                CaseMatch.case_id == existing.id,
                CaseMatch.verification_status == VerificationStatus.VERIFIED,
            )
            .count()
        )
        return CaseResponse(
            id=existing.id,
            fir_number=existing.fir_number,
            reported_plate=existing.reported_plate,
            theft_datetime=existing.theft_datetime,
            theft_latitude=existing.theft_latitude,
            theft_longitude=existing.theft_longitude,
            theft_location_name=existing.theft_location_name,
            vehicle_type=existing.vehicle_type,
            vehicle_color=existing.vehicle_color,
            make=existing.make,
            model=existing.model,
            distinctive_features=existing.distinctive_features,
            status=existing.status.value if hasattr(existing.status, "value") else str(existing.status),
            investigating_officer=existing.investigating_officer,
            police_station=existing.police_station,
            reference_image_path=existing.reference_image_path,
            total_candidate_sightings=total_candidates,
            verified_sightings=verified,
            created_at=existing.created_at,
            updated_at=existing.updated_at,
        )

    # Save reference image if provided as base64 with magic byte validation (SEC-003)
    ref_image_path = None
    if payload.reference_image_base64:
        try:
            image_data = base64.b64decode(payload.reference_image_base64)
            ext = None
            if image_data.startswith(b"\xff\xd8\xff"):
                ext = "jpg"
            elif image_data.startswith(b"\x89PNG\r\n\x1a\n"):
                ext = "png"
            elif image_data.startswith(b"RIFF") and len(image_data) >= 12 and image_data[8:12] == b"WEBP":
                ext = "webp"

            if not ext:
                logger.warning(f"Invalid reference image format for FIR {payload.fir_number}: disallowed magic bytes")
            else:
                safe_fir = "".join(c for c in payload.fir_number if c.isalnum() or c in ("_", "-"))
                file_name = f"ref_{safe_fir}.{ext}"
                full_path = settings.EVIDENCE_DIR / file_name
                with open(full_path, "wb") as f:
                    f.write(image_data)
                ref_image_path = f"/evidence/{file_name}"
        except Exception as e:
            logger.warning(f"Could not decode reference image for FIR {payload.fir_number}: {e}")

    new_case = Case(
        fir_number=payload.fir_number,
        reported_plate=payload.reported_plate.upper().strip(),
        theft_datetime=payload.theft_datetime,
        theft_latitude=payload.theft_latitude,
        theft_longitude=payload.theft_longitude,
        theft_location_name=payload.theft_location_name,
        vehicle_type=payload.vehicle_type.lower().strip(),
        vehicle_color=payload.vehicle_color.lower().strip(),
        make=payload.make,
        model=payload.model,
        distinctive_features=payload.distinctive_features,
        reference_image_path=ref_image_path,
        status=CaseStatus.OPEN,
        investigating_officer=payload.investigating_officer,
        police_station=payload.police_station,
    )
    db.add(new_case)
    db.commit()
    db.refresh(new_case)

    # Initial candidate matching against existing sightings
    target_clean = payload.reported_plate.upper().replace(" ", "").replace("-", "")
    potential_sightings = (
        db.query(Sighting)
        .filter(
            or_(
                Sighting.plate_text == payload.reported_plate.upper().strip(),
                Sighting.plate_text.like(f"%{target_clean[:6]}%"),
                Sighting.vehicle_color == payload.vehicle_color.lower().strip(),
            )
        )
        .all()
    )

    matched_count = 0
    for sighting in potential_sightings:
        c_score = calculate_sighting_composite_score(
            reported_plate=payload.reported_plate,
            sighting_plate=sighting.plate_text,
            plate_conf=sighting.plate_confidence,
            reported_type=payload.vehicle_type,
            sighting_type=sighting.vehicle_type,
            reported_color=payload.vehicle_color,
            sighting_color=sighting.vehicle_color,
        )
        if c_score >= 0.40:
            match_entry = CaseMatch(
                case_id=new_case.id,
                sighting_id=sighting.id,
                plate_score=sighting.plate_confidence,
                visual_score=0.90,
                attribute_score=1.0,
                composite_score=c_score,
                verification_status=VerificationStatus.PENDING,
            )
            db.add(match_entry)
            matched_count += 1

    if matched_count > 0:
        new_case.status = CaseStatus.TRACKING
    db.commit()

    # Record Audit Log
    client_ip = request.client.host if request.client else "127.0.0.1"
    log_audit_event(
        db=db,
        user_badge_id=payload.investigating_officer,
        user_name=payload.investigating_officer,
        action="CASE_REGISTER",
        resource_type="case",
        resource_id=new_case.fir_number,
        endpoint="/api/report",
        ip_address=client_ip,
        details={
            "fir_number": new_case.fir_number,
            "reported_plate": new_case.reported_plate,
            "initial_matches_found": matched_count,
        },
    )

    return CaseResponse(
        id=new_case.id,
        fir_number=new_case.fir_number,
        reported_plate=new_case.reported_plate,
        theft_datetime=new_case.theft_datetime,
        theft_latitude=new_case.theft_latitude,
        theft_longitude=new_case.theft_longitude,
        theft_location_name=new_case.theft_location_name,
        vehicle_type=new_case.vehicle_type,
        vehicle_color=new_case.vehicle_color,
        make=new_case.make,
        model=new_case.model,
        distinctive_features=new_case.distinctive_features,
        status=new_case.status.value if hasattr(new_case.status, "value") else str(new_case.status),
        investigating_officer=new_case.investigating_officer,
        police_station=new_case.police_station,
        reference_image_path=new_case.reference_image_path,
        total_candidate_sightings=matched_count,
        verified_sightings=0,
        created_at=new_case.created_at,
        updated_at=new_case.updated_at,
    )


# -----------------------------------------------------------------------------
# 3. Multi-Criteria Sighting Search (GET /api/search)
# -----------------------------------------------------------------------------
@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Search Vehicle Sightings across Camera Network",
)
def search_sightings(
    request: Request,
    plate: Optional[str] = Query(None, description="License plate number or substring"),
    vehicle_type: Optional[str] = Query(None, description="Vehicle category"),
    color: Optional[str] = Query(None, description="Vehicle color"),
    camera_id: Optional[str] = Query(None, description="Specific camera node ID"),
    start_time: Optional[datetime] = Query(None, description="Start timestamp"),
    end_time: Optional[datetime] = Query(None, description="End timestamp"),
    min_confidence: float = Query(0.50, ge=0.0, le=1.0, description="Minimum OCR plate confidence"),
    min_visual_similarity: Optional[float] = Query(None, ge=0.0, le=1.0),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Performs multi-criteria search over captured vehicle sightings with pagination,
    plate fuzzy filtering, attribute filters, and camera location joins.
    """
    query = db.query(Sighting)

    if plate:
        clean_p = plate.upper().strip()
        query = query.filter(
            or_(
                Sighting.plate_text == clean_p,
                Sighting.plate_text.like(f"%{clean_p}%"),
            )
        )

    if vehicle_type:
        query = query.filter(Sighting.vehicle_type.ilike(f"%{vehicle_type.strip()}%"))

    if color:
        query = query.filter(Sighting.vehicle_color.ilike(f"%{color.strip()}%"))

    if camera_id:
        query = query.filter(Sighting.camera_id == camera_id.strip())

    if start_time:
        query = query.filter(Sighting.timestamp >= start_time)

    if end_time:
        query = query.filter(Sighting.timestamp <= end_time)

    if min_confidence > 0:
        query = query.filter(Sighting.plate_confidence >= min_confidence)

    total_count = query.count()
    offset = (page - 1) * limit
    results_raw = query.order_by(Sighting.timestamp.desc()).offset(offset).limit(limit).all()

    items: List[SightingDTO] = []
    for s in results_raw:
        cam = s.camera or db.query(Camera).filter(Camera.id == s.camera_id).first()
        cam_name = cam.name if cam else s.camera_id
        cam_lat = cam.latitude if cam else 0.0
        cam_lon = cam.longitude if cam else 0.0
        road_name = cam.road_name if cam else "Corridor Location"

        # Determine composite score and verification status from linked case_matches
        comp_score = s.plate_confidence or 0.0
        v_status = "pending"
        if s.case_matches:
            best_match = max(s.case_matches, key=lambda m: m.composite_score or 0.0)
            comp_score = best_match.composite_score or comp_score
            if hasattr(best_match.verification_status, "value"):
                v_status = best_match.verification_status.value
            else:
                v_status = str(best_match.verification_status)

        items.append(
            SightingDTO(
                id=s.id,
                camera_id=s.camera_id,
                camera_name=cam_name,
                camera_lat=cam_lat,
                camera_lon=cam_lon,
                road_name=road_name,
                timestamp=s.timestamp,
                track_id=s.track_id,
                plate_text=s.plate_text,
                plate_confidence=s.plate_confidence or 0.0,
                plate_box=s.plate_box,
                vehicle_box=s.vehicle_box,
                vehicle_type=s.vehicle_type,
                vehicle_color=s.vehicle_color,
                make=s.make,
                model=s.model,
                composite_score=round(comp_score, 4),
                image_url=s.image_path,
                crop_url=s.crop_path,
                plate_crop_url=s.plate_crop_path,
                sha256_hash=s.sha256_hash,
                direction_travel=s.direction_travel,
                speed_estimate_kmh=s.speed_estimate_kmh,
                verification_status=v_status,
                created_at=s.created_at,
            )
        )

    # Log search audit event
    client_ip = request.client.host if request.client else "127.0.0.1"
    log_audit_event(
        db=db,
        user_badge_id="OFFICER_SEARCH",
        user_name="Command Center Operator",
        action="SEARCH_SIGHTINGS",
        resource_type="sighting",
        resource_id=plate or "MULTI_FILTER",
        endpoint="/api/search",
        ip_address=client_ip,
        details={
            "plate": plate,
            "color": color,
            "type": vehicle_type,
            "results_count": len(items),
            "total_count": total_count,
        },
    )

    return SearchResponse(
        total_count=total_count,
        page=page,
        limit=limit,
        results=items,
    )


# -----------------------------------------------------------------------------
# 4. Spatio-Temporal Route Reconstruction (GET /api/route)
# -----------------------------------------------------------------------------
@router.get(
    "/route",
    response_model=RouteReconstructionResponse,
    summary="Reconstruct Chronological Movement Route for Case",
)
def get_case_route(
    case_id: int = Query(..., description="Stolen vehicle case ID"),
    db: Session = Depends(get_db),
):
    """
    Reconstructs chronological trajectory for a reported stolen vehicle case,
    validates velocity feasibility between consecutive sightings, and returns waypoints and segments.
    """
    case_obj = db.query(Case).filter(Case.id == case_id).first()
    if not case_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case with ID {case_id} not found.",
        )

    # Find candidate or verified sightings linked via CaseMatch or matching plate
    matches = (
        db.query(CaseMatch)
        .filter(CaseMatch.case_id == case_id)
        .all()
    )

    sighting_records = []
    if matches:
        for m in matches:
            if m.sighting:
                sighting_records.append((m.sighting, m))
    else:
        # Fallback: query sightings with matching plate
        direct_sightings = (
            db.query(Sighting)
            .filter(Sighting.plate_text == case_obj.reported_plate)
            .order_by(Sighting.timestamp.asc())
            .all()
        )
        for s in direct_sightings:
            sighting_records.append((s, None))

    # Sort sightings strictly chronologically
    sighting_records.sort(key=lambda x: x[0].timestamp)

    waypoints: List[RouteWaypointDTO] = []
    segments: List[RouteSegmentDTO] = []
    total_dist_km = 0.0

    prev_lat: Optional[float] = case_obj.theft_latitude
    prev_lon: Optional[float] = case_obj.theft_longitude
    prev_time: Optional[datetime] = case_obj.theft_datetime
    prev_cam_id: str = "THEFT_ORIGIN"

    for idx, (sighting, match_obj) in enumerate(sighting_records, start=1):
        cam = sighting.camera or db.query(Camera).filter(Camera.id == sighting.camera_id).first()
        cam_lat = cam.latitude if cam else case_obj.theft_latitude
        cam_lon = cam.longitude if cam else case_obj.theft_longitude
        cam_name = cam.name if cam else sighting.camera_id
        road_name = cam.road_name if cam else "Corridor Location"

        dist_from_prev = 0.0
        time_delta_sec = 0.0
        speed_kmh = 0.0
        is_feasible = True

        if prev_lat is not None and prev_lon is not None and prev_time is not None:
            dist_from_prev = haversine_km(prev_lat, prev_lon, cam_lat, cam_lon)
            time_delta_sec = (sighting.timestamp - prev_time).total_seconds()

            if time_delta_sec < 0:
                speed_kmh = 999.0
                is_feasible = False
            elif time_delta_sec > 0:
                speed_kmh = dist_from_prev / (time_delta_sec / 3600.0)
                is_feasible = speed_kmh <= settings.MAX_SPEED_KMH
            elif dist_from_prev > 0.05:
                speed_kmh = 999.0
                is_feasible = False
            else:
                speed_kmh = 0.0
                is_feasible = True

            # Create route segment
            segments.append(
                RouteSegmentDTO(
                    from_camera_id=prev_cam_id,
                    to_camera_id=sighting.camera_id,
                    from_coords=[prev_lat, prev_lon],
                    to_coords=[cam_lat, cam_lon],
                    distance_km=round(dist_from_prev, 3),
                    elapsed_time_sec=round(time_delta_sec, 1),
                    speed_kmh=round(speed_kmh, 1),
                    feasible=is_feasible,
                )
            )
            total_dist_km += dist_from_prev

        v_status = "pending"
        c_score = 0.90
        if match_obj:
            v_status = (
                match_obj.verification_status.value
                if hasattr(match_obj.verification_status, "value")
                else str(match_obj.verification_status)
            )
            c_score = match_obj.composite_score or 0.90

        waypoints.append(
            RouteWaypointDTO(
                sequence=idx,
                sighting_id=sighting.id,
                camera_id=sighting.camera_id,
                camera_name=cam_name,
                latitude=cam_lat,
                longitude=cam_lon,
                road_name=road_name,
                timestamp=sighting.timestamp,
                plate_text=sighting.plate_text,
                plate_confidence=sighting.plate_confidence or 0.0,
                composite_score=round(c_score, 4),
                speed_from_prev_kmh=round(speed_kmh, 1) if idx > 1 else None,
                distance_from_prev_km=round(dist_from_prev, 3) if idx > 1 else None,
                time_delta_sec=round(time_delta_sec, 1) if idx > 1 else None,
                is_spatially_feasible=is_feasible,
                verification_status=v_status,
                vehicle_crop_url=sighting.crop_path,
                plate_crop_url=sighting.plate_crop_path,
            )
        )

        prev_lat = cam_lat
        prev_lon = cam_lon
        prev_time = sighting.timestamp
        prev_cam_id = sighting.camera_id

    # Compute duration
    est_duration_min = 0.0
    if len(sighting_records) >= 1:
        first_time = case_obj.theft_datetime
        last_time = sighting_records[-1][0].timestamp
        est_duration_min = max(0.0, round((last_time - first_time).total_seconds() / 60.0, 1))

    last_cam = waypoints[-1].camera_id if waypoints else None
    last_time_val = waypoints[-1].timestamp if waypoints else None

    return RouteReconstructionResponse(
        case_id=case_obj.id,
        fir_number=case_obj.fir_number,
        reported_plate=case_obj.reported_plate,
        theft_origin={
            "latitude": case_obj.theft_latitude,
            "longitude": case_obj.theft_longitude,
            "location_name": case_obj.theft_location_name,
            "datetime": case_obj.theft_datetime.isoformat(),
        },
        waypoints=waypoints,
        segments=segments,
        total_distance_km=round(total_dist_km, 3),
        estimated_duration_min=est_duration_min,
        route_status="Active Lead" if waypoints else "No Detections",
        last_known_camera=last_cam,
        last_seen_timestamp=last_time_val,
    )


# -----------------------------------------------------------------------------
# 5. Sighting History Feed (GET /api/sightings)
# -----------------------------------------------------------------------------
@router.get(
    "/sightings",
    response_model=SearchResponse,
    summary="Get Paginated Feed of Camera Sightings",
)
def get_sightings_feed(
    camera_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Returns paginated stream of recent vehicle sightings."""
    query = db.query(Sighting)
    if camera_id:
        query = query.filter(Sighting.camera_id == camera_id)

    total_count = query.count()
    records = query.order_by(Sighting.timestamp.desc()).offset(offset).limit(limit).all()

    results = []
    for s in records:
        cam = s.camera or db.query(Camera).filter(Camera.id == s.camera_id).first()
        cam_name = cam.name if cam else s.camera_id
        cam_lat = cam.latitude if cam else 0.0
        cam_lon = cam.longitude if cam else 0.0
        road_name = cam.road_name if cam else "Corridor Location"

        # Determine composite score and verification status from linked case_matches
        comp_score = s.plate_confidence or 0.0
        v_status = "pending"
        if s.case_matches:
            best_match = max(s.case_matches, key=lambda m: m.composite_score or 0.0)
            comp_score = best_match.composite_score or comp_score
            if hasattr(best_match.verification_status, "value"):
                v_status = best_match.verification_status.value
            else:
                v_status = str(best_match.verification_status)

        results.append(
            SightingDTO(
                id=s.id,
                camera_id=s.camera_id,
                camera_name=cam_name,
                camera_lat=cam_lat,
                camera_lon=cam_lon,
                road_name=road_name,
                timestamp=s.timestamp,
                track_id=s.track_id,
                plate_text=s.plate_text,
                plate_confidence=s.plate_confidence or 0.0,
                plate_box=s.plate_box,
                vehicle_box=s.vehicle_box,
                vehicle_type=s.vehicle_type,
                vehicle_color=s.vehicle_color,
                make=s.make,
                model=s.model,
                composite_score=round(comp_score, 4),
                image_url=s.image_path,
                crop_url=s.crop_path,
                plate_crop_url=s.plate_crop_path,
                sha256_hash=s.sha256_hash,
                direction_travel=s.direction_travel,
                speed_estimate_kmh=s.speed_estimate_kmh,
                verification_status=v_status,
                created_at=s.created_at,
            )
        )

    page = (offset // limit) + 1
    return SearchResponse(
        total_count=total_count,
        page=page,
        limit=limit,
        results=results,
    )


# -----------------------------------------------------------------------------
# 6. Officer Sighting Verification (POST /api/verify)
# -----------------------------------------------------------------------------
@router.post(
    "/verify",
    response_model=VerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Officer Sighting Confirmation / Rejection Action",
)
def verify_sighting(
    payload: VerificationRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Updates the verification status of a candidate sighting match to 'verified' or 'rejected'.
    Maintains officer accountability and records tamper-evident audit logs.
    """
    target_status = payload.status.lower().strip()
    if target_status not in ["verified", "rejected", "flagged", "pending"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid verification status: '{payload.status}'. Must be 'verified' or 'rejected'.",
        )

    # Find existing match
    match = (
        db.query(CaseMatch)
        .filter(
            CaseMatch.case_id == payload.case_id,
            CaseMatch.sighting_id == payload.sighting_id,
        )
        .first()
    )

    status_map = {
        "verified": VerificationStatus.VERIFIED,
        "rejected": VerificationStatus.REJECTED,
        "flagged": VerificationStatus.FLAGGED,
        "pending": VerificationStatus.PENDING,
    }
    enum_status = status_map.get(target_status, VerificationStatus.PENDING)

    if not match:
        # Create match record if not existing
        sighting = db.query(Sighting).filter(Sighting.id == payload.sighting_id).first()
        if not sighting:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sighting with ID {payload.sighting_id} not found.",
            )
        match = CaseMatch(
            case_id=payload.case_id,
            sighting_id=payload.sighting_id,
            plate_score=sighting.plate_confidence or 0.90,
            visual_score=0.90,
            attribute_score=1.0,
            composite_score=0.90,
            verification_status=enum_status,
            reviewed_by=payload.officer_badge_id,
            reviewed_at=utc_now(),
            review_notes=payload.notes,
        )
        db.add(match)
    else:
        match.verification_status = enum_status
        match.reviewed_by = payload.officer_badge_id
        match.reviewed_at = utc_now()
        if payload.notes:
            match.review_notes = payload.notes

    # Update case status if verified
    case_obj = db.query(Case).filter(Case.id == payload.case_id).first()
    if case_obj and target_status == "verified" and case_obj.status == CaseStatus.OPEN:
        case_obj.status = CaseStatus.TRACKING

    db.commit()
    db.refresh(match)

    # Log to forensic audit trail
    client_ip = request.client.host if request.client else "127.0.0.1"
    log_audit_event(
        db=db,
        user_badge_id=payload.officer_badge_id,
        user_name=f"Officer {payload.officer_badge_id}",
        action="VERIFY_SIGHTING",
        resource_type="case_match",
        resource_id=str(match.id),
        endpoint="/api/verify",
        ip_address=client_ip,
        details={
            "case_id": payload.case_id,
            "sighting_id": payload.sighting_id,
            "status": target_status,
            "notes": payload.notes,
        },
    )

    return VerificationResponse(
        match_id=match.id,
        case_id=payload.case_id,
        sighting_id=payload.sighting_id,
        status=target_status,
        updated_at=match.reviewed_at or utc_now(),
        message=f"Sighting {payload.sighting_id} successfully marked as '{target_status}'.",
    )


# -----------------------------------------------------------------------------
# 7. Dashboard Command Telemetry (GET /api/dashboard/stats)
# -----------------------------------------------------------------------------
@router.get(
    "/dashboard/stats",
    response_model=DashboardStatsResponse,
    summary="Police Command Center Dashboard Telemetry & KPI Metrics",
)
def get_dashboard_stats(db: Session = Depends(get_db)):
    """
    Returns real-time command metrics: active cases, total cameras, online cameras,
    detections today, matches today, recovery rate percentage, and recent alerts.
    """
    total_cams = db.query(Camera).count()
    active_cams = db.query(Camera).filter(Camera.status == CameraStatus.ACTIVE.value).count()

    active_cases = (
        db.query(Case)
        .filter(
            or_(
                Case.status == CaseStatus.OPEN,
                Case.status == CaseStatus.TRACKING,
            )
        )
        .count()
    )

    recovered_cases = db.query(Case).filter(Case.status == CaseStatus.RECOVERED).count()
    total_cases = db.query(Case).count()
    recovery_rate = round((recovered_cases / total_cases * 100.0) if total_cases > 0 else 0.0, 1)

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    detections_today = db.query(Sighting).filter(Sighting.timestamp >= today_start).count()
    matches_today = db.query(CaseMatch).filter(CaseMatch.created_at >= today_start).count()

    # Query all cameras with detection counts
    cameras_raw = db.query(Camera).all()
    cam_status_list: List[CameraStatusDTO] = []
    for cam in cameras_raw:
        det_count = db.query(Sighting).filter(Sighting.camera_id == cam.id).count()
        cam_status_list.append(
            CameraStatusDTO(
                id=cam.id,
                name=cam.name,
                latitude=cam.latitude,
                longitude=cam.longitude,
                road_name=cam.road_name,
                status=cam.status,
                total_detections_today=det_count,
                last_ping=cam.updated_at or cam.created_at,
                camera_type=cam.camera_type,
                direction_bearing=cam.direction_bearing,
            )
        )

    # Recent Alerts (latest high-score matches or latest sightings)
    recent_matches = (
        db.query(CaseMatch)
        .order_by(CaseMatch.created_at.desc())
        .limit(5)
        .all()
    )

    recent_alerts = []
    for m in recent_matches:
        c_obj = m.case
        s_obj = m.sighting
        if c_obj and s_obj:
            recent_alerts.append(
                {
                    "alert_id": m.id,
                    "case_id": c_obj.id,
                    "fir_number": c_obj.fir_number,
                    "reported_plate": c_obj.reported_plate,
                    "camera_id": s_obj.camera_id,
                    "timestamp": s_obj.timestamp.isoformat() if s_obj.timestamp else None,
                    "composite_score": m.composite_score,
                    "verification_status": (
                        m.verification_status.value
                        if hasattr(m.verification_status, "value")
                        else str(m.verification_status)
                    ),
                    "vehicle_crop_url": s_obj.crop_path,
                }
            )

    return DashboardStatsResponse(
        active_cases=active_cases,
        total_cameras=total_cams,
        active_cameras=active_cams,
        detections_today=detections_today,
        matches_today=matches_today,
        recovery_rate_pct=recovery_rate,
        recent_alerts=recent_alerts,
        active_camera_list=cam_status_list,
    )


# -----------------------------------------------------------------------------
# 8. Camera Management Endpoints (GET/POST /api/cameras)
# -----------------------------------------------------------------------------
@router.get(
    "/cameras",
    response_model=List[CameraStatusDTO],
    summary="List All Monitored Surveillance Cameras",
)
def list_cameras(db: Session = Depends(get_db)):
    """Returns list of all registered CCTV surveillance nodes in the network."""
    cameras = db.query(Camera).order_by(Camera.id.asc()).all()
    results = []
    for cam in cameras:
        det_count = db.query(Sighting).filter(Sighting.camera_id == cam.id).count()
        results.append(
            CameraStatusDTO(
                id=cam.id,
                name=cam.name,
                latitude=cam.latitude,
                longitude=cam.longitude,
                road_name=cam.road_name,
                status=cam.status,
                total_detections_today=det_count,
                last_ping=cam.updated_at or cam.created_at,
                camera_type=cam.camera_type,
                direction_bearing=cam.direction_bearing,
            )
        )
    return results


@router.post(
    "/cameras",
    response_model=CameraStatusDTO,
    status_code=status.HTTP_201_CREATED,
    summary="Register New Camera Node",
)
def create_camera(payload: CameraCreateRequest, db: Session = Depends(get_db)):
    """Registers a new surveillance camera / toll plaza node into the network."""
    existing = db.query(Camera).filter(Camera.id == payload.id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Camera with ID '{payload.id}' already registered.",
        )

    cam = Camera(
        id=payload.id,
        name=payload.name,
        latitude=payload.latitude,
        longitude=payload.longitude,
        road_name=payload.road_name,
        direction_bearing=payload.direction_bearing,
        camera_type=payload.camera_type,
        stream_url=payload.stream_url,
        status=payload.status,
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)

    return CameraStatusDTO(
        id=cam.id,
        name=cam.name,
        latitude=cam.latitude,
        longitude=cam.longitude,
        road_name=cam.road_name,
        status=cam.status,
        total_detections_today=0,
        last_ping=cam.created_at,
        camera_type=cam.camera_type,
        direction_bearing=cam.direction_bearing,
    )


@router.get("/cameras/{camera_id}", response_model=CameraStatusDTO, summary="Get Camera Details")
def get_camera_details(camera_id: str, db: Session = Depends(get_db)):
    """Retrieves metadata and status for a specific camera node."""
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    det_count = db.query(Sighting).filter(Sighting.camera_id == cam.id).count()
    return CameraStatusDTO(
        id=cam.id,
        name=cam.name,
        latitude=cam.latitude,
        longitude=cam.longitude,
        road_name=cam.road_name,
        status=cam.status,
        total_detections_today=det_count,
        last_ping=cam.updated_at or cam.created_at,
        camera_type=cam.camera_type,
        direction_bearing=cam.direction_bearing,
    )


@router.get("/cameras/{camera_id}/frame", summary="Get Latest Camera Frame Image")
def get_camera_frame(camera_id: str):
    """Returns the latest CCTV frame image for a camera node."""
    import glob
    from fastapi.responses import FileResponse as FR
    frames_dir = settings.EVIDENCE_DIR / "frames"
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="No frames available.")
    pattern = str(frames_dir / f"{camera_id}_*.jpg")
    matches = sorted(glob.glob(pattern), reverse=True)
    if not matches:
        raise HTTPException(status_code=404, detail=f"No frame found for camera {camera_id}")
    return FR(matches[0], media_type="image/jpeg")


# -----------------------------------------------------------------------------
# 9. Case Management Endpoints (GET /api/cases)
# -----------------------------------------------------------------------------
@router.get("/cases", response_model=List[CaseResponse], summary="List All FIR Cases")
def list_cases(
    limit: Optional[int] = Query(None, ge=1, le=500, description="Max cases to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
):
    """Retrieves registered stolen vehicle FIR cases with candidate counts and optional pagination."""
    query = db.query(Case).order_by(Case.created_at.desc())
    if limit is not None:
        query = query.offset(offset).limit(limit)
    cases = query.all()
    results = []
    for c in cases:
        total_cand = db.query(CaseMatch).filter(CaseMatch.case_id == c.id).count()
        verified = (
            db.query(CaseMatch)
            .filter(
                CaseMatch.case_id == c.id,
                CaseMatch.verification_status == VerificationStatus.VERIFIED,
            )
            .count()
        )
        results.append(
            CaseResponse(
                id=c.id,
                fir_number=c.fir_number,
                reported_plate=c.reported_plate,
                theft_datetime=c.theft_datetime,
                theft_latitude=c.theft_latitude,
                theft_longitude=c.theft_longitude,
                theft_location_name=c.theft_location_name,
                vehicle_type=c.vehicle_type,
                vehicle_color=c.vehicle_color,
                make=c.make,
                model=c.model,
                distinctive_features=c.distinctive_features,
                status=c.status.value if hasattr(c.status, "value") else str(c.status),
                investigating_officer=c.investigating_officer,
                police_station=c.police_station,
                reference_image_path=c.reference_image_path,
                total_candidate_sightings=total_cand,
                verified_sightings=verified,
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
        )
    return results


@router.patch(
    "/cases/{case_id}/status",
    response_model=CaseResponse,
    summary="Update Case Lifecycle Status (Recovered / Closed / Tracking / Open)",
)
def update_case_status(
    case_id: int,
    payload: Dict[str, Any],
    request: Request,
    db: Session = Depends(get_db),
):
    """Allows authorized police officers to update case lifecycle status (e.g. mark RECOVERED or CLOSED)."""
    case_obj = db.query(Case).filter(Case.id == case_id).first()
    if not case_obj:
        raise HTTPException(status_code=404, detail=f"Case #{case_id} not found.")

    new_status_str = str(payload.get("status", "")).lower().strip()
    status_map = {
        "open": CaseStatus.OPEN,
        "tracking": CaseStatus.TRACKING,
        "recovered": CaseStatus.RECOVERED,
        "closed": CaseStatus.CLOSED,
    }
    if new_status_str not in status_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{new_status_str}'. Allowed statuses: {list(status_map.keys())}",
        )

    old_status = case_obj.status.value if hasattr(case_obj.status, "value") else str(case_obj.status)
    case_obj.status = status_map[new_status_str]
    case_obj.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()
    db.refresh(case_obj)

    # Log to audit trail
    client_ip = request.client.host if request.client else "127.0.0.1"
    officer_badge = str(payload.get("officer_badge_id") or case_obj.investigating_officer or "OFFICER")
    log_audit_event(
        db=db,
        user_badge_id=officer_badge,
        user_name=f"Officer {officer_badge}",
        action="CASE_STATUS_UPDATE",
        resource_type="case",
        resource_id=case_obj.fir_number,
        endpoint=f"/api/cases/{case_id}/status",
        ip_address=client_ip,
        details={
            "case_id": case_obj.id,
            "old_status": old_status,
            "new_status": new_status_str,
            "notes": payload.get("notes"),
        },
    )

    total_cand = db.query(CaseMatch).filter(CaseMatch.case_id == case_obj.id).count()
    verified = (
        db.query(CaseMatch)
        .filter(
            CaseMatch.case_id == case_obj.id,
            CaseMatch.verification_status == VerificationStatus.VERIFIED,
        )
        .count()
    )
    return CaseResponse(
        id=case_obj.id,
        fir_number=case_obj.fir_number,
        reported_plate=case_obj.reported_plate,
        theft_datetime=case_obj.theft_datetime,
        theft_latitude=case_obj.theft_latitude,
        theft_longitude=case_obj.theft_longitude,
        theft_location_name=case_obj.theft_location_name,
        vehicle_type=case_obj.vehicle_type,
        vehicle_color=case_obj.vehicle_color,
        make=case_obj.make,
        model=case_obj.model,
        distinctive_features=case_obj.distinctive_features,
        status=case_obj.status.value if hasattr(case_obj.status, "value") else str(case_obj.status),
        investigating_officer=case_obj.investigating_officer,
        police_station=case_obj.police_station,
        reference_image_path=case_obj.reference_image_path,
        total_candidate_sightings=total_cand,
        verified_sightings=verified,
        created_at=case_obj.created_at,
        updated_at=case_obj.updated_at,
    )


@router.get("/cases/{case_id}", response_model=CaseResponse, summary="Get Case Details")
def get_case(case_id: int, db: Session = Depends(get_db)):
    """Retrieves details of a single case by ID."""
    c = db.query(Case).filter(Case.id == case_id).first()
    if not c:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")
    total_cand = db.query(CaseMatch).filter(CaseMatch.case_id == c.id).count()
    verified = (
        db.query(CaseMatch)
        .filter(
            CaseMatch.case_id == c.id,
            CaseMatch.verification_status == VerificationStatus.VERIFIED,
        )
        .count()
    )
    return CaseResponse(
        id=c.id,
        fir_number=c.fir_number,
        reported_plate=c.reported_plate,
        theft_datetime=c.theft_datetime,
        theft_latitude=c.theft_latitude,
        theft_longitude=c.theft_longitude,
        theft_location_name=c.theft_location_name,
        vehicle_type=c.vehicle_type,
        vehicle_color=c.vehicle_color,
        make=c.make,
        model=c.model,
        distinctive_features=c.distinctive_features,
        status=c.status.value if hasattr(c.status, "value") else str(c.status),
        investigating_officer=c.investigating_officer,
        police_station=c.police_station,
        reference_image_path=c.reference_image_path,
        total_candidate_sightings=total_cand,
        verified_sightings=verified,
        created_at=c.created_at,
        updated_at=c.updated_at,
    )


# -----------------------------------------------------------------------------
# 10. Forensic Evidence Dossier Export (GET /api/evidence/export & /api/evidence/html)
# -----------------------------------------------------------------------------
@router.get(
    "/evidence/export",
    response_model=EvidenceExportResponse,
    summary="Export Court-Admissible Section 65B Electronic Dossier",
)
def export_evidence_dossier(
    request: Request,
    case_id: int = Query(..., description="Stolen vehicle case ID"),
    db: Session = Depends(get_db),
):
    """
    Compiles complete court-admissible electronic dossier with Section 65B SHA-256
    cryptographic digests, chronological photographic timeline, and chain of custody.
    """
    client_ip = request.client.host if request.client else "127.0.0.1"
    dossier = generate_evidence_dossier(db, case_id, client_ip=client_ip)
    if not dossier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case with ID {case_id} not found.",
        )

    return EvidenceExportResponse(
        fir_number=dossier["fir_number"],
        case_id=dossier["case_id"],
        export_timestamp=dossier["export_timestamp"],
        generated_by=dossier["generated_by"],
        evidence_hash_sha256=dossier["evidence_hash_sha256"],
        case_summary=dossier["case_summary"],
        chain_of_custody=dossier["chain_of_custody"],
        verified_timeline=dossier["verified_timeline"],
        download_url_pdf=dossier["download_url_pdf"],
        html_printable_view=dossier["html_printable_view"],
    )


@router.get(
    "/evidence/html",
    response_class=HTMLResponse,
    summary="Direct Printable View of Court Evidence Dossier",
)
def view_printable_evidence_html(
    case_id: int = Query(..., description="Stolen vehicle case ID"),
    db: Session = Depends(get_db),
):
    """Renders formatted printable Section 65B court dossier view."""
    dossier = generate_evidence_dossier(db, case_id)
    if not dossier:
        raise HTTPException(status_code=404, detail="Case not found.")
    return HTMLResponse(content=dossier["html_printable_view"])


# -----------------------------------------------------------------------------
# 12. Real-Time ANPR Live Scan Endpoints
# -----------------------------------------------------------------------------
_live_anpr_processor = None

def _get_anpr_processor():
    global _live_anpr_processor
    if _live_anpr_processor is None:
        from vision.anpr_live import LiveANPRProcessor
        _live_anpr_processor = LiveANPRProcessor(gpu=False)
    return _live_anpr_processor


@router.post("/anpr/scan", summary="Real-Time ANPR Scan on Uploaded Image")
async def anpr_scan_image(request: Request):
    """
    Process an uploaded image through real EasyOCR-based ANPR.
    Accepts multipart form with 'file' field or raw JSON with base64 'image' field.
    Returns detected license plates with confidence scores.
    """
    import cv2
    import numpy as np
    import base64 as b64

    frame = None
    content_type = request.headers.get("content-type", "")

    if "multipart" in content_type:
        form = await request.form()
        file = form.get("file")
        if not file:
            raise HTTPException(status_code=400, detail="No 'file' field in form data.")
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        body = await request.json()
        img_b64 = body.get("image")
        if not img_b64:
            raise HTTPException(status_code=400, detail="No 'image' base64 field.")
        data = b64.b64decode(img_b64)
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    try:
        from vision.anpr_live import compute_frame_hash
        processor = _get_anpr_processor()
        detections = processor.detect_plates_in_frame(frame)
        frame_hash = compute_frame_hash(frame)
    except ImportError:
        raise HTTPException(status_code=500, detail="ANPR module not available. Install easyocr.")

    results = []
    for det in detections:
        crop_b64 = None
        if det.get('plate_crop') is not None:
            _, buf = cv2.imencode('.jpg', det['plate_crop'])
            crop_b64 = b64.b64encode(buf.tobytes()).decode('utf-8')

        results.append({
            "plate_text": det['plate_text'],
            "confidence": det['confidence'],
            "bbox": det['bbox'],
            "plate_crop_base64": crop_b64,
        })

    return {
        "detections": results,
        "total_plates_found": len(results),
        "frame_hash_sha256": frame_hash,
        "processing_engine": "EasyOCR",
    }


@router.get("/anpr/scan-camera/{camera_id}", summary="Real-Time ANPR Scan on Camera Frame")
def anpr_scan_camera(camera_id: str, db: Session = Depends(get_db)):
    """
    Process the latest frame from a camera through real ANPR detection.
    Returns detected plates from the camera's most recent captured frame.
    """
    import cv2
    import glob
    import base64 as b64

    # Verify camera exists
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    # Find latest frame
    frames_dir = settings.EVIDENCE_DIR / "frames"
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="No frames directory.")

    pattern = str(frames_dir / f"{camera_id}_*.jpg")
    matches = sorted(glob.glob(pattern), reverse=True)
    if not matches:
        raise HTTPException(status_code=404, detail=f"No frame for camera {camera_id}.")

    frame = cv2.imread(matches[0])
    if frame is None:
        raise HTTPException(status_code=500, detail="Could not read frame image.")

    try:
        from vision.anpr_live import compute_frame_hash
        processor = _get_anpr_processor()
        detections = processor.detect_plates_in_frame(frame)
        frame_hash = compute_frame_hash(frame)
    except ImportError:
        raise HTTPException(status_code=500, detail="ANPR module not available.")

    results = []
    for det in detections:
        crop_b64 = None
        if det.get('plate_crop') is not None:
            _, buf = cv2.imencode('.jpg', det['plate_crop'])
            crop_b64 = b64.b64encode(buf.tobytes()).decode('utf-8')

        results.append({
            "plate_text": det['plate_text'],
            "confidence": det['confidence'],
            "bbox": det['bbox'],
            "plate_crop_base64": crop_b64,
        })

    return {
        "camera_id": camera_id,
        "camera_name": cam.name,
        "frame_file": os.path.basename(matches[0]),
        "detections": results,
        "total_plates_found": len(results),
        "frame_hash_sha256": frame_hash,
        "processing_engine": "EasyOCR",
    }

