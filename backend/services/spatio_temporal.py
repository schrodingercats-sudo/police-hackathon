"""
Spatio-Temporal Association, Geodesic Kinematics & Route Trajectory Reconstruction.
Implements:
- Geodesic great-circle Haversine distance on Earth (R = 6371.0 km)
- Forward azimuth / heading bearing calculations (0.0 - 360.0 degrees)
- Kinematic spatio-temporal transition validation (v <= 120 km/h threshold & teleportation rejection)
- Directed Acyclic Graph (DAG) chronological route reconstruction with waypoints and segment velocities
"""

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

# Earth mean radius in kilometers
EARTH_RADIUS_KM: float = 6371.0
DEFAULT_MAX_SPEED_KMH: float = 120.0


# ----------------------------------------------------------------------------
# 1. GEODESIC DISTANCE & BEARING CALCULATIONS
# ----------------------------------------------------------------------------

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the great-circle geodesic distance in kilometers between two GPS points
    using the Haversine formula on a spherical Earth (R = 6371.0 km).
    """
    if abs(lat1 - lat2) < 1e-7 and abs(lon1 - lon2) < 1e-7:
        return 0.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    )
    a = max(0.0, min(1.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    distance = EARTH_RADIUS_KM * c

    return round(float(distance), 4)


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the forward azimuth / initial bearing in degrees [0.0, 360.0)
    from (lat1, lon1) to (lat2, lon2).
    """
    if abs(lat1 - lat2) < 1e-7 and abs(lon1 - lon2) < 1e-7:
        return 0.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)

    theta = math.atan2(y, x)
    bearing_deg = (math.degrees(theta) + 360.0) % 360.0

    return round(float(bearing_deg), 2)


# ----------------------------------------------------------------------------
# 2. SPATIO-TEMPORAL TRANSITION VALIDATION
# ----------------------------------------------------------------------------

def validate_spatio_temporal_transition(
    lat1: float,
    lon1: float,
    time1: datetime,
    lat2: float,
    lon2: float,
    time2: datetime,
    max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
) -> Tuple[bool, float, float, float]:
    """
    Validates physical feasibility of a transition between two camera sightings.
    
    Returns:
        (is_feasible, distance_km, time_delta_sec, speed_kmh)
    
    Feasibility criteria:
    - Time delta must be non-negative (time2 >= time1)
    - Calculated travel speed must be <= max_speed_kmh (default 120.0 km/h)
    - Teleportation (> 0.05 km at delta_t = 0) is rejected as physically impossible
    """
    dist_km = haversine_distance_km(lat1, lon1, lat2, lon2)
    time_delta_sec = (time2 - time1).total_seconds()

    # Case 1: Time paradox (time2 before time1)
    if time_delta_sec < 0:
        return (False, round(dist_km, 4), time_delta_sec, 0.0)

    # Case 2: Zero elapsed time
    if time_delta_sec == 0:
        if dist_km <= 0.05:
            return (True, round(dist_km, 4), 0.0, 0.0)
        else:
            return (False, round(dist_km, 4), 0.0, float('inf'))

    # Case 3: Positive elapsed time
    hours = time_delta_sec / 3600.0
    speed_kmh = dist_km / hours
    is_feasible = (speed_kmh <= max_speed_kmh)

    return (
        is_feasible,
        round(dist_km, 4),
        round(time_delta_sec, 2),
        round(speed_kmh, 2),
    )


# ----------------------------------------------------------------------------
# 3. DIRECTED ACYCLIC GRAPH (DAG) ROUTE TRAJECTORY RECONSTRUCTION
# ----------------------------------------------------------------------------

def _parse_sighting_entry(entry: Any) -> Dict[str, Any]:
    """Extract standard fields from dict or SQLAlchemy model."""
    if isinstance(entry, dict):
        ts = entry.get('timestamp')
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except Exception:
                pass
        return {
            'sighting_id': entry.get('sighting_id') or entry.get('id'),
            'camera_id': entry.get('camera_id') or 'UNKNOWN_CAM',
            'camera_name': entry.get('camera_name') or f"Camera {entry.get('camera_id', '')}",
            'latitude': float(entry.get('latitude', 0.0)),
            'longitude': float(entry.get('longitude', 0.0)),
            'timestamp': ts,
            'composite_score': float(entry.get('composite_score', 1.0)),
            'plate_text': entry.get('plate_text'),
            'vehicle_type': entry.get('vehicle_type'),
            'vehicle_color': entry.get('vehicle_color'),
            'crop_path': entry.get('crop_path'),
            'image_path': entry.get('image_path'),
        }

    cam = getattr(entry, 'camera', None)
    ts = getattr(entry, 'timestamp', None)
    lat = cam.latitude if cam else 0.0
    lon = cam.longitude if cam else 0.0
    cam_name = cam.name if cam else f"Camera {getattr(entry, 'camera_id', '')}"

    return {
        'sighting_id': getattr(entry, 'id', None),
        'camera_id': getattr(entry, 'camera_id', 'UNKNOWN_CAM'),
        'camera_name': cam_name,
        'latitude': lat,
        'longitude': lon,
        'timestamp': ts,
        'composite_score': float(getattr(entry, 'composite_score', 1.0) or 1.0),
        'plate_text': getattr(entry, 'plate_text', None),
        'vehicle_type': getattr(entry, 'vehicle_type', None),
        'vehicle_color': getattr(entry, 'vehicle_color', None),
        'crop_path': getattr(entry, 'crop_path', None),
        'image_path': getattr(entry, 'image_path', None),
    }


def reconstruct_route_dag(
    sightings: List[Any],
    max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
    origin_coords: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Reconstructs chronological route trajectory across camera nodes as a Directed Acyclic Graph.
    
    Generates:
    - Ordered waypoints with timestamps and confidence scores
    - Step-by-step segments with velocities, distances, bearings, and kinematic feasibility
    - Summary telemetry: total distance, total elapsed time, average speed, route validity
    """
    if not sightings:
        return {
            'waypoints': [],
            'segments': [],
            'total_distance_km': 0.0,
            'total_duration_sec': 0.0,
            'average_speed_kmh': 0.0,
            'is_valid_route': True,
            'waypoint_count': 0,
            'segment_count': 0,
        }

    parsed = [_parse_sighting_entry(s) for s in sightings]
    parsed = [p for p in parsed if p['timestamp'] is not None]
    parsed.sort(key=lambda x: x['timestamp'])

    waypoints = []
    for idx, item in enumerate(parsed):
        ts_val = item['timestamp']
        ts_str = ts_val.isoformat() if isinstance(ts_val, datetime) else str(ts_val)
        waypoints.append({
            'sequence_order': idx + 1,
            'sighting_id': item['sighting_id'],
            'camera_id': item['camera_id'],
            'camera_name': item['camera_name'],
            'latitude': item['latitude'],
            'longitude': item['longitude'],
            'timestamp': ts_str,
            'composite_score': item['composite_score'],
            'plate_text': item.get('plate_text'),
            'vehicle_type': item.get('vehicle_type'),
            'vehicle_color': item.get('vehicle_color'),
            'crop_path': item.get('crop_path'),
            'image_path': item.get('image_path'),
        })

    segments = []
    total_dist = 0.0
    for i in range(len(parsed) - 1):
        w1, w2 = parsed[i], parsed[i + 1]
        t1 = w1['timestamp']
        t2 = w2['timestamp']

        is_feas, dist, dt, speed = validate_spatio_temporal_transition(
            w1['latitude'], w1['longitude'], t1,
            w2['latitude'], w2['longitude'], t2,
            max_speed_kmh=max_speed_kmh,
        )
        bearing = calculate_bearing(
            w1['latitude'], w1['longitude'],
            w2['latitude'], w2['longitude']
        )

        total_dist += dist
        segments.append({
            'from_sequence': i + 1,
            'to_sequence': i + 2,
            'from_sighting_id': w1['sighting_id'],
            'to_sighting_id': w2['sighting_id'],
            'from_camera_id': w1['camera_id'],
            'to_camera_id': w2['camera_id'],
            'from_camera_name': w1['camera_name'],
            'to_camera_name': w2['camera_name'],
            'distance_km': round(dist, 3),
            'time_delta_sec': dt,
            'speed_kmh': speed,
            'bearing_deg': bearing,
            'feasible': is_feas,
        })

    total_duration_sec = 0.0
    if len(parsed) > 1:
        total_duration_sec = (parsed[-1]['timestamp'] - parsed[0]['timestamp']).total_seconds()

    all_feasible = all(seg['feasible'] for seg in segments) if segments else True
    avg_speed = 0.0
    if total_duration_sec > 0:
        avg_speed = round(total_dist / (total_duration_sec / 3600.0), 2)

    return {
        'waypoints': waypoints,
        'segments': segments,
        'total_distance_km': round(total_dist, 3),
        'total_duration_sec': total_duration_sec,
        'average_speed_kmh': avg_speed,
        'is_valid_route': all_feasible,
        'waypoint_count': len(waypoints),
        'segment_count': len(segments),
    }


def reconstruct_route_trajectory(
    theft_origin: Optional[Dict[str, Any]],
    sightings_list: List[Any],
    max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
) -> Dict[str, Any]:
    """Alias / extended wrapper for DAG route trajectory reconstruction."""
    return reconstruct_route_dag(sightings_list, max_speed_kmh=max_speed_kmh)


# Re-export matching functions & confusion map for direct spatio_temporal module imports
try:
    from backend.services.matching_service import (
        OCR_CONFUSION_MAP,
        modified_levenshtein_similarity,
        compute_cosine_similarity,
        compute_composite_matching_score,
    )
except ImportError:
    try:
        from stolen_vehicle_ai.backend.services.matching_service import (
            OCR_CONFUSION_MAP,
            modified_levenshtein_similarity,
            compute_cosine_similarity,
            compute_composite_matching_score,
        )
    except ImportError:
        pass

