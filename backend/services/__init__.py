"""
Backend Services Package for Indian Police Stolen Vehicle AI Command Center.
Exports:
- spatio_temporal: Geodesic distances, kinematics, DAG route builder
- matching_service: OCR confusion matrix matching, 512-D cosine similarity, composite score, candidate ranking
"""

try:
    from backend.services.spatio_temporal import (
        EARTH_RADIUS_KM,
        DEFAULT_MAX_SPEED_KMH,
        haversine_distance_km,
        calculate_bearing,
        validate_spatio_temporal_transition,
        reconstruct_route_dag,
        reconstruct_route_trajectory,
    )
    from backend.services.matching_service import (
        OCR_CONFUSION_MAP,
        are_confusable,
        modified_levenshtein_similarity,
        compute_cosine_similarity,
        compute_composite_matching_score,
        rank_candidate_sightings,
        MatchingEngine,
    )
except ImportError:
    from stolen_vehicle_ai.backend.services.spatio_temporal import (
        EARTH_RADIUS_KM,
        DEFAULT_MAX_SPEED_KMH,
        haversine_distance_km,
        calculate_bearing,
        validate_spatio_temporal_transition,
        reconstruct_route_dag,
        reconstruct_route_trajectory,
    )
    from stolen_vehicle_ai.backend.services.matching_service import (
        OCR_CONFUSION_MAP,
        are_confusable,
        modified_levenshtein_similarity,
        compute_cosine_similarity,
        compute_composite_matching_score,
        rank_candidate_sightings,
        MatchingEngine,
    )

__all__ = [
    'EARTH_RADIUS_KM',
    'DEFAULT_MAX_SPEED_KMH',
    'haversine_distance_km',
    'calculate_bearing',
    'validate_spatio_temporal_transition',
    'reconstruct_route_dag',
    'reconstruct_route_trajectory',
    'OCR_CONFUSION_MAP',
    'are_confusable',
    'modified_levenshtein_similarity',
    'compute_cosine_similarity',
    'compute_composite_matching_score',
    'rank_candidate_sightings',
    'MatchingEngine',
]

