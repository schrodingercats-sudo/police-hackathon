"""
Matching Engine & Multi-Modal Composite Scorer for Indian Police Stolen Vehicle AI System.
Implements:
- Positional OCR confusion matrix distance compensation (0<->O/Q/D, 1<->I/L/T, 8<->B/3, 5<->S, 2<->Z, 6<->G/C, A<->4)
- 512-D L2-normalized cosine similarity computation
- Dynamic weight-shifting composite matching engine:
    * Clear plate: 0.50 * Plate + 0.35 * Re-ID + 0.15 * Attributes
    * Occluded plate: 0.65 * Re-ID + 0.20 * Color + 0.15 * Type
- Database candidate sighting ranking and CaseMatch record management
"""

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
from sqlalchemy.orm import Session

try:
    from backend.database.models import Sighting, Case, CaseMatch, Camera, VerificationStatus, CaseStatus
    from backend.config import settings
except ImportError:
    try:
        from stolen_vehicle_ai.backend.database.models import Sighting, Case, CaseMatch, Camera, VerificationStatus, CaseStatus
        from stolen_vehicle_ai.backend.config import settings
    except ImportError:
        pass


# ----------------------------------------------------------------------------
# 1. OCR CONFUSION MATRIX & CHARACTER MAPPING
# ----------------------------------------------------------------------------

OCR_CONFUSION_MAP: Dict[str, Set[str]] = {
    '0': {'O', 'Q', 'D', '0'},
    'O': {'0', 'Q', 'D', 'O'},
    'Q': {'0', 'O', 'D', 'Q'},
    'D': {'0', 'O', 'Q', 'D'},
    '1': {'I', 'L', 'T', '1', '|', '/'},
    'I': {'1', 'L', 'T', 'I', '|', '/'},
    'L': {'1', 'I', 'T', 'L'},
    'T': {'1', 'I', 'L', 'T'},
    '8': {'B', '3', '8'},
    'B': {'8', '3', 'B'},
    '3': {'8', 'B', '3'},
    '5': {'S', '5'},
    'S': {'5', 'S'},
    '2': {'Z', '2'},
    'Z': {'2', 'Z'},
    '6': {'G', '6', 'C'},
    'G': {'6', 'G', 'C'},
    'C': {'G', '6', 'C'},
    'A': {'4', 'A'},
    '4': {'A', '4'},
    'V': {'U', 'V', 'Y'},
    'U': {'V', 'U'},
    'Y': {'V', 'Y'},
}


def are_confusable(c1: str, c2: str) -> bool:
    """Checks if two characters are prone to Indian ANPR OCR confusion."""
    u1, u2 = str(c1).upper(), str(c2).upper()
    if u1 == u2:
        return True
    return u2 in OCR_CONFUSION_MAP.get(u1, set()) or u1 in OCR_CONFUSION_MAP.get(u2, set())


def modified_levenshtein_similarity(
    target: Optional[str],
    candidate: Optional[str],
    confusion_penalty: float = 0.30,
    mismatch_penalty: float = 1.0,
) -> float:
    """
    Computes Levenshtein similarity [0.0, 1.0] with OCR confusion matrix discount.
    Known OCR substitutions (e.g. 0/O, 1/I, 8/B) incur a reduced penalty (default 0.30)
    instead of full character replacement penalty (1.0).
    """
    if target is None and candidate is None:
        return 1.0
    if target is None or candidate is None:
        return 0.0

    clean_t = "".join(c for c in str(target).upper() if c.isalnum())
    clean_c = "".join(c for c in str(candidate).upper() if c.isalnum())

    if not clean_t and not clean_c:
        return 1.0
    if not clean_t or not clean_c:
        return 0.0
    if clean_t == clean_c:
        return 1.0

    n, m = len(clean_t), len(clean_c)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = float(i)
    for j in range(m + 1):
        dp[0][j] = float(j)

    for i in range(1, n + 1):
        char_t = clean_t[i - 1]
        for j in range(1, m + 1):
            char_c = clean_c[j - 1]
            if char_t == char_c:
                sub_cost = 0.0
            elif are_confusable(char_t, char_c):
                sub_cost = confusion_penalty
            else:
                sub_cost = mismatch_penalty

            dp[i][j] = min(
                dp[i - 1][j] + 1.0,        # Deletion
                dp[i][j - 1] + 1.0,        # Insertion
                dp[i - 1][j - 1] + sub_cost # Substitution
            )

    max_len = max(n, m)
    dist = dp[n][m]
    similarity = max(0.0, 1.0 - (dist / float(max_len)))
    return round(float(similarity), 4)


# ----------------------------------------------------------------------------
# 2. VISUAL EMBEDDING COSINE SIMILARITY
# ----------------------------------------------------------------------------

def compute_cosine_similarity(
    vec_a: Optional[List[float]],
    vec_b: Optional[List[float]],
) -> float:
    """
    Computes cosine similarity between two 512-D L2-normalized float vectors.
    Returns a float clamped to [0.0, 1.0].
    """
    if vec_a is None or vec_b is None:
        return 0.0
    if len(vec_a) == 0 or len(vec_b) == 0 or len(vec_a) != len(vec_b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        fa, fb = float(a), float(b)
        dot += fa * fb
        norm_a += fa * fa
        norm_b += fb * fb

    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0

    sim = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    return round(float(max(0.0, min(1.0, sim))), 4)


# ----------------------------------------------------------------------------
# 3. DYNAMIC WEIGHT-SHIFTING COMPOSITE MATCHING SCORE
# ----------------------------------------------------------------------------

def compute_composite_matching_score(
    reported_plate: str,
    sighting_plate: Optional[str] = None,
    plate_confidence: float = 0.0,
    reference_embedding: Optional[List[float]] = None,
    sighting_embedding: Optional[List[float]] = None,
    reported_type: Optional[str] = None,
    sighting_type: Optional[str] = None,
    reported_color: Optional[str] = None,
    sighting_color: Optional[str] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Computes a multi-modal composite matching score between reported stolen case
    and candidate sighting.
    
    Dynamic Weight Allocation:
    1. Clear & Confident Plate (Confidence >= 0.50 & Plate Similarity >= 0.40):
       Score = 0.50 * Plate + 0.35 * Visual Re-ID + 0.15 * Attributes
    2. Occluded / Missing / Degraded Plate:
       Score = 0.65 * Visual Re-ID + 0.20 * Color + 0.15 * Vehicle Type
    """
    # 1. Plate matching
    if sighting_plate is None or not str(sighting_plate).strip():
        plate_sim = 0.0
        plate_score = 0.0
    else:
        plate_sim = modified_levenshtein_similarity(reported_plate, sighting_plate)
        conf = float(plate_confidence) if plate_confidence is not None else 1.0
        conf = max(0.0, min(1.0, conf))
        plate_score = plate_sim * (conf if conf > 0 else 1.0)

    # 2. Visual Re-ID matching
    if reference_embedding and sighting_embedding:
        visual_score = compute_cosine_similarity(reference_embedding, sighting_embedding)
    else:
        visual_score = 0.0

    # 3. Attribute matching (Vehicle Category + Color)
    type_score = 0.0
    if reported_type and sighting_type:
        type_score = 1.0 if str(reported_type).strip().lower() == str(sighting_type).strip().lower() else 0.0
    elif not reported_type or not sighting_type:
        type_score = 0.5

    color_score = 0.0
    if reported_color and sighting_color:
        rc = str(reported_color).strip().lower()
        sc = str(sighting_color).strip().lower()
        if rc == sc or rc in sc or sc in rc:
            color_score = 1.0
        elif rc in ("silver", "grey", "gray") and sc in ("silver", "grey", "gray", "silver/grey"):
            color_score = 1.0
        else:
            color_score = 0.0
    elif not reported_color or not sighting_color:
        color_score = 0.5

    attribute_score = 0.5 * type_score + 0.5 * color_score

    # 4. Dynamic weight evaluation
    has_confident_plate = (
        sighting_plate is not None
        and bool(str(sighting_plate).strip())
        and plate_confidence is not None
        and float(plate_confidence) >= 0.50
        and plate_sim >= 0.40
    )

    if has_confident_plate:
        composite = (0.50 * plate_score) + (0.35 * visual_score) + (0.15 * attribute_score)
    else:
        # Occluded plate / missing plate / low OCR confidence
        composite = (0.65 * visual_score) + (0.20 * color_score) + (0.15 * type_score)

    composite = round(max(0.0, min(1.0, float(composite))), 4)
    breakdown = {
        "plate_score": round(float(plate_score), 4),
        "plate_similarity": round(float(plate_sim), 4),
        "visual_score": round(float(visual_score), 4),
        "attribute_score": round(float(attribute_score), 4),
        "type_score": round(float(type_score), 4),
        "color_score": round(float(color_score), 4),
        "plate_occluded": not has_confident_plate,
    }
    return composite, breakdown


# ----------------------------------------------------------------------------
# 4. CANDIDATE SIGHTING RANKING & MATCH PERSISTENCE
# ----------------------------------------------------------------------------

def rank_candidate_sightings(
    db_session: Session,
    case: Any,
    top_k: int = 20,
    min_score: float = 0.40,
    max_speed_kmh: float = 120.0,
    persist_matches: bool = True,
) -> List[Any]:
    """
    Queries database sightings, computes composite score against case specifications,
    filters by spatio-temporal feasibility from theft origin, and saves/updates CaseMatch records.
    """
    try:
        from backend.services.spatio_temporal import validate_spatio_temporal_transition
    except ImportError:
        from stolen_vehicle_ai.backend.services.spatio_temporal import validate_spatio_temporal_transition

    sightings = db_session.query(Sighting).all()
    candidates = []

    for s in sightings:
        score, breakdown = compute_composite_matching_score(
            reported_plate=case.reported_plate,
            sighting_plate=s.plate_text,
            plate_confidence=s.plate_confidence if s.plate_confidence is not None else 0.0,
            reference_embedding=case.reference_embedding,
            sighting_embedding=s.embedding,
            reported_type=case.vehicle_type,
            sighting_type=s.vehicle_type,
            reported_color=case.vehicle_color,
            sighting_color=s.vehicle_color,
        )

        spatio_feasible = True
        cam = s.camera or db_session.query(Camera).filter(Camera.id == s.camera_id).first()
        if cam and case.theft_latitude and case.theft_longitude and case.theft_datetime:
            is_feas, dist, dt, speed = validate_spatio_temporal_transition(
                case.theft_latitude,
                case.theft_longitude,
                case.theft_datetime,
                cam.latitude,
                cam.longitude,
                s.timestamp,
                max_speed_kmh=max_speed_kmh,
            )
            if dt > 0 and not is_feas:
                spatio_feasible = False
            elif dt < -3600:
                spatio_feasible = False

        st_score = 1.0 if spatio_feasible else 0.0
        final_score = score * (1.0 if spatio_feasible else 0.2)

        if final_score >= min_score:
            candidates.append({
                "sighting": s,
                "score": final_score,
                "breakdown": breakdown,
                "spatio_score": st_score,
            })

    candidates.sort(key=lambda x: (x["score"], x["sighting"].timestamp), reverse=True)
    top_candidates = candidates[:top_k]

    ordered_by_time = sorted(top_candidates, key=lambda x: x["sighting"].timestamp)
    time_order_map = {c["sighting"].id: idx + 1 for idx, c in enumerate(ordered_by_time)}

    matches = []
    for c in top_candidates:
        s = c["sighting"]
        score = c["score"]
        bd = c["breakdown"]

        match = db_session.query(CaseMatch).filter(
            CaseMatch.case_id == case.id,
            CaseMatch.sighting_id == s.id
        ).first()

        if match is None:
            match = CaseMatch(
                case_id=case.id,
                sighting_id=s.id,
                plate_score=bd["plate_score"],
                visual_score=bd["visual_score"],
                attribute_score=bd["attribute_score"],
                spatio_temporal_score=c["spatio_score"],
                composite_score=score,
                sequence_order=time_order_map.get(s.id, 1),
                verification_status=VerificationStatus.PENDING,
            )
            if persist_matches:
                db_session.add(match)
        else:
            match.plate_score = bd["plate_score"]
            match.visual_score = bd["visual_score"]
            match.attribute_score = bd["attribute_score"]
            match.spatio_temporal_score = c["spatio_score"]
            match.composite_score = score
            match.sequence_order = time_order_map.get(s.id, 1)

        matches.append(match)

    if persist_matches:
        db_session.commit()

    return matches


# ----------------------------------------------------------------------------
# 5. MATCHING ENGINE CLASS
# ----------------------------------------------------------------------------

class MatchingEngine:
    """High-level Matching Engine for Stolen Vehicle AI Command Center."""

    def __init__(self, db: Optional[Session] = None, max_speed_kmh: float = 120.0):
        self.db = db
        self.max_speed_kmh = max_speed_kmh

    def compute_composite_matching_score(self, *args, **kwargs) -> Tuple[float, Dict[str, float]]:
        return compute_composite_matching_score(*args, **kwargs)

    def rank_candidate_sightings(
        self,
        case: Any,
        top_k: int = 20,
        min_score: float = 0.40,
        db: Optional[Session] = None
    ) -> List[Any]:
        session = db or self.db
        if not session:
            raise ValueError("Database session required for ranking candidates")
        return rank_candidate_sightings(
            session,
            case,
            top_k=top_k,
            min_score=min_score,
            max_speed_kmh=self.max_speed_kmh
        )

    def evaluate_sighting(self, case: Any, sighting: Any) -> Tuple[float, Dict[str, float]]:
        return compute_composite_matching_score(
            reported_plate=case.reported_plate,
            sighting_plate=sighting.plate_text,
            plate_confidence=sighting.plate_confidence if sighting.plate_confidence is not None else 0.0,
            reference_embedding=case.reference_embedding,
            sighting_embedding=sighting.embedding,
            reported_type=case.vehicle_type,
            sighting_type=sighting.vehicle_type,
            reported_color=case.vehicle_color,
            sighting_color=sighting.vehicle_color,
        )

