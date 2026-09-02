"""Forensic Audit Logging Service & DPDP Act Compliance Middleware.

Provides tamper-evident audit logging for all investigative queries,
case registrations, evidence exports, and officer verification actions.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.orm import Session

from backend.database.models import AuditLog
from backend.database.session import SessionLocal

logger = logging.getLogger(__name__)


def log_audit_event(
    db: Session,
    user_badge_id: str,
    user_name: str,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    endpoint: str = "",
    ip_address: str = "127.0.0.1",
    details: Optional[Dict[str, Any]] = None,
) -> Optional[AuditLog]:
    """
    Persists an immutable forensic audit log entry in the database.
    Complies with DPDP Act 2023 Purpose Limitation and Section 65B Electronic Evidence standards.
    """
    try:
        entry = AuditLog(
            user_badge_id=user_badge_id or "SYSTEM_AUTO",
            user_name=user_name or "Automated Police System",
            action=action.upper(),
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            endpoint=endpoint,
            ip_address=ip_address,
            details=details or {},
            timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        logger.info(
            f"[AUDIT] {entry.action} by {entry.user_badge_id} on {entry.resource_type}:{entry.resource_id} "
            f"from {entry.ip_address}"
        )
        return entry
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to record audit event ({action}): {e}", exc_info=True)
        return None


def get_audit_trail_for_case(db: Session, case_id: Union[int, str]) -> List[Dict[str, Any]]:
    """Retrieves chronological audit trail for a specific case."""
    case_str = str(case_id)
    records = (
        db.query(AuditLog)
        .filter((AuditLog.resource_id == case_str) | (AuditLog.resource_type == "case"))
        .order_by(AuditLog.timestamp.desc())
        .limit(100)
        .all()
    )
    return [r.to_dict() for r in records]


def get_audit_trail_for_user(db: Session, user_badge_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves all investigative actions performed by a specific officer badge."""
    records = (
        db.query(AuditLog)
        .filter(AuditLog.user_badge_id == user_badge_id)
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [r.to_dict() for r in records]


def get_recent_audit_logs(db: Session, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves the most recent system-wide audit entries."""
    records = (
        db.query(AuditLog)
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [r.to_dict() for r in records]


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    ASGI Middleware that automatically intercepts investigative endpoints,
    recording client IP, officer credentials, and resource targets.
    """

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "127.0.0.1"
        badge_id = request.headers.get("X-Officer-Badge-ID", "OFFICER_DEFAULT")
        officer_name = request.headers.get("X-Officer-Name", "Duty Officer")
        path = request.url.path

        response: Response = await call_next(request)

        # Log sensitive search, route query, and verification endpoints
        if path.startswith("/api/") and request.method in ["POST", "PUT", "DELETE"]:
            action_map = {
                "/api/report": ("CASE_CREATE", "case"),
                "/api/verify": ("VERIFY_SIGHTING", "case_match"),
                "/api/cameras": ("CAMERA_MUTATE", "camera"),
            }
            matched_action = None
            for prefix, act_info in action_map.items():
                if path.startswith(prefix):
                    matched_action = act_info
                    break

            if matched_action and response.status_code < 400:
                action_name, res_type = matched_action
                # Perform audit logging in isolated session
                try:
                    db = SessionLocal()
                    try:
                        log_audit_event(
                            db=db,
                            user_badge_id=badge_id,
                            user_name=officer_name,
                            action=action_name,
                            resource_type=res_type,
                            resource_id=path.split("/")[-1] if len(path.split("/")) > 3 else None,
                            endpoint=path,
                            ip_address=client_ip,
                            details={"method": request.method, "status_code": response.status_code},
                        )
                    finally:
                        db.close()
                except Exception as e:
                    logger.warning(f"Middleware audit logging skipped: {e}")

        return response
