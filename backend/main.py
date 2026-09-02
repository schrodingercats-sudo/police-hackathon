"""FastAPI Application Entrypoint for Indian Police Stolen Vehicle AI Command Center.

Initializes database schema, seeds baseline Delhi NCR surveillance topology on first boot,
configures CORS and DPDP audit middleware, mounts evidence and frontend static assets,
and exposes all REST API services.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.config import settings
from backend.database.models import Camera
from backend.database.seed_data import seed_all
from backend.database.session import SessionLocal, init_db
from backend.api.routes import router as api_router
from backend.services.audit_service import AuditLoggingMiddleware

# Configure logging format
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stolen_vehicle_ai.backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Event Handler.
    Initializes database schema and populates baseline Delhi NCR camera network
    and mock demonstration data on initial cold boot.
    """
    logger.info("Starting Indian Police Stolen Vehicle AI Command Center Backend...")
    try:
        # Initialize database tables
        init_db()

        # Seed initial cameras and demo data if DB is empty
        db = SessionLocal()
        try:
            camera_count = db.query(Camera).count()
            if camera_count == 0:
                logger.info("Empty database detected; seeding baseline Delhi NCR surveillance network...")
                seed_all(db=db, reset=False)
            else:
                logger.info(f"Database contains {camera_count} surveillance camera nodes. Ready.")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error during database startup initialization: {e}", exc_info=True)

    yield

    logger.info("Shutting down Indian Police Stolen Vehicle AI Command Center Backend...")


# -----------------------------------------------------------------------------
# FastAPI App Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-grade Indian Police command platform for automated ANPR, "
        "visual Re-ID vehicle fingerprinting, spatio-temporal route reconstruction, "
        "and Section 65B court-admissible electronic evidence dossiers."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# -----------------------------------------------------------------------------
# Middleware Configuration
# -----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(AuditLoggingMiddleware)


# -----------------------------------------------------------------------------
# Static Media & Web Assets Mounting
# -----------------------------------------------------------------------------
evidence_path = settings.EVIDENCE_DIR
if not evidence_path.exists():
    os.makedirs(evidence_path, exist_ok=True)
app.mount("/evidence", StaticFiles(directory=str(evidence_path)), name="evidence")
# Also mount at /data/evidence since mock_stream stores DB paths like /data/evidence/crops/...
app.mount("/data/evidence", StaticFiles(directory=str(evidence_path)), name="evidence_data")

frontend_dir = settings.BASE_DIR / "frontend"
static_dir = frontend_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# -----------------------------------------------------------------------------
# API Router Registration
# -----------------------------------------------------------------------------
app.include_router(api_router)


# -----------------------------------------------------------------------------
# Root UI Route
# -----------------------------------------------------------------------------
@app.get("/", summary="Command Center Web Interface", include_in_schema=False)
async def serve_dashboard():
    """Serves the main Police Command Center web application."""
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        content={
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "status": "operational",
            "api_docs": "/docs",
            "health_check": "/api/health",
        }
    )


# -----------------------------------------------------------------------------
# Exception Handlers
# -----------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception catcher providing sanitized JSON error responses."""
    logger.error(f"Unhandled Exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing the command request.",
            "path": request.url.path,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
