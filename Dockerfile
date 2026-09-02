# =============================================================================
# Multi-Stage Dockerfile for Indian Police Stolen Vehicle AI Command Platform
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build Dependencies & Wheels
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# Set build environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install system compilation prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --user --no-warn-script-location -r requirements.txt


# -----------------------------------------------------------------------------
# Stage 2: Final Production Runtime Image
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runner

# Metadata
LABEL maintainer="Indian Police CCTNS AI Division" \
      version="1.0.0" \
      description="Production AI System for Stolen Vehicle Detection, Re-ID, Spatio-Temporal Association & Evidence Dossiers"

# Set runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/home/police_app/.local/bin:$PATH \
    HOST=0.0.0.0 \
    PORT=8000 \
    DATABASE_URL=sqlite:////app/data/stolen_vehicle_ai.db \
    DEBUG=False

WORKDIR /app

# Install runtime system libraries required by OpenCV headless and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create secure, unprivileged system user
RUN groupadd -g 1000 police_app && \
    useradd -u 1000 -g police_app -m -s /bin/bash police_app

# Copy installed Python packages from builder stage to user local directory
COPY --from=builder --chown=police_app:police_app /root/.local /home/police_app/.local

# Copy application source tree
COPY --chown=police_app:police_app . /app

# Create necessary persistent data directories with proper permissions
RUN mkdir -p /app/data/evidence/frames \
             /app/data/evidence/crops \
             /app/data/evidence/plates \
             /app/data/cameras \
             /app/data/synthetic_feeds && \
    chown -R police_app:police_app /app/data

# Switch to non-root user for security hardening
USER police_app

# Expose FastAPI backend and dashboard port
EXPOSE 8000

# Container Healthcheck
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default Command: Start FastAPI application with Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
