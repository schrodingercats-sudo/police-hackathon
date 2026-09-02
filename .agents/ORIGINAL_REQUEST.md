# Original User Request

## Initial Request — 2026-09-01T05:48:00Z

Build production-grade AI system for Indian police to detect, track, re-identify stolen vehicles across multi-camera CCTV feeds, reconstruct movement routes on map, and generate verifiable evidence packages.

Working directory: c:/Users/prath/OneDrive/Desktop/hackathon/police hackathon/stolen_vehicle_ai
Integrity mode: development

## Requirements

### R1. Multi-Stage Computer Vision & ANPR Pipeline
Build modular Python vision pipeline:
- Vehicle detection for cars, motorcycles, trucks, buses using YOLO.
- Single-camera multi-object tracking (ByteTrack / Sort).
- License plate detection and OCR optimized for Indian license plate formats (HSRP, state codes like MH, DL, KA, GJ, multi-line / commercial plates).
- Visual feature extraction and vehicle fingerprinting (color, body type, visual embedding for Re-ID when plates are occluded or unreadable).

### R2. Cross-Camera Spatio-Temporal Association & Route Reconstruction
Build graph/temporal matching engine:
- Connect sightings of same vehicle across distinct camera locations using composite score (plate match weight, visual embedding cosine similarity, color/type match).
- Spatio-temporal feasibility filtering (enforce maximum speed limits and minimum travel time between GPS camera coordinates).
- Reconstruct chronological route trajectory with camera IDs, timestamps, directions, and match confidence scores.

### R3. Backend REST API & Relational Database
Build FastAPI backend service:
- Database schema (SQLAlchemy / SQLite / PostgreSQL): Cameras (id, lat, lon, desc), Vehicles, Sightings (timestamp, camera_id, plate, bbox, embedding, image_path, confidence), Cases (stolen report, vehicle details, status), AuditLogs (user, action, timestamp).
- Endpoints:
  - POST /api/report: Register stolen vehicle case.
  - GET /api/search: Query sightings by plate, visual similarity, color, date-time range.
  - GET /api/route: Return reconstructed route coordinates and evidence points for case.
  - GET /api/sightings: Fetch detection history with evidence snapshots.
  - POST /api/verify: Police officer review/verification toggle (confirm/reject sighting).
  - GET /api/dashboard/stats: System metrics (active cases, sightings today, camera statuses).

### R4. Police Web Dashboard UI
Build modern, responsive police investigation dashboard (React + Leaflet / MapLibre or standalone web interface):
- Interactive map plotting camera nodes, detection markers, and route paths with directional indicators.
- Chronological timeline sidebar showing vehicle snapshots, plate crops, timestamps, confidence scores.
- Search interface supporting exact plate search, color/type filters, and reference image upload.
- Evidence package export (printable/PDF report view) and human-in-the-loop verification buttons.

### R5. Mock Multi-Camera Video Pipeline & Automated Test Suite
Provide complete runnable evaluation harness:
- Synthetic/mock multi-camera video and image stream generator simulating stolen vehicle travelling through simulated city intersection cameras.
- Comprehensive PyTest suite covering detection, OCR parsing, tracking continuity, spatio-temporal validation, API endpoints, and database operations.

### R6. Containerization & Documentation
- Dockerfile and docker-compose.yml for single-command deployment.
- Complete README.md with architecture diagram, API documentation, CLI test commands, and quickstart guide.

## Acceptance Criteria

### Vision & ANPR Pipeline
- [ ] Vehicle detection detects vehicles with bounding boxes and classes.
- [ ] OCR module correctly extracts alphanumeric characters from Indian plate crops with confidence scoring.
- [ ] Re-ID extractor computes normalized feature embeddings for vehicle crops.

### Cross-Camera Association & Route Engine
- [ ] Spatio-temporal filter correctly rejects impossible sightings (speed > threshold between distant cameras).
- [ ] Route builder produces chronologically sorted GPS waypoints matching simulated vehicle path.

### Backend & API
- [ ] FastAPI app boots and serves all endpoints defined in spec.
- [ ] Stolen vehicle registration creates case and triggers search against existing sightings.
- [ ] Database correctly stores and indexes cameras, sightings, cases, and audit logs.

### Dashboard UI
- [ ] Web dashboard renders Leaflet map with camera markers and route polyline.
- [ ] Sighting timeline displays thumbnail image crops and plate strings.
- [ ] Police verification action updates sighting status in database.

### Automated Tests & Deployment
- [ ] PyTest suite executes and passes 100% of test cases.
- [ ] docker-compose.yml or local runner script starts backend and frontend cleanly.
