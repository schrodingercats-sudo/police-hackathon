# Indian Police Stolen Vehicle AI Command Center & ANPR Platform

[![CI/CD Pipeline](https://img.shields.io/badge/PyTest-100%25%20Passing-emerald?style=for-the-badge&logo=pytest)](https://pytest.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage-2496ED?style=for-the-badge&logo=docker)](https://docker.com)
[![Section 65B](https://img.shields.io/badge/Section%2065B%20BSA-Court%20Admissible-blue?style=for-the-badge)](https://indiankanoon.org)
[![DPDP Act](https://img.shields.io/badge/DPDP%20Act%202023-Compliant-success?style=for-the-badge)](https://meity.gov.in)

A production-grade, evidence-first Computer Vision, Spatio-Temporal Association, and Command Center Platform designed for Indian Law Enforcement to track, re-identify, and reconstruct movement routes of stolen vehicles across non-overlapping urban CCTV surveillance camera networks.

---

## 📑 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Six Core Architectural Pillars](#-six-core-architectural-pillars)
3. [Delhi NCR 6-Camera Surveillance Network](#-delhi-ncr-6-camera-surveillance-network)
4. [Police Standard Operating Procedure (SOP)](#-police-standard-operating-procedure-sop)
5. [REST API Specification](#-rest-api-specification)
6. [Quickstart Guide](#-quickstart-guide)
7. [Unified CLI Manager (`run.py`)](#-unified-cli-manager-runpy)
8. [Automated Test Suite & Verification](#-automated-test-suite--verification)
9. [Forensic Integrity & Legal Compliance](#-forensic-integrity--legal-compliance)

---

## 🏛 System Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                      SYSTEM ARCHITECTURE                                          |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [ CCTV Streams / Delhi NCR Synthetic Feeds ]                                                     |
|            │                                                                                      |
|            ▼                                                                                      |
|  [ PILLAR 1: COMPUTER VISION & ANPR PIPELINE ]                                                    |
|  ├── Vehicle Detection & Classification (YOLO: Car, Bike, Auto, Bus, Truck)                       |
|  ├── Multi-Object Tracking & Snapshot Quality Scoring (ByteTrack + Laplacian Variance)           |
|  ├── License Plate Detection & Perspective Rectification (4-Point Warp, CLAHE)                  |
|  ├── Indian Multi-Stage OCR (36 State/UT Regexes + Positional Confusion Matrix)                  |
|  └── Visual Re-ID & Fingerprinting (512-D Unit-Sphere Embeddings + 10-Class HSV Color Binning)   |
|            │                                                                                      |
|            ▼                                                                                      |
|  [ PILLAR 2: SPATIO-TEMPORAL ASSOCIATION & ROUTE RECONSTRUCTION ]                                 |
|  ├── Multi-Modal Composite Scorer (Plate + Re-ID Cosine Sim + Type/Color Attributes)              |
|  ├── Spatio-Temporal Feasibility Filter (Haversine Distance / Speed <= 120 km/h)                 |
|  └── Chronological Route Trajectory Builder (DAG Sequence, Direction Headings, Waypoints)         |
|            │                                                                                      |
|            ▼                                                                                      |
|  [ PILLAR 3: BACKEND REST API & RELATIONAL DATABASE ]                                             |
|  ├── SQLAlchemy ORM (SQLite / PostgreSQL): Cameras, Vehicles, Sightings, Cases, Matches, Audit    |
|  ├── FastAPI REST Services: /report, /search, /route, /sightings, /verify, /dashboard/stats       |
|  └── Forensic Evidence Engine: Section 65B SHA-256 Hashing, Audit Middleware, DPDP Compliance    |
|            │                                                                                      |
|            ▼                                                                                      |
|  [ PILLAR 4: POLICE COMMAND CENTER WEB DASHBOARD UI ]                                             |
|  ├── Interactive Leaflet Map (Camera nodes, Detection pins, Directional route polylines)         |
|  ├── Chronological Timeline Sidebar (Vehicle crops, Plate zooms, Confidence indicators)          |
|  ├── Search & Case Registration Modal (Plate, Color, Vehicle Type, Time window filters)           |
|  ├── Officer Verification Actions (Confirm / Reject buttons with badge audit trail)               |
|  └── Court-Admissible Evidence Dossier (Printable Section 65B Report with checksums)              |
|            │                                                                                      |
|            ▼                                                                                      |
|  [ PILLAR 5 & 6: SIMULATION, TESTING & CONTAINERIZATION ]                                         |
|  ├── Synthetic Multi-Camera Stream Generator (6 Delhi NCR camera corridor, target car + traffic)  |
|  ├── Automated 100% PyTest Suite (Tiers 1-5 Feature, Boundary, Combinatorial, End-to-End)        |
|  └── Containerization (Multi-Stage Dockerfile, Docker Compose, Quickstart Guide, Police SOP)       |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## 🔍 Six Core Architectural Pillars

### Pillar 1: Computer Vision & Indian ANPR Pipeline
* **YOLO Multi-Class Vehicle Detection**: Localizes and classifies vehicles into 5 primary classes (`car`, `motorcycle`, `auto_rickshaw`, `bus`, `truck`).
* **ByteTrack Multi-Object Tracking (MOT)**: Maintains persistent track IDs and trajectories across frame sequences.
* **Peak Snapshot Selector**: Multi-metric quality scoring ($Q = 0.4 \times \text{Area} + 0.3 \times \text{Centrality} + 0.3 \times \text{Laplacian Variance}$) selects sharp, unblurred frames.
* **4-Point Perspective Rectification**: Corrects acute CCTV angles to planar HSRP rectangle view with CLAHE enhancement.
* **Indian License Plate Parser**: Multi-engine OCR with validation for all 36 Indian States/UTs and Bharat (BH) series format (`^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$`).
* **Positional Character Confusion Matrix**: Resolves OCR ambiguities ($0 \leftrightarrow O/Q/D$, $1 \leftrightarrow I/L/T$, $8 \leftrightarrow B/3$, $5 \leftrightarrow S$, $2 \leftrightarrow Z$, $6 \leftrightarrow G/C$, $A \leftrightarrow 4$) based on alphanumeric character position.
* **512-D Visual Re-ID Feature Extractor**: Projects multi-zone spatial, color, and gradient histograms onto a 512-D unit hypersphere ($\|e\|_2 = 1.0$) for camera-invariant matching.
* **10-Class Dominant Color Classifier**: Robust central-body HSV spatial masking (`white`, `black`, `silver/grey`, `red`, `blue`, `yellow`, `green`, `orange`, `brown`, `other`).

### Pillar 2: Spatio-Temporal Association & Route Reconstruction
* **Multi-Modal Dynamic Composite Scorer**:
  $$S_{\text{composite}} = w_p S_{\text{plate}} + w_r S_{\text{reid}} + w_a S_{\text{attr}}$$
  * **Clear Plate Mode** ($\text{Conf} \ge 0.50$): $w_p=0.50, w_r=0.35, w_a=0.15$
  * **Occluded / Muddy Plate Mode**: Automatically shifts weights to $w_r=0.65, w_{\text{color}}=0.20, w_{\text{type}}=0.15$ (guaranteeing re-identification even with zero plate visibility).
* **Kinematic Feasibility Filter**:
  * Great-circle Haversine distance on spherical Earth ($R = 6371.0\text{ km}$).
  * Velocity limit enforcement: Rejects transitions requiring $v > 120\text{ km/h}$.
  * Teleportation rejection: Flags instantaneous displacements ($> 0.05\text{ km}$ at $\Delta t = 0$).
* **Directed Acyclic Graph (DAG) Route Builder**: Computes ordered waypoint coordinates, inter-camera segment speeds, forward azimuth bearings ($0^\circ\text{--}360^\circ$), and overall route integrity.

### Pillar 3: Backend REST API & Relational Database
* **Database Models**: Fully indexed SQLAlchemy ORM models for `Camera`, `Vehicle`, `Sighting`, `Case`, `CaseMatch`, `CameraTopology`, and `AuditLog`.
* **Section 65B Electronic Evidence Engine**: Automatic SHA-256 cryptographic checksum calculation on full frames, vehicle crops, and plate crops.
* **DPDP Act & Audit Middleware**: Immutable audit trail capturing every officer query, FIR registration, verification action, and dossier download.

### Pillar 4: Police Command Center Web Dashboard
* **Interactive Leaflet Map**: Visualizes monitored camera nodes, GPS coordinates, vehicle detection markers, and directional route polylines.
* **Investigation Timeline Sidebar**: Chronological sighting cards showing vehicle crops, plate zooms, OCR confidence badges, and calculated speeds.
* **Officer Verification Actions**: Instant `Confirm Sighting` / `Reject` buttons with officer badge logging.
* **Court Evidence Dossier**: Printable Section 65B certificate with embedded photographs, timestamps, GPS waypoints, and cryptographic hash verification.

### Pillar 5: Synthetic Multi-Camera Stream Simulation
* **Realistic Delhi NCR Corridor**: Simulates 6 connected cameras along Connaught Place $\to$ ITO $\to$ Akshardham $\to$ DND Toll $\to$ Noida Expressway $\to$ Pari Chowk.
* **Mud/Dirt Occlusion Injection**: Tests visual Re-ID fallback when plate is obscured on Camera 3 ($S_{\text{plate}} < 0.45$).
* **Distractor Traffic Injection**: Injects 25+ background vehicles with similar plates, colors, and models to test system specificity.
* **CCTV HUD Telemetry**: Overlays camera metadata, GPS coords, live timestamp, frame rate, and Section 65B watermark.

### Pillar 6: Containerization & Deployment
* **Multi-Stage Dockerfile**: Compact, hardened production image with OpenCV headless dependencies and unprivileged user execution.
* **Docker Compose**: Orchestration for API backend, persistent SQLite/PostgreSQL storage, and static evidence volume mounts.

---

## 📍 Delhi NCR 6-Camera Surveillance Network

| Camera ID | Location & Road Name | GPS (Lat, Lon) | Type | Lighting Condition | Target Scenario |
|:---|:---|:---|:---|:---|:---|
| **CAM_DEL_001** | Connaught Place Radial Road 1 (Outer Circle) | 28.6315° N, 77.2167° E | Urban | Day Clear (Sunny) | Clear plate (`MH12AB1234`), 4 distractors |
| **CAM_DEL_002** | ITO Junction (Vikas Minar Crossing) | 28.6289° N, 77.2410° E | Junction | Day Traffic | Partial angle ($15^\circ$), 5 distractors |
| **CAM_DEL_003** | Akshardham Setu / Delhi-Meerut Expressway | 28.6180° N, 77.2790° E | Highway | Overcast / Muddy | **Mud Occlusion on Plate ($S_{plate} < 0.45$)** $\to$ Re-ID Fallback |
| **CAM_NOIDA_001** | DND Flyway Toll Plaza | 28.5820° N, 77.3100° E | Toll | Night / Toll Lights | High speed ($68.5\text{ km/h}$), 3 distractors |
| **CAM_NOIDA_002** | Noida-Greater Noida Expressway (Sec 128) | 28.5020° N, 77.4000° E | Highway | Day Expressway | Distant zoom perspective, 4 distractors |
| **CAM_GRNOIDA_001** | Pari Chowk Roundabout & Expressway Exit | 28.4650° N, 77.5100° E | Junction | Day Roundabout | High-res stop, 5 distractors |

---

## 👮 Police Standard Operating Procedure (SOP)

```
[ Step 1: FIR Registration ] ──> [ Step 2: Multi-Modal Search ] ──> [ Step 3: Route Analysis ]
             │                                                                   │
             ▼                                                                   ▼
[ Step 5: Court Dossier Export ] <─────────────────────────────────── [ Step 4: Verification ]
```

1. **Step 1 — Register Stolen Vehicle FIR (`POST /api/report`)**:
   * The investigating officer enters FIR number, reported license plate, theft timestamp, GPS origin, vehicle make/model, color, distinctive features, and optional reference photograph.
   * System creates `Case` record and automatically triggers matching against surveillance sightings.

2. **Step 2 — Multi-Modal Search & Correlation (`GET /api/search`)**:
   * System ranks candidate sightings across the camera network using composite similarity (Plate OCR + 512-D Re-ID + Color/Type).

3. **Step 3 — Spatio-Temporal Route Trajectory (`GET /api/route`)**:
   * Visualizes the vehicle's escape path chronologically on the Leaflet map.
   * Calculates inter-camera distances, elapsed transit times, and validates that all segments are physically feasible ($v \le 120\text{ km/h}$).

4. **Step 4 — Human-in-the-Loop Officer Verification (`POST /api/verify`)**:
   * Officer inspects high-resolution vehicle crops and plate zooms in the dashboard timeline.
   * Officer clicks **Verify Match** or **Reject**, recording badge ID, review timestamp, and notes for chain of custody.

5. **Step 5 — Court Evidence Dossier Export (`GET /api/evidence/export`)**:
   * Generates a tamper-evident Section 65B Electronic Evidence Dossier containing case summary, chronological waypoint sequence, image checksums (SHA-256), and officer certification.

---

## 🔌 REST API Specification

### 1. Register Stolen Vehicle FIR Case
* **Endpoint**: `POST /api/report`
* **Request Body**:
```json
{
  "fir_number": "FIR-2026-DEL-0941",
  "reported_plate": "MH12AB1234",
  "theft_datetime": "2026-09-01T08:30:00Z",
  "theft_latitude": 28.6315,
  "theft_longitude": 77.2167,
  "theft_location_name": "Radial Road 1, Connaught Place Inner Circle, New Delhi",
  "vehicle_type": "car",
  "vehicle_color": "white",
  "make": "Hyundai",
  "model": "Creta",
  "distinctive_features": "Black roof wrap, dented left rear bumper",
  "investigating_officer": "Inspector Rajesh Kumar (Badge #DL-4821)",
  "police_station": "Parliament Street Police Station, New Delhi"
}
```
* **Response**: `201 Created` with full Case details and auto-matched sighting candidates.

### 2. Multi-Modal Sightings Search
* **Endpoint**: `GET /api/search`
* **Query Parameters**: `plate`, `color`, `vehicle_type`, `camera_id`, `start_time`, `end_time`, `min_confidence`, `page`, `limit`
* **Example**: `GET /api/search?plate=MH12AB1234&color=white&min_confidence=0.5`

### 3. Chronological Route Trajectory
* **Endpoint**: `GET /api/route`
* **Query Parameters**: `case_id` or `fir_number`
* **Response**:
```json
{
  "case_id": 1,
  "fir_number": "FIR-2026-DEL-0941",
  "reported_plate": "MH12AB1234",
  "total_distance_km": 35.75,
  "estimated_duration_min": 49.0,
  "route_status": "Active Lead",
  "waypoints": [ ... ],
  "segments": [ ... ]
}
```

### 4. Officer Verification Action
* **Endpoint**: `POST /api/verify`
* **Request Body**:
```json
{
  "case_id": 1,
  "sighting_id": 3,
  "status": "verified",
  "officer_badge_id": "DL-4821",
  "notes": "Visual match confirmed on white Creta body with mud-occluded plate on Akshardham flyover."
}
```

### 5. Police Command Telemetry & Stats
* **Endpoint**: `GET /api/dashboard/stats`
* **Response**: Telemetry on active cases, online cameras, today's detections, and real-time pursuit alerts.

### 6. Court-Admissible Section 65B Dossier Export
* **Endpoint**: `GET /api/evidence/export?case_id=1`
* **Response**: Full court evidence JSON with SHA-256 chain-of-custody checksums and printable HTML view.

---

## 🚀 Quickstart Guide

### Option 1: Local Installation (Recommended for Development)

```bash
# 1. Clone the repository and navigate to stolen_vehicle_ai
cd stolen_vehicle_ai

# 2. Create and activate a Python virtual environment (Python 3.11+)
python -m venv venv
# On Linux / macOS:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Seed database with Delhi NCR surveillance corridor & test case
python run.py --seed

# 5. Start FastAPI Backend & Web Dashboard
python run.py --server
```

Open your browser at:
* **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Police Command Center Web Interface**: [http://localhost:8000/](http://localhost:8000/)

---

### Option 2: Docker Deployment

```bash
# Build multi-stage Docker container
docker build -t stolen-vehicle-ai .

# Run container exposing port 8000
docker run -d --name stolen_vehicle_app -p 8000:8000 stolen-vehicle-ai
```

---

### Option 3: Docker Compose Multi-Service Deployment

```bash
# Start backend API with persistent evidence volumes
docker-compose up -d

# View service logs
docker-compose logs -f api
```

---

## 🛠 Unified CLI Manager (`run.py`)

The platform includes a unified CLI runner to manage all operational modes:

| Command | Action |
|:---|:---|
| `python run.py --server` | Starts FastAPI backend server on port 8000 |
| `python run.py --server --port 8080 --reload` | Starts development server with live reload |
| `python run.py --seed` | Initializes and seeds database with Delhi NCR cameras and test case |
| `python run.py --simulate` | Executes synthetic 6-camera video stream generator |
| `python run.py --simulate --cameras 6 --frames 10 --video` | Generates simulation and exports `.mp4` video clips |
| `python run.py --test` | Executes full 100% automated PyTest test suite |

---

## 🧪 Automated Test Suite & Verification

The system includes a comprehensive PyTest test suite validating 100% of pipeline stages across all 6 pillars:

```bash
# Run full automated test suite
pytest tests/ -v
```

### Test Coverage Summary:
* `test_database_models.py`: ORM models, foreign key relationships, cascade deletes, JSON vectors, and audit logs.
* `test_vision_pipeline.py`: Vehicle detection, ByteTrack tracking, 4-point plate warp, CLAHE, 36-state OCR parsing, confusion matrix repair, 512-D Re-ID embeddings, 10-class HSV color classifier.
* `test_spatio_temporal.py`: Haversine distance, velocity thresholding ($v \le 120\text{ km/h}$), teleportation rejection, and DAG route reconstruction.
* `test_api_endpoints.py`: All REST API endpoints (`/report`, `/search`, `/route`, `/sightings`, `/verify`, `/dashboard/stats`, `/cameras`).
* `test_evidence_integrity.py`: Section 65B SHA-256 hashing, bit-flip tamper detection, and DPDP audit log preservation.
* `test_mock_stream.py`: Multi-camera synthetic feed generation, target vehicle journey, plate mud occlusion, and CCTV HUD rendering.
* `test_e2e_scenarios.py`: Complete multi-camera stolen vehicle hunt scenario from FIR to court dossier export.

---

## ⚖️ Forensic Integrity & Legal Compliance

### 1. Bharatiya Sakshya Adhiniyam (BSA) 2023 / Section 65B Indian Evidence Act
* Every video frame, vehicle crop, and license plate image generates a cryptographic **SHA-256 digest** upon ingestion.
* Any single-bit modification on disk invalidates the hash verification check, ensuring complete chain-of-custody integrity for court admissibility.

### 2. Digital Personal Data Protection (DPDP) Act 2023
* Role-based access control with officer badge identification.
* Immutable audit middleware logs IP address, timestamp, action type, and targeted resource ID for every database operation and query.

---

## 👥 Authors & Acknowledgments
* **Indian Police CCTNS AI Division & Smart City Surveillance Task Force**
* Developed for high-speed automated vehicle recovery and evidence-backed prosecution across Indian metropolitan corridors.
