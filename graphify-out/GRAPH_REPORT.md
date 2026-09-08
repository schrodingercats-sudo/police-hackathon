# Graph Report - stolen_vehicle_ai  (2026-09-07)

## Corpus Check
- 53 files · ~51,770 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 883 nodes · 1703 edges · 79 communities (65 shown, 14 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 225 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 65
- Community 66
- Community 67
- Community 68
- Community 69
- Community 70
- Community 71
- Community 72
- Community 73
- Community 74
- Community 75
- Community 76
- Community 77
- Community 78

## God Nodes (most connected - your core abstractions)
1. `Sighting` - 62 edges
2. `Case` - 62 edges
3. `Camera` - 61 edges
4. `CaseMatch` - 46 edges
5. `AuditLog` - 31 edges
6. `VerificationStatus` - 30 edges
7. `CaseStatus` - 23 edges
8. `Vehicle` - 21 edges
9. `generate_evidence_dossier()` - 20 edges
10. `CameraTopology` - 19 edges

## Surprising Connections (you probably didn't know these)
- `test_snapshot_manager_hashing_and_storage()` --uses--> `SnapshotManager`  [INFERRED]
  tests/test_vision_pipeline.py → pipeline/snapshot_manager.py
- `test_yolo_vehicle_detector_fallback()` --uses--> `YOLOVehicleDetector`  [INFERRED]
  tests/test_vision_pipeline.py → vision/detector.py
- `test_license_plate_detector()` --uses--> `LicensePlateDetector`  [INFERRED]
  tests/test_vision_pipeline.py → vision/plate_detector.py
- `test_positional_character_confusion_repair()` --calls--> `repair_indian_plate()`  [INFERRED]
  tests/test_vision_pipeline.py → vision/indian_lp_parser.py
- `test_two_line_plate_parsing()` --calls--> `parse_two_line_plate()`  [INFERRED]
  tests/test_vision_pipeline.py → vision/indian_lp_parser.py

## Import Cycles
- None detected.

## Communities (79 total, 14 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (34): get_dashboard_stats(), Returns real-time command metrics: active cases, total cameras, online cameras,…, Case, CaseMatch, Individual vehicle detection event captured by a camera node., Police First Information Report (FIR) stolen vehicle tracking case., Association between a stolen vehicle case and a candidate camera sighting., Sighting (+26 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (39): calculate_sighting_composite_score(), create_case_report(), FastAPI REST API Route Handlers for Indian Police Stolen Vehicle AI Command…, Registers a new stolen vehicle FIR case, decodes and saves reference image (if…, Computes a baseline matching score for candidate association., Application Configuration & Settings for Indian Police Stolen Vehicle AI System., compute_sha256(), generate_mock_embedding() (+31 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (30): test_binarize_plate(), test_clahe_enhance(), test_gamma_correction_and_auto_gamma(), test_laplacian_variance_and_blur(), OCREngine, Any, ndarray, Multi-Engine OCR and Indian License Plate Character Recognition. Provides: -… (+22 more)

### Community 3 - "Community 3"
Cohesion: 0.10
Nodes (30): Updates the verification status of a candidate sighting match to 'verified' or…, verify_sighting(), Database Package Initialization., CameraStatus, CaseStatus, datetime, str, SQLAlchemy ORM Models for Indian Police Stolen Vehicle AI Command Center. (+22 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (24): AuditLog, Camera, CameraTopology, Spatio-temporal adjacency graph edge between two surveillance camera nodes., Tamper-evident forensic audit log complying with Section 65B BSA & DPDP Act., CCTV / ANPR camera node in the surveillance network., Base, populated_db() (+16 more)

### Community 5 - "Community 5"
Cohesion: 0.12
Nodes (17): anpr_scan_image(), Request, Process an uploaded image through real EasyOCR-based ANPR. Accepts multipart…, compute_frame_hash(), LiveANPRProcessor, Any, ndarray, Real-Time ANPR Processor using EasyOCR. Processes camera frames/images to… (+9 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (22): generate_synthetic_streams(), main(), Convenience functional interface to trigger synthetic stream generation., Command Line Entrypoint for Synthetic Multi-Camera Stream Generator., Path, main(), Executes the full automated PyTest test suite., Unified CLI Management Tool for Indian Police Stolen Vehicle AI Command Center.… (+14 more)

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (17): generate_evidence_dossier(), generate_sha256_hash(), Session, Section 65B Bharatiya Sakshya Adhiniyam / Indian Evidence Act Forensic Service.…, Generates a full court-admissible evidence dossier for a given stolen vehicle…, Computes cryptographic SHA-256 hexadecimal digest for raw media bytes or…, Verifies whether the SHA-256 digest of the provided data matches the expected…, verify_evidence_hash() (+9 more)

### Community 8 - "Community 8"
Cohesion: 0.15
Nodes (19): Backend Services Package for Indian Police Stolen Vehicle AI Command Center.…, are_confusable(), compute_composite_matching_score(), compute_cosine_similarity(), modified_levenshtein_similarity(), Matching Engine & Multi-Modal Composite Scorer for Indian Police Stolen Vehicle…, Computes cosine similarity between two 512-D L2-normalized float vectors.…, Computes a multi-modal composite matching score between reported stolen case… (+11 more)

### Community 9 - "Community 9"
Cohesion: 0.15
Nodes (13): ABC, test_mock_vehicle_detector_detection(), BaseVehicleDetector, MockVehicleDetector, ndarray, Vehicle Detection Module for Indian Police Stolen Vehicle AI System. Provides…, YOLO-based Vehicle Detector utilizing Ultralytics YOLO models (v8/v11). Falls…, Classifies whether a compact vehicle crop has classic Indian auto-rickshaw… (+5 more)

### Community 10 - "Community 10"
Cohesion: 0.12
Nodes (15): Any, datetime, ndarray, Evidence Snapshot Manager and Section 65B SHA-256 Cryptographic Digest Engine.…, Verify that an evidence file on disk matches its Section 65B cryptographic…, Evidence Snapshot Manager. Saves vehicle and license plate image crops and…, Encode image to bytes and compute SHA-256 hexadecimal hash., Compute SHA-256 hash of an existing file on disk. (+7 more)

### Community 11 - "Community 11"
Cohesion: 0.19
Nodes (18): Adversarial Stress Test Suite & Edge Case Harvester for Indian Police Stolen…, test_all_36_state_codes_validation(), test_hsrp_and_bh_regex_validation(), clean_plate_string(), format_plate_display(), is_valid_bh_series(), is_valid_hsrp(), is_valid_indian_plate() (+10 more)

### Community 12 - "Community 12"
Cohesion: 0.10
Nodes (11): POST /api/report with latitude < -90.0 triggers 422., POST /api/report with longitude > 180.0 triggers 422., POST /api/report with longitude < -180.0 triggers 422., POST /api/report with non-date string triggers 422., GET /api/search with min_confidence > 1.0 or < 0.0 triggers 422., GET /api/search with page < 1 or limit > 100 triggers 422., GET /api/sightings with offset < 0 or limit > 100 triggers 422., POST /api/cameras with invalid coordinates triggers 422. (+3 more)

### Community 13 - "Community 13"
Cohesion: 0.11
Nodes (19): camera_nodes_data(), db_engine(), db_session(), distractor_embeddings(), make_unit_embedding(), fixture, Raw dictionary definitions of the 6 Delhi NCR CCTV surveillance cameras., Generate a reproducible, mathematically exact L2 unit-norm 512-D embedding. (+11 more)

### Community 14 - "Community 14"
Cohesion: 0.13
Nodes (14): test_compute_iou(), test_compute_snapshot_quality(), ByteTrackTracker, compute_iou(), compute_iou_matrix(), compute_snapshot_quality(), ndarray, Multi-Object Tracking and Snapshot Quality Selection Module. Implements: -… (+6 more)

### Community 15 - "Community 15"
Cohesion: 0.16
Nodes (16): anpr_scan_camera(), export_evidence_dossier(), get_camera_details(), get_camera_frame(), health_check(), list_cameras(), get, Session (+8 more)

### Community 16 - "Community 16"
Cohesion: 0.13
Nodes (8): Any, Serialize model to dictionary., Serialize model to dictionary., Serialize model to dictionary., Serialize model to dictionary., Serialize model to dictionary., Serialize model to dictionary., Serialize model to dictionary.

### Community 17 - "Community 17"
Cohesion: 0.13
Nodes (13): Unit and Integration Tests for Computer Vision & Indian ANPR Pipeline…, test_all_ten_dominant_color_classes(), test_bytetrack_continuity(), test_bytetrack_multi_vehicle_tracking(), test_dominant_color_classifier(), test_format_plate_display(), test_license_plate_detector(), test_ocr_engine_with_synthetic_plate() (+5 more)

### Community 18 - "Community 18"
Cohesion: 0.14
Nodes (8): Empirically verifies 404 Not Found handling on non-existent resources., GET /api/route with non-existent case_id returns 404., GET /api/cases/{case_id} with non-existent ID returns 404., GET /api/cameras/{camera_id} with non-existent camera ID returns 404., GET /api/evidence/export with non-existent case_id returns 404., GET /api/evidence/html with non-existent case_id returns 404., POST /api/verify with non-existent sighting_id returns 404., TestApi404NotFoundHandlers

### Community 19 - "Community 19"
Cohesion: 0.19
Nodes (11): test_four_point_transform(), test_order_points(), four_point_transform(), LicensePlateDetector, order_points(), ndarray, License Plate Detection and Perspective Rectification Module. Provides: -…, Orders 4 polygon points in order: top-left, top-right, bottom-right, bottom-… (+3 more)

### Community 20 - "Community 20"
Cohesion: 0.16
Nodes (8): ndarray, Visual Re-Identification (Re-ID) and Dominant Vehicle Color Classifier.…, Compute cosine similarity between two L2-normalized embeddings. Since ||vec1||…, Extract vehicle body region and classify into one of 10 dominant colors. Args:…, 512-Dimensional Visual Re-Identification Feature Extractor. Extracts dense…, Generates a deterministic orthonormal random projection matrix for metric…, Extract 512-dimensional L2-normalized float embedding from a vehicle image…, ReIDFeatureExtractor

### Community 21 - "Community 21"
Cohesion: 0.15
Nodes (13): CaseCreateRequest, DashboardStatsResponse, EvidenceExportResponse, Officer sighting verification payload., Response confirming officer verification action., Real-time operational command center metrics., Court-admissible electronic evidence dossier with Section 65B checksums., Payload for registering a new First Information Report (FIR) stolen vehicle… (+5 more)

### Community 22 - "Community 22"
Cohesion: 0.17
Nodes (8): ndarray, Calculates SHA-256 digest of JPEG-encoded image., Renders the road environment and background according to lighting condition., Renders a vehicle onto the frame within box_rect = (x1, y1, x2, y2). Returns:…, Applies realistic brown/mud splatter over a target bounding box (e.g. license…, Configuration for a vehicle rendered in a synthetic CCTV stream., SyntheticVehicleConfig, Verifies mud splatter modifies target pixel region.

### Community 23 - "Community 23"
Cohesion: 0.18
Nodes (4): Standardized vehicle detection representation., VehicleDetection, Representation of a tracked vehicle across multiple frames. Maintains best…, Tracklet

### Community 24 - "Community 24"
Cohesion: 0.24
Nodes (7): MatchingEngine, Any, Session, rank_candidate_sightings(), Queries database sightings, computes composite score against case…, High-level Matching Engine for Stolen Vehicle AI Command Center., Verify mathematical weight allocation between Clear Plate vs Occluded Plate: -…

### Community 25 - "Community 25"
Cohesion: 0.21
Nodes (8): build_delhi_ncr_stream_network(), Constructs the 6-camera synthetic surveillance corridor across Delhi NCR with…, Validates the 6-camera corridor topology and vehicle scenario parameters., Verifies exactly 6 cameras configured across Delhi NCR corridor., Ensures target stolen vehicle (MH12AB1234) is present in all 6 cameras., Verifies Camera 3 (Akshardham) has mud/dirt occlusion on target plate with…, Verifies each node has multiple distractors (3 to 5 distractors per camera)., TestSyntheticCorridorConfiguration

### Community 26 - "Community 26"
Cohesion: 0.18
Nodes (7): parametrize, Stress-test OCR parser against extreme noise, muddy plates, and character…, Verify state code recognition under standard optical OCR confusions across…, Stress two-line license plates with line inversions and whitespace variations., Verify parser never throws uncaught exceptions on arbitrary adversarial strings., Verify that OCR confusable pairs (e.g. 0/O, 1/I, 8/B) achieve significantly…, TestAdversarialOCRCorruptions

### Community 27 - "Community 27"
Cohesion: 0.17
Nodes (7): Stress-test speed limit bounds, teleportation detection, and route DAG graph…, Test the exact boundary at 120.0 km/h: - 119.0 km in 3600 sec = 119.0 km/h ->…, Test delta_t = 0 corner cases across small and large distances., Verify that reverse-timestamp transitions (time2 < time1) are rejected., Verify that route DAG constructor correctly sorts out-of-order waypoints and…, Test Haversine distance on extreme GPS edge cases., TestAdversarialKinematicsAndSpeedLimits

### Community 28 - "Community 28"
Cohesion: 0.17
Nodes (7): Tier 3: Camera CRUD endpoints., Verify GET /api/cameras returns all registered camera nodes., Verify POST /api/cameras creates a new camera node., Verify 400 Bad Request when registering duplicate camera ID., Verify GET /api/cameras/{camera_id} returns correct node., Verify 404 for non-existent camera ID., TestCameraManagementEndpoints

### Community 29 - "Community 29"
Cohesion: 0.22
Nodes (10): _parse_sighting_entry(), Any, Extract standard fields from dict or SQLAlchemy model., Reconstructs chronological route trajectory across camera nodes as a Directed…, Alias / extended wrapper for DAG route trajectory reconstruction., reconstruct_route_dag(), reconstruct_route_trajectory(), Tier 1: Chronological route trajectory building. (+2 more)

### Community 30 - "Community 30"
Cohesion: 0.18
Nodes (8): mock_db(), fixture, Automated PyTest Suite for Synthetic Multi-Camera Stream Generator (Milestone…, Tests 512-D Re-ID embedding properties and Section 65B SHA-256 digests., Verifies high visual Re-ID cosine similarity (>0.85) between target sightings…, Verifies SHA-256 calculation produces 64-character hexadecimal digest., Initializes and provides clean database session., TestReIDAndCryptographicHashing

### Community 31 - "Community 31"
Cohesion: 0.22
Nodes (6): KalmanBoxTracker, Kalman Filter for single-object bounding box tracking. State: [u, v, a, h,…, Initialize tracker with initial bounding box (x1, y1, x2, y2)., Advance state vector and return predicted bounding box., Update state vector with observed bounding box., Convert current state vector [u, v, a, h] to (x1, y1, x2, y2).

### Community 32 - "Community 32"
Cohesion: 0.20
Nodes (10): get_case_route(), haversine_km(), Reconstructs chronological trajectory for a reported stolen vehicle case,…, Great-circle distance in kilometers using the Haversine formula., Chronological waypoint on the reconstructed vehicle trajectory., Kinematic transition between consecutive camera detections., Complete chronological route trajectory and kinematic analysis., RouteReconstructionResponse (+2 more)

### Community 33 - "Community 33"
Cohesion: 0.31
Nodes (9): CameraStatusEnum, CaseStatusEnum, str, Pydantic v2 Request & Response Schemas for Indian Police Stolen Vehicle AI…, Officer verification statuses., Investigation case statuses., Surveillance camera operational statuses., VerificationStatusEnum (+1 more)

### Community 34 - "Community 34"
Cohesion: 0.27
Nodes (7): datetime, ndarray, End-to-End Multi-Camera Video and Image Stream Processing Pipeline.…, Process a video file stream and extract all vehicle sightings. Args:…, Standardized atomic sighting result produced by the computer vision pipeline.…, Process a single video frame through the full computer vision & ANPR pipeline.…, SightingResult

### Community 35 - "Community 35"
Cohesion: 0.20
Nodes (6): Empirically verifies 400 Bad Request and duplicate handling across API & DB…, POST /api/verify with unknown status returns 400 Bad Request., POST /api/cameras with already registered ID returns 400., POST /api/report with duplicate FIR returns existing case record gracefully., Direct DB insert of duplicate FIR number enforces UNIQUE constraint &…, TestApi400AndDuplicateConflicts

### Community 36 - "Community 36"
Cohesion: 0.25
Nodes (9): get_sightings_feed(), datetime, Performs multi-criteria search over captured vehicle sightings with pagination,…, Returns paginated stream of recent vehicle sightings., search_sightings(), Data Transfer Object representing a vehicle detection event., Paginated search response for vehicle sightings., SearchResponse (+1 more)

### Community 37 - "Community 37"
Cohesion: 0.22
Nodes (6): get_target_base_embedding(), Generates the reference L2-normalized 512-D embedding for White Hyundai Creta., Verifies 512-D base embedding satisfies L2 unit-norm condition (||e|| == 1.0)., Tests full generator execution and database persistence., Verifies 6-camera simulation creates files and populates database., TestEndToEndSimulationAndDatabaseIngestion

### Community 38 - "Community 38"
Cohesion: 0.28
Nodes (6): Renders photorealistic synthetic CCTV camera frames with: - Asphalt road with…, SyntheticFrameRenderer, Verifies CCTV HUD telemetry bar is drawn at top of frame., Tests image rendering, HUD overlays, and mud occlusion mechanics., Verifies background rendering produces valid 3-channel image., TestSyntheticRendererAndHUD

### Community 39 - "Community 39"
Cohesion: 0.32
Nodes (8): get_audit_trail_for_case(), get_audit_trail_for_user(), get_recent_audit_logs(), Any, Session, Retrieves chronological audit trail for a specific case., Retrieves all investigative actions performed by a specific officer badge., Retrieves the most recent system-wide audit entries.

### Community 40 - "Community 40"
Cohesion: 0.25
Nodes (5): Tier 3: Court-admissible dossier export GET /api/evidence/export., Verify GET /api/evidence/export?case_id=X includes Section 65B SHA-256 hash., Verify 404 Not Found when exporting dossier for non-existent case., Verify GET /api/evidence/html?case_id=X returns HTML document., TestEvidenceExportEndpoint

### Community 41 - "Community 41"
Cohesion: 0.25
Nodes (5): Tier 1 & 2: Multi-modal composite scoring with dynamic weights., High plate confidence + matching visual Re-ID -> Composite > 0.90., Mud on plate (no plate) but matching Re-ID + Color/Type -> Composite retained >…, Different plate, distinct embedding, different color -> Composite < 0.40., TestCompositeMatchingScore

### Community 42 - "Community 42"
Cohesion: 0.25
Nodes (5): Tier 1: Geodesic great-circle distance math., Verify identical coordinates return 0.0 km., Verify distance between Connaught Place and ITO Junction (~2.38 km)., Verify distance between Connaught Place and Pari Chowk Greater Noida (~34-38…, TestHaversineDistance

### Community 43 - "Community 43"
Cohesion: 0.25
Nodes (5): Tier 1 & 2: Velocity thresholding (v <= 120 km/h) and impossible transitions., 5 km in 10 minutes = 30 km/h -> Feasible., 30 km in 2 minutes = 900 km/h -> Infeasible (Rejection)., Distance > 1 km at delta_t = 0 -> Infeasible., TestSpatioTemporalFeasibility

### Community 44 - "Community 44"
Cohesion: 0.29
Nodes (7): create_camera(), Registers a new surveillance camera / toll plaza node into the network., CameraCreateRequest, CameraStatusDTO, Payload for registering a new surveillance camera node., Surveillance camera node telemetry status., post

### Community 45 - "Community 45"
Cohesion: 0.33
Nodes (5): perturb_embedding(), Any, Generates synthetic frame sequence for a camera node. Returns a list of…, Executes full synthetic generation across the Delhi NCR corridor. Optionally…, Applies small environmental perturbation while maintaining L2-normalization.

### Community 46 - "Community 46"
Cohesion: 0.29
Nodes (5): fixture, Stress-test ranking against massive swarms of visually identical distractor…, Provides an isolated in-memory SQLite database session., Test Cosine Similarity against zero vectors, orthogonal vectors, and dimension…, TestAdversarialDistractorRankingAndReID

### Community 47 - "Community 47"
Cohesion: 0.33
Nodes (6): get_case(), list_cases(), Retrieves details of a single case by ID., Retrieves all registered stolen vehicle FIR cases with candidate counts., CaseResponse, Response model for stolen vehicle FIR case details.

### Community 48 - "Community 48"
Cohesion: 0.33
Nodes (4): Tier 3: Case registration endpoint POST /api/report., Verify POST /api/report creates new case and returns CaseResponse., Verify 422 Unprocessable Content when required fields are missing., TestReportFIRCaseEndpoint

### Community 49 - "Community 49"
Cohesion: 0.33
Nodes (4): Tier 3: Multi-criteria sightings search GET /api/search., Verify GET /api/search?plate=MH12AB1234 returns matched sightings., Verify GET /api/search?color=white&type=car filters appropriately., TestSearchSightingsEndpoint

### Community 50 - "Community 50"
Cohesion: 0.33
Nodes (4): Tier 1: Camera node model tests., Verify camera node creation with GPS coordinates and bearing., Verify camera ID uniqueness is strictly enforced., TestCameraModel

### Community 51 - "Community 51"
Cohesion: 0.33
Nodes (4): Tier 1: Unique vehicle registry tests., Verify vehicle creation with attributes., Verify vehicle plate number is unique in registry., TestVehicleModel

### Community 52 - "Community 52"
Cohesion: 0.33
Nodes (4): Tier 1 & 2: Section 65B SHA-256 cryptographic evidence hashing., Verify SHA-256 produces exact 64-hex character string., Verify modifying single byte in evidence payload alters hash (Avalanche Effect)., TestSection65BForensicHashing

### Community 53 - "Community 53"
Cohesion: 0.33
Nodes (4): Tier 1 & 2: String matching with OCR confusion discount., Identical plates return similarity 1.0., Substitution of O for 0 incurs less penalty than random letter., TestModifiedLevenshteinSimilarity

### Community 54 - "Community 54"
Cohesion: 0.40
Nodes (5): global_exception_handler(), Request, Global exception catcher providing sanitized JSON error responses., Exception, exception_handler

### Community 55 - "Community 55"
Cohesion: 0.40
Nodes (4): AuditLoggingMiddleware, Request, ASGI Middleware that automatically intercepts investigative endpoints,…, BaseHTTPMiddleware

### Community 56 - "Community 56"
Cohesion: 0.40
Nodes (5): generate_printable_html_dossier(), generate_section_65b_certificate(), Any, Generates a formal Section 65B Certificate under the Indian Evidence Act, 1872…, Generates a clean, court-ready printable HTML document with Indian Police…

### Community 57 - "Community 57"
Cohesion: 0.40
Nodes (4): CameraNodeConfig, datetime, Configuration for a surveillance camera node in the synthetic corridor., Renders a professional, court-admissible CCTV telemetry HUD banner. Includes…

### Community 59 - "Community 59"
Cohesion: 0.50
Nodes (3): Tier 3: Police Command Center stats telemetry GET /api/dashboard/stats., Verify GET /api/dashboard/stats returns active counts., TestDashboardStatsEndpoint

### Community 60 - "Community 60"
Cohesion: 0.50
Nodes (3): Tier 3: System liveness and telemetry., Verify GET /api/health returns 200 and healthy status., TestHealthEndpoint

### Community 61 - "Community 61"
Cohesion: 0.50
Nodes (3): Tier 3: Case listing and single retrieval., Verify GET /api/cases and GET /api/cases/{case_id}., TestCasesManagementEndpoints

### Community 62 - "Community 62"
Cohesion: 0.50
Nodes (3): Tier 1: Immutable forensic audit trail models., Verify audit log stores user badge, action, endpoint, and JSON payload., TestAuditLogModel

### Community 63 - "Community 63"
Cohesion: 0.50
Nodes (3): Tier 1 & 3: Immutable audit logs for DPDP & police chain of custody., Verify audit log captures all critical police actions with badge IDs., TestImmutableAuditLogging

### Community 64 - "Community 64"
Cohesion: 0.67
Nodes (3): Core application settings with environment variable overrides., Settings, BaseSettings

### Community 65 - "Community 65"
Cohesion: 0.67
Nodes (3): get, Serves the main Police Command Center web application., serve_dashboard()

## Knowledge Gaps
- **4 isolated node(s):** `App`, `EvidenceExport`, `MapView`, `TimelineView`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 420 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Case` connect `Community 0` to `Community 32`, `Community 1`, `Community 3`, `Community 4`, `Community 35`, `Community 7`, `Community 8`, `Community 40`, `Community 11`, `Community 46`, `Community 47`, `Community 16`, `Community 18`, `Community 61`?**
  _High betweenness centrality (0.139) - this node is a cross-community bridge._
- **Why does `ReIDFeatureExtractor` connect `Community 20` to `Community 17`, `Community 10`, `Community 3`, `Community 37`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Why does `Camera` connect `Community 4` to `Community 32`, `Community 1`, `Community 0`, `Community 3`, `Community 36`, `Community 35`, `Community 7`, `Community 8`, `Community 11`, `Community 44`, `Community 46`, `Community 15`, `Community 16`, `Community 49`, `Community 50`, `Community 24`?**
  _High betweenness centrality (0.121) - this node is a cross-community bridge._
- **Are the 26 inferred relationships involving `Sighting` (e.g. with `create_case_report()` and `get_camera_details()`) actually correct?**
  _`Sighting` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `Case` (e.g. with `create_case_report()` and `get_case()`) actually correct?**
  _`Case` has 24 INFERRED edges - model-reasoned connections that need verification._
- **Are the 29 inferred relationships involving `Camera` (e.g. with `anpr_scan_camera()` and `create_camera()`) actually correct?**
  _`Camera` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `CaseMatch` (e.g. with `create_case_report()` and `get_case()`) actually correct?**
  _`CaseMatch` has 16 INFERRED edges - model-reasoned connections that need verification._