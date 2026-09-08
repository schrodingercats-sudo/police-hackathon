#!/usr/bin/env python3
"""
Unified CLI Management Tool for Indian Police Stolen Vehicle AI Command Center.

Usage:
  python run.py --server       # Start FastAPI server on port 8000
  python run.py --seed         # Initialize & seed database with Delhi NCR camera network and test case
  python run.py --simulate     # Run synthetic 6-camera video stream generator
  python run.py --test         # Execute full PyTest automated test suite
"""

import argparse
import logging
import os
import sys
import subprocess
import shutil
from pathlib import Path

# Add project root directory to Python path
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

# Add parent directory as well to allow both "backend.main" and "stolen_vehicle_ai.backend.main"
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))


BANNER = """
===============================================================================
     INDIAN POLICE STOLEN VEHICLE AI COMMAND CENTER & ANPR SYSTEM
          CCTNS-Compliant Multi-Camera Tracking & Route Reconstruction
===============================================================================
"""


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Starts the FastAPI application using Uvicorn."""
    print(BANNER)
    print(f"[*] Starting Police AI Command Center Backend on http://{host}:{port}")
    print("[*] Interactive API Documentation: http://localhost:8000/docs")
    print("[*] Web Command Dashboard:       http://localhost:8000/\n")

    try:
        import uvicorn
        uvicorn.run(
            "backend.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except ImportError:
        print("[!] Error: uvicorn is not installed. Please run: pip install -r requirements.txt")
        sys.exit(1)


def seed_database(reset: bool = False):
    """Initializes and seeds database with Delhi NCR cameras, vehicles, and test cases."""
    print(BANNER)
    print(f"[*] Initializing and seeding database (reset={reset})...")

    try:
        from backend.database.session import init_db, reset_db
        from backend.database.seed_data import seed_all

        if reset:
            reset_db()
        else:
            init_db()

        seed_all(reset=False)
        print("\n[+] Database seeded successfully with 6 Delhi NCR cameras, sample traffic, and FIR-2026-DEL-0941.\n")
    except Exception as e:
        print(f"[!] Error seeding database: {e}")
        logging.exception(e)
        sys.exit(1)


def run_simulation(cameras: int = 6, frames_per_camera: int = 10, save_video: bool = False, db_insert: bool = True):
    """Executes the multi-camera synthetic stream generator."""
    print(BANNER)
    print(f"[*] Running Synthetic Multi-Camera Stream Generator...")
    print(f"    - Surveillance Cameras : {cameras} nodes across Delhi NCR corridor")
    print(f"    - Frames per camera    : {frames_per_camera}")
    print(f"    - Save Video Feeds     : {save_video}")
    print(f"    - Ingest to Database   : {db_insert}\n")

    try:
        from backend.services.mock_stream import generate_synthetic_streams

        result = generate_synthetic_streams(
            cameras=cameras,
            frames_per_camera=frames_per_camera,
            save_video=save_video,
            insert_to_db=db_insert,
        )

        print("\n[+] Synthetic Multi-Camera Simulation Completed Successfully!")
        print(f"    Total Cameras Simulated : {result.get('total_cameras')}")
        print(f"    Total Vehicle Sightings : {result.get('total_sightings_generated')}")
        print(f"    Target Stolen Sightings : {result.get('target_sightings')}")
        print(f"    Distractor Sightings    : {result.get('distractor_sightings')}")
        print(f"    DB Records Populated    : {result.get('db_records_inserted')}\n")
    except Exception as e:
        print(f"[!] Error running simulation: {e}")
        logging.exception(e)
        sys.exit(1)


def run_tests(verbose: bool = True, extra_args: list = None):
    """Executes the full automated PyTest test suite."""
    print(BANNER)
    print("[*] Executing Automated PyTest Test Suite across all pillars...")

    tests_dir = _current_dir / "tests"
    pytest_bin = shutil.which("pytest")
    if pytest_bin:
        cmd = [pytest_bin, str(tests_dir)]
    else:
        cmd = [sys.executable, "-m", "pytest", str(tests_dir)]
    if verbose:
        cmd.append("-v")
    if extra_args:
        cmd.extend(extra_args)

    print(f"[*] Command: {' '.join(cmd)}\n")
    exit_code = subprocess.call(cmd, cwd=str(_parent_dir))

    if exit_code == 0:
        print("\n[+] 100% OF TESTS PASSED SUCCESSFULLY! ALL PIPELINE PILLARS VERIFIED.\n")
    else:
        print(f"\n[!] Test suite failed with exit code {exit_code}.\n")
    sys.exit(exit_code)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Unified CLI Manager for Indian Police Stolen Vehicle AI Command Center",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --server                     # Start API server on http://localhost:8000
  python run.py --server --port 8080 --reload
  python run.py --seed                       # Seed Delhi NCR cameras & sample FIR case
  python run.py --simulate --cameras 6       # Run synthetic CCTV video simulation
  python run.py --test                       # Run 100% PyTest test suite
        """,
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-s", "--server", action="store_true", help="Start FastAPI backend server")
    group.add_argument("--seed", action="store_true", help="Initialize and seed database")
    group.add_argument("--simulate", action="store_true", help="Run synthetic multi-camera stream generator")
    group.add_argument("-t", "--test", action="store_true", help="Execute automated test suite")

    # Server options
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address for API server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable live auto-reload for development")

    # Seed options
    parser.add_argument("--reset", action="store_true", help="Drop existing tables before seeding (WARNING: wipes data)")
    parser.add_argument("--no-reset", action="store_true", help="Do not drop existing tables before seeding")

    # Simulation options
    parser.add_argument("--cameras", type=int, default=6, help="Number of cameras to simulate (1 to 6, default: 6)")
    parser.add_argument("--frames", type=int, default=10, help="Frames per camera (min 1, default: 10)")
    parser.add_argument("--video", action="store_true", help="Save output video feeds (.mp4)")
    parser.add_argument("--no-db", action="store_true", help="Do not insert simulation records into database")

    args, unknown = parser.parse_known_args()

    if args.server:
        start_server(host=args.host, port=args.port, reload=args.reload)
    elif args.seed:
        should_reset = args.reset and not args.no_reset
        seed_database(reset=should_reset)
    elif args.simulate:
        if args.cameras < 1 or args.cameras > 6:
            parser.error("--cameras must be between 1 and 6.")
        if args.frames < 1:
            parser.error("--frames must be at least 1.")
        run_simulation(
            cameras=args.cameras,
            frames_per_camera=args.frames,
            save_video=args.video,
            db_insert=not args.no_db,
        )
    elif args.test:
        run_tests(verbose=True, extra_args=unknown)
    else:
        # Default: print banner and help
        print(BANNER)
        parser.print_help()


if __name__ == "__main__":
    main()
