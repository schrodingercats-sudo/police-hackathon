"""
Vercel Serverless ASGI Application Entrypoint.
Exposes the FastAPI app for Vercel's Python runtime.
Handles /tmp SQLite persistence for read-write operations in serverless execution environments.
"""

import os
import sys
import shutil
from pathlib import Path

# Ensure project root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Handle Vercel Serverless read-only filesystem by cloning SQLite database to /tmp
if os.environ.get("VERCEL"):
    tmp_db = Path("/tmp/stolen_vehicle_ai.db")
    src_db = ROOT_DIR / "stolen_vehicle_ai.db"
    
    if not tmp_db.exists() and src_db.exists():
        try:
            shutil.copy2(src_db, tmp_db)
        except Exception as e:
            print(f"Warning: Failed to copy database to /tmp: {e}")

    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_db}"

    # Ensure evidence directories exist in /tmp
    tmp_evidence = Path("/tmp/data/evidence")
    for subdir in ["crops", "frames", "plates"]:
        (tmp_evidence / subdir).mkdir(parents=True, exist_ok=True)
    src_evidence = ROOT_DIR / "data" / "evidence"
    if src_evidence.exists():
        try:
            for subdir in ["crops", "frames", "plates"]:
                src_sub = src_evidence / subdir
                if src_sub.exists():
                    shutil.copytree(src_sub, tmp_evidence / subdir, dirs_exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to copy evidence files: {e}")

from backend.main import app as base_app

class VercelPathFixMiddleware:
    """
    ASGI middleware that restores original incoming request path
    when Vercel rewrites or internal routing alters scope['path'].
    """
    def __init__(self, inner_app):
        self.inner_app = inner_app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            headers = dict(scope.get("headers", []))
            matched_path = headers.get(b"x-matched-path", b"").decode("utf-8", errors="ignore")
            forwarded_uri = headers.get(b"x-forwarded-uri", b"").decode("utf-8", errors="ignore")
            invoke_path = headers.get(b"x-invoke-path", b"").decode("utf-8", errors="ignore")

            real_path = matched_path or forwarded_uri or invoke_path
            if real_path and not real_path.startswith("/api/index"):
                clean_path = real_path.split("?")[0]
                scope["path"] = clean_path
                scope["raw_path"] = clean_path.encode("utf-8")
        await self.inner_app(scope, receive, send)

app = VercelPathFixMiddleware(base_app)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)
