"""
Health & Info Routes

Endpoints:
  GET /           — API metadata
  GET /health     — Liveness probe
  GET /app        — Serve the compiled Next.js frontend
  GET /api/stats  — Overall system statistics
"""

import logging
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.services.student_service import StudentService
from backend.api.dependencies import get_attendance_system

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
async def root():
    """Return basic API metadata."""
    return {
        "name": "Face Recognition Attendance System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "students": "/api/students",
            "attendance": "/api/attendance",
            "statistics": "/api/statistics",
        },
    }


@router.get("/health")
async def health_check():
    """Liveness probe — returns 200 when the server is up."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected",
        "face_recognition": "loaded",
    }


@router.get("/app", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the compiled Next.js frontend (production build only)."""
    index = Path(__file__).parents[3] / "frontend" / "out" / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse(
        "<h1>Frontend not found</h1>"
        "<p>Run <code>npm run build</code> inside <code>frontend/</code>, "
        "or use Next.js dev server on port 3000.</p>",
        status_code=404,
    )


@router.get("/api/stats")
async def get_system_stats(db: Session = Depends(get_db)):
    """
    Return overall system statistics.

    Merges live DB student count with face-recognition engine stats so the
    dashboard always reflects the current state.
    """
    try:
        system = get_attendance_system()
        stats = system.get_system_stats()

        # Patch recognizer.total_faces from the live DB to stay in sync
        live_students = StudentService(db).get_all_students(active_only=True)
        recognizer_stats = stats.get("face_recognition", {}).get("recognizer", {})
        recognizer_stats["total_faces"] = len(live_students)
        recognizer_stats.setdefault("model_name", "HOG+Cosine")

        return {"success": True, "data": {**stats, "recognizer": recognizer_stats}}
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
