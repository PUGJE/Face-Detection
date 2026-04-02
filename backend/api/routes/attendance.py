"""
Attendance & Statistics Routes

Endpoints:
  POST /api/attendance/mark         — Mark attendance from a full uploaded image
  POST /api/attendance/mark-crop    — Mark attendance from a browser-detected face crop
  GET  /api/attendance/today        — Today's attendance records
  GET  /api/attendance/date/{date}  — Records for a specific date (YYYY-MM-DD)
  GET  /api/attendance/student/{id} — Attendance history for one student
  GET  /api/attendance/absent       — Students absent on a given date

  GET  /api/statistics/overall          — Overall attendance statistics
  GET  /api/statistics/student/{id}     — Statistics for a specific student
"""

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
import time

# 10s cooldown cache for optimistic attendance latency reduction
_RECENT_SCANS = {}
SCAN_COOLDOWN_SECONDS = 10

from backend.database.connection import get_db
from backend.services.attendance_service import AttendanceService
from backend.ml.recognition_pipeline import embed_face_crop
from backend.api.dependencies import get_attendance_system, decode_image_upload

router = APIRouter(tags=["attendance"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attendance marking
# ---------------------------------------------------------------------------

@router.post("/api/attendance/mark")
async def mark_attendance_full_pipeline(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Mark attendance from a full uploaded photo.

    Runs the complete server-side pipeline:
    RetinaFace detection → anti-spoofing → ArcFace recognition → DB write.
    """
    try:
        image = await decode_image_upload(file)
        system = get_attendance_system()
        result = system.mark_attendance_from_image(image)

        if not result["success"]:
            if result.get("duplicate"):
                return {
                    "success": False,
                    "duplicate": True,
                    "message": "Attendance already marked today",
                    "data": result,
                }
            raise HTTPException(status_code=400, detail=result.get("error"))

        return {"success": True, "message": "Attendance marked successfully", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking attendance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/attendance/mark-crop")
async def mark_attendance_browser_crop(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Mark attendance from a browser-detected face crop (~5 ms server-side).
    Uses optimistic caching and BackgroundTasks to eliminate database latency!
    """
    try:
        global _RECENT_SCANS
        now = time.time()
        
        # Cleanup expired cache loosely
        _RECENT_SCANS = {k: v for k, v in _RECENT_SCANS.items() if now - v < SCAN_COOLDOWN_SECONDS * 2}

        crop = await decode_image_upload(file)

        embedding = embed_face_crop(crop)
        if embedding is None:
            return {"success": False, "error": "Could not extract embedding from crop"}

        system = get_attendance_system()
        match = system.face_pipeline.recognizer.recognize_from_embedding(embedding)
        if match is None:
            return {"success": False, "error": "Face not recognised"}

        student_id = match["student_id"]

        # Cache check to prevent rapid-fire requests
        last_scan = _RECENT_SCANS.get(student_id, 0)
        if now - last_scan < SCAN_COOLDOWN_SECONDS:
            return {
                "success": False,
                "duplicate": True,
                "message": "Recently marked (optimistic buffer)"
            }

        _RECENT_SCANS[student_id] = now

        # We need the user's name for the frontend UI. 1 fast SQL lookup here.
        from backend.models.student import Student
        student = db.query(Student).filter(Student.student_id == student_id).first()
        student_name = student.name if student else student_id

        def _bg_mark_attendance(sid, conf, dist):
            from backend.database.connection import db_manager
            try:
                with db_manager.session_scope() as session:
                    AttendanceService(session).mark_attendance(
                        student_id=sid,
                        recognition_confidence=conf,
                        recognition_distance=dist,
                    )
            except Exception as e:
                logger.error(f"Background DB write failed for {sid}: {e}")

        # Dispatch background task so the frontend unblocks IMMEDIATELY
        background_tasks.add_task(
            _bg_mark_attendance,
            student_id,
            match["confidence"],
            match["distance"]
        )

        return {
            "success": True,
            "message": "Attendance queued successfully",
            "data": {
                "student_id": student_id,
                "student_name": student_name,
                "status": "present", # Optimistic status
                "recognition_confidence": match["confidence"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in mark-crop: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Attendance queries
# ---------------------------------------------------------------------------

@router.get("/api/attendance/today")
async def get_today_attendance(db: Session = Depends(get_db)):
    """Return all attendance records for today."""
    try:
        records = AttendanceService(db).get_attendance_by_date()
        return {
            "success": True,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "count": len(records),
            "data": records,
        }
    except Exception as e:
        logger.error(f"Error fetching today's attendance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/attendance/date/{date}")
async def get_attendance_by_date(date: str, db: Session = Depends(get_db)):
    """Return all attendance records for a specific date (YYYY-MM-DD)."""
    try:
        records = AttendanceService(db).get_attendance_by_date(date)
        return {"success": True, "date": date, "count": len(records), "data": records}
    except Exception as e:
        logger.error(f"Error fetching attendance for {date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/attendance/student/{student_id}")
async def get_student_attendance_history(
    student_id: str,
    days: int = 30,
    db: Session = Depends(get_db),
):
    """Return attendance history for a student over the past N days (default 30)."""
    try:
        records = AttendanceService(db).get_student_attendance_history(student_id, days)
        return {
            "success": True,
            "student_id": student_id,
            "days": days,
            "count": len(records),
            "data": records,
        }
    except Exception as e:
        logger.error(f"Error fetching history for {student_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/attendance/absent")
async def get_absent_students(
    date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Return the list of students who are absent on a given date (today if omitted)."""
    try:
        absent = AttendanceService(db).get_absent_students(date)
        return {
            "success": True,
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "count": len(absent),
            "data": absent,
        }
    except Exception as e:
        logger.error(f"Error fetching absent students: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@router.get("/api/statistics/overall")
async def get_overall_statistics(db: Session = Depends(get_db)):
    """Return overall attendance statistics across all students."""
    try:
        stats = AttendanceService(db).get_attendance_statistics()
        return {"success": True, "data": stats}
    except Exception as e:
        logger.error(f"Error fetching overall statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/statistics/student/{student_id}")
async def get_student_statistics(student_id: str, db: Session = Depends(get_db)):
    """Return attendance statistics for a specific student."""
    try:
        stats = AttendanceService(db).get_attendance_statistics(student_id)
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        return {"success": True, "data": stats}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching statistics for {student_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
