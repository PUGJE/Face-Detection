"""
Attendance & Statistics Routes

Endpoints:
  POST /api/attendance/mark         — Mark attendance from a full uploaded image
  POST /api/attendance/mark-crop    — Mark attendance from a browser-detected face crop
  GET  /api/attendance/today        — Today's attendance records
  GET  /api/attendance/date/{date}  — Records for a specific date (YYYY-MM-DD)
  GET  /api/attendance/student/{id} — Attendance history for one student
  GET  /api/attendance/absent       — Students absent on a given date
  GET  /api/attendance/summary      — Timetable attendance summary (admin)
  GET  /api/attendance/report/matrix— Student × Subject attendance matrix

  GET  /api/statistics/overall          — Overall attendance statistics
  GET  /api/statistics/student/{id}     — Statistics for a specific student

  GET    /api/timetable              — List all timetable slots
  POST   /api/timetable              — Create a new timetable slot
  GET    /api/timetable/active       — Return the currently active slot (by server time)
  PUT    /api/timetable/{id}         — Update a timetable slot
  DELETE /api/timetable/{id}         — Delete a timetable slot
"""

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
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
# Pydantic schemas
# ---------------------------------------------------------------------------

class TimetableCreate(BaseModel):
    subject_name: str
    day_of_week: str      # e.g. "Monday"
    start_time: str       # "HH:MM" or "HH:MM:SS"
    end_time: str
    teacher_id: Optional[str] = None


class TimetableUpdate(BaseModel):
    subject_name: Optional[str] = None
    day_of_week: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    teacher_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_time(t: str):
    """Accept HH:MM or HH:MM:SS and return a time object."""
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(t, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse time '{t}'. Use HH:MM or HH:MM:SS.")


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
                "status": "present",  # Optimistic status
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


@router.get("/api/attendance/summary")
async def get_timetable_attendance_summary(db: Session = Depends(get_db)):
    """Admin dashboard stats showing attendance % grouped by Timetable subjects."""
    try:
        stats = AttendanceService(db).get_timetable_summary_statistics()
        return {"success": True, "data": stats}
    except Exception as e:
        logger.error(f"Error fetching timetable summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/attendance/report/matrix")
async def get_student_subject_matrix(db: Session = Depends(get_db)):
    """
    Return a 2D attendance matrix: each student's count per timetable subject.
    Used by the Admin Dashboard to render the per-student × per-subject table.
    """
    try:
        matrix = AttendanceService(db).get_student_subject_matrix()
        return {"success": True, "data": matrix}
    except Exception as e:
        logger.error(f"Error fetching student-subject matrix: {e}")
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


# ---------------------------------------------------------------------------
# Timetable CRUD
# ---------------------------------------------------------------------------

@router.get("/api/timetable")
async def list_timetables(db: Session = Depends(get_db)):
    """Return all timetable slots ordered by day then start time."""
    try:
        from backend.models.student import Timetable
        from sqlalchemy import asc
        slots = db.query(Timetable).order_by(asc(Timetable.day_of_week), asc(Timetable.start_time)).all()
        return {"success": True, "count": len(slots), "data": [s.to_dict() for s in slots]}
    except Exception as e:
        logger.error(f"Error listing timetables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/timetable/active")
async def get_active_timetable(db: Session = Depends(get_db)):
    """
    Return the timetable slot that is currently active based on server time.
    Compares using HH:MM string format to avoid SQLAlchemy Time-type edge cases.
    """
    try:
        from backend.models.student import Timetable

        now          = datetime.now()
        day_name     = now.strftime("%A")   # "Monday", "Tuesday", …
        current_hhmm = now.strftime("%H:%M") # e.g. "14:30"

        # Fetch all slots for today and filter in Python using string comparison.
        # This avoids any SQLAlchemy / Postgres driver quirks with Time columns.
        today_slots = (
            db.query(Timetable)
            .filter(Timetable.day_of_week == day_name)
            .all()
        )

        slot = None
        for s in today_slots:
            start_hhmm = s.start_time.strftime("%H:%M") if s.start_time else "00:00"
            end_hhmm   = s.end_time.strftime("%H:%M")   if s.end_time   else "23:59"
            if start_hhmm <= current_hhmm <= end_hhmm:
                slot = s
                break

        logger.info(
            f"Active class check: day={day_name} time={current_hhmm} "
            f"slots_today={len(today_slots)} active={'yes' if slot else 'no'}"
        )

        return {
            "success": True,
            "active": slot is not None,
            "data": slot.to_dict() if slot else None,
            "server_time": current_hhmm,
            "day": day_name,
        }
    except Exception as e:
        logger.error(f"Error fetching active timetable: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/timetable")
async def create_timetable(data: TimetableCreate, db: Session = Depends(get_db)):
    """Create a new timetable slot."""
    try:
        from backend.models.student import Timetable
        t = Timetable(
            subject_name=data.subject_name,
            day_of_week=data.day_of_week,
            start_time=_parse_time(data.start_time),
            end_time=_parse_time(data.end_time),
            teacher_id=data.teacher_id,
        )
        db.add(t)
        db.commit()
        db.refresh(t)
        return {"success": True, "data": t.to_dict()}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Error creating timetable: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/timetable/{timetable_id}")
async def update_timetable(
    timetable_id: int,
    data: TimetableUpdate,
    db: Session = Depends(get_db),
):
    """Update an existing timetable slot (partial updates supported)."""
    try:
        from backend.models.student import Timetable
        slot = db.query(Timetable).filter(Timetable.id == timetable_id).first()
        if not slot:
            raise HTTPException(status_code=404, detail=f"Timetable {timetable_id} not found")

        if data.subject_name is not None:
            slot.subject_name = data.subject_name
        if data.day_of_week is not None:
            slot.day_of_week = data.day_of_week
        if data.start_time is not None:
            slot.start_time = _parse_time(data.start_time)
        if data.end_time is not None:
            slot.end_time = _parse_time(data.end_time)
        if data.teacher_id is not None:
            slot.teacher_id = data.teacher_id

        db.commit()
        db.refresh(slot)
        return {"success": True, "data": slot.to_dict()}
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Error updating timetable {timetable_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/timetable/{timetable_id}")
async def delete_timetable(timetable_id: int, db: Session = Depends(get_db)):
    """
    Delete a timetable slot.
    WARNING: associated attendance records will have their timetable_id set to NULL.
    """
    try:
        from backend.models.student import Timetable, Attendance

        slot = db.query(Timetable).filter(Timetable.id == timetable_id).first()
        if not slot:
            raise HTTPException(status_code=404, detail=f"Timetable {timetable_id} not found")

        # Null-out FK references so attendance history is preserved
        db.query(Attendance).filter(Attendance.timetable_id == timetable_id).update(
            {"timetable_id": None}, synchronize_session="fetch"
        )
        db.delete(slot)
        db.commit()
        return {"success": True, "message": f"Timetable slot {timetable_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting timetable {timetable_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
