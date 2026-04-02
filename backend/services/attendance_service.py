"""
Attendance Service

Handles all attendance business logic: marking attendance, querying
records, generating reports, and determining late/present status.

All methods require a live SQLAlchemy session passed at construction time.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.connection import db_manager
from backend.models.student import Attendance, Student

logger = logging.getLogger(__name__)


def _is_late(now: datetime) -> bool:
    """
    Return True if ``now`` is after the configured late-arrival threshold.

    The threshold is read from ``settings.late_threshold_time`` (HH:MM).
    Defaults to 09:30 if the setting cannot be parsed.
    """
    try:
        threshold = datetime.strptime(settings.late_threshold_time, "%H:%M").time()
    except ValueError:
        logger.warning(
            f"Could not parse late_threshold_time '{settings.late_threshold_time}'. "
            "Falling back to 09:30."
        )
        threshold = datetime.strptime("09:30", "%H:%M").time()
    return now.time() > threshold


class AttendanceService:
    """
    Service layer for attendance operations.

    Accepts an optional SQLAlchemy session; creates one from the global
    db_manager if none is provided (useful for background tasks / CLI use).
    """

    def __init__(self, session: Session = None) -> None:
        self.session = session
        self._own_session = session is None
        if self._own_session:
            self.session = db_manager.get_session()

    def __del__(self) -> None:
        if self._own_session and self.session:
            self.session.close()

    # ------------------------------------------------------------------
    # Marking attendance
    # ------------------------------------------------------------------

    def mark_attendance(
        self,
        student_id: str,
        recognition_confidence: float = None,
        recognition_distance: float = None,
        detection_confidence: float = None,
        notes: str = None,
    ) -> Dict[str, Any]:
        """
        Write an attendance record for ``student_id``.

        Guards:
        - Student must exist and be active.
        - Only one record per student per calendar day (duplicate check).
        - Status is "late" if the arrival time exceeds ``late_threshold_time``.

        Returns:
            Dict with ``success`` bool plus record details on success,
            or ``success=False`` with an ``error`` / ``duplicate=True`` on failure.
        """
        try:
            student = (
                self.session.query(Student)
                .filter(Student.student_id == student_id)
                .first()
            )

            if not student:
                return {"success": False, "error": f"Student '{student_id}' not found.", "student_id": student_id}

            if not student.is_active:
                return {"success": False, "error": f"Student '{student_id}' is inactive.", "student_id": student_id}

            now = datetime.now()
            today_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            # Duplicate check: one record per student per day
            existing = (
                self.session.query(Attendance)
                .filter(
                    and_(
                        Attendance.student_id == student.id,
                        Attendance.date == today_str,
                    )
                )
                .first()
            )
            if existing:
                return {
                    "success": False,
                    "duplicate": True,
                    "error": "Attendance already marked today.",
                    "student_id": student_id,
                    "student_name": student.name,
                    "existing_record": existing.to_dict(),
                }

            status = "late" if _is_late(now) else "present"

            record = Attendance(
                student_id=student.id,
                timestamp=now,
                date=today_str,
                time=time_str,
                recognition_confidence=recognition_confidence,
                recognition_distance=recognition_distance,
                detection_confidence=detection_confidence,
                status=status,
                marked_by="system",
                notes=notes,
            )
            self.session.add(record)
            self.session.commit()

            logger.info(f"Attendance marked — {student_id} ({student.name}) [{status}]")

            return {
                "success": True,
                "student_id": student_id,
                "student_name": student.name,
                "date": today_str,
                "time": time_str,
                "status": status,
                "recognition_confidence": recognition_confidence,
                "attendance_id": record.id,
            }

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error marking attendance for '{student_id}': {e}")
            return {"success": False, "error": str(e), "student_id": student_id}

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_attendance_by_date(self, target_date: str = None) -> List[Dict[str, Any]]:
        """
        Return all attendance records for a given date.

        Args:
            target_date: Date string in YYYY-MM-DD format. Defaults to today.
        """
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")

        records = (
            self.session.query(Attendance)
            .filter(Attendance.date == target_date)
            .all()
        )
        return [r.to_dict() for r in records]

    def get_student_attendance_history(
        self, student_id: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Return attendance records for one student over the past ``days`` days.

        Returns an empty list if the student is not found.
        """
        student = (
            self.session.query(Student)
            .filter(Student.student_id == student_id)
            .first()
        )
        if not student:
            return []

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        records = (
            self.session.query(Attendance)
            .filter(
                and_(
                    Attendance.student_id == student.id,
                    Attendance.timestamp >= start_date,
                    Attendance.timestamp <= end_date,
                )
            )
            .order_by(Attendance.timestamp.desc())
            .all()
        )
        return [r.to_dict() for r in records]

    def get_absent_students(self, target_date: str = None) -> List[Dict[str, Any]]:
        """
        Return active students who have no attendance record on ``target_date``.

        Args:
            target_date: Date string in YYYY-MM-DD format. Defaults to today.
        """
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")

        all_active = (
            self.session.query(Student).filter(Student.is_active.is_(True)).all()
        )

        # IDs (internal) of students who already have a record today
        present_internal_ids = {
            row[0]
            for row in self.session.query(Attendance.student_id).filter(
                Attendance.date == target_date
            )
        }

        return [
            s.to_dict()
            for s in all_active
            if s.id not in present_internal_ids
        ]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_attendance_statistics(
        self, student_id: str = None
    ) -> Dict[str, Any]:
        """
        Return attendance statistics.

        If ``student_id`` is provided, returns per-student stats.
        Otherwise returns overall stats across all active students.
        """
        if student_id:
            return self._student_statistics(student_id)
        return self._overall_statistics()

    def _student_statistics(self, student_id: str) -> Dict[str, Any]:
        student = (
            self.session.query(Student)
            .filter(Student.student_id == student_id)
            .first()
        )
        if not student:
            return {"error": "Student not found"}

        total = (
            self.session.query(Attendance)
            .filter(Attendance.student_id == student.id)
            .count()
        )
        present = (
            self.session.query(Attendance)
            .filter(
                and_(
                    Attendance.student_id == student.id,
                    Attendance.status == "present",
                )
            )
            .count()
        )
        late = (
            self.session.query(Attendance)
            .filter(
                and_(
                    Attendance.student_id == student.id,
                    Attendance.status == "late",
                )
            )
            .count()
        )

        return {
            "student_id": student_id,
            "student_name": student.name,
            "total_days": total,
            "present_days": present,
            "late_days": late,
            "attendance_percentage": round((present / total * 100), 1) if total else 0.0,
        }

    def _overall_statistics(self) -> Dict[str, Any]:
        total_students = (
            self.session.query(Student).filter(Student.is_active.is_(True)).count()
        )
        total_records = self.session.query(Attendance).count()
        today_str = datetime.now().strftime("%Y-%m-%d")
        today_records = (
            self.session.query(Attendance)
            .filter(Attendance.date == today_str)
            .count()
        )

        return {
            "total_students": total_students,
            "total_attendance_records": total_records,
            "today_attendance": today_records,
            "today_percentage": round(
                (today_records / total_students * 100), 1
            ) if total_students else 0.0,
        }


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from backend.database.connection import init_database

    print("=" * 60)
    print("ATTENDANCE SERVICE TEST")
    print("=" * 60)
    init_database()
    service = AttendanceService()
    stats = service.get_attendance_statistics()
    print(f"\nTotal students:  {stats['total_students']}")
    print(f"Total records:   {stats['total_attendance_records']}")
    print(f"Today's count:   {stats['today_attendance']}")
    print("\n✓ Test complete.")
