"""
AttendanceSystem — High-level Facade

Combines face recognition and attendance persistence into a single object,
used as a process-wide singleton (see backend/api/dependencies.py).

The HTTP API creates per-request database sessions via FastAPI's Depends()
mechanism. This class does NOT hold a long-lived session of its own;
it only owns the face-recognition pipeline (which is stateful / in-memory).

Direct DB operations inside methods here use short-lived sessions via
`db_manager.session_scope()`.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from backend.config import settings
from backend.database.connection import db_manager, init_database
from backend.ml.recognition_pipeline import FaceRecognitionPipeline

logger = logging.getLogger(__name__)


class AttendanceSystem:
    """
    Process-wide singleton that owns the face-recognition pipeline.

    Responsibilities:
    - Load and persist ArcFace embeddings.
    - Expose high-level `register_new_student` and `mark_attendance_from_image`
      methods for callers that need the full pipeline in a single call.
    - Provide system statistics.

    Database writes are performed through short-lived sessions so this object
    never holds an open SQLAlchemy session.
    """

    def __init__(self) -> None:
        init_database()
        self.face_pipeline = FaceRecognitionPipeline()
        self.face_pipeline.load_database()
        logger.info("AttendanceSystem initialised.")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_new_student(
        self,
        student_id: str,
        name: str,
        image: np.ndarray,
        email: str = None,
        enrollment_number: str = None,
        department: str = None,
        year: str = None,
    ) -> Dict[str, Any]:
        """
        Register a new student with a face image in one call.

        Creates the DB record, embeds the face, saves the image to disk,
        and persists the embedding database.  Rolls back the DB record if
        face registration fails.

        Args:
            student_id: Unique student identifier.
            name: Student's full name.
            image: BGR image containing the student's face.
            email, enrollment_number, department, year: Optional metadata.

        Returns:
            Dict with ``success`` bool and relevant data.
        """
        # Import here to avoid circular imports at module load time
        from backend.services.student_service import StudentService

        with db_manager.session_scope() as session:
            service = StudentService(session)
            db_result = service.create_student(
                student_id=student_id,
                name=name,
                email=email,
                enrollment_number=enrollment_number,
                department=department,
                year=year,
            )

        if not db_result["success"]:
            return db_result

        # Embed face
        face_result = self.face_pipeline.register_student(student_id, image)
        if not face_result["success"]:
            # Roll back the DB record so state stays consistent
            with db_manager.session_scope() as session:
                StudentService(session).delete_student(student_id, soft_delete=False)
            return face_result

        # Persist the face image to disk
        face_dir = Path(settings.student_images_path) / student_id
        face_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(face_dir / "face.jpg"), image)

        # Mark the student as face-registered
        with db_manager.session_scope() as session:
            StudentService(session).register_face(
                student_id, str(face_dir / "face.jpg")
            )

        self.face_pipeline.save_database()
        logger.info(f"New student registered: {student_id} — {name}")

        return {
            "success": True,
            "student_id": student_id,
            "name": name,
            "face_registered": True,
            "face_confidence": face_result.get("face_confidence"),
        }

    # ------------------------------------------------------------------
    # Attendance marking
    # ------------------------------------------------------------------

    def mark_attendance_from_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run the full server-side attendance pipeline on a single image.

        Steps:
        1. Detect the face (RetinaFace via InsightFace).
        2. Run anti-spoofing / liveness check.
        3. Identify the student (ArcFace + FAISS).
        4. Write the attendance record to the database.

        Args:
            image: BGR image containing the student's face.

        Returns:
            Dict with ``success`` bool and attendance details.
        """
        from backend.services.attendance_service import AttendanceService

        recognition = self.face_pipeline.recognize_with_liveness(image)

        if not recognition["success"]:
            return recognition

        if not recognition.get("is_live", True):
            return {
                "success": False,
                "error": "Liveness check failed — possible spoofing attack detected.",
                "recognized": False,
                "is_live": False,
                "liveness_score": recognition.get("liveness_score", 0),
            }

        if not recognition["recognized"]:
            return {
                "success": False,
                "error": "Student not recognised.",
                "recognized": False,
                "is_live": recognition.get("is_live", True),
            }

        with db_manager.session_scope() as session:
            att_result = AttendanceService(session).mark_attendance(
                student_id=recognition["student_id"],
                recognition_confidence=recognition.get("recognition_confidence"),
                recognition_distance=recognition.get("recognition_distance"),
                detection_confidence=recognition.get("detection_confidence"),
            )

        if att_result["success"]:
            att_result["is_live"] = recognition.get("is_live", True)
            att_result["liveness_score"] = recognition.get("liveness_score", 1.0)

        return att_result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_system_stats(self) -> Dict[str, Any]:
        """Return combined face-recognition and attendance statistics."""
        from backend.services.attendance_service import AttendanceService

        face_stats = self.face_pipeline.get_stats()
        with db_manager.session_scope() as session:
            attendance_stats = AttendanceService(session).get_attendance_statistics()

        return {"face_recognition": face_stats, "attendance": attendance_stats}


# ---------------------------------------------------------------------------
# NOTE: Optional webcam-based attendance scanning (not used by the HTTP API).
#
# If you want batch attendance via webcam from the command line, instantiate
# AttendanceSystem() directly and call process_webcam_attendance().
# ---------------------------------------------------------------------------

def run_webcam_attendance(duration_seconds: int = 60) -> List[Dict[str, Any]]:
    """
    Standalone webcam attendance loop — NOT used by the HTTP API.

    Reads frames from the default camera, identifies faces, and writes
    attendance records until `duration_seconds` have elapsed.

    Args:
        duration_seconds: How long to keep the camera open.

    Returns:
        List of attendance result dicts for every student identified.
    """
    import time

    system = AttendanceSystem()
    from backend.services.attendance_service import AttendanceService

    results: List[Dict[str, Any]] = []
    processed: set = set()

    cap = cv2.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera.")
        return results

    start = time.time()
    try:
        while (time.time() - start) < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                continue

            for rec in system.face_pipeline.process_attendance_frame(frame):
                if not rec["recognized"] or rec["student_id"] in processed:
                    continue

                with db_manager.session_scope() as session:
                    att = AttendanceService(session).mark_attendance(
                        student_id=rec["student_id"],
                        recognition_confidence=rec.get("confidence"),
                        recognition_distance=rec.get("distance"),
                    )

                if att["success"]:
                    results.append(att)
                    processed.add(rec["student_id"])
                    logger.info(f"Attendance marked: {rec['student_id']}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return results


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ATTENDANCE SYSTEM TEST")
    print("=" * 60)

    system = AttendanceSystem()
    stats = system.get_system_stats()

    print(f"\nRegistered faces:   {stats['face_recognition']['recognizer']['total_faces']}")
    print(f"Total DB students:  {stats['attendance']['total_students']}")
    print(f"Today's attendance: {stats['attendance']['today_attendance']}")
    print("\n✓ Test complete.")
