"""
Face Registration Routes

Endpoints for registering face embeddings for students.

Two registration paths are supported:

1. Server-side (full pipeline):
   POST /api/students/{student_id}/register-face
   - Uploads a full photo; the server runs RetinaFace detection + ArcFace.

2. Browser-crop (fast path):
   POST /api/students/{student_id}/register-face-crop
   - The browser (MediaPipe) already cropped the face; the server runs ArcFace only.
   - Saves ~60-80% inference time vs. the full-pipeline path.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.services.student_service import StudentService
from backend.ml.recognition_pipeline import embed_face_crop
from backend.api.dependencies import get_attendance_system, decode_image_upload

router = APIRouter(prefix="/api/students", tags=["faces"])
logger = logging.getLogger(__name__)


@router.post("/{student_id}/register-face")
async def register_face_full_pipeline(
    student_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Register a student's face using a full uploaded photo.

    The server runs RetinaFace to locate the face, then ArcFace to embed it.
    Use this when the client cannot do client-side detection.
    """
    try:
        image = await decode_image_upload(file)

        service = StudentService(db)
        if not service.get_student(student_id):
            raise HTTPException(status_code=404, detail="Student not found")

        system = get_attendance_system()
        result = system.face_pipeline.register_student(student_id, image)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])

        service.register_face(student_id)
        system.face_pipeline.save_database()

        return {
            "success": True,
            "message": "Face registered successfully",
            "data": {
                "student_id": student_id,
                "face_confidence": result.get("face_confidence"),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in register-face: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{student_id}/register-face-crop")
async def register_face_browser_crop(
    student_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Register a student's face from a browser-detected face crop.

    The frontend (MediaPipe) detects and crops the face before upload,
    so the server only runs ArcFace — skipping the heavier RetinaFace step.
    """
    try:
        crop = await decode_image_upload(file)

        service = StudentService(db)
        if not service.get_student(student_id):
            raise HTTPException(status_code=404, detail="Student not found")

        embedding = embed_face_crop(crop)
        if embedding is None:
            raise HTTPException(
                status_code=400,
                detail="ArcFace embedding failed — check crop quality or image size.",
            )

        system = get_attendance_system()
        if not system.face_pipeline.recognizer.add_embedding(student_id, embedding):
            raise HTTPException(status_code=500, detail="Failed to store embedding.")

        service.register_face(student_id)
        system.face_pipeline.save_database()

        return {
            "success": True,
            "message": "Face registered (browser-crop path)",
            "student_id": student_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in register-face-crop: {e}")
        raise HTTPException(status_code=500, detail=str(e))
