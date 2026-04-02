"""
Shared FastAPI Dependencies

This module contains reusable dependencies and helper functions
injected into route handlers via FastAPI's Depends() system.

- decode_image_upload: Decodes a raw UploadFile into an OpenCV BGR image.
- get_attendance_system: Returns the singleton AttendanceSystem instance.
"""

import cv2
import numpy as np
import logging
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import to avoid a circular-import cycle at startup
# (AttendanceSystem → recognition_pipeline → config, all of which import
#  from backend.* which is already loading when this module is first used.)
# ---------------------------------------------------------------------------
_attendance_system = None


def get_attendance_system():
    """
    Return the process-wide AttendanceSystem singleton.

    Creates it on first call so that the heavy ML models are loaded
    only once, not on every request.
    """
    global _attendance_system
    if _attendance_system is None:
        from backend.attendance_system import AttendanceSystem
        _attendance_system = AttendanceSystem()
        logger.info("AttendanceSystem singleton created.")
    return _attendance_system


async def decode_image_upload(file: UploadFile) -> np.ndarray:
    """
    Read an uploaded image file and decode it into an OpenCV BGR ndarray.

    Raises:
        HTTPException(400): If the file cannot be decoded as a valid image.
    """
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file.")
    return image
