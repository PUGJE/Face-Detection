"""
Student Management Routes

All CRUD operations for students and their face registrations.

Endpoints:
  POST   /api/students                               — Create a student
  GET    /api/students                               — List all students
  GET    /api/students/{student_id}                  — Get one student
  PUT    /api/students/{student_id}                  — Update a student
  DELETE /api/students/{student_id}                  — Delete (soft by default)
  GET    /api/students/search/{query}                — Search students
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.services.student_service import StudentService

router = APIRouter(prefix="/api/students", tags=["students"])
logger = logging.getLogger(__name__)


@router.post("")
async def create_student(
    student_id: str,
    name: str,
    email: Optional[str] = None,
    enrollment_number: Optional[str] = None,
    department: Optional[str] = None,
    year: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Create a new student record."""
    try:
        result = StudentService(db).create_student(
            student_id=student_id,
            name=name,
            email=email,
            enrollment_number=enrollment_number,
            department=department,
            year=year,
        )
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return {"success": True, "message": "Student created successfully", "data": result["student"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating student: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def get_all_students(active_only: bool = True, db: Session = Depends(get_db)):
    """Return a list of all students (active only by default)."""
    try:
        students = StudentService(db).get_all_students(active_only=active_only)
        return {"success": True, "count": len(students), "data": students}
    except Exception as e:
        logger.error(f"Error listing students: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/{query}")
async def search_students(query: str, db: Session = Depends(get_db)):
    """Search students by name, ID, or email."""
    try:
        students = StudentService(db).search_students(query)
        return {"success": True, "query": query, "count": len(students), "data": students}
    except Exception as e:
        logger.error(f"Error searching students: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{student_id}")
async def get_student(student_id: str, db: Session = Depends(get_db)):
    """Return a single student by their unique ID."""
    try:
        student = StudentService(db).get_student(student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        return {"success": True, "data": student}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching student: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{student_id}")
async def update_student(
    student_id: str,
    name: Optional[str] = None,
    email: Optional[str] = None,
    department: Optional[str] = None,
    year: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db),
):
    """Update one or more fields on an existing student."""
    try:
        # Build a dict of only the fields that were explicitly provided
        updates = {
            key: val
            for key, val in {
                "name": name,
                "email": email,
                "department": department,
                "year": year,
                "is_active": is_active,
            }.items()
            if val is not None
        }
        result = StudentService(db).update_student(student_id, **updates)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return {"success": True, "message": "Student updated successfully", "data": result["student"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating student: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{student_id}")
async def delete_student(
    student_id: str,
    hard_delete: bool = False,
    db: Session = Depends(get_db),
):
    """
    Delete a student.

    By default performs a *soft delete* (sets is_active=False).
    Pass ?hard_delete=true to permanently remove the record.
    """
    try:
        result = StudentService(db).delete_student(student_id, soft_delete=not hard_delete)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return {"success": True, "message": "Student deleted successfully", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting student: {e}")
        raise HTTPException(status_code=500, detail=str(e))
