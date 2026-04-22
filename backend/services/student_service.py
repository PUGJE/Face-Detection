"""
Student Service — CRUD operations for student management.
"""


from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Dict, Any, Optional
import logging

from backend.models.student import Student
from backend.database.connection import db_manager

logger = logging.getLogger(__name__)


class StudentService:
    """
    Student Service for managing student operations
    """
    
    def __init__(self, session: Session = None):
        """Accept an optional DB session; creates one from db_manager if None."""
        self.session = session
        self._own_session = False
        if self.session is None:
            self.session = db_manager.get_session()
            self._own_session = True
    
    def __del__(self):
        """Cleanup session if we created it"""
        if self._own_session and self.session:
            self.session.close()
    
    def create_student(
        self,
        student_id: str,
        name: str,
        email: str = None,
        enrollment_number: str = None,
        department: str = None,
        year: str = None,
    ) -> Dict[str, Any]:
        """Create and persist a new student record. Returns success dict."""
        try:
            existing = self.session.query(Student).filter(
                Student.student_id == student_id
            ).first()
            if existing:
                return {
                    'success': False,
                    'error': f'Student with ID {student_id} already exists',
                    'student_id': student_id
                }

            student = Student(
                student_id=student_id,
                name=name,
                email=email,
                enrollment_number=enrollment_number,
                department=department,
                year=year,
                is_active=True
            )
            self.session.add(student)
            self.session.commit()
            logger.info(f"Student created: {student_id} - {name}")
            return {'success': True, 'student': student.to_dict()}
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating student: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Return a student dict by ID, or None if not found."""
        student = self.session.query(Student).filter(
            Student.student_id == student_id
        ).first()
        return student.to_dict() if student else None

    def get_all_students(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Return all students, optionally filtered to active-only."""
        query = self.session.query(Student)
        if active_only:
            query = query.filter(Student.is_active == True)
        return [s.to_dict() for s in query.all()]
    
    def update_student(self, student_id: str, **kwargs) -> Dict[str, Any]:
        """Update one or more fields on a student. Returns success dict."""
        try:
            student = self.session.query(Student).filter(
                Student.student_id == student_id
            ).first()
            if not student:
                return {'success': False, 'error': f'Student {student_id} not found'}
            for key, value in kwargs.items():
                if hasattr(student, key):
                    setattr(student, key, value)
            self.session.commit()
            logger.info(f"Student updated: {student_id}")
            return {'success': True, 'student': student.to_dict()}
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating student: {e}")
            return {'success': False, 'error': str(e)}
    
    def register_face(self, student_id: str, face_image_path: str = None) -> Dict[str, Any]:
        """Mark the student as face-registered and store the image path."""
        return self.update_student(
            student_id,
            face_registered=True,
            face_image_path=face_image_path
        )
    
    def delete_student(self, student_id: str, soft_delete: bool = True) -> Dict[str, Any]:
        """
        Delete a student (soft delete sets is_active=False; hard delete removes the record).
        """
        try:
            student = self.session.query(Student).filter(
                Student.student_id == student_id
            ).first()
            if not student:
                return {'success': False, 'error': f'Student {student_id} not found'}
            if soft_delete:
                student.is_active = False
                self.session.commit()
                logger.info(f"Student deactivated: {student_id}")
            else:
                self.session.delete(student)
                self.session.commit()
                logger.info(f"Student deleted: {student_id}")
            return {'success': True, 'student_id': student_id, 'deleted': not soft_delete}
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting student: {e}")
            return {'success': False, 'error': str(e)}
    
    def search_students(self, query: str) -> List[Dict[str, Any]]:
        """Search students by name, ID, or email."""
        return [
            s.to_dict()
            for s in self.session.query(Student).filter(
                or_(
                    Student.student_id.like(f'%{query}%'),
                    Student.name.like(f'%{query}%'),
                    Student.email.like(f'%{query}%'),
                )
            ).all()
        ]
