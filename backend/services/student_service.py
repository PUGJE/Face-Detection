"""
Student Service

This module handles student management operations.

Features:
- Register new students
- Update student information
- Link face recognition data
- Query student records

Author: Face Recognition Team
Date: January 2026
"""

from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from backend.models.student import Student
from backend.database.connection import db_manager

logger = logging.getLogger(__name__)


class StudentService:
    """
    Student Service for managing student operations
    """
    
    def __init__(self, session: Session = None):
        """
        Initialize student service
        
        Args:
            session (Session): Database session (creates new if None)
        """
        self.session = session
        self._own_session = False
        
        if self.session is None:
            self.session = db_manager.get_session()
            self._own_session = True
    
    def __del__(self):
        """Cleanup session if we created it"""
        if self._own_session and self.session:
            self.session.close()
    
    def create_student(self,
                      student_id: str,
                      name: str,
                      email: str = None,
                      enrollment_number: str = None,
                      department: str = None,
                      year: str = None) -> Dict[str, Any]:
        """
        Create a new student
        
        Args:
            student_id (str): Unique student ID
            name (str): Student name
            email (str): Email address
            enrollment_number (str): Enrollment number
            department (str): Department
            year (str): Year/class
        
        Returns:
            Dict: Result with student data
        """
        try:
            # Check if student already exists
            existing = self.session.query(Student).filter(
                Student.student_id == student_id
            ).first()
            
            if existing:
                return {
                    'success': False,
                    'error': f'Student with ID {student_id} already exists',
                    'student_id': student_id
                }
            
            # Create student
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
            
            return {
                'success': True,
                'student': student.to_dict()
            }
        
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating student: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """
        Get student by ID
        
        Args:
            student_id (str): Student ID
        
        Returns:
            Optional[Dict]: Student data or None
        """
        student = self.session.query(Student).filter(
            Student.student_id == student_id
        ).first()
        
        return student.to_dict() if student else None
    
    def get_all_students(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all students
        
        Args:
            active_only (bool): Return only active students
        
        Returns:
            List[Dict]: List of students
        """
        query = self.session.query(Student)
        
        if active_only:
            query = query.filter(Student.is_active == True)
        
        students = query.all()
        return [student.to_dict() for student in students]
    
    def update_student(self, student_id: str, **kwargs) -> Dict[str, Any]:
        """
        Update student information
        
        Args:
            student_id (str): Student ID
            **kwargs: Fields to update
        
        Returns:
            Dict: Result with updated student data
        """
        try:
            student = self.session.query(Student).filter(
                Student.student_id == student_id
            ).first()
            
            if not student:
                return {
                    'success': False,
                    'error': f'Student {student_id} not found'
                }
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(student, key):
                    setattr(student, key, value)
            
            self.session.commit()
            
            logger.info(f"Student updated: {student_id}")
            
            return {
                'success': True,
                'student': student.to_dict()
            }
        
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating student: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def register_face(self, student_id: str, face_image_path: str = None) -> Dict[str, Any]:
        """
        Mark student as having face registered
        
        Args:
            student_id (str): Student ID
            face_image_path (str): Path to face image
        
        Returns:
            Dict: Result
        """
        return self.update_student(
            student_id,
            face_registered=True,
            face_image_path=face_image_path
        )
    
    def delete_student(self, student_id: str, soft_delete: bool = True) -> Dict[str, Any]:
        """
        Delete a student
        
        Args:
            student_id (str): Student ID
            soft_delete (bool): Soft delete (deactivate) or hard delete
        
        Returns:
            Dict: Result
        """
        try:
            student = self.session.query(Student).filter(
                Student.student_id == student_id
            ).first()
            
            if not student:
                return {
                    'success': False,
                    'error': f'Student {student_id} not found'
                }
            
            if soft_delete:
                # Soft delete - just deactivate
                student.is_active = False
                self.session.commit()
                logger.info(f"Student deactivated: {student_id}")
            else:
                # Hard delete
                self.session.delete(student)
                self.session.commit()
                logger.info(f"Student deleted: {student_id}")
            
            return {
                'success': True,
                'student_id': student_id,
                'deleted': not soft_delete
            }
        
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting student: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def search_students(self, query: str) -> List[Dict[str, Any]]:
        """
        Search students by name, ID, or email
        
        Args:
            query (str): Search query
        
        Returns:
            List[Dict]: Matching students
        """
        students = self.session.query(Student).filter(
            or_(
                Student.student_id.like(f'%{query}%'),
                Student.name.like(f'%{query}%'),
                Student.email.like(f'%{query}%')
            )
        ).all()
        
        return [student.to_dict() for student in students]


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("STUDENT SERVICE TEST")
    print("=" * 60)
    
    from backend.database.connection import init_database
    
    # Initialize database
    init_database()
    
    # Create service
    service = StudentService()
    
    print("\nStudent Service initialized successfully!")
    
    # Get all students
    students = service.get_all_students()
    print(f"\nTotal students: {len(students)}")
    
    print("\n✓ Test completed!")
