"""
Attendance Service

This module handles attendance marking logic and business rules.

Features:
- Mark attendance with duplicate prevention
- Query attendance records
- Generate attendance reports
- Handle late arrivals

Author: Face Recognition Team
Date: January 2026
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import logging

from backend.models.student import Student, Attendance
from backend.database.connection import db_manager
from backend.config import settings

logger = logging.getLogger(__name__)


class AttendanceService:
    """
    Attendance Service for managing attendance operations
    
    Handles attendance marking, querying, and reporting
    """
    
    def __init__(self, session: Session = None):
        """
        Initialize attendance service
        
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
    
    def mark_attendance(self, 
                       student_id: str,
                       recognition_confidence: float = None,
                       recognition_distance: float = None,
                       detection_confidence: float = None,
                       notes: str = None) -> Dict[str, Any]:
        """
        Mark attendance for a student
        
        Args:
            student_id (str): Student ID
            recognition_confidence (float): Recognition confidence score
            recognition_distance (float): Recognition distance
            detection_confidence (float): Detection confidence
            notes (str): Additional notes
        
        Returns:
            Dict: Result with status and details
        """
        try:
            # Find student
            student = self.session.query(Student).filter(
                Student.student_id == student_id
            ).first()
            
            if not student:
                return {
                    'success': False,
                    'error': f'Student {student_id} not found',
                    'student_id': student_id
                }
            
            if not student.is_active:
                return {
                    'success': False,
                    'error': f'Student {student_id} is inactive',
                    'student_id': student_id
                }
            
            # Get current date and time
            now = datetime.now()
            current_date = now.strftime('%Y-%m-%d')
            current_time = now.strftime('%H:%M:%S')
            
            # Check if attendance already marked today
            existing = self.session.query(Attendance).filter(
                and_(
                    Attendance.student_id == student.id,
                    Attendance.date == current_date
                )
            ).first()
            
            if existing:
                return {
                    'success': False,
                    'error': 'Attendance already marked today',
                    'student_id': student_id,
                    'student_name': student.name,
                    'existing_record': existing.to_dict(),
                    'duplicate': True
                }
            
            # Determine status (present/late)
            status = 'present'
            attendance_start_time = settings.attendance_start_time
            attendance_end_time = settings.attendance_end_time
            
            # Parse times
            current_hour = now.hour
            current_minute = now.minute
            
            # Simple late check (after 9:30 AM is late)
            if current_hour > 9 or (current_hour == 9 and current_minute > 30):
                status = 'late'
            
            # Create attendance record
            attendance = Attendance(
                student_id=student.id,
                timestamp=now,
                date=current_date,
                time=current_time,
                recognition_confidence=recognition_confidence,
                recognition_distance=recognition_distance,
                detection_confidence=detection_confidence,
                status=status,
                marked_by='system',
                notes=notes
            )
            
            self.session.add(attendance)
            self.session.commit()
            
            logger.info(f"Attendance marked for {student_id} ({student.name}) - {status}")
            
            return {
                'success': True,
                'student_id': student_id,
                'student_name': student.name,
                'date': current_date,
                'time': current_time,
                'status': status,
                'recognition_confidence': recognition_confidence,
                'attendance_id': attendance.id
            }
        
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error marking attendance: {e}")
            return {
                'success': False,
                'error': str(e),
                'student_id': student_id
            }
    
    def get_attendance_by_date(self, target_date: str = None) -> List[Dict[str, Any]]:
        """
        Get all attendance records for a specific date
        
        Args:
            target_date (str): Date in YYYY-MM-DD format (today if None)
        
        Returns:
            List[Dict]: List of attendance records
        """
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        records = self.session.query(Attendance).filter(
            Attendance.date == target_date
        ).all()
        
        return [record.to_dict() for record in records]
    
    def get_student_attendance_history(self, 
                                      student_id: str,
                                      days: int = 30) -> List[Dict[str, Any]]:
        """
        Get attendance history for a student
        
        Args:
            student_id (str): Student ID
            days (int): Number of days to look back
        
        Returns:
            List[Dict]: Attendance history
        """
        student = self.session.query(Student).filter(
            Student.student_id == student_id
        ).first()
        
        if not student:
            return []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        records = self.session.query(Attendance).filter(
            and_(
                Attendance.student_id == student.id,
                Attendance.timestamp >= start_date,
                Attendance.timestamp <= end_date
            )
        ).order_by(Attendance.timestamp.desc()).all()
        
        return [record.to_dict() for record in records]
    
    def get_attendance_statistics(self, student_id: str = None) -> Dict[str, Any]:
        """
        Get attendance statistics
        
        Args:
            student_id (str): Student ID (all students if None)
        
        Returns:
            Dict: Statistics
        """
        if student_id:
            student = self.session.query(Student).filter(
                Student.student_id == student_id
            ).first()
            
            if not student:
                return {'error': 'Student not found'}
            
            total = self.session.query(Attendance).filter(
                Attendance.student_id == student.id
            ).count()
            
            present = self.session.query(Attendance).filter(
                and_(
                    Attendance.student_id == student.id,
                    Attendance.status == 'present'
                )
            ).count()
            
            late = self.session.query(Attendance).filter(
                and_(
                    Attendance.student_id == student.id,
                    Attendance.status == 'late'
                )
            ).count()
            
            return {
                'student_id': student_id,
                'student_name': student.name,
                'total_days': total,
                'present_days': present,
                'late_days': late,
                'attendance_percentage': (present / total * 100) if total > 0 else 0
            }
        else:
            # Overall statistics
            total_students = self.session.query(Student).filter(
                Student.is_active == True
            ).count()
            
            total_records = self.session.query(Attendance).count()
            
            today = datetime.now().strftime('%Y-%m-%d')
            today_records = self.session.query(Attendance).filter(
                Attendance.date == today
            ).count()
            
            return {
                'total_students': total_students,
                'total_attendance_records': total_records,
                'today_attendance': today_records,
                'today_percentage': (today_records / total_students * 100) if total_students > 0 else 0
            }
    
    def get_absent_students(self, target_date: str = None) -> List[Dict[str, Any]]:
        """
        Get list of students who are absent on a specific date
        
        Args:
            target_date (str): Date in YYYY-MM-DD format (today if None)
        
        Returns:
            List[Dict]: List of absent students
        """
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get all active students
        all_students = self.session.query(Student).filter(
            Student.is_active == True
        ).all()
        
        # Get students who marked attendance
        present_student_ids = self.session.query(Attendance.student_id).filter(
            Attendance.date == target_date
        ).all()
        
        present_ids = [sid[0] for sid in present_student_ids]
        
        # Find absent students
        absent_students = [
            student.to_dict() 
            for student in all_students 
            if student.id not in present_ids
        ]
        
        return absent_students


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("ATTENDANCE SERVICE TEST")
    print("=" * 60)
    
    from backend.database.connection import init_database
    
    # Initialize database
    init_database()
    
    # Create service
    service = AttendanceService()
    
    print("\nAttendance Service initialized successfully!")
    
    # Get statistics
    stats = service.get_attendance_statistics()
    print(f"\nOverall Statistics:")
    print(f"  Total Students: {stats['total_students']}")
    print(f"  Total Records: {stats['total_attendance_records']}")
    print(f"  Today's Attendance: {stats['today_attendance']}")
    
    print("\n✓ Test completed!")
