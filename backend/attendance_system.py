"""
Complete Attendance System

This module integrates face recognition with attendance management.

Combines:
- Face detection
- Face recognition
- Student management
- Attendance marking

Author: Face Recognition Team
Date: January 2026
"""

import cv2
import numpy as np
from typing import Dict, Any, List
import logging
from pathlib import Path

from backend.ml.recognition_pipeline import FaceRecognitionPipeline
from backend.services.student_service import StudentService
from backend.services.attendance_service import AttendanceService
from backend.database.connection import init_database, db_manager
from backend.config import settings

logger = logging.getLogger(__name__)


class AttendanceSystem:
    """
    Complete Attendance System
    
    Integrates all components for a complete attendance solution
    """
    
    def __init__(self):
        """Initialize the attendance system"""
        # Initialize database
        init_database()
        
        # Initialize face recognition pipeline
        self.face_pipeline = FaceRecognitionPipeline()
        
        # Initialize services
        self.student_service = StudentService()
        self.attendance_service = AttendanceService()
        
        # Load face recognition database
        self.face_pipeline.load_database()
        
        logger.info("Attendance System initialized")
    
    def register_new_student(self,
                            student_id: str,
                            name: str,
                            image: np.ndarray,
                            email: str = None,
                            enrollment_number: str = None,
                            department: str = None,
                            year: str = None) -> Dict[str, Any]:
        """
        Register a new student with face recognition
        
        Args:
            student_id (str): Unique student ID
            name (str): Student name
            image (np.ndarray): Face image
            email (str): Email address
            enrollment_number (str): Enrollment number
            department (str): Department
            year (str): Year/class
        
        Returns:
            Dict: Registration result
        """
        # Create student in database
        result = self.student_service.create_student(
            student_id=student_id,
            name=name,
            email=email,
            enrollment_number=enrollment_number,
            department=department,
            year=year
        )
        
        if not result['success']:
            return result
        
        # Register face
        face_result = self.face_pipeline.register_student(student_id, image)
        
        if not face_result['success']:
            # Rollback student creation if face registration fails
            self.student_service.delete_student(student_id, soft_delete=False)
            return face_result
        
        # Save face image
        face_image_dir = Path(settings.student_images_path) / student_id
        face_image_dir.mkdir(parents=True, exist_ok=True)
        face_image_path = face_image_dir / "face.jpg"
        cv2.imwrite(str(face_image_path), image)
        
        # Update student with face info
        self.student_service.register_face(student_id, str(face_image_path))
        
        # Save face recognition database
        self.face_pipeline.save_database()
        
        logger.info(f"Student registered: {student_id} - {name}")
        
        return {
            'success': True,
            'student_id': student_id,
            'name': name,
            'face_registered': True,
            'face_confidence': face_result.get('face_confidence')
        }
    
    def mark_attendance_from_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Mark attendance from a single image with anti-spoofing
        
        Args:
            image (np.ndarray): Image containing student face
        
        Returns:
            Dict: Attendance result
        """
        # Recognize student with liveness check
        recognition_result = self.face_pipeline.recognize_with_liveness(image)
        
        if not recognition_result['success']:
            return recognition_result
        
        # Check liveness
        if not recognition_result.get('is_live', True):
            return {
                'success': False,
                'error': 'Liveness check failed - possible spoofing attack detected',
                'recognized': False,
                'is_live': False,
                'liveness_score': recognition_result.get('liveness_score', 0)
            }
        
        if not recognition_result['recognized']:
            return {
                'success': False,
                'error': 'Student not recognized',
                'recognized': False,
                'is_live': recognition_result.get('is_live', True)
            }
        
        student_id = recognition_result['student_id']
        
        # Mark attendance
        attendance_result = self.attendance_service.mark_attendance(
            student_id=student_id,
            recognition_confidence=recognition_result.get('recognition_confidence'),
            recognition_distance=recognition_result.get('recognition_distance'),
            detection_confidence=recognition_result.get('detection_confidence')
        )
        
        # Add liveness info to result
        if attendance_result['success']:
            attendance_result['is_live'] = recognition_result.get('is_live', True)
            attendance_result['liveness_score'] = recognition_result.get('liveness_score', 1.0)
        
        return attendance_result
    
    def process_webcam_attendance(self, duration_seconds: int = 60) -> List[Dict[str, Any]]:
        """
        Process webcam feed for attendance marking
        
        Args:
            duration_seconds (int): How long to process (seconds)
        
        Returns:
            List[Dict]: List of attendance results
        """
        import time
        
        results = []
        cap = cv2.VideoCapture(settings.camera_index)
        
        if not cap.isOpened():
            logger.error("Could not open camera")
            return results
        
        start_time = time.time()
        processed_students = set()  # Track who we've already processed
        
        try:
            while (time.time() - start_time) < duration_seconds:
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Process frame
                recognition_results = self.face_pipeline.process_attendance_frame(frame)
                
                # Mark attendance for recognized faces
                for rec_result in recognition_results:
                    if rec_result['recognized']:
                        student_id = rec_result['student_id']
                        
                        # Skip if already processed
                        if student_id in processed_students:
                            continue
                        
                        # Mark attendance
                        att_result = self.attendance_service.mark_attendance(
                            student_id=student_id,
                            recognition_confidence=rec_result.get('confidence'),
                            recognition_distance=rec_result.get('distance')
                        )
                        
                        if att_result['success']:
                            results.append(att_result)
                            processed_students.add(student_id)
                            logger.info(f"Attendance marked: {student_id}")
                
                # Optional: Display frame
                # cv2.imshow('Attendance', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return results
    
    def get_today_attendance(self) -> List[Dict[str, Any]]:
        """
        Get today's attendance records
        
        Returns:
            List[Dict]: Attendance records
        """
        return self.attendance_service.get_attendance_by_date()
    
    def get_student_stats(self, student_id: str) -> Dict[str, Any]:
        """
        Get statistics for a student
        
        Args:
            student_id (str): Student ID
        
        Returns:
            Dict: Statistics
        """
        return self.attendance_service.get_attendance_statistics(student_id)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get overall system statistics
        
        Returns:
            Dict: System statistics
        """
        face_stats = self.face_pipeline.get_stats()
        attendance_stats = self.attendance_service.get_attendance_statistics()
        
        return {
            'face_recognition': face_stats,
            'attendance': attendance_stats
        }


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("ATTENDANCE SYSTEM TEST")
    print("=" * 60)
    
    # Initialize system
    system = AttendanceSystem()
    
    print("\nAttendance System initialized successfully!")
    
    # Get system stats
    stats = system.get_system_stats()
    print(f"\nSystem Statistics:")
    print(f"  Registered Students (Face): {stats['face_recognition']['recognizer']['total_faces']}")
    print(f"  Total Students (DB): {stats['attendance']['total_students']}")
    print(f"  Today's Attendance: {stats['attendance']['today_attendance']}")
    
    print("\n✓ Test completed!")
