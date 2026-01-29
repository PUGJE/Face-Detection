"""
Backend Services Package

This package contains business logic services.
"""

from backend.services.student_service import StudentService
from backend.services.attendance_service import AttendanceService

__all__ = ['StudentService', 'AttendanceService']
