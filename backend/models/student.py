"""
Database Models for Attendance System

This module defines the SQLAlchemy models for the attendance system.

Models:
- Student: Student information and face embeddings
- Attendance: Attendance records
- User: Admin/staff users

Author: Face Recognition Team
Date: January 2026
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Student(Base):
    """
    Student Model
    
    Stores student information and metadata for face recognition
    """
    __tablename__ = 'students'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=True)
    enrollment_number = Column(String(50), unique=True, nullable=True)
    department = Column(String(100), nullable=True)
    year = Column(String(20), nullable=True)
    
    # Face recognition metadata
    face_registered = Column(Boolean, default=False)
    face_image_path = Column(String(255), nullable=True)
    registration_date = Column(DateTime, default=datetime.utcnow)
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    attendance_records = relationship("Attendance", back_populates="student", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Student(id={self.student_id}, name={self.name})>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'name': self.name,
            'email': self.email,
            'enrollment_number': self.enrollment_number,
            'department': self.department,
            'year': self.year,
            'face_registered': self.face_registered,
            'is_active': self.is_active,
            'registration_date': self.registration_date.isoformat() if self.registration_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Attendance(Base):
    """
    Attendance Model
    
    Stores attendance records with timestamps and confidence scores
    """
    __tablename__ = 'attendance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey('students.id'), nullable=False, index=True)
    
    # Attendance details
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    date = Column(String(10), nullable=False, index=True)  # YYYY-MM-DD format
    time = Column(String(8), nullable=False)  # HH:MM:SS format
    
    # Recognition details
    recognition_confidence = Column(Float, nullable=True)
    recognition_distance = Column(Float, nullable=True)
    detection_confidence = Column(Float, nullable=True)
    
    # Status
    status = Column(String(20), default='present')  # present, absent, late
    marked_by = Column(String(50), default='system')  # system, manual, admin
    
    # Additional info
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    student = relationship("Student", back_populates="attendance_records")
    
    def __repr__(self):
        return f"<Attendance(student_id={self.student_id}, date={self.date}, time={self.time})>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'student_name': self.student.name if self.student else None,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'date': self.date,
            'time': self.time,
            'recognition_confidence': self.recognition_confidence,
            'recognition_distance': self.recognition_distance,
            'detection_confidence': self.detection_confidence,
            'status': self.status,
            'marked_by': self.marked_by,
            'notes': self.notes
        }


class User(Base):
    """
    User Model
    
    Stores admin/staff user information for authentication
    """
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    
    # User details
    full_name = Column(String(100), nullable=True)
    role = Column(String(20), default='admin')  # admin, staff, viewer
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<User(username={self.username}, role={self.role})>"
    
    def to_dict(self):
        """Convert to dictionary (excluding password)"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("DATABASE MODELS TEST")
    print("=" * 60)
    
    # Create sample student
    student = Student(
        student_id="STUDENT_001",
        name="John Doe",
        email="john@example.com",
        enrollment_number="EN2024001",
        department="Computer Science",
        year="3rd Year"
    )
    
    print("\nSample Student:")
    print(student)
    print(student.to_dict())
    
    # Create sample attendance
    attendance = Attendance(
        student_id=1,
        date="2026-01-29",
        time="09:30:00",
        recognition_confidence=0.95,
        status="present"
    )
    
    print("\nSample Attendance:")
    print(attendance)
    
    print("\n✓ Models defined successfully!")
