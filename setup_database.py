"""
Database Setup and Verification Script

This script initializes the database and verifies all components.

Author: Face Recognition Team
Date: January 2026
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("DATABASE SETUP AND VERIFICATION")
print("=" * 70)

# Test 1: Import modules
print("\n[1/5] Testing imports...")
try:
    from backend.models.student import Student, Attendance, User, Base
    from backend.database.connection import db_manager, init_database
    from backend.services.student_service import StudentService
    from backend.services.attendance_service import AttendanceService
    from backend.attendance_system import AttendanceSystem
    print("✓ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Initialize database
print("\n[2/5] Initializing database...")
try:
    init_database()
    print("✓ Database initialized")
    print(f"  Location: {db_manager.database_url}")
except Exception as e:
    print(f"❌ Database initialization failed: {e}")
    sys.exit(1)

# Test 3: Test student service
print("\n[3/5] Testing student service...")
try:
    student_service = StudentService()
    
    # Get all students
    students = student_service.get_all_students()
    print(f"✓ Student service working")
    print(f"  Total students: {len(students)}")
    
except Exception as e:
    print(f"❌ Student service test failed: {e}")
    sys.exit(1)

# Test 4: Test attendance service
print("\n[4/5] Testing attendance service...")
try:
    attendance_service = AttendanceService()
    
    # Get statistics
    stats = attendance_service.get_attendance_statistics()
    print(f"✓ Attendance service working")
    print(f"  Total students: {stats['total_students']}")
    print(f"  Total records: {stats['total_attendance_records']}")
    print(f"  Today's attendance: {stats['today_attendance']}")
    
except Exception as e:
    print(f"❌ Attendance service test failed: {e}")
    sys.exit(1)

# Test 5: Test complete system
print("\n[5/5] Testing complete attendance system...")
try:
    system = AttendanceSystem()
    
    # Get system stats
    sys_stats = system.get_system_stats()
    print(f"✓ Attendance system working")
    print(f"  Registered faces: {sys_stats['face_recognition']['recognizer']['total_faces']}")
    print(f"  Database students: {sys_stats['attendance']['total_students']}")
    
except Exception as e:
    print(f"❌ Attendance system test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE - ALL TESTS PASSED ✓")
print("=" * 70)
print("\nDatabase Schema Created:")
print("  ✓ students table")
print("  ✓ attendance table")
print("  ✓ users table")
print("\nServices Ready:")
print("  ✓ StudentService - Manage students")
print("  ✓ AttendanceService - Mark and query attendance")
print("  ✓ AttendanceSystem - Complete integrated system")
print("\nNext steps:")
print("1. Register students with faces")
print("2. Mark attendance using the system")
print("3. Query attendance records and statistics")
print("=" * 70)
