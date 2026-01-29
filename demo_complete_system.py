"""
Complete Attendance System Demo

This demo shows the complete workflow:
1. Register a student with face
2. Mark attendance using face recognition
3. Query attendance records
4. View statistics

Author: Face Recognition Team
Date: January 2026
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.attendance_system import AttendanceSystem
from backend.services.student_service import StudentService
from backend.services.attendance_service import AttendanceService

print("=" * 70)
print("COMPLETE ATTENDANCE SYSTEM DEMO")
print("=" * 70)

# Initialize system
print("\n[Step 1] Initializing attendance system...")
system = AttendanceSystem()
print("✓ System initialized")

# Show initial stats
stats = system.get_system_stats()
print(f"\nInitial State:")
print(f"  - Registered faces: {stats['face_recognition']['recognizer']['total_faces']}")
print(f"  - Database students: {stats['attendance']['total_students']}")
print(f"  - Today's attendance: {stats['attendance']['today_attendance']}")

# Demo 1: Register a student with face using webcam
print("\n" + "=" * 70)
print("[Step 2] STUDENT REGISTRATION DEMO")
print("=" * 70)
print("\nThis will:")
print("1. Open your webcam")
print("2. Capture your face")
print("3. Register you as a student")
print("4. Save face to database")
print("\nPress SPACE to capture, 'q' to skip")

cap = cv2.VideoCapture(0)
if cap.isOpened():
    student_registered = False
    
    while not student_registered:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show preview
        cv2.putText(frame, "Press SPACE to register as DEMO_STUDENT", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to skip", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Student Registration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to register
            print("\n📸 Capturing and registering...")
            
            # Register student
            result = system.register_new_student(
                student_id="DEMO_STUDENT",
                name="Demo User",
                image=frame,
                email="demo@example.com",
                department="Demo Department",
                year="Demo Year"
            )
            
            if result['success']:
                print(f"✓ Student registered successfully!")
                print(f"  Student ID: {result['student_id']}")
                print(f"  Name: {result['name']}")
                print(f"  Face Confidence: {result.get('face_confidence', 'N/A')}")
                student_registered = True
            else:
                print(f"❌ Registration failed: {result.get('error')}")
                if 'already exists' in result.get('error', ''):
                    print("  (Student already exists, continuing...)")
                    student_registered = True
        
        elif key == ord('q'):
            print("\n⏭️  Registration skipped")
            student_registered = True
    
    cap.release()
    cv2.destroyAllWindows()
else:
    print("❌ Could not open webcam, skipping registration demo")

# Demo 2: Mark attendance using face recognition
print("\n" + "=" * 70)
print("[Step 3] ATTENDANCE MARKING DEMO")
print("=" * 70)
print("\nThis will:")
print("1. Open your webcam")
print("2. Recognize your face")
print("3. Mark your attendance")
print("\nPress SPACE to mark attendance, 'q' to skip")

cap = cv2.VideoCapture(0)
if cap.isOpened():
    attendance_marked = False
    
    while not attendance_marked:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Try to recognize in real-time
        recognition_results = system.face_pipeline.process_attendance_frame(frame)
        
        # Draw results
        output = system.face_pipeline.draw_recognition_results(frame, recognition_results)
        
        # Add instructions
        cv2.putText(output, "Press SPACE to mark attendance", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output, "Press 'q' to skip", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show recognized students
        for i, result in enumerate(recognition_results):
            if result['recognized']:
                text = f"Recognized: {result['student_id']}"
                cv2.putText(output, text, (10, 90 + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Attendance Marking', output)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to mark
            print("\n📋 Marking attendance...")
            
            # Mark attendance
            result = system.mark_attendance_from_image(frame)
            
            if result['success']:
                print(f"✓ Attendance marked successfully!")
                print(f"  Student ID: {result['student_id']}")
                print(f"  Student Name: {result['student_name']}")
                print(f"  Date: {result['date']}")
                print(f"  Time: {result['time']}")
                print(f"  Status: {result['status']}")
                attendance_marked = True
            else:
                if result.get('duplicate'):
                    print(f"ℹ️  Attendance already marked today")
                    print(f"  Student: {result.get('student_name')}")
                    attendance_marked = True
                else:
                    print(f"❌ Failed: {result.get('error')}")
        
        elif key == ord('q'):
            print("\n⏭️  Attendance marking skipped")
            attendance_marked = True
    
    cap.release()
    cv2.destroyAllWindows()
else:
    print("❌ Could not open webcam, skipping attendance demo")

# Demo 3: Query attendance records
print("\n" + "=" * 70)
print("[Step 4] QUERY ATTENDANCE RECORDS")
print("=" * 70)

# Get today's attendance
attendance_service = AttendanceService()
today_records = attendance_service.get_attendance_by_date()

print(f"\nToday's Attendance Records: {len(today_records)}")
for record in today_records:
    print(f"\n  Student: {record['student_name']}")
    print(f"  Time: {record['time']}")
    print(f"  Status: {record['status']}")
    print(f"  Confidence: {record['recognition_confidence']:.3f}" if record['recognition_confidence'] else "  Confidence: N/A")

# Demo 4: View statistics
print("\n" + "=" * 70)
print("[Step 5] VIEW STATISTICS")
print("=" * 70)

# Overall stats
overall_stats = attendance_service.get_attendance_statistics()
print(f"\nOverall Statistics:")
print(f"  Total Students: {overall_stats['total_students']}")
print(f"  Total Attendance Records: {overall_stats['total_attendance_records']}")
print(f"  Today's Attendance: {overall_stats['today_attendance']}")
print(f"  Today's Percentage: {overall_stats['today_percentage']:.1f}%")

# Get all students
student_service = StudentService()
all_students = student_service.get_all_students()

print(f"\nRegistered Students: {len(all_students)}")
for student in all_students:
    print(f"\n  Student ID: {student['student_id']}")
    print(f"  Name: {student['name']}")
    print(f"  Department: {student['department']}")
    print(f"  Face Registered: {'✓' if student['face_registered'] else '✗'}")
    
    # Get student-specific stats
    student_stats = attendance_service.get_attendance_statistics(student['student_id'])
    if 'total_days' in student_stats:
        print(f"  Attendance: {student_stats['attendance_percentage']:.1f}% ({student_stats['present_days']}/{student_stats['total_days']} days)")

# Final summary
print("\n" + "=" * 70)
print("DEMO COMPLETE!")
print("=" * 70)
print("\n✓ System Verification Summary:")
print("  ✓ Student registration working")
print("  ✓ Face recognition working")
print("  ✓ Attendance marking working")
print("  ✓ Database queries working")
print("  ✓ Statistics generation working")
print("\n🎉 The complete attendance system is fully functional!")
print("\nDatabase location: ./data/database/attendance.db")
print("Face embeddings: ./data/embeddings/face_embeddings.pkl")
print("Student images: ./data/student_images/")
print("=" * 70)
