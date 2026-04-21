import os
import sys

# Ensure backend modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, date, time
from unittest.mock import patch

from backend.database.connection import db_manager, init_database
from backend.models.student import Student, Timetable
from backend.services.attendance_service import AttendanceService

def setup_test_data():
    with db_manager.session_scope() as s:
        # Create a test student
        student = Student(
            student_id="STUDENT_TEST_001",
            name="Alice Tester",
            is_active=True
        )
        s.add(student)
        
        # Create a test timetable for Math 101, occurring on the current day of the week
        today_name = datetime.now().strftime("%A")
        
        # Class from 10:00 to 11:00
        tt = Timetable(
            subject_name="Math 101",
            day_of_week=today_name,
            start_time=time(10, 0),
            end_time=time(11, 0),
            teacher_id="Prof. Smith"
        )
        s.add(tt)
        s.commit()
        return student.student_id

@patch('backend.services.attendance_service.datetime')
def test_timetable_matching(mock_datetime):
    print("=" * 50)
    print("TESTING TIMETABLE MATCHING LOGIC")
    print("=" * 50)
    
    # Reset DB and init
    db_manager.reset_database()
    init_database()
    
    student_id = setup_test_data()
    
    # Test Case 1: Student recognized at 10:05 am (During Math 101, on time)
    # Target time: Today at 10:05 AM
    target_time_1 = datetime.combine(date.today(), time(10, 5))
    mock_datetime.now.return_value = target_time_1
    mock_datetime.combine = datetime.combine
    
    service = AttendanceService()
    res = service.mark_attendance(student_id)
    
    print("\n[Test 1] 10:05 AM (Present)")
    print("Result:", res)
    assert res['success'] is True, "Expected success marking attendance."
    assert res['status'] == "present", "Expected student to be marked present."
    
    # Test Case 2: Duplicate scan at 10:07 am (Should fail duplicate check)
    target_time_2 = datetime.combine(date.today(), time(10, 7))
    mock_datetime.now.return_value = target_time_2
    
    res2 = service.mark_attendance(student_id)
    print("\n[Test 2] 10:07 AM (Duplicate Check)")
    print("Result:", res2)
    assert res2['success'] is False, "Expected failure on duplicate scan."
    assert res2['duplicate'] is True, "Expected to be flagged as duplicate."
    
    print("\nAll internal logic tests passed successfully!")

if __name__ == "__main__":
    test_timetable_matching()
