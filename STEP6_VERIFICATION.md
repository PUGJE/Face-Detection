# Step 6 Verification Guide

## How to Verify the Attendance System & Database

This guide shows you how to verify that Step 6 (Attendance System & Database) is working correctly.

---

## Quick Verification (Already Done ✓)

You've already run this successfully:

```bash
python setup_database.py
```

**Result:** All tests passed ✓

---

## Complete System Demo (Interactive)

Run the complete interactive demo:

```bash
python demo_complete_system.py
```

### What This Demo Does:

**Step 1:** Initialize system
- Loads database
- Loads face recognition
- Shows initial statistics

**Step 2:** Student Registration
- Opens webcam
- Press SPACE to capture your face
- Registers you as "DEMO_STUDENT"
- Saves face to database

**Step 3:** Attendance Marking
- Opens webcam again
- Recognizes your face in real-time
- Press SPACE to mark attendance
- Prevents duplicate marking

**Step 4:** Query Records
- Shows today's attendance
- Displays all records

**Step 5:** View Statistics
- Overall system stats
- Per-student attendance percentage
- Today's attendance rate

---

## Manual Verification Steps

### 1. Check Database Created

```bash
# Check if database file exists
ls data/database/attendance.db
```

**Expected:** File should exist

### 2. Check Tables Created

```python
python -c "from backend.database.connection import db_manager, init_database; init_database(); print('✓ Database tables created')"
```

**Expected:** "✓ Database tables created"

### 3. Test Student Service

```python
python backend/services/student_service.py
```

**Expected:** Shows student count (0 initially)

### 4. Test Attendance Service

```python
python backend/services/attendance_service.py
```

**Expected:** Shows attendance statistics

### 5. Test Complete System

```python
python backend/attendance_system.py
```

**Expected:** Shows system initialized with stats

---

## Verify Database Contents

You can inspect the database using Python:

```python
from backend.database.connection import db_manager
from backend.models.student import Student, Attendance

with db_manager.session_scope() as session:
    # Count students
    student_count = session.query(Student).count()
    print(f"Students: {student_count}")
    
    # Count attendance
    attendance_count = session.query(Attendance).count()
    print(f"Attendance records: {attendance_count}")
    
    # List all students
    students = session.query(Student).all()
    for student in students:
        print(f"  - {student.student_id}: {student.name}")
```

---

## What Should Work

After Step 6, you should be able to:

✅ **Create students** in the database
✅ **Register faces** for students
✅ **Mark attendance** using face recognition
✅ **Query attendance** by date
✅ **Get statistics** (overall and per-student)
✅ **Prevent duplicates** (one attendance per day)
✅ **Detect late arrivals** (after 9:30 AM)
✅ **Search students** by name/ID/email
✅ **Soft delete** students (deactivate)

---

## Key Files Created

### Database Models
- `backend/models/student.py` - Student, Attendance, User models

### Database Connection
- `backend/database/connection.py` - Database manager

### Services
- `backend/services/student_service.py` - Student CRUD
- `backend/services/attendance_service.py` - Attendance logic

### Complete System
- `backend/attendance_system.py` - Integrated system

### Verification Scripts
- `setup_database.py` - Database setup & verification
- `demo_complete_system.py` - Interactive demo

---

## Database Schema

The database has 3 tables:

### students
- id (Primary Key)
- student_id (Unique)
- name, email, enrollment_number
- department, year
- face_registered, face_image_path
- is_active, created_at, updated_at

### attendance
- id (Primary Key)
- student_id (Foreign Key → students.id)
- timestamp, date, time
- recognition_confidence, recognition_distance
- status (present/late/absent)
- marked_by (system/manual/admin)

### users
- id (Primary Key)
- username, email (Unique)
- password_hash
- full_name, role
- is_active, created_at, last_login

---

## Troubleshooting

### Database not found
```bash
python setup_database.py
```

### Import errors
Make sure you're in the virtual environment:
```bash
venv\Scripts\activate  # Windows
```

### Webcam not working
- Check camera permissions
- Try different camera_index in .env file
- Skip webcam demos (press 'q')

---

## Success Criteria

✅ `setup_database.py` runs without errors
✅ Database file created at `./data/database/attendance.db`
✅ Can register students
✅ Can mark attendance
✅ Can query records
✅ Duplicate prevention works
✅ Statistics are accurate

---

## Next Steps

After verifying Step 6, you can:

1. **Proceed to Step 7** - Build FastAPI REST API
2. **Proceed to Step 8** - Create web frontend
3. **Test the system** - Register multiple students and test

---

**Step 6 Status: ✅ COMPLETE**
