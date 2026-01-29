# API Documentation

## Face Recognition Attendance System API

**Base URL:** `http://127.0.0.1:8000`

**Interactive Documentation:**
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

## Starting the Server

```bash
python main.py
```

The server will start on `http://127.0.0.1:8000`

---

## API Endpoints

### Health & Info

#### GET `/`
Get API information

**Response:**
```json
{
  "name": "Face Recognition Attendance System API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "endpoints": {
    "students": "/api/students",
    "attendance": "/api/attendance",
    "statistics": "/api/statistics"
  }
}
```

#### GET `/health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-29T22:51:21.903077",
  "database": "connected",
  "face_recognition": "loaded"
}
```

#### GET `/api/stats`
Get overall system statistics

**Response:**
```json
{
  "success": true,
  "data": {
    "face_recognition": {
      "recognizer": {
        "total_faces": 2,
        "student_ids": ["STUDENT_001", "DEMO_STUDENT"]
      }
    },
    "attendance": {
      "total_students": 1,
      "total_attendance_records": 0,
      "today_attendance": 0
    }
  }
}
```

---

### Student Management

#### POST `/api/students`
Create a new student

**Parameters:**
- `student_id` (required): Unique student ID
- `name` (required): Student name
- `email` (optional): Email address
- `enrollment_number` (optional): Enrollment number
- `department` (optional): Department
- `year` (optional): Year/class

**Example:**
```bash
curl -X POST "http://127.0.0.1:8000/api/students?student_id=CS2024001&name=John%20Doe&email=john@example.com&department=Computer%20Science&year=3rd%20Year"
```

**Response:**
```json
{
  "success": true,
  "message": "Student created successfully",
  "data": {
    "id": 1,
    "student_id": "CS2024001",
    "name": "John Doe",
    "email": "john@example.com",
    "department": "Computer Science",
    "year": "3rd Year",
    "face_registered": false,
    "is_active": true
  }
}
```

#### GET `/api/students`
Get all students

**Parameters:**
- `active_only` (optional, default: true): Return only active students

**Example:**
```bash
curl "http://127.0.0.1:8000/api/students"
```

**Response:**
```json
{
  "success": true,
  "count": 1,
  "data": [
    {
      "id": 1,
      "student_id": "CS2024001",
      "name": "John Doe",
      "face_registered": false
    }
  ]
}
```

#### GET `/api/students/{student_id}`
Get student by ID

**Example:**
```bash
curl "http://127.0.0.1:8000/api/students/CS2024001"
```

#### PUT `/api/students/{student_id}`
Update student information

**Parameters:**
- `name` (optional)
- `email` (optional)
- `department` (optional)
- `year` (optional)
- `is_active` (optional)

**Example:**
```bash
curl -X PUT "http://127.0.0.1:8000/api/students/CS2024001?department=IT"
```

#### DELETE `/api/students/{student_id}`
Delete student

**Parameters:**
- `hard_delete` (optional, default: false): Hard delete vs soft delete

**Example:**
```bash
curl -X DELETE "http://127.0.0.1:8000/api/students/CS2024001"
```

#### GET `/api/students/search/{query}`
Search students by name, ID, or email

**Example:**
```bash
curl "http://127.0.0.1:8000/api/students/search/John"
```

---

### Face Registration

#### POST `/api/students/{student_id}/register-face`
Register face for a student

**Body:** Multipart form data with image file

**Example (Python):**
```python
import requests

with open('student_photo.jpg', 'rb') as f:
    files = {'file': ('photo.jpg', f, 'image/jpeg')}
    response = requests.post(
        'http://127.0.0.1:8000/api/students/CS2024001/register-face',
        files=files
    )
```

**Example (curl):**
```bash
curl -X POST "http://127.0.0.1:8000/api/students/CS2024001/register-face" \
  -F "file=@student_photo.jpg"
```

**Response:**
```json
{
  "success": true,
  "message": "Face registered successfully",
  "data": {
    "student_id": "CS2024001",
    "face_confidence": 0.95
  }
}
```

---

### Attendance

#### POST `/api/attendance/mark`
Mark attendance using uploaded image

**Body:** Multipart form data with image file

**Example (Python):**
```python
import requests

with open('webcam_capture.jpg', 'rb') as f:
    files = {'file': ('capture.jpg', f, 'image/jpeg')}
    response = requests.post(
        'http://127.0.0.1:8000/api/attendance/mark',
        files=files
    )
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Attendance marked successfully",
  "data": {
    "student_id": "CS2024001",
    "student_name": "John Doe",
    "date": "2026-01-29",
    "time": "09:15:30",
    "status": "present",
    "recognition_confidence": 0.95
  }
}
```

**Response (Duplicate):**
```json
{
  "success": false,
  "duplicate": true,
  "message": "Attendance already marked today",
  "data": {
    "student_id": "CS2024001",
    "existing_record": {...}
  }
}
```

#### GET `/api/attendance/today`
Get today's attendance records

**Example:**
```bash
curl "http://127.0.0.1:8000/api/attendance/today"
```

**Response:**
```json
{
  "success": true,
  "date": "2026-01-29",
  "count": 5,
  "data": [
    {
      "student_id": 1,
      "student_name": "John Doe",
      "time": "09:15:30",
      "status": "present",
      "recognition_confidence": 0.95
    }
  ]
}
```

#### GET `/api/attendance/date/{date}`
Get attendance for a specific date (YYYY-MM-DD)

**Example:**
```bash
curl "http://127.0.0.1:8000/api/attendance/date/2026-01-29"
```

#### GET `/api/attendance/student/{student_id}`
Get attendance history for a student

**Parameters:**
- `days` (optional, default: 30): Number of days to look back

**Example:**
```bash
curl "http://127.0.0.1:8000/api/attendance/student/CS2024001?days=7"
```

#### GET `/api/attendance/absent`
Get list of absent students

**Parameters:**
- `date` (optional): Date in YYYY-MM-DD format (today if not specified)

**Example:**
```bash
curl "http://127.0.0.1:8000/api/attendance/absent"
```

---

### Statistics

#### GET `/api/statistics/overall`
Get overall attendance statistics

**Example:**
```bash
curl "http://127.0.0.1:8000/api/statistics/overall"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total_students": 10,
    "total_attendance_records": 50,
    "today_attendance": 8,
    "today_percentage": 80.0
  }
}
```

#### GET `/api/statistics/student/{student_id}`
Get statistics for a specific student

**Example:**
```bash
curl "http://127.0.0.1:8000/api/statistics/student/CS2024001"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "student_id": "CS2024001",
    "student_name": "John Doe",
    "total_days": 20,
    "present_days": 18,
    "late_days": 2,
    "attendance_percentage": 90.0
  }
}
```

---

## Testing the API

### Using curl (Command Line)

```bash
# Get all students
curl http://127.0.0.1:8000/api/students

# Create student
curl -X POST "http://127.0.0.1:8000/api/students?student_id=TEST001&name=Test%20User"

# Upload face image
curl -X POST "http://127.0.0.1:8000/api/students/TEST001/register-face" \
  -F "file=@photo.jpg"

# Mark attendance
curl -X POST "http://127.0.0.1:8000/api/attendance/mark" \
  -F "file=@webcam.jpg"
```

### Using Python

```python
import requests

# Get students
response = requests.get('http://127.0.0.1:8000/api/students')
print(response.json())

# Create student
response = requests.post(
    'http://127.0.0.1:8000/api/students',
    params={
        'student_id': 'TEST001',
        'name': 'Test User'
    }
)
print(response.json())

# Upload face
with open('photo.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://127.0.0.1:8000/api/students/TEST001/register-face',
        files=files
    )
print(response.json())
```

### Using the Test Script

```bash
python test_api.py
```

This will test all endpoints automatically.

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "detail": "Error message here"
}
```

Common HTTP status codes:
- `200` - Success
- `400` - Bad request (invalid parameters)
- `404` - Not found
- `500` - Server error

---

## CORS

CORS is enabled for all origins in development. For production, update the `allow_origins` in `main.py`.

---

## Rate Limiting

Currently no rate limiting. Consider adding in production.

---

## Authentication

Authentication endpoints will be added in future updates.
