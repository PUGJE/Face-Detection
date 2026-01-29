"""
FastAPI Application - Main Entry Point

This is the main FastAPI application for the attendance system.

Features:
- Student management endpoints
- Attendance marking endpoints
- Query and statistics endpoints
- File upload for face registration
- CORS support

Author: Face Recognition Team
Date: January 2026
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List, Optional
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import logging

from backend.database.connection import get_db, init_database
from backend.services.student_service import StudentService
from backend.services.attendance_service import AttendanceService
from backend.attendance_system import AttendanceSystem
from backend.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Face Recognition Attendance System",
    description="Smart attendance system with face recognition and anti-spoofing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")
    logger.info(f"Static files mounted from: {frontend_path / 'static'}")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and system on startup"""
    logger.info("Starting up application...")
    init_database()
    logger.info("Database initialized")
    logger.info(f"Application started on {settings.host}:{settings.port}")
    logger.info(f"Frontend available at: http://{settings.host}:{settings.port}/app")

# Initialize attendance system (singleton)
attendance_system = AttendanceSystem()

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected",
        "face_recognition": "loaded"
    }

@app.get("/app", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend application"""
    frontend_file = Path(__file__).parent / "frontend" / "index.html"
    if frontend_file.exists():
        return FileResponse(frontend_file)
    else:
        return HTMLResponse("<h1>Frontend not found</h1><p>Please ensure frontend files are in the 'frontend' directory</p>", status_code=404)

@app.get("/api/stats")
async def get_system_stats():
    """Get overall system statistics"""
    try:
        stats = attendance_system.get_system_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# STUDENT MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/api/students")
async def create_student(
    student_id: str,
    name: str,
    email: Optional[str] = None,
    enrollment_number: Optional[str] = None,
    department: Optional[str] = None,
    year: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Create a new student"""
    try:
        service = StudentService(db)
        result = service.create_student(
            student_id=student_id,
            name=name,
            email=email,
            enrollment_number=enrollment_number,
            department=department,
            year=year
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return {
            "success": True,
            "message": "Student created successfully",
            "data": result['student']
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating student: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/students")
async def get_all_students(
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """Get all students"""
    try:
        service = StudentService(db)
        students = service.get_all_students(active_only=active_only)
        
        return {
            "success": True,
            "count": len(students),
            "data": students
        }
    
    except Exception as e:
        logger.error(f"Error getting students: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/students/{student_id}")
async def get_student(
    student_id: str,
    db: Session = Depends(get_db)
):
    """Get student by ID"""
    try:
        service = StudentService(db)
        student = service.get_student(student_id)
        
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        return {
            "success": True,
            "data": student
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting student: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/students/{student_id}")
async def update_student(
    student_id: str,
    name: Optional[str] = None,
    email: Optional[str] = None,
    department: Optional[str] = None,
    year: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Update student information"""
    try:
        service = StudentService(db)
        
        # Build update dict
        updates = {}
        if name is not None:
            updates['name'] = name
        if email is not None:
            updates['email'] = email
        if department is not None:
            updates['department'] = department
        if year is not None:
            updates['year'] = year
        if is_active is not None:
            updates['is_active'] = is_active
        
        result = service.update_student(student_id, **updates)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return {
            "success": True,
            "message": "Student updated successfully",
            "data": result['student']
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating student: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/students/{student_id}")
async def delete_student(
    student_id: str,
    hard_delete: bool = False,
    db: Session = Depends(get_db)
):
    """Delete student (soft delete by default)"""
    try:
        service = StudentService(db)
        result = service.delete_student(student_id, soft_delete=not hard_delete)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return {
            "success": True,
            "message": "Student deleted successfully",
            "data": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting student: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/students/search/{query}")
async def search_students(
    query: str,
    db: Session = Depends(get_db)
):
    """Search students by name, ID, or email"""
    try:
        service = StudentService(db)
        students = service.search_students(query)
        
        return {
            "success": True,
            "query": query,
            "count": len(students),
            "data": students
        }
    
    except Exception as e:
        logger.error(f"Error searching students: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# FACE REGISTRATION ENDPOINTS
# ============================================================================

@app.post("/api/students/{student_id}/register-face")
async def register_student_face(
    student_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Register face for a student using uploaded image"""
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Check if student exists
        service = StudentService(db)
        student = service.get_student(student_id)
        
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        # Register face
        result = attendance_system.face_pipeline.register_student(student_id, image)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Update student record
        service.register_face(student_id)
        
        # Save face database
        attendance_system.face_pipeline.save_database()
        
        return {
            "success": True,
            "message": "Face registered successfully",
            "data": {
                "student_id": student_id,
                "face_confidence": result.get('face_confidence')
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering face: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ATTENDANCE ENDPOINTS
# ============================================================================

@app.post("/api/attendance/mark")
async def mark_attendance_from_upload(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Mark attendance using uploaded image"""
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Mark attendance
        result = attendance_system.mark_attendance_from_image(image)
        
        if not result['success']:
            if result.get('duplicate'):
                return {
                    "success": False,
                    "duplicate": True,
                    "message": "Attendance already marked today",
                    "data": result
                }
            else:
                raise HTTPException(status_code=400, detail=result.get('error'))
        
        return {
            "success": True,
            "message": "Attendance marked successfully",
            "data": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking attendance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/attendance/today")
async def get_today_attendance(db: Session = Depends(get_db)):
    """Get today's attendance records"""
    try:
        service = AttendanceService(db)
        records = service.get_attendance_by_date()
        
        return {
            "success": True,
            "date": datetime.now().strftime('%Y-%m-%d'),
            "count": len(records),
            "data": records
        }
    
    except Exception as e:
        logger.error(f"Error getting attendance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/attendance/date/{date}")
async def get_attendance_by_date(
    date: str,
    db: Session = Depends(get_db)
):
    """Get attendance records for a specific date (YYYY-MM-DD)"""
    try:
        service = AttendanceService(db)
        records = service.get_attendance_by_date(date)
        
        return {
            "success": True,
            "date": date,
            "count": len(records),
            "data": records
        }
    
    except Exception as e:
        logger.error(f"Error getting attendance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/attendance/student/{student_id}")
async def get_student_attendance_history(
    student_id: str,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get attendance history for a student"""
    try:
        service = AttendanceService(db)
        records = service.get_student_attendance_history(student_id, days)
        
        return {
            "success": True,
            "student_id": student_id,
            "days": days,
            "count": len(records),
            "data": records
        }
    
    except Exception as e:
        logger.error(f"Error getting student attendance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/attendance/absent")
async def get_absent_students(
    date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get list of absent students for a date"""
    try:
        service = AttendanceService(db)
        absent = service.get_absent_students(date)
        
        return {
            "success": True,
            "date": date or datetime.now().strftime('%Y-%m-%d'),
            "count": len(absent),
            "data": absent
        }
    
    except Exception as e:
        logger.error(f"Error getting absent students: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# STATISTICS ENDPOINTS
# ============================================================================

@app.get("/api/statistics/overall")
async def get_overall_statistics(db: Session = Depends(get_db)):
    """Get overall attendance statistics"""
    try:
        service = AttendanceService(db)
        stats = service.get_attendance_statistics()
        
        return {
            "success": True,
            "data": stats
        }
    
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics/student/{student_id}")
async def get_student_statistics(
    student_id: str,
    db: Session = Depends(get_db)
):
    """Get attendance statistics for a specific student"""
    try:
        service = AttendanceService(db)
        stats = service.get_attendance_statistics(student_id)
        
        if 'error' in stats:
            raise HTTPException(status_code=404, detail=stats['error'])
        
        return {
            "success": True,
            "data": stats
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting student statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )
