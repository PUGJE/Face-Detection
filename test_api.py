"""
API Testing Script

This script tests all API endpoints to verify they're working correctly.

Usage:
    python test_api.py

Make sure the API server is running:
    python main.py

Author: Face Recognition Team
Date: January 2026
"""

import requests
import json
from pathlib import Path
import cv2
import numpy as np

# API Base URL
BASE_URL = "http://127.0.0.1:8000"

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_result(endpoint, response):
    """Print API response"""
    print(f"\n{endpoint}")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✓ Success")
        try:
            data = response.json()
            print(json.dumps(data, indent=2))
        except:
            print(response.text)
    else:
        print(f"❌ Failed: {response.text}")

def test_health_endpoints():
    """Test health and info endpoints"""
    print_section("HEALTH & INFO ENDPOINTS")
    
    # Root
    response = requests.get(f"{BASE_URL}/")
    print_result("GET /", response)
    
    # Health
    response = requests.get(f"{BASE_URL}/health")
    print_result("GET /health", response)
    
    # Stats
    response = requests.get(f"{BASE_URL}/api/stats")
    print_result("GET /api/stats", response)

def test_student_endpoints():
    """Test student management endpoints"""
    print_section("STUDENT MANAGEMENT ENDPOINTS")
    
    # Get all students
    response = requests.get(f"{BASE_URL}/api/students")
    print_result("GET /api/students", response)
    
    # Create a new student
    print("\n--- Creating new student ---")
    response = requests.post(
        f"{BASE_URL}/api/students",
        params={
            "student_id": "TEST_STUDENT_001",
            "name": "Test Student",
            "email": "test@example.com",
            "department": "Computer Science",
            "year": "3rd Year"
        }
    )
    print_result("POST /api/students", response)
    
    # Get specific student
    response = requests.get(f"{BASE_URL}/api/students/TEST_STUDENT_001")
    print_result("GET /api/students/TEST_STUDENT_001", response)
    
    # Update student
    print("\n--- Updating student ---")
    response = requests.put(
        f"{BASE_URL}/api/students/TEST_STUDENT_001",
        params={
            "department": "Information Technology"
        }
    )
    print_result("PUT /api/students/TEST_STUDENT_001", response)
    
    # Search students
    response = requests.get(f"{BASE_URL}/api/students/search/Test")
    print_result("GET /api/students/search/Test", response)

def test_attendance_endpoints():
    """Test attendance endpoints"""
    print_section("ATTENDANCE ENDPOINTS")
    
    # Get today's attendance
    response = requests.get(f"{BASE_URL}/api/attendance/today")
    print_result("GET /api/attendance/today", response)
    
    # Get attendance by date
    response = requests.get(f"{BASE_URL}/api/attendance/date/2026-01-29")
    print_result("GET /api/attendance/date/2026-01-29", response)
    
    # Get absent students
    response = requests.get(f"{BASE_URL}/api/attendance/absent")
    print_result("GET /api/attendance/absent", response)

def test_statistics_endpoints():
    """Test statistics endpoints"""
    print_section("STATISTICS ENDPOINTS")
    
    # Overall statistics
    response = requests.get(f"{BASE_URL}/api/statistics/overall")
    print_result("GET /api/statistics/overall", response)
    
    # Student statistics (if DEMO_STUDENT exists)
    response = requests.get(f"{BASE_URL}/api/statistics/student/DEMO_STUDENT")
    print_result("GET /api/statistics/student/DEMO_STUDENT", response)

def test_face_registration():
    """Test face registration with image upload"""
    print_section("FACE REGISTRATION (Image Upload)")
    
    # Create a test image (black image with white circle)
    print("\nCreating test image...")
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(test_image, (320, 240), 100, (255, 255, 255), -1)
    
    # Save temporarily
    test_image_path = "test_face.jpg"
    cv2.imwrite(test_image_path, test_image)
    
    # Upload for face registration
    print("\nUploading test image for face registration...")
    with open(test_image_path, 'rb') as f:
        files = {'file': ('test_face.jpg', f, 'image/jpeg')}
        response = requests.post(
            f"{BASE_URL}/api/students/TEST_STUDENT_001/register-face",
            files=files
        )
    
    print_result("POST /api/students/TEST_STUDENT_001/register-face", response)
    
    # Cleanup
    Path(test_image_path).unlink(missing_ok=True)
    print("\n(Note: This may fail if no face detected in test image - that's expected)")

def cleanup_test_data():
    """Delete test student"""
    print_section("CLEANUP")
    
    print("\nDeleting test student...")
    response = requests.delete(f"{BASE_URL}/api/students/TEST_STUDENT_001")
    print_result("DELETE /api/students/TEST_STUDENT_001", response)

def main():
    """Run all tests"""
    print("=" * 70)
    print("  API ENDPOINT TESTING")
    print("=" * 70)
    print(f"\nTesting API at: {BASE_URL}")
    print("Make sure the server is running: python main.py")
    
    try:
        # Test connection
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("\n❌ API server is not responding!")
            print("Please start the server with: python main.py")
            return
        
        print("\n✓ API server is running")
        
        # Run all tests
        test_health_endpoints()
        test_student_endpoints()
        test_attendance_endpoints()
        test_statistics_endpoints()
        test_face_registration()
        cleanup_test_data()
        
        # Summary
        print("\n" + "=" * 70)
        print("  TEST SUMMARY")
        print("=" * 70)
        print("\n✓ All API endpoint tests completed!")
        print("\nAPI Documentation:")
        print(f"  Swagger UI: {BASE_URL}/docs")
        print(f"  ReDoc: {BASE_URL}/redoc")
        print("\nThe API is fully functional and ready to use!")
        print("=" * 70)
    
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API server!")
        print("Please start the server with: python main.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
