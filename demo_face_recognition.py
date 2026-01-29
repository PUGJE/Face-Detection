"""
Face Recognition Demo Application

Interactive demo for testing face registration and recognition.

Usage:
    python demo_face_recognition.py --mode register --id STUDENT_001
    python demo_face_recognition.py --mode recognize
    python demo_face_recognition.py --mode verify --id STUDENT_001

Author: Face Recognition Team
Date: January 2026
"""

import argparse
import cv2
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.ml.recognition_pipeline import FaceRecognitionPipeline
from backend.config import settings


def demo_register(student_id: str):
    """
    Register a student using webcam
    
    Args:
        student_id (str): Student ID to register
    """
    print("=" * 60)
    print("STUDENT REGISTRATION")
    print("=" * 60)
    print(f"Student ID: {student_id}")
    print("\nInstructions:")
    print("- Look at the camera")
    print("- Press SPACE to capture")
    print("- Press 'q' to quit")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = FaceRecognitionPipeline()
    
    # Load existing database
    pipeline.load_database()
    
    # Open webcam
    cap = cv2.VideoCapture(settings.camera_index)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera")
        return
    
    registered = False
    
    try:
        while not registered:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Detect face for preview
            faces = pipeline.detector.detect_faces(frame)
            preview = pipeline.detector.draw_detections(frame, faces)
            
            # Add instruction text
            cv2.putText(preview, "Press SPACE to capture", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(preview, f"Student ID: {student_id}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Registration', preview)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                print("\n📸 Capturing...")
                
                # Register student
                result = pipeline.register_student(student_id, frame)
                
                if result['success']:
                    print(f"✓ Student {student_id} registered successfully!")
                    print(f"  Face confidence: {result['face_confidence']:.3f}")
                    
                    # Save database
                    if pipeline.save_database():
                        print(f"✓ Database saved")
                    
                    registered = True
                else:
                    print(f"❌ Registration failed: {result.get('error', 'Unknown error')}")
                    print("  Please try again...")
            
            elif key == ord('q'):
                print("\n✓ Registration cancelled")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def demo_recognize():
    """
    Recognize students using webcam
    """
    print("=" * 60)
    print("STUDENT RECOGNITION")
    print("=" * 60)
    print("\nInstructions:")
    print("- Look at the camera")
    print("- Press 'q' to quit")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = FaceRecognitionPipeline()
    
    # Load database
    if not pipeline.load_database():
        print("❌ No database found. Please register students first.")
        return
    
    stats = pipeline.get_stats()
    print(f"\nLoaded {stats['recognizer']['total_faces']} student(s)")
    print(f"Students: {', '.join(stats['recognizer']['student_ids'])}")
    print()
    
    # Open webcam
    cap = cv2.VideoCapture(settings.camera_index)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Process frame for recognition
            results = pipeline.process_attendance_frame(frame)
            
            # Draw results
            output = pipeline.draw_recognition_results(frame, results)
            
            # Add info text
            recognized_count = sum(1 for r in results if r['recognized'])
            info_text = f"Detected: {len(results)} | Recognized: {recognized_count}"
            cv2.putText(output, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Face Recognition', output)
            
            # Print recognition results
            for result in results:
                if result['recognized']:
                    student_id = result['student_id']
                    confidence = result['confidence']
                    print(f"✓ Recognized: {student_id} (confidence: {confidence:.3f})")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✓ Recognition stopped")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def demo_verify(student_id: str):
    """
    Verify a specific student using webcam
    
    Args:
        student_id (str): Student ID to verify
    """
    print("=" * 60)
    print("STUDENT VERIFICATION")
    print("=" * 60)
    print(f"Verifying: {student_id}")
    print("\nInstructions:")
    print("- Look at the camera")
    print("- Press SPACE to verify")
    print("- Press 'q' to quit")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = FaceRecognitionPipeline()
    
    # Load database
    if not pipeline.load_database():
        print("❌ No database found. Please register students first.")
        return
    
    # Check if student exists
    stats = pipeline.get_stats()
    if student_id not in stats['recognizer']['student_ids']:
        print(f"❌ Student {student_id} not found in database")
        print(f"Available students: {', '.join(stats['recognizer']['student_ids'])}")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(settings.camera_index)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Detect face for preview
            faces = pipeline.detector.detect_faces(frame)
            preview = pipeline.detector.draw_detections(frame, faces)
            
            # Add instruction text
            cv2.putText(preview, "Press SPACE to verify", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(preview, f"Verifying: {student_id}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Verification', preview)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to verify
                print("\n🔍 Verifying...")
                
                # Verify student
                result = pipeline.verify_student(student_id, frame)
                
                if result['success']:
                    if result['verified']:
                        print(f"✓ VERIFIED: This is {student_id}")
                        print(f"  Distance: {result['distance']:.3f}")
                    else:
                        print(f"❌ NOT VERIFIED: This is NOT {student_id}")
                        print(f"  Distance: {result['distance']:.3f}")
                else:
                    print(f"❌ Verification failed: {result.get('error', 'Unknown error')}")
            
            elif key == ord('q'):
                print("\n✓ Verification stopped")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Face Recognition Demo Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register a student
  python demo_face_recognition.py --mode register --id STUDENT_001
  
  # Recognize students
  python demo_face_recognition.py --mode recognize
  
  # Verify a specific student
  python demo_face_recognition.py --mode verify --id STUDENT_001
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['register', 'recognize', 'verify'],
        required=True,
        help='Demo mode'
    )
    
    parser.add_argument(
        '--id',
        type=str,
        help='Student ID (required for register and verify modes)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['register', 'verify'] and not args.id:
        parser.error(f"--id is required for {args.mode} mode")
    
    # Run demo
    try:
        if args.mode == 'register':
            demo_register(args.id)
        elif args.mode == 'recognize':
            demo_recognize()
        elif args.mode == 'verify':
            demo_verify(args.id)
    
    except KeyboardInterrupt:
        print("\n\n✓ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
