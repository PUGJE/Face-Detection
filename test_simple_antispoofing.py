"""
Simple Anti-Spoofing Test

Quick test for texture-based anti-spoofing detection.

Usage:
    python test_simple_antispoofing.py

Author: Face Recognition Team
Date: January 2026
"""

import cv2
from backend.ml.anti_spoofing import AntiSpoofingDetector

def main():
    print("=" * 60)
    print("SIMPLE ANTI-SPOOFING TEST")
    print("=" * 60)
    print("\nThis test uses TEXTURE ANALYSIS ONLY")
    print("(Motion detection disabled for reliability)")
    print("\nInstructions:")
    print("- Press 's' to check liveness")
    print("- Press 'q' to quit")
    print("\nTest scenarios:")
    print("1. Your real face (should PASS with score > 0.60)")
    print("2. Photo on phone screen (should FAIL with score < 0.60)")
    print("3. Printed photo (should FAIL with score < 0.60)")
    print("\n" + "=" * 60)
    
    # Initialize detector
    detector = AntiSpoofingDetector(texture_threshold=0.60)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return
    
    print("\nWebcam opened successfully!")
    print("Press 's' to test, 'q' to quit\n")
    
    test_count = 0
    pass_count = 0
    fail_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display instructions on frame
        display_frame = frame.copy()
        cv2.putText(display_frame, "Anti-Spoofing Test", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(display_frame, "Press 's' to test, 'q' to quit", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Tests: {test_count} | Pass: {pass_count} | Fail: {fail_count}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Anti-Spoofing Test', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # First, detect face
            from backend.ml.face_detection import FaceDetector
            face_detector = FaceDetector()
            
            face_data = face_detector.detect_single_face(frame)
            
            if face_data is None:
                print("\n" + "=" * 50)
                print(f"TEST #{test_count + 1}")
                print("=" * 50)
                print("ERROR: No face detected!")
                print("Please ensure your face is visible in the camera.")
                
                # Show error on screen
                error_frame = frame.copy()
                cv2.putText(error_frame, "NO FACE DETECTED", (50, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(error_frame, "Show your face to camera", (50, 260),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow('Result', error_frame)
                cv2.waitKey(2000)
                cv2.destroyWindow('Result')
                continue
            
            # Face detected, now check liveness
            bbox = face_data['bbox']
            test_count += 1
            
            # Check liveness with face region
            result = detector.check_liveness(frame, face_region=bbox)
            
            print("\n" + "=" * 50)
            print(f"TEST #{test_count}")
            print("=" * 50)
            print(f"Face Detection: ✓ (confidence: {face_data['confidence']:.3f})")
            
            if result['success']:
                is_live = result['is_live']
                score = result['liveness_score']
                texture = result['texture_score']
                
                print(f"Liveness Score: {score:.3f}")
                print(f"Texture Score:  {texture:.3f}")
                print(f"Threshold:      0.600")
                print(f"Result:         {'✓ REAL FACE (PASS)' if is_live else '✗ FAKE/SPOOFING (FAIL)'}")
                
                if is_live:
                    pass_count += 1
                    color = (0, 255, 0)  # Green
                    text = "REAL FACE"
                else:
                    fail_count += 1
                    color = (0, 0, 255)  # Red
                    text = "FAKE/SPOOFING"
                
                # Show result on screen with bounding box
                result_frame = frame.copy()
                x, y, w, h = bbox
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(result_frame, text, (50, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
                cv2.putText(result_frame, f"Score: {score:.3f}", (50, 280),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                cv2.putText(result_frame, f"Threshold: 0.600", (50, 340),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                
                cv2.imshow('Result', result_frame)
                cv2.waitKey(2000)  # Show for 2 seconds
                cv2.destroyWindow('Result')
            else:
                print(f"ERROR: {result.get('error', 'Unknown error')}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {test_count}")
    print(f"Passed:      {pass_count} ({pass_count/test_count*100 if test_count > 0 else 0:.1f}%)")
    print(f"Failed:      {fail_count} ({fail_count/test_count*100 if test_count > 0 else 0:.1f}%)")
    print("=" * 60)
    print("\n✓ Test completed!")


if __name__ == "__main__":
    main()
