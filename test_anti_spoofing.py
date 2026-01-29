"""
Anti-Spoofing Test Script

Test the anti-spoofing detection with various scenarios.

Usage:
    python test_anti_spoofing.py

Author: Face Recognition Team
Date: January 2026
"""

import cv2
import numpy as np
from backend.ml.anti_spoofing import AntiSpoofingDetector
from backend.ml.liveness_detector import LivenessDetector
import time

def test_texture_analysis():
    """Test texture-based anti-spoofing"""
    print("=" * 60)
    print("TEXTURE ANALYSIS TEST")
    print("=" * 60)
    
    detector = AntiSpoofingDetector()
    
    print("\nTesting with webcam...")
    print("Press 's' to check liveness, 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display frame
        cv2.putText(frame, "Press 's' to check liveness", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Anti-Spoofing Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Check liveness
            result = detector.check_liveness(frame)
            
            print("\n" + "=" * 40)
            print("LIVENESS CHECK RESULT")
            print("=" * 40)
            print(f"Is Live: {result['is_live']}")
            print(f"Liveness Score: {result['liveness_score']:.3f}")
            print(f"Texture Score: {result['texture_score']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            if result['is_live']:
                print("✓ REAL FACE DETECTED")
                color = (0, 255, 0)
                text = "REAL FACE"
            else:
                print("✗ SPOOFING ATTACK DETECTED")
                color = (0, 0, 255)
                text = "FAKE/SPOOFING"
            
            # Show result on frame
            result_frame = frame.copy()
            cv2.putText(result_frame, text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(result_frame, f"Score: {result['liveness_score']:.2f}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow('Result', result_frame)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyWindow('Result')
    
    cap.release()
    cv2.destroyAllWindows()


def test_motion_detection():
    """Test motion-based anti-spoofing"""
    print("\n" + "=" * 60)
    print("MOTION DETECTION TEST")
    print("=" * 60)
    
    detector = AntiSpoofingDetector()
    
    print("\nCollecting frames for motion analysis...")
    print("Move your head naturally")
    print("Press 'q' to quit, 's' to check motion")
    
    cap = cv2.VideoCapture(0)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add frame to buffer
        detector.add_frame(frame)
        frame_count += 1
        
        # Display frame
        cv2.putText(frame, f"Frames collected: {len(detector.frame_buffer)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 's' to check motion", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Motion Detection Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Check motion
            motion_result = detector.check_motion()
            
            if motion_result['success']:
                print("\n" + "=" * 40)
                print("MOTION ANALYSIS RESULT")
                print("=" * 40)
                print(f"Has Natural Motion: {motion_result['has_motion']}")
                print(f"Motion Score: {motion_result['motion_score']:.3f}")
                print(f"Average Movement: {motion_result['avg_motion']:.2f}")
                print(f"Movement Variance: {motion_result['motion_variance']:.2f}")
                
                if motion_result['has_motion']:
                    print("✓ NATURAL MOTION DETECTED")
                else:
                    print("✗ SUSPICIOUS MOTION PATTERN")
    
    cap.release()
    cv2.destroyAllWindows()


def test_combined_detection():
    """Test combined texture + motion detection"""
    print("\n" + "=" * 60)
    print("COMBINED DETECTION TEST")
    print("=" * 60)
    
    detector = AntiSpoofingDetector()
    
    print("\nTesting combined anti-spoofing...")
    print("Move naturally and press 's' to test")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add frame for motion tracking
        detector.add_frame(frame)
        
        # Display frame
        cv2.putText(frame, "Combined Anti-Spoofing Test", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Frames: {len(detector.frame_buffer)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow('Combined Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Check texture
            texture_result = detector.check_liveness(frame)
            
            # Check motion
            motion_result = detector.check_motion()
            
            print("\n" + "=" * 50)
            print("COMBINED ANTI-SPOOFING RESULT")
            print("=" * 50)
            
            print("\nTexture Analysis:")
            print(f"  Is Live: {texture_result['is_live']}")
            print(f"  Score: {texture_result['liveness_score']:.3f}")
            
            if motion_result['success']:
                print("\nMotion Analysis:")
                print(f"  Has Motion: {motion_result['has_motion']}")
                print(f"  Score: {motion_result['motion_score']:.3f}")
                
                # Combined decision - favor texture over motion
                combined_score = (
                    0.7 * texture_result['liveness_score'] +
                    0.3 * motion_result['motion_score']
                )
                is_live = combined_score > 0.5  # Lowered threshold
            else:
                combined_score = texture_result['liveness_score']
                is_live = texture_result['is_live']
            
            print(f"\nCombined Score: {combined_score:.3f}")
            print(f"Final Decision: {'✓ REAL FACE' if is_live else '✗ SPOOFING DETECTED'}")
            
            # Show result
            result_frame = frame.copy()
            color = (0, 255, 0) if is_live else (0, 0, 255)
            text = "REAL" if is_live else "FAKE"
            cv2.putText(result_frame, text, (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
            cv2.imshow('Result', result_frame)
            cv2.waitKey(2000)
            cv2.destroyWindow('Result')
    
    cap.release()
    cv2.destroyAllWindows()


def print_instructions():
    """Print test instructions"""
    print("\n" + "=" * 60)
    print("ANTI-SPOOFING TEST INSTRUCTIONS")
    print("=" * 60)
    print("\nThis script will test anti-spoofing detection.")
    print("\nTest Scenarios:")
    print("1. Real Face - Look at camera normally")
    print("2. Printed Photo - Hold a printed photo in front of camera")
    print("3. Phone Screen - Display a photo on phone screen")
    print("4. Video Replay - Play a video of yourself")
    print("\nFor best results:")
    print("- Ensure good lighting")
    print("- Face camera directly")
    print("- Move naturally (for motion tests)")
    print("\nPress Enter to continue...")
    input()


def main():
    """Run all tests"""
    print_instructions()
    
    print("\n" + "=" * 60)
    print("SELECT TEST")
    print("=" * 60)
    print("\n1. Texture Analysis Only")
    print("2. Motion Detection Only")
    print("3. Combined Detection (Recommended)")
    print("4. Run All Tests")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        test_texture_analysis()
    elif choice == '2':
        test_motion_detection()
    elif choice == '3':
        test_combined_detection()
    elif choice == '4':
        test_texture_analysis()
        test_motion_detection()
        test_combined_detection()
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "=" * 60)
    print("✓ ANTI-SPOOFING TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
