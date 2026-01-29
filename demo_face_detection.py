"""
Face Detection Demo Application

A simple command-line demo to test face detection functionality.
Supports webcam and image file testing.

Usage:
    python demo_face_detection.py --mode webcam
    python demo_face_detection.py --mode image --path "path/to/image.jpg"

Author: Face Recognition Team
Date: January 2026
"""

import argparse
import cv2
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.ml.face_detection import FaceDetector
from backend.config import settings


def demo_webcam(confidence: float = 0.5):
    """
    Run face detection demo with webcam
    
    Args:
        confidence (float): Minimum detection confidence
    """
    print("=" * 60)
    print("FACE DETECTION - WEBCAM DEMO")
    print("=" * 60)
    print(f"Confidence Threshold: {confidence}")
    print("Press 'q' to quit")
    print("=" * 60)
    
    # Initialize detector
    detector = FaceDetector(min_detection_confidence=confidence)
    
    # Open webcam
    cap = cv2.VideoCapture(settings.camera_index)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {settings.camera_index}")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            # Draw detections
            output_frame = detector.draw_detections(frame, faces, draw_keypoints=True)
            
            # Add info text
            info_text = f"Faces: {len(faces)} | Frame: {frame_count}"
            cv2.putText(output_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Face Detection Demo', output_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✓ User quit")
                break
            elif key == ord('s') and len(faces) > 0:
                # Save screenshot
                filename = f"face_detection_screenshot_{frame_count}.jpg"
                cv2.imwrite(filename, output_frame)
                print(f"✓ Screenshot saved: {filename}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Processed {frame_count} frames")


def demo_image(image_path: str, confidence: float = 0.5):
    """
    Run face detection demo with image file
    
    Args:
        image_path (str): Path to image file
        confidence (float): Minimum detection confidence
    """
    print("=" * 60)
    print("FACE DETECTION - IMAGE DEMO")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Confidence Threshold: {confidence}")
    print("=" * 60)
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return
    
    # Initialize detector
    detector = FaceDetector(min_detection_confidence=confidence)
    
    # Detect faces
    image, faces = detector.detect_from_file(image_path)
    
    if image is None:
        print("❌ Error: Failed to load image")
        return
    
    print(f"\n✓ Detected {len(faces)} face(s)")
    
    # Display face details
    for i, face in enumerate(faces):
        print(f"\nFace {i+1}:")
        print(f"  Confidence: {face['confidence']:.3f}")
        print(f"  Bounding Box: {face['bbox']}")
        print(f"  Keypoints: {len(face['keypoints'])}")
        
        # Crop and save face
        cropped = detector.crop_face(image, face['bbox'], padding=0.2)
        if cropped is not None:
            crop_filename = f"face_{i+1}_cropped.jpg"
            cv2.imwrite(crop_filename, cropped)
            print(f"  Saved cropped face: {crop_filename}")
    
    # Draw detections
    output = detector.draw_detections(image, faces, draw_keypoints=True)
    
    # Save output
    output_filename = "face_detection_result.jpg"
    cv2.imwrite(output_filename, output)
    print(f"\n✓ Result saved: {output_filename}")
    
    # Display
    print("\nPress any key to close window...")
    cv2.imshow('Face Detection Result', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Face Detection Demo Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam demo
  python demo_face_detection.py --mode webcam
  
  # Image demo
  python demo_face_detection.py --mode image --path photo.jpg
  
  # Custom confidence
  python demo_face_detection.py --mode webcam --confidence 0.7
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['webcam', 'image'],
        required=True,
        help='Demo mode: webcam or image'
    )
    
    parser.add_argument(
        '--path',
        type=str,
        help='Path to image file (required for image mode)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Minimum detection confidence (0.0-1.0, default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'image' and not args.path:
        parser.error("--path is required for image mode")
    
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0.0 and 1.0")
    
    # Run demo
    try:
        if args.mode == 'webcam':
            demo_webcam(confidence=args.confidence)
        else:
            demo_image(image_path=args.path, confidence=args.confidence)
    
    except KeyboardInterrupt:
        print("\n\n✓ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
