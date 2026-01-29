"""
Face Detection Module using MediaPipe

This module provides face detection functionality using Google's MediaPipe library.
It supports both image and video stream processing with configurable confidence thresholds.

Features:
- Real-time face detection from camera/video
- Face detection from static images
- Face cropping and preprocessing
- Bounding box extraction
- Confidence scoring

Author: Face Recognition Team
Date: January 2026
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face Detection class using MediaPipe Face Detection
    
    This class provides methods to detect faces in images and video streams
    with configurable confidence thresholds and preprocessing options.
    """
    
    def __init__(self, min_detection_confidence: float = 0.5, model_selection: int = 0):
        """
        Initialize the Face Detector
        
        Args:
            min_detection_confidence (float): Minimum confidence value [0.0, 1.0] for face detection
                                             to be considered successful. Default is 0.5.
            model_selection (int): 0 for short-range detection (within 2 meters),
                                  1 for full-range detection (within 5 meters).
                                  Default is 0 for better performance.
        """
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Create face detection object
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=self.min_detection_confidence,
            model_selection=self.model_selection
        )
        
        logger.info(f"FaceDetector initialized with confidence={min_detection_confidence}, model={model_selection}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image
        
        Args:
            image (np.ndarray): Input image in BGR format (OpenCV format)
        
        Returns:
            List[Dict]: List of detected faces with bounding boxes and confidence scores
                       Each dict contains: 'bbox', 'confidence', 'keypoints'
        """
        if image is None or image.size == 0:
            logger.warning("Empty or invalid image provided")
            return []
        
        # Convert BGR to RGB (MediaPipe uses RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_detection.process(image_rgb)
        
        detected_faces = []
        
        if results.detections:
            height, width, _ = image.shape
            
            for detection in results.detections:
                # Extract bounding box
                bboxC = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute pixel coordinates
                x = int(bboxC.xmin * width)
                y = int(bboxC.ymin * height)
                w = int(bboxC.width * width)
                h = int(bboxC.height * height)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                
                # Extract confidence score
                confidence = detection.score[0]
                
                # Extract keypoints (right eye, left eye, nose, mouth, right ear, left ear)
                keypoints = []
                if detection.location_data.relative_keypoints:
                    for keypoint in detection.location_data.relative_keypoints:
                        kp_x = int(keypoint.x * width)
                        kp_y = int(keypoint.y * height)
                        keypoints.append((kp_x, kp_y))
                
                face_data = {
                    'bbox': (x, y, w, h),  # (x, y, width, height)
                    'confidence': float(confidence),
                    'keypoints': keypoints
                }
                
                detected_faces.append(face_data)
            
            logger.info(f"Detected {len(detected_faces)} face(s)")
        else:
            logger.info("No faces detected")
        
        return detected_faces
    
    def detect_single_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect a single face in an image (returns the face with highest confidence)
        
        Args:
            image (np.ndarray): Input image in BGR format
        
        Returns:
            Optional[Dict]: Face data dict or None if no face detected
        """
        faces = self.detect_faces(image)
        
        if not faces:
            return None
        
        # Return the face with highest confidence
        return max(faces, key=lambda x: x['confidence'])
    
    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                  padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Crop face from image using bounding box with optional padding
        
        Args:
            image (np.ndarray): Input image
            bbox (Tuple): Bounding box (x, y, width, height)
            padding (float): Padding ratio to add around face (default 0.2 = 20%)
        
        Returns:
            Optional[np.ndarray]: Cropped face image or None if invalid
        """
        x, y, w, h = bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate new coordinates with padding
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        # Crop the face
        face_crop = image[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            logger.warning("Failed to crop face - empty result")
            return None
        
        return face_crop
    
    def draw_detections(self, image: np.ndarray, faces: List[Dict[str, Any]], 
                       draw_keypoints: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and keypoints on image
        
        Args:
            image (np.ndarray): Input image
            faces (List[Dict]): List of detected faces
            draw_keypoints (bool): Whether to draw facial keypoints
        
        Returns:
            np.ndarray: Image with drawn detections
        """
        output_image = image.copy()
        
        for face in faces:
            x, y, w, h = face['bbox']
            confidence = face['confidence']
            
            # Draw bounding box
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Face: {confidence:.2f}"
            cv2.putText(output_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw keypoints
            if draw_keypoints and face['keypoints']:
                for kp in face['keypoints']:
                    cv2.circle(output_image, kp, 3, (0, 0, 255), -1)
        
        return output_image
    
    def detect_from_file(self, image_path: str) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        """
        Detect faces from an image file
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            Tuple[Optional[np.ndarray], List[Dict]]: (image, detected_faces)
        """
        if not Path(image_path).exists():
            logger.error(f"Image file not found: {image_path}")
            return None, []
        
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return None, []
        
        # Detect faces
        faces = self.detect_faces(image)
        
        return image, faces
    
    def process_video_stream(self, camera_index: int = 0, 
                           display: bool = True) -> None:
        """
        Process video stream from camera with real-time face detection
        
        Args:
            camera_index (int): Camera index (default 0 for primary camera)
            display (bool): Whether to display the video feed
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_index}")
            return
        
        logger.info("Starting video stream... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Draw detections
                output_frame = self.draw_detections(frame, faces)
                
                # Display frame
                if display:
                    cv2.imshow('Face Detection', output_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested quit")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Video stream stopped")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()


# Utility functions for face preprocessing
def align_face(image: np.ndarray, keypoints: List[Tuple[int, int]]) -> Optional[np.ndarray]:
    """
    Align face based on eye positions (basic alignment)
    
    Args:
        image (np.ndarray): Input face image
        keypoints (List[Tuple]): Facial keypoints (right_eye, left_eye, ...)
    
    Returns:
        Optional[np.ndarray]: Aligned face image
    """
    if len(keypoints) < 2:
        return image
    
    # Get eye positions (keypoints[0] = right eye, keypoints[1] = left eye)
    right_eye = keypoints[0]
    left_eye = keypoints[1]
    
    # Calculate angle between eyes
    dY = left_eye[1] - right_eye[1]
    dX = left_eye[0] - right_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Get image center
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Rotate image to align eyes horizontally
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    
    return aligned


def resize_face(image: np.ndarray, target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    """
    Resize face image to target size
    
    Args:
        image (np.ndarray): Input face image
        target_size (Tuple[int, int]): Target size (width, height)
    
    Returns:
        np.ndarray: Resized face image
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_face(image: np.ndarray) -> np.ndarray:
    """
    Normalize face image to [0, 1] range
    
    Args:
        image (np.ndarray): Input face image
    
    Returns:
        np.ndarray: Normalized face image
    """
    return image.astype(np.float32) / 255.0


# Demo/Test function
if __name__ == "__main__":
    """
    Test the face detection module
    """
    print("=" * 60)
    print("FACE DETECTION MODULE TEST")
    print("=" * 60)
    
    # Initialize detector
    detector = FaceDetector(min_detection_confidence=0.5)
    
    print("\nOptions:")
    print("1. Test with webcam (real-time)")
    print("2. Test with image file")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\nStarting webcam test...")
        print("Press 'q' to quit")
        detector.process_video_stream(camera_index=0, display=True)
    
    elif choice == "2":
        image_path = input("Enter image path: ").strip()
        image, faces = detector.detect_from_file(image_path)
        
        if image is not None:
            print(f"\nDetected {len(faces)} face(s)")
            
            for i, face in enumerate(faces):
                print(f"\nFace {i+1}:")
                print(f"  Confidence: {face['confidence']:.2f}")
                print(f"  Bounding Box: {face['bbox']}")
                print(f"  Keypoints: {len(face['keypoints'])}")
            
            # Draw and display
            output = detector.draw_detections(image, faces)
            cv2.imshow('Face Detection Result', output)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    else:
        print("Invalid choice!")
    
    print("\n✓ Test completed!")
