"""
Face Detection Module — OpenCV Haar Cascade

Provides face detection using OpenCV's built-in Haar Cascade classifier.
This is kept as a fast, dependency-free fallback for when InsightFace
(RetinaFace) is unavailable.

Note: InsightFace (recognition_pipeline.py) is the primary detection path
in production. This module is used only during registration when InsightFace
fails, and for standalone testing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# NOTE: Do NOT call logging.basicConfig() here.
# Logging configuration is the responsibility of the application entry point (main.py).
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face Detection class using OpenCV Cascade Classifier
    
    This class maintains the same API as the old MediaPipe implementation
    but operates completely independently natively.
    """
    
    def __init__(self, min_detection_confidence: float = 0.5, model_selection: int = 0):
        self.min_detection_confidence = min_detection_confidence
        
        # Load OpenCV native frontal-face Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        
        if self.detector.empty():
            logger.error("Failed to load native Haar cascade from OpenCV!")
            
        logger.info("FaceDetector initialized smoothly via native OpenCV.")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image natively via Haar cascade
        """
        if image is None or image.size == 0:
            logger.warning("Empty or invalid image provided")
            return []
        
        # Convert to grayscale for Haar detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Map original confidence threshold to Haar's minNeighbors
        min_neighbors = int(3 + (self.min_detection_confidence * 4))
        
        faces = self.detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=max(3, min_neighbors), 
            minSize=(40, 40)
        )
        
        detected_faces = []
        for (x, y, w, h) in faces:
            # Haar cascades don't expose strict 0-1 floats, map a mock pseudo-confidence
            confidence = min(0.99, 0.85 + (min_neighbors * 0.01))
            
            face_data = {
                'bbox': (int(x), int(y), int(w), int(h)),
                'confidence': float(confidence),
                'keypoints': []  # Native cascades do not generate facial landmarks out-of-the-box
            }
            detected_faces.append(face_data)
            
        if detected_faces:
            logger.info(f"Native Detector found {len(detected_faces)} face(s)")
            
        return detected_faces
    
    def detect_single_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        faces = self.detect_faces(image)
        if not faces:
            return None
        # Return largest bounding box dynamically
        return max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
    
    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                  padding: float = 0.2) -> Optional[np.ndarray]:
        x, y, w, h = bbox
        
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        return face_crop
    
    def draw_detections(self, image: np.ndarray, faces: List[Dict[str, Any]], 
                       draw_keypoints: bool = True) -> np.ndarray:
        output_image = image.copy()
        for face in faces:
            x, y, w, h = face['bbox']
            confidence = face['confidence']
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Face: {confidence:.2f}"
            cv2.putText(output_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return output_image
    
    def detect_from_file(self, image_path: str) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        if not Path(image_path).exists():
            return None, []
        image = cv2.imread(image_path)
        if image is None:
            return None, []
        return image, self.detect_faces(image)
        
    def process_video_stream(self, camera_index: int = 0, display: bool = True) -> None:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return
            
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                faces = self.detect_faces(frame)
                output_frame = self.draw_detections(frame, faces)
                
                if display:
                    cv2.imshow('Face Detection', output_frame)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
    def __del__(self):
        pass


def align_face(
    image: np.ndarray,
    keypoints: List[Tuple[int, int]],
) -> np.ndarray:
    """
    No-op alignment shim.

    Haar cascades do not produce facial landmarks, so this function simply
    returns the image unchanged. Kept for API compatibility.
    """
    return image


def resize_face(
    image: np.ndarray,
    target_size: Tuple[int, int] = (160, 160),
) -> np.ndarray:
    """Resize a face crop to ``target_size`` using area interpolation."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_face(image: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0.0, 1.0]."""
    return image.astype(np.float32) / 255.0
