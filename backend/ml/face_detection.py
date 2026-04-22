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
from typing import Any, Dict, List, Optional

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
    
