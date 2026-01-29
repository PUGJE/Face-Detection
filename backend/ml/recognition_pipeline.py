"""
Integrated Face Recognition Pipeline

This module combines face detection and face recognition into a complete pipeline
for the attendance system.

Author: Face Recognition Team
Date: January 2026
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

from backend.ml.face_detection import FaceDetector
from backend.ml.face_recognition import FaceRecognizer
from backend.ml.anti_spoofing import AntiSpoofingDetector
from backend.config import settings

logger = logging.getLogger(__name__)


class FaceRecognitionPipeline:
    """
    Complete face recognition pipeline combining detection and recognition
    
    This class provides end-to-end functionality for:
    - Detecting faces in images/video
    - Recognizing detected faces
    - Managing the recognition database
    """
    
    def __init__(self,
                 detection_confidence: float = None,
                 recognition_threshold: float = None,
                 model_name: str = "Facenet",
                 enable_anti_spoofing: bool = True):
        """
        Initialize the face recognition pipeline
        
        Args:
            detection_confidence (float): Face detection confidence threshold
            recognition_threshold (float): Face recognition distance threshold
            model_name (str): Face recognition model name
            enable_anti_spoofing (bool): Enable anti-spoofing detection
        """
        # Use config values if not provided
        if detection_confidence is None:
            detection_confidence = settings.face_detection_confidence
        if recognition_threshold is None:
            recognition_threshold = settings.face_recognition_threshold
        
        # Initialize face detector
        self.detector = FaceDetector(min_detection_confidence=detection_confidence)
        
        # Initialize face recognizer
        self.recognizer = FaceRecognizer(
            model_name=model_name,
            distance_metric="cosine",
            recognition_threshold=recognition_threshold
        )
        
        # Initialize anti-spoofing detector
        self.enable_anti_spoofing = enable_anti_spoofing
        if self.enable_anti_spoofing:
            self.anti_spoof = AntiSpoofingDetector(
                texture_threshold=0.60,  # Balanced security and usability
                motion_threshold=0.4
            )
            logger.info("Anti-spoofing enabled")
        else:
            self.anti_spoof = None
            logger.info("Anti-spoofing disabled")
        
        logger.info("Face Recognition Pipeline initialized")
    
    def register_student(self, student_id: str, image: np.ndarray) -> Dict[str, Any]:
        """
        Register a new student by detecting and storing their face
        
        Args:
            student_id (str): Unique student identifier
            image (np.ndarray): Image containing student's face
        
        Returns:
            Dict: Registration result with status and details
        """
        # Detect face in image
        face_data = self.detector.detect_single_face(image)
        
        if face_data is None:
            return {
                'success': False,
                'error': 'No face detected in image',
                'student_id': student_id
            }
        
        # Crop face
        face_crop = self.detector.crop_face(image, face_data['bbox'], padding=0.2)
        
        if face_crop is None:
            return {
                'success': False,
                'error': 'Failed to crop face',
                'student_id': student_id
            }
        
        # Add to recognition database
        success = self.recognizer.add_face_to_database(student_id, face_crop)
        
        if success:
            return {
                'success': True,
                'student_id': student_id,
                'face_confidence': face_data['confidence'],
                'bbox': face_data['bbox']
            }
        else:
            return {
                'success': False,
                'error': 'Failed to generate face embedding',
                'student_id': student_id
            }
    
    def recognize_student(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Recognize a student from an image
        
        Args:
            image (np.ndarray): Image containing student's face
        
        Returns:
            Dict: Recognition result with student_id and confidence
        """
        # Detect face
        face_data = self.detector.detect_single_face(image)
        
        if face_data is None:
            return {
                'success': False,
                'error': 'No face detected',
                'recognized': False
            }
        
        # Crop face
        face_crop = self.detector.crop_face(image, face_data['bbox'], padding=0.2)
        
        if face_crop is None:
            return {
                'success': False,
                'error': 'Failed to crop face',
                'recognized': False
            }
        
        # Recognize face
        recognition_result = self.recognizer.recognize_face(face_crop)
        
        if recognition_result is None:
            return {
                'success': True,
                'recognized': False,
                'error': 'Face not recognized',
                'detection_confidence': face_data['confidence']
            }
        
        return {
            'success': True,
            'recognized': True,
            'student_id': recognition_result['student_id'],
            'recognition_confidence': recognition_result['confidence'],
            'recognition_distance': recognition_result['distance'],
            'detection_confidence': face_data['confidence'],
            'bbox': face_data['bbox']
        }
    
    def recognize_with_liveness(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Recognize a student with liveness detection (anti-spoofing)
        
        Args:
            image (np.ndarray): Image containing student's face
        
        Returns:
            Dict: Recognition result with liveness check
        """
        # Detect face
        face_data = self.detector.detect_single_face(image)
        
        if face_data is None:
            return {
                'success': False,
                'error': 'No face detected',
                'recognized': False,
                'is_live': False
            }
        
        # Check liveness if enabled
        if self.enable_anti_spoofing and self.anti_spoof:
            liveness_result = self.anti_spoof.check_liveness(
                image,
                face_region=face_data['bbox']
            )
            
            if not liveness_result.get('is_live', False):
                return {
                    'success': True,
                    'recognized': False,
                    'is_live': False,
                    'liveness_score': liveness_result.get('liveness_score', 0),
                    'error': 'Liveness check failed - possible spoofing attack',
                    'detection_confidence': face_data['confidence']
                }
        else:
            liveness_result = {'is_live': True, 'liveness_score': 1.0}
        
        # Crop face
        face_crop = self.detector.crop_face(image, face_data['bbox'], padding=0.2)
        
        if face_crop is None:
            return {
                'success': False,
                'error': 'Failed to crop face',
                'recognized': False,
                'is_live': liveness_result['is_live']
            }
        
        # Recognize face
        recognition_result = self.recognizer.recognize_face(face_crop)
        
        if recognition_result is None:
            return {
                'success': True,
                'recognized': False,
                'is_live': liveness_result['is_live'],
                'liveness_score': liveness_result.get('liveness_score', 1.0),
                'error': 'Face not recognized',
                'detection_confidence': face_data['confidence']
            }
        
        return {
            'success': True,
            'recognized': True,
            'is_live': liveness_result['is_live'],
            'liveness_score': liveness_result.get('liveness_score', 1.0),
            'student_id': recognition_result['student_id'],
            'recognition_confidence': recognition_result['confidence'],
            'recognition_distance': recognition_result['distance'],
            'detection_confidence': face_data['confidence'],
            'bbox': face_data['bbox']
        }
    
    def verify_student(self, student_id: str, image: np.ndarray) -> Dict[str, Any]:
        """
        Verify if an image matches a specific student (1:1 verification)
        
        Args:
            student_id (str): Student ID to verify against
            image (np.ndarray): Image to verify
        
        Returns:
            Dict: Verification result
        """
        # Detect face
        face_data = self.detector.detect_single_face(image)
        
        if face_data is None:
            return {
                'success': False,
                'verified': False,
                'error': 'No face detected'
            }
        
        # Crop face
        face_crop = self.detector.crop_face(image, face_data['bbox'], padding=0.2)
        
        if face_crop is None:
            return {
                'success': False,
                'verified': False,
                'error': 'Failed to crop face'
            }
        
        # Verify face
        is_verified, distance = self.recognizer.verify_face(face_crop, student_id)
        
        return {
            'success': True,
            'verified': is_verified,
            'student_id': student_id,
            'distance': float(distance),
            'detection_confidence': face_data['confidence']
        }
    
    def process_attendance_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process a video frame for attendance marking
        Detects all faces and attempts to recognize each one
        
        Args:
            frame (np.ndarray): Video frame
        
        Returns:
            List[Dict]: List of recognition results for each detected face
        """
        # Detect all faces
        faces = self.detector.detect_faces(frame)
        
        results = []
        
        for face_data in faces:
            # Crop face
            face_crop = self.detector.crop_face(frame, face_data['bbox'], padding=0.2)
            
            if face_crop is None:
                continue
            
            # Try to recognize
            recognition_result = self.recognizer.recognize_face(face_crop)
            
            if recognition_result:
                results.append({
                    'recognized': True,
                    'student_id': recognition_result['student_id'],
                    'confidence': recognition_result['confidence'],
                    'distance': recognition_result['distance'],
                    'bbox': face_data['bbox']
                })
            else:
                results.append({
                    'recognized': False,
                    'bbox': face_data['bbox'],
                    'detection_confidence': face_data['confidence']
                })
        
        return results
    
    def draw_recognition_results(self, image: np.ndarray, 
                                results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw recognition results on image
        
        Args:
            image (np.ndarray): Input image
            results (List[Dict]): Recognition results
        
        Returns:
            np.ndarray: Image with drawn results
        """
        output = image.copy()
        
        for result in results:
            bbox = result['bbox']
            x, y, w, h = bbox
            
            if result['recognized']:
                # Green box for recognized faces
                color = (0, 255, 0)
                student_id = result['student_id']
                confidence = result['confidence']
                label = f"{student_id} ({confidence:.2f})"
            else:
                # Red box for unrecognized faces
                color = (0, 0, 255)
                label = "Unknown"
            
            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            cv2.putText(output, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return output
    
    def save_database(self, file_path: str = None) -> bool:
        """
        Save the recognition database
        
        Args:
            file_path (str): Path to save database (uses default if None)
        
        Returns:
            bool: True if successful
        """
        if file_path is None:
            file_path = f"{settings.embeddings_path}/face_embeddings.pkl"
        
        return self.recognizer.save_embeddings(file_path)
    
    def load_database(self, file_path: str = None) -> bool:
        """
        Load the recognition database
        
        Args:
            file_path (str): Path to load database from (uses default if None)
        
        Returns:
            bool: True if successful
        """
        if file_path is None:
            file_path = f"{settings.embeddings_path}/face_embeddings.pkl"
        
        return self.recognizer.load_embeddings(file_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics
        
        Returns:
            Dict: Statistics about the pipeline
        """
        return {
            'detector': {
                'confidence_threshold': self.detector.min_detection_confidence,
                'model_selection': self.detector.model_selection
            },
            'recognizer': self.recognizer.get_database_stats()
        }


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("FACE RECOGNITION PIPELINE TEST")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = FaceRecognitionPipeline()
    
    print("\nPipeline initialized successfully!")
    
    # Show stats
    stats = pipeline.get_stats()
    print(f"\nPipeline Stats:")
    print(f"  Detection Confidence: {stats['detector']['confidence_threshold']}")
    print(f"  Recognition Threshold: {stats['recognizer']['threshold']}")
    print(f"  Registered Students: {stats['recognizer']['total_faces']}")
    
    print("\n✓ Test completed!")
