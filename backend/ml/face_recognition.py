"""
Face Recognition Module using FaceNet/DeepFace

This module provides face recognition functionality using deep learning embeddings.
It generates face embeddings and performs face matching for identity verification.

Features:
- Generate 128-dimensional face embeddings using FaceNet
- Compare face embeddings using distance metrics
- Face matching with configurable thresholds
- Support for multiple face recognition models
- Embedding storage and retrieval

Author: Face Recognition Team
Date: January 2026
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging
import pickle
import json
from deepface import DeepFace
from scipy.spatial.distance import cosine, euclidean

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Face Recognition class using DeepFace with FaceNet
    
    This class generates face embeddings and performs face matching
    for identity verification in the attendance system.
    """
    
    def __init__(self, 
                 model_name: str = "Facenet",
                 distance_metric: str = "cosine",
                 recognition_threshold: float = 0.6):
        """
        Initialize the Face Recognizer
        
        Args:
            model_name (str): Model to use for face recognition
                             Options: 'Facenet', 'VGG-Face', 'OpenFace', 'DeepFace'
                             Default: 'Facenet' (best accuracy)
            distance_metric (str): Distance metric for face comparison
                                  Options: 'cosine', 'euclidean'
                                  Default: 'cosine'
            recognition_threshold (float): Maximum distance for face match
                                          Lower = stricter matching
                                          Default: 0.6
        """
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.recognition_threshold = recognition_threshold
        
        # Embedding database (in-memory)
        self.embeddings_db = {}  # {student_id: embedding_vector}
        
        logger.info(f"FaceRecognizer initialized with model={model_name}, "
                   f"metric={distance_metric}, threshold={recognition_threshold}")
    
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding from face image
        
        Args:
            face_image (np.ndarray): Face image (BGR format)
        
        Returns:
            Optional[np.ndarray]: Face embedding vector or None if failed
        """
        try:
            # DeepFace expects RGB format
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_image
            
            # Generate embedding using DeepFace
            # DeepFace.represent returns a list of dicts with 'embedding' key
            result = DeepFace.represent(
                img_path=face_rgb,
                model_name=self.model_name,
                enforce_detection=False,  # Don't fail if face not detected
                detector_backend='skip'    # Skip detection (we already have cropped face)
            )
            
            # Extract embedding from result
            if result and len(result) > 0:
                embedding = np.array(result[0]['embedding'])
                logger.info(f"Generated embedding of shape {embedding.shape}")
                return embedding
            else:
                logger.warning("Failed to generate embedding")
                return None
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def calculate_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate distance between two face embeddings
        
        Args:
            embedding1 (np.ndarray): First face embedding
            embedding2 (np.ndarray): Second face embedding
        
        Returns:
            float: Distance between embeddings (lower = more similar)
        """
        if self.distance_metric == "cosine":
            # Cosine distance (0 = identical, 2 = opposite)
            return float(cosine(embedding1, embedding2))
        elif self.distance_metric == "euclidean":
            # Euclidean distance
            return float(euclidean(embedding1, embedding2))
        else:
            logger.warning(f"Unknown metric {self.distance_metric}, using cosine")
            return float(cosine(embedding1, embedding2))
    
    def is_match(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Tuple[bool, float]:
        """
        Check if two face embeddings match
        
        Args:
            embedding1 (np.ndarray): First face embedding
            embedding2 (np.ndarray): Second face embedding
        
        Returns:
            Tuple[bool, float]: (is_match, distance)
        """
        distance = self.calculate_distance(embedding1, embedding2)
        is_match = distance < self.recognition_threshold
        
        return is_match, distance
    
    def recognize_face(self, face_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Recognize a face by comparing against stored embeddings
        
        Args:
            face_image (np.ndarray): Face image to recognize
        
        Returns:
            Optional[Dict]: Recognition result with student_id, distance, confidence
                           or None if no match found
        """
        # Generate embedding for input face
        query_embedding = self.generate_embedding(face_image)
        
        if query_embedding is None:
            logger.warning("Failed to generate embedding for query face")
            return None
        
        if not self.embeddings_db:
            logger.warning("No embeddings in database")
            return None
        
        # Compare against all stored embeddings
        best_match = None
        min_distance = float('inf')
        
        for student_id, stored_embedding in self.embeddings_db.items():
            distance = self.calculate_distance(query_embedding, stored_embedding)
            
            if distance < min_distance:
                min_distance = distance
                best_match = student_id
        
        # Check if best match is within threshold
        if min_distance < self.recognition_threshold:
            confidence = 1.0 - (min_distance / self.recognition_threshold)
            
            result = {
                'student_id': best_match,
                'distance': float(min_distance),
                'confidence': float(confidence),
                'matched': True
            }
            
            logger.info(f"Face recognized: {best_match} (distance={min_distance:.3f})")
            return result
        else:
            logger.info(f"No match found (min_distance={min_distance:.3f})")
            return None
    
    def add_face_to_database(self, student_id: str, face_image: np.ndarray) -> bool:
        """
        Add a face to the recognition database
        
        Args:
            student_id (str): Unique student identifier
            face_image (np.ndarray): Face image
        
        Returns:
            bool: True if successful, False otherwise
        """
        embedding = self.generate_embedding(face_image)
        
        if embedding is None:
            logger.error(f"Failed to add face for student {student_id}")
            return False
        
        self.embeddings_db[student_id] = embedding
        logger.info(f"Added face for student {student_id} to database")
        return True
    
    def add_face_from_file(self, student_id: str, image_path: str) -> bool:
        """
        Add a face to database from image file
        
        Args:
            student_id (str): Unique student identifier
            image_path (str): Path to face image
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not Path(image_path).exists():
            logger.error(f"Image file not found: {image_path}")
            return False
        
        face_image = cv2.imread(image_path)
        
        if face_image is None:
            logger.error(f"Failed to read image: {image_path}")
            return False
        
        return self.add_face_to_database(student_id, face_image)
    
    def remove_face_from_database(self, student_id: str) -> bool:
        """
        Remove a face from the recognition database
        
        Args:
            student_id (str): Student identifier to remove
        
        Returns:
            bool: True if removed, False if not found
        """
        if student_id in self.embeddings_db:
            del self.embeddings_db[student_id]
            logger.info(f"Removed student {student_id} from database")
            return True
        else:
            logger.warning(f"Student {student_id} not found in database")
            return False
    
    def save_embeddings(self, file_path: str) -> bool:
        """
        Save embeddings database to file
        
        Args:
            file_path (str): Path to save embeddings
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if needed
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save as pickle file
            with open(file_path, 'wb') as f:
                pickle.dump(self.embeddings_db, f)
            
            logger.info(f"Saved {len(self.embeddings_db)} embeddings to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False
    
    def load_embeddings(self, file_path: str) -> bool:
        """
        Load embeddings database from file
        
        Args:
            file_path (str): Path to embeddings file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not Path(file_path).exists():
                logger.warning(f"Embeddings file not found: {file_path}")
                return False
            
            with open(file_path, 'rb') as f:
                self.embeddings_db = pickle.load(f)
            
            logger.info(f"Loaded {len(self.embeddings_db)} embeddings from {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embeddings database
        
        Returns:
            Dict: Database statistics
        """
        return {
            'total_faces': len(self.embeddings_db),
            'student_ids': list(self.embeddings_db.keys()),
            'model_name': self.model_name,
            'distance_metric': self.distance_metric,
            'threshold': self.recognition_threshold
        }
    
    def verify_face(self, face_image: np.ndarray, student_id: str) -> Tuple[bool, float]:
        """
        Verify if a face belongs to a specific student (1:1 verification)
        
        Args:
            face_image (np.ndarray): Face image to verify
            student_id (str): Student ID to verify against
        
        Returns:
            Tuple[bool, float]: (is_verified, distance)
        """
        if student_id not in self.embeddings_db:
            logger.warning(f"Student {student_id} not in database")
            return False, float('inf')
        
        query_embedding = self.generate_embedding(face_image)
        
        if query_embedding is None:
            return False, float('inf')
        
        stored_embedding = self.embeddings_db[student_id]
        is_match, distance = self.is_match(query_embedding, stored_embedding)
        
        return is_match, distance


# Utility functions
def compare_faces(face1_path: str, face2_path: str, 
                 model_name: str = "Facenet") -> Dict[str, Any]:
    """
    Compare two face images
    
    Args:
        face1_path (str): Path to first face image
        face2_path (str): Path to second face image
        model_name (str): Model to use for comparison
    
    Returns:
        Dict: Comparison result with distance and match status
    """
    try:
        result = DeepFace.verify(
            img1_path=face1_path,
            img2_path=face2_path,
            model_name=model_name,
            enforce_detection=False
        )
        
        return {
            'verified': result['verified'],
            'distance': result['distance'],
            'threshold': result['threshold'],
            'model': model_name
        }
    
    except Exception as e:
        logger.error(f"Error comparing faces: {e}")
        return None


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("FACE RECOGNITION MODULE TEST")
    print("=" * 60)
    
    # Initialize recognizer
    recognizer = FaceRecognizer(
        model_name="Facenet",
        distance_metric="cosine",
        recognition_threshold=0.6
    )
    
    print("\nFace Recognizer initialized successfully!")
    print(f"Model: {recognizer.model_name}")
    print(f"Distance Metric: {recognizer.distance_metric}")
    print(f"Threshold: {recognizer.recognition_threshold}")
    
    # Test with sample images
    print("\nTo test face recognition:")
    print("1. Add faces: recognizer.add_face_from_file('student_001', 'path/to/face.jpg')")
    print("2. Recognize: result = recognizer.recognize_face(face_image)")
    print("3. Save database: recognizer.save_embeddings('embeddings.pkl')")
    
    # Show database stats
    stats = recognizer.get_database_stats()
    print(f"\nDatabase Stats: {stats}")
    
    print("\n✓ Test completed!")
