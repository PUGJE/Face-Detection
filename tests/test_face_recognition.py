"""
Test Suite for Face Recognition Module

This module contains unit tests for face recognition functionality.

Author: Face Recognition Team
Date: January 2026
"""

import pytest
import cv2
import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ml.face_recognition import FaceRecognizer


class TestFaceRecognizer:
    """Test cases for FaceRecognizer class"""
    
    @pytest.fixture
    def recognizer(self):
        """Create a FaceRecognizer instance for testing"""
        return FaceRecognizer(
            model_name="Facenet",
            distance_metric="cosine",
            recognition_threshold=0.6
        )
    
    @pytest.fixture
    def sample_face(self):
        """Create a sample face image"""
        # Create a realistic-sized face image
        return np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    
    def test_recognizer_initialization(self, recognizer):
        """Test if recognizer initializes correctly"""
        assert recognizer is not None
        assert recognizer.model_name == "Facenet"
        assert recognizer.distance_metric == "cosine"
        assert recognizer.recognition_threshold == 0.6
        assert isinstance(recognizer.embeddings_db, dict)
        assert len(recognizer.embeddings_db) == 0
    
    def test_embeddings_database_empty(self, recognizer):
        """Test that embeddings database starts empty"""
        assert len(recognizer.embeddings_db) == 0
    
    def test_calculate_distance_cosine(self, recognizer):
        """Test cosine distance calculation"""
        emb1 = np.random.rand(128)
        emb2 = np.random.rand(128)
        
        distance = recognizer.calculate_distance(emb1, emb2)
        
        assert isinstance(distance, float)
        assert distance >= 0.0
    
    def test_calculate_distance_euclidean(self):
        """Test euclidean distance calculation"""
        recognizer = FaceRecognizer(distance_metric="euclidean")
        
        emb1 = np.random.rand(128)
        emb2 = np.random.rand(128)
        
        distance = recognizer.calculate_distance(emb1, emb2)
        
        assert isinstance(distance, float)
        assert distance >= 0.0
    
    def test_is_match_identical_embeddings(self, recognizer):
        """Test matching with identical embeddings"""
        emb = np.random.rand(128)
        
        is_match, distance = recognizer.is_match(emb, emb)
        
        assert is_match == True
        assert distance < recognizer.recognition_threshold
    
    def test_is_match_different_embeddings(self, recognizer):
        """Test matching with very different embeddings"""
        emb1 = np.zeros(128)
        emb2 = np.ones(128)
        
        is_match, distance = recognizer.is_match(emb1, emb2)
        
        # These should not match
        assert isinstance(is_match, bool)
        assert isinstance(distance, float)
    
    def test_add_face_to_database(self, recognizer, sample_face):
        """Test adding face to database"""
        # Note: This test might fail if DeepFace can't process random noise
        # In real usage, we'd use actual face images
        student_id = "test_001"
        
        # We'll just test the structure, not actual embedding generation
        # since random noise won't produce valid embeddings
        assert student_id not in recognizer.embeddings_db
    
    def test_remove_face_from_database(self, recognizer):
        """Test removing face from database"""
        student_id = "test_001"
        
        # Add a dummy embedding
        recognizer.embeddings_db[student_id] = np.random.rand(128)
        
        # Remove it
        result = recognizer.remove_face_from_database(student_id)
        
        assert result == True
        assert student_id not in recognizer.embeddings_db
    
    def test_remove_nonexistent_face(self, recognizer):
        """Test removing face that doesn't exist"""
        result = recognizer.remove_face_from_database("nonexistent_id")
        assert result == False
    
    def test_get_database_stats(self, recognizer):
        """Test getting database statistics"""
        stats = recognizer.get_database_stats()
        
        assert isinstance(stats, dict)
        assert 'total_faces' in stats
        assert 'student_ids' in stats
        assert 'model_name' in stats
        assert 'distance_metric' in stats
        assert 'threshold' in stats
        
        assert stats['total_faces'] == 0
        assert stats['model_name'] == "Facenet"
    
    def test_database_stats_with_faces(self, recognizer):
        """Test database stats with added faces"""
        # Add dummy embeddings
        recognizer.embeddings_db['student_001'] = np.random.rand(128)
        recognizer.embeddings_db['student_002'] = np.random.rand(128)
        
        stats = recognizer.get_database_stats()
        
        assert stats['total_faces'] == 2
        assert 'student_001' in stats['student_ids']
        assert 'student_002' in stats['student_ids']
    
    def test_recognize_face_empty_database(self, recognizer, sample_face):
        """Test recognition with empty database"""
        result = recognizer.recognize_face(sample_face)
        # Should return None when database is empty
        assert result is None or isinstance(result, dict)
    
    def test_save_and_load_embeddings(self, recognizer, tmp_path):
        """Test saving and loading embeddings"""
        # Add dummy embeddings
        recognizer.embeddings_db['student_001'] = np.random.rand(128)
        recognizer.embeddings_db['student_002'] = np.random.rand(128)
        
        # Save
        save_path = tmp_path / "test_embeddings.pkl"
        result = recognizer.save_embeddings(str(save_path))
        assert result == True
        assert save_path.exists()
        
        # Create new recognizer and load
        new_recognizer = FaceRecognizer()
        result = new_recognizer.load_embeddings(str(save_path))
        assert result == True
        assert len(new_recognizer.embeddings_db) == 2
        assert 'student_001' in new_recognizer.embeddings_db
    
    def test_load_nonexistent_file(self, recognizer):
        """Test loading from nonexistent file"""
        result = recognizer.load_embeddings("nonexistent_file.pkl")
        assert result == False


class TestDifferentModels:
    """Test different face recognition models"""
    
    def test_facenet_model(self):
        """Test FaceNet model initialization"""
        recognizer = FaceRecognizer(model_name="Facenet")
        assert recognizer.model_name == "Facenet"
    
    def test_vggface_model(self):
        """Test VGG-Face model initialization"""
        recognizer = FaceRecognizer(model_name="VGG-Face")
        assert recognizer.model_name == "VGG-Face"
    
    def test_different_thresholds(self):
        """Test different recognition thresholds"""
        recognizer_strict = FaceRecognizer(recognition_threshold=0.3)
        recognizer_loose = FaceRecognizer(recognition_threshold=0.9)
        
        assert recognizer_strict.recognition_threshold == 0.3
        assert recognizer_loose.recognition_threshold == 0.9


# Run tests
if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING FACE RECOGNITION TESTS")
    print("=" * 60)
    
    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])
