"""
Test Suite for Face Detection Module

This module contains unit tests for the face detection functionality.

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

from backend.ml.face_detection import FaceDetector, align_face, resize_face, normalize_face
from backend.ml.utils import validate_image, calculate_image_quality


class TestFaceDetector:
    """Test cases for FaceDetector class"""
    
    @pytest.fixture
    def detector(self):
        """Create a FaceDetector instance for testing"""
        return FaceDetector(min_detection_confidence=0.5)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        # Create a simple test image (640x480, 3 channels)
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self, detector):
        """Test if detector initializes correctly"""
        assert detector is not None
        assert detector.min_detection_confidence == 0.5
        assert detector.model_selection == 0
    
    def test_detect_faces_with_empty_image(self, detector):
        """Test detection with empty image"""
        empty_image = np.array([])
        faces = detector.detect_faces(empty_image)
        assert faces == []
    
    def test_detect_faces_with_none(self, detector):
        """Test detection with None image"""
        faces = detector.detect_faces(None)
        assert faces == []
    
    def test_detect_faces_returns_list(self, detector, sample_image):
        """Test that detect_faces returns a list"""
        faces = detector.detect_faces(sample_image)
        assert isinstance(faces, list)
    
    def test_face_data_structure(self, detector, sample_image):
        """Test the structure of detected face data"""
        faces = detector.detect_faces(sample_image)
        
        for face in faces:
            assert 'bbox' in face
            assert 'confidence' in face
            assert 'keypoints' in face
            assert len(face['bbox']) == 4  # x, y, w, h
            assert 0.0 <= face['confidence'] <= 1.0
    
    def test_crop_face_with_valid_bbox(self, detector, sample_image):
        """Test face cropping with valid bounding box"""
        bbox = (100, 100, 200, 200)
        cropped = detector.crop_face(sample_image, bbox, padding=0.1)
        assert cropped is not None
        assert cropped.size > 0
    
    def test_crop_face_with_invalid_bbox(self, detector, sample_image):
        """Test face cropping with invalid bounding box"""
        bbox = (1000, 1000, 200, 200)  # Outside image bounds
        cropped = detector.crop_face(sample_image, bbox)
        # Should still return something or None
        assert cropped is None or cropped.size > 0
    
    def test_draw_detections(self, detector, sample_image):
        """Test drawing detections on image"""
        faces = [
            {
                'bbox': (100, 100, 200, 200),
                'confidence': 0.95,
                'keypoints': [(150, 150), (250, 150)]
            }
        ]
        output = detector.draw_detections(sample_image, faces)
        assert output is not None
        assert output.shape == sample_image.shape


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_resize_face(self):
        """Test face resizing"""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        resized = resize_face(image, target_size=(160, 160))
        assert resized.shape == (160, 160, 3)
    
    def test_normalize_face(self):
        """Test face normalization"""
        image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        normalized = normalize_face(image)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_validate_image_with_valid_image(self):
        """Test image validation with valid image"""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        assert validate_image(image, min_size=(50, 50)) == True
    
    def test_validate_image_with_small_image(self):
        """Test image validation with too small image"""
        image = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        assert validate_image(image, min_size=(50, 50)) == False
    
    def test_validate_image_with_none(self):
        """Test image validation with None"""
        assert validate_image(None) == False
    
    def test_calculate_image_quality(self):
        """Test image quality calculation"""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        quality = calculate_image_quality(image)
        assert isinstance(quality, float)
        assert quality >= 0.0


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    def detector(self):
        return FaceDetector()
    
    def test_detect_single_face_no_faces(self, detector):
        """Test single face detection when no faces present"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        face = detector.detect_single_face(image)
        # Could be None if no face detected
        assert face is None or isinstance(face, dict)
    
    def test_different_confidence_thresholds(self):
        """Test detector with different confidence thresholds"""
        detector_low = FaceDetector(min_detection_confidence=0.3)
        detector_high = FaceDetector(min_detection_confidence=0.9)
        
        assert detector_low.min_detection_confidence == 0.3
        assert detector_high.min_detection_confidence == 0.9
    
    def test_different_model_selections(self):
        """Test detector with different model selections"""
        detector_short = FaceDetector(model_selection=0)
        detector_full = FaceDetector(model_selection=1)
        
        assert detector_short.model_selection == 0
        assert detector_full.model_selection == 1


# Run tests
if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING FACE DETECTION TESTS")
    print("=" * 60)
    
    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])
