"""
Quick Verification Script for Face Detection Module

This script verifies that the face detection module is working correctly
without requiring pytest.

Author: Face Recognition Team
Date: January 2026
"""

import sys
from pathlib import Path
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("FACE DETECTION MODULE - VERIFICATION")
print("=" * 70)

# Test 1: Import modules
print("\n[1/5] Testing imports...")
try:
    from backend.ml.face_detection import FaceDetector, resize_face, normalize_face
    from backend.ml.utils import validate_image, calculate_image_quality
    from backend.config import settings
    print("✓ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize detector
print("\n[2/5] Testing detector initialization...")
try:
    detector = FaceDetector(min_detection_confidence=0.5)
    print(f"✓ Detector initialized")
    print(f"  - Confidence threshold: {detector.min_detection_confidence}")
    print(f"  - Model selection: {detector.model_selection}")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    sys.exit(1)

# Test 3: Test with sample image
print("\n[3/5] Testing face detection with sample image...")
try:
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect faces
    faces = detector.detect_faces(test_image)
    
    print(f"✓ Detection completed")
    print(f"  - Detected faces: {len(faces)}")
    print(f"  - Return type: {type(faces)}")
except Exception as e:
    print(f"❌ Detection failed: {e}")
    sys.exit(1)

# Test 4: Test utility functions
print("\n[4/5] Testing utility functions...")
try:
    # Test resize
    resized = resize_face(test_image, target_size=(160, 160))
    assert resized.shape == (160, 160, 3), "Resize failed"
    print(f"✓ resize_face: {test_image.shape} → {resized.shape}")
    
    # Test normalize
    normalized = normalize_face(test_image)
    assert 0.0 <= normalized.min() and normalized.max() <= 1.0, "Normalize failed"
    print(f"✓ normalize_face: range [{normalized.min():.2f}, {normalized.max():.2f}]")
    
    # Test validate
    is_valid = validate_image(test_image, min_size=(50, 50))
    assert is_valid == True, "Validation failed"
    print(f"✓ validate_image: {is_valid}")
    
    # Test quality
    quality = calculate_image_quality(test_image)
    assert isinstance(quality, float), "Quality calculation failed"
    print(f"✓ calculate_image_quality: {quality:.2f}")
    
except Exception as e:
    print(f"❌ Utility test failed: {e}")
    sys.exit(1)

# Test 5: Test configuration
print("\n[5/5] Testing configuration...")
try:
    print(f"✓ Camera index: {settings.camera_index}")
    print(f"✓ Camera resolution: {settings.camera_width}x{settings.camera_height}")
    print(f"✓ Detection confidence: {settings.face_detection_confidence}")
except Exception as e:
    print(f"❌ Configuration test failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE - ALL TESTS PASSED ✓")
print("=" * 70)
print("\nNext steps:")
print("1. Test with webcam: python demo_face_detection.py --mode webcam")
print("2. Test with image:  python demo_face_detection.py --mode image --path <image_path>")
print("3. Run full tests:   python -m pytest tests/test_face_detection.py -v")
print("=" * 70)
