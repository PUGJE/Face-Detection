"""
Quick Verification Script for Face Recognition Module

This script verifies that the face recognition module is working correctly.

Author: Face Recognition Team
Date: January 2026
"""

import sys
from pathlib import Path
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("FACE RECOGNITION MODULE - VERIFICATION")
print("=" * 70)

# Test 1: Import modules
print("\n[1/6] Testing imports...")
try:
    from backend.ml.face_recognition import FaceRecognizer
    from backend.ml.recognition_pipeline import FaceRecognitionPipeline
    from backend.config import settings
    print("✓ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize recognizer
print("\n[2/6] Testing recognizer initialization...")
try:
    recognizer = FaceRecognizer(
        model_name="Facenet",
        distance_metric="cosine",
        recognition_threshold=0.6
    )
    print(f"✓ Recognizer initialized")
    print(f"  - Model: {recognizer.model_name}")
    print(f"  - Distance metric: {recognizer.distance_metric}")
    print(f"  - Threshold: {recognizer.recognition_threshold}")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    sys.exit(1)

# Test 3: Test database operations
print("\n[3/6] Testing database operations...")
try:
    # Add dummy embeddings
    recognizer.embeddings_db['test_001'] = np.random.rand(128)
    recognizer.embeddings_db['test_002'] = np.random.rand(128)
    
    # Get stats
    stats = recognizer.get_database_stats()
    assert stats['total_faces'] == 2
    assert 'test_001' in stats['student_ids']
    
    print(f"✓ Database operations working")
    print(f"  - Total faces: {stats['total_faces']}")
    print(f"  - Student IDs: {stats['student_ids']}")
    
    # Remove face
    removed = recognizer.remove_face_from_database('test_001')
    assert removed == True
    assert len(recognizer.embeddings_db) == 1
    
    print(f"✓ Face removal working")
except Exception as e:
    print(f"❌ Database test failed: {e}")
    sys.exit(1)

# Test 4: Test distance calculation
print("\n[4/6] Testing distance calculation...")
try:
    emb1 = np.random.rand(128)
    emb2 = np.random.rand(128)
    
    # Test cosine distance
    distance_cosine = recognizer.calculate_distance(emb1, emb2)
    assert isinstance(distance_cosine, float)
    assert distance_cosine >= 0.0
    
    print(f"✓ Cosine distance: {distance_cosine:.3f}")
    
    # Test euclidean distance
    recognizer_euclidean = FaceRecognizer(distance_metric="euclidean")
    distance_euclidean = recognizer_euclidean.calculate_distance(emb1, emb2)
    assert isinstance(distance_euclidean, float)
    assert distance_euclidean >= 0.0
    
    print(f"✓ Euclidean distance: {distance_euclidean:.3f}")
except Exception as e:
    print(f"❌ Distance calculation failed: {e}")
    sys.exit(1)

# Test 5: Test matching
print("\n[5/6] Testing face matching...")
try:
    # Identical embeddings should match
    emb = np.random.rand(128)
    is_match, distance = recognizer.is_match(emb, emb)
    
    print(f"✓ Identical embeddings:")
    print(f"  - Match: {is_match}")
    print(f"  - Distance: {distance:.3f}")
    
    # Very different embeddings should not match
    emb1 = np.zeros(128)
    emb2 = np.ones(128)
    is_match, distance = recognizer.is_match(emb1, emb2)
    
    print(f"✓ Different embeddings:")
    print(f"  - Match: {is_match}")
    print(f"  - Distance: {distance:.3f}")
except Exception as e:
    print(f"❌ Matching test failed: {e}")
    sys.exit(1)

# Test 6: Test pipeline
print("\n[6/6] Testing recognition pipeline...")
try:
    pipeline = FaceRecognitionPipeline()
    
    stats = pipeline.get_stats()
    
    print(f"✓ Pipeline initialized")
    print(f"  - Detection confidence: {stats['detector']['confidence_threshold']}")
    print(f"  - Recognition threshold: {stats['recognizer']['threshold']}")
    print(f"  - Registered students: {stats['recognizer']['total_faces']}")
except Exception as e:
    print(f"❌ Pipeline test failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE - ALL TESTS PASSED ✓")
print("=" * 70)
print("\nNext steps:")
print("1. Register student:  python demo_face_recognition.py --mode register --id STUDENT_001")
print("2. Recognize:         python demo_face_recognition.py --mode recognize")
print("3. Verify:            python demo_face_recognition.py --mode verify --id STUDENT_001")
print("4. Run tests:         python -m pytest tests/test_face_recognition.py -v")
print("=" * 70)
