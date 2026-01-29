"""
Liveness Detector - Blink and Motion Detection

This module implements blink detection and head movement tracking
for liveness verification.

Author: Face Recognition Team
Date: January 2026
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)


class LivenessDetector:
    """
    Detect liveness through blink detection and head movement
    """
    
    def __init__(
        self,
        ear_threshold: float = 0.21,
        consecutive_frames: int = 2,
        blink_timeout: float = 3.0
    ):
        """
        Initialize liveness detector
        
        Args:
            ear_threshold: Eye Aspect Ratio threshold for blink detection
            consecutive_frames: Number of consecutive frames for blink
            blink_timeout: Maximum time to wait for blink (seconds)
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.blink_timeout = blink_timeout
        
        # Blink tracking
        self.blink_counter = 0
        self.total_blinks = 0
        self.last_blink_time = None
        self.start_time = time.time()
        
        # Head position tracking
        self.head_positions = []
        self.max_positions = 30
        
        logger.info(
            f"LivenessDetector initialized with "
            f"ear_threshold={ear_threshold}, "
            f"blink_timeout={blink_timeout}s"
        )
    
    def calculate_eye_aspect_ratio(
        self,
        eye_landmarks: np.ndarray
    ) -> float:
        """
        Calculate Eye Aspect Ratio (EAR)
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            eye_landmarks: Array of 6 (x,y) coordinates for eye
        
        Returns:
            Eye aspect ratio
        """
        # Compute euclidean distances
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C + 1e-7)
        
        return ear
    
    def detect_blink_from_landmarks(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray
    ) -> Dict:
        """
        Detect blink from eye landmarks
        
        Args:
            left_eye: Left eye landmarks (6 points)
            right_eye: Right eye landmarks (6 points)
        
        Returns:
            Dictionary with blink detection results
        """
        try:
            # Calculate EAR for both eyes
            left_ear = self.calculate_eye_aspect_ratio(left_eye)
            right_ear = self.calculate_eye_aspect_ratio(right_eye)
            
            # Average EAR
            ear = (left_ear + right_ear) / 2.0
            
            # Check if eyes are closed
            if ear < self.ear_threshold:
                self.blink_counter += 1
            else:
                # Eyes opened after being closed
                if self.blink_counter >= self.consecutive_frames:
                    self.total_blinks += 1
                    self.last_blink_time = time.time()
                    logger.info(f"Blink detected! Total: {self.total_blinks}")
                
                self.blink_counter = 0
            
            # Check if blink occurred within timeout
            elapsed_time = time.time() - self.start_time
            has_blinked = self.total_blinks > 0
            
            if elapsed_time > self.blink_timeout and not has_blinked:
                is_live = False
                message = "No blink detected within timeout"
            else:
                is_live = has_blinked
                message = f"Blinks detected: {self.total_blinks}"
            
            return {
                'success': True,
                'is_live': is_live,
                'ear': float(ear),
                'blink_count': self.total_blinks,
                'elapsed_time': float(elapsed_time),
                'message': message
            }
            
        except Exception as e:
            logger.error(f"Blink detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'is_live': False
            }
    
    def detect_blink_from_face(
        self,
        face_landmarks: Dict
    ) -> Dict:
        """
        Detect blink from MediaPipe face landmarks
        
        Args:
            face_landmarks: MediaPipe face landmarks
        
        Returns:
            Dictionary with blink detection results
        """
        try:
            # Extract eye landmarks from MediaPipe
            # Left eye indices: 33, 160, 158, 133, 153, 144
            # Right eye indices: 362, 385, 387, 263, 373, 380
            
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            
            landmarks = face_landmarks.landmark
            
            # Get left eye coordinates
            left_eye = np.array([
                [landmarks[i].x, landmarks[i].y]
                for i in left_eye_indices
            ])
            
            # Get right eye coordinates
            right_eye = np.array([
                [landmarks[i].x, landmarks[i].y]
                for i in right_eye_indices
            ])
            
            return self.detect_blink_from_landmarks(left_eye, right_eye)
            
        except Exception as e:
            logger.error(f"Blink detection from face failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'is_live': False
            }
    
    def track_head_movement(
        self,
        face_center: Tuple[int, int]
    ) -> Dict:
        """
        Track head movement to detect liveness
        
        Args:
            face_center: (x, y) coordinates of face center
        
        Returns:
            Dictionary with movement analysis
        """
        try:
            # Add position to history
            self.head_positions.append(face_center)
            if len(self.head_positions) > self.max_positions:
                self.head_positions.pop(0)
            
            if len(self.head_positions) < 5:
                return {
                    'success': False,
                    'error': 'Insufficient position history',
                    'has_movement': False
                }
            
            # Calculate movement statistics
            positions = np.array(self.head_positions)
            
            # Calculate displacement
            displacements = np.diff(positions, axis=0)
            distances = np.linalg.norm(displacements, axis=1)
            
            avg_movement = np.mean(distances)
            movement_variance = np.var(distances)
            
            # Natural head movement characteristics:
            # - Average displacement: 1-5 pixels per frame
            # - Variance: 0.5-3.0 (some variation but not erratic)
            
            # Static photo: avg < 0.5, variance < 0.2
            # Video replay: might have movement but too consistent
            # Real person: moderate movement with natural variance
            
            has_natural_movement = (
                0.5 < avg_movement < 8.0 and
                0.2 < movement_variance < 5.0
            )
            
            # Calculate overall position variance
            position_std = np.std(positions, axis=0)
            total_variance = np.mean(position_std)
            
            return {
                'success': True,
                'has_movement': has_natural_movement,
                'avg_movement': float(avg_movement),
                'movement_variance': float(movement_variance),
                'position_variance': float(total_variance),
                'num_positions': len(self.head_positions)
            }
            
        except Exception as e:
            logger.error(f"Head movement tracking failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'has_movement': False
            }
    
    def reset(self):
        """Reset detector state"""
        self.blink_counter = 0
        self.total_blinks = 0
        self.last_blink_time = None
        self.start_time = time.time()
        self.head_positions = []
        logger.info("LivenessDetector reset")
    
    def get_stats(self) -> Dict:
        """Get detector statistics"""
        elapsed = time.time() - self.start_time
        return {
            'total_blinks': self.total_blinks,
            'elapsed_time': elapsed,
            'blink_rate': self.total_blinks / (elapsed + 1e-7),
            'head_positions_tracked': len(self.head_positions),
            'last_blink_time': self.last_blink_time
        }


class BlinkDetectorSimple:
    """
    Simplified blink detector using frame differencing
    (when facial landmarks are not available)
    """
    
    def __init__(self, threshold: float = 0.15):
        """
        Initialize simple blink detector
        
        Args:
            threshold: Threshold for eye region change
        """
        self.threshold = threshold
        self.previous_eye_region = None
        self.blink_count = 0
        self.state = "open"  # "open" or "closed"
        
    def detect_blink(
        self,
        eye_region: np.ndarray
    ) -> Dict:
        """
        Detect blink from eye region image
        
        Args:
            eye_region: Cropped eye region image
        
        Returns:
            Dictionary with blink detection results
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate average intensity
            avg_intensity = np.mean(gray)
            
            if self.previous_eye_region is not None:
                # Calculate difference
                prev_gray = cv2.cvtColor(
                    self.previous_eye_region,
                    cv2.COLOR_BGR2GRAY
                )
                
                diff = cv2.absdiff(gray, prev_gray)
                change_ratio = np.mean(diff) / 255.0
                
                # Detect state change
                if change_ratio > self.threshold:
                    if self.state == "open" and avg_intensity < 100:
                        self.state = "closed"
                    elif self.state == "closed" and avg_intensity > 100:
                        self.state = "open"
                        self.blink_count += 1
                        logger.info(f"Blink detected (simple)! Total: {self.blink_count}")
            
            self.previous_eye_region = eye_region.copy()
            
            return {
                'success': True,
                'blink_count': self.blink_count,
                'current_state': self.state,
                'avg_intensity': float(avg_intensity)
            }
            
        except Exception as e:
            logger.error(f"Simple blink detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'blink_count': 0
            }
    
    def reset(self):
        """Reset detector"""
        self.previous_eye_region = None
        self.blink_count = 0
        self.state = "open"


if __name__ == "__main__":
    """Test liveness detector"""
    print("=" * 60)
    print("Liveness Detector Test")
    print("=" * 60)
    
    # Initialize detector
    detector = LivenessDetector()
    
    print("\nSimulated blink test...")
    
    # Simulate eye landmarks for open eyes
    open_eye = np.array([
        [0.0, 0.0],  # Left corner
        [0.1, -0.05],  # Top left
        [0.2, -0.05],  # Top right
        [0.3, 0.0],  # Right corner
        [0.2, 0.05],  # Bottom right
        [0.1, 0.05]  # Bottom left
    ])
    
    # Simulate eye landmarks for closed eyes
    closed_eye = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.2, 0.0],
        [0.3, 0.0],
        [0.2, 0.0],
        [0.1, 0.0]
    ])
    
    # Simulate blink sequence
    print("\nSimulating blink sequence...")
    for i in range(10):
        if 3 <= i <= 5:
            # Eyes closed
            result = detector.detect_blink_from_landmarks(closed_eye, closed_eye)
        else:
            # Eyes open
            result = detector.detect_blink_from_landmarks(open_eye, open_eye)
        
        print(f"Frame {i}: EAR={result.get('ear', 0):.3f}, Blinks={result.get('blink_count', 0)}")
        time.sleep(0.1)
    
    print(f"\n✓ Total blinks detected: {detector.total_blinks}")
    print(f"✓ Is live: {detector.total_blinks > 0}")
    
    # Test head movement
    print("\nTesting head movement tracking...")
    for i in range(20):
        # Simulate natural head movement
        x = 100 + np.random.randint(-3, 4)
        y = 100 + np.random.randint(-3, 4)
        detector.track_head_movement((x, y))
    
    movement_result = detector.track_head_movement((100, 100))
    print(f"Has natural movement: {movement_result.get('has_movement', False)}")
    print(f"Average movement: {movement_result.get('avg_movement', 0):.2f}")
    
    print("\n✓ Test complete!")
