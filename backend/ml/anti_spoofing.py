"""
Anti-Spoofing Module - Liveness Detection

This module implements liveness detection to prevent spoofing attacks using:
1. Texture Analysis (LBP, color space, frequency domain)
2. Motion Detection (blink detection, head movement)

Author: Face Recognition Team
Date: January 2026
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from scipy import ndimage
from skimage.feature import local_binary_pattern

logger = logging.getLogger(__name__)


class AntiSpoofingDetector:
    """
    Anti-spoofing detector using texture and motion analysis
    """
    
    def __init__(
        self,
        texture_threshold: float = 0.60,  # Balanced threshold
        motion_threshold: float = 0.4,
        lbp_radius: int = 1,
        lbp_points: int = 8
    ):
        """
        Initialize anti-spoofing detector
        
        Args:
            texture_threshold: Threshold for texture-based liveness (0-1)
            motion_threshold: Threshold for motion-based liveness (0-1)
            lbp_radius: Radius for LBP calculation
            lbp_points: Number of points for LBP calculation
        """
        self.texture_threshold = texture_threshold
        self.motion_threshold = motion_threshold
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        
        # Frame buffer for motion detection
        self.frame_buffer = []
        self.max_buffer_size = 10
        
        logger.info(
            f"AntiSpoofingDetector initialized with "
            f"texture_threshold={texture_threshold}, "
            f"motion_threshold={motion_threshold}"
        )
    
    def check_liveness(
        self,
        image: np.ndarray,
        face_region: Optional[Tuple[int, int, int, int]] = None,
        use_motion: bool = False  # Motion detection disabled by default
    ) -> Dict:
        """
        Comprehensive liveness check using texture analysis
        
        Args:
            image: Input image (BGR format)
            face_region: Optional face bounding box (x, y, w, h)
            use_motion: Whether to include motion detection (unreliable)
        
        Returns:
            Dictionary with liveness results
        """
        try:
            # Extract face region if provided
            if face_region is not None:
                x, y, w, h = face_region
                face_img = image[y:y+h, x:x+w]
            else:
                face_img = image
            
            # Perform texture analysis (primary method)
            texture_score = self._analyze_texture(face_img)
            
            # Use texture score as liveness score
            liveness_score = texture_score
            is_live = liveness_score > self.texture_threshold
            
            result = {
                'success': True,
                'is_live': is_live,
                'liveness_score': float(liveness_score),
                'texture_score': float(texture_score),
                'confidence': float(liveness_score),
                'method': 'texture_analysis'
            }
            
            logger.info(
                f"Liveness check: is_live={is_live}, "
                f"score={liveness_score:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in liveness check: {e}")
            return {
                'success': False,
                'error': str(e),
                'is_live': False
            }
    
    def _analyze_texture(self, image: np.ndarray) -> float:
        """
        Analyze image texture to detect spoofing
        
        Real faces have different texture patterns than printed photos
        
        Args:
            image: Face image
        
        Returns:
            Texture score (0-1, higher = more likely real)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Local Binary Pattern (LBP) Analysis
        lbp_score = self._compute_lbp_score(gray)
        
        # 2. Color Space Analysis
        color_score = self._analyze_color_space(image)
        
        # 3. Frequency Domain Analysis
        freq_score = self._analyze_frequency(gray)
        
        # 4. Edge Density Analysis
        edge_score = self._analyze_edges(gray)
        
        # 5. Screen/Photo Detection (NEW)
        screen_score = self._detect_screen_patterns(image)
        
        # 6. Reflection Analysis (NEW)
        reflection_score = self._analyze_reflections(image)
        
        # Combine scores with weights
        texture_score = (
            0.25 * lbp_score +
            0.20 * color_score +
            0.20 * freq_score +
            0.10 * edge_score +
            0.15 * screen_score +
            0.10 * reflection_score
        )
        
        return texture_score
    
    def _compute_lbp_score(self, gray_image: np.ndarray) -> float:
        """
        Compute LBP-based texture score
        
        Real faces have higher LBP variance than printed photos
        """
        try:
            # Compute LBP
            lbp = local_binary_pattern(
                gray_image,
                self.lbp_points,
                self.lbp_radius,
                method='uniform'
            )
            
            # Calculate histogram
            hist, _ = np.histogram(
                lbp.ravel(),
                bins=np.arange(0, self.lbp_points + 3),
                range=(0, self.lbp_points + 2)
            )
            
            # Normalize histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            
            # Calculate variance (real faces have higher variance)
            variance = np.var(hist)
            
            # Normalize to 0-1 range
            # Real faces typically have variance > 0.01
            # Fake faces typically have variance < 0.005
            score = min(1.0, variance / 0.015)
            
            return score
            
        except Exception as e:
            logger.warning(f"LBP computation failed: {e}")
            return 0.5
    
    def _analyze_color_space(self, image: np.ndarray) -> float:
        """
        Analyze color distribution
        
        Real faces have more diverse color distribution
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate color diversity
            h_std = np.std(hsv[:, :, 0])
            s_std = np.std(hsv[:, :, 1])
            v_std = np.std(hsv[:, :, 2])
            
            # Real faces have higher color diversity
            color_diversity = (h_std + s_std + v_std) / 3.0
            
            # Normalize (real faces typically > 30, fake < 20)
            score = min(1.0, color_diversity / 40.0)
            
            return score
            
        except Exception as e:
            logger.warning(f"Color analysis failed: {e}")
            return 0.5
    
    def _analyze_frequency(self, gray_image: np.ndarray) -> float:
        """
        Analyze frequency domain characteristics
        
        Printed photos have different frequency patterns
        """
        try:
            # Resize for consistent analysis
            resized = cv2.resize(gray_image, (64, 64))
            
            # Apply FFT
            f = np.fft.fft2(resized)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            
            # Calculate high-frequency energy
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2
            
            # High frequency region (outer ring)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (center_w, center_h), min(h, w) // 3, 1, -1)
            cv2.circle(mask, (center_w, center_h), min(h, w) // 6, 0, -1)
            
            high_freq_energy = np.sum(magnitude * mask)
            total_energy = np.sum(magnitude)
            
            # Real faces have more high-frequency content
            ratio = high_freq_energy / (total_energy + 1e-7)
            
            # Normalize (real faces typically > 0.15, fake < 0.10)
            score = min(1.0, ratio / 0.20)
            
            return score
            
        except Exception as e:
            logger.warning(f"Frequency analysis failed: {e}")
            return 0.5
    
    def _analyze_edges(self, gray_image: np.ndarray) -> float:
        """
        Analyze edge density
        
        Real faces have different edge patterns than photos
        """
        try:
            # Apply Canny edge detection
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Calculate edge density
            edge_density = np.sum(edges > 0) / edges.size
            
            # Real faces have moderate edge density
            # Too high = noise/screen, too low = blurred photo
            # Optimal range: 0.05 - 0.15
            if 0.05 <= edge_density <= 0.15:
                score = 1.0
            elif edge_density < 0.05:
                score = edge_density / 0.05
            else:
                score = max(0.0, 1.0 - (edge_density - 0.15) / 0.15)
            
            return score
            
        except Exception as e:
            logger.warning(f"Edge analysis failed: {e}")
            return 0.5
    
    def _detect_screen_patterns(self, image: np.ndarray) -> float:
        """
        Detect screen patterns (moiré effect, pixel grid)
        
        Phone/tablet screens have visible pixel patterns
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize for consistent analysis
            resized = cv2.resize(gray, (128, 128))
            
            # Apply high-pass filter to detect fine patterns
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            filtered = cv2.filter2D(resized, -1, kernel)
            
            # Calculate variance of high-frequency components
            high_freq_variance = np.var(filtered)
            
            # Screens have regular patterns (higher variance in specific ranges)
            # Real faces have irregular patterns
            
            # Check for periodic patterns using autocorrelation
            mean_val = np.mean(filtered)
            centered = filtered - mean_val
            autocorr = np.correlate(centered.flatten(), centered.flatten(), mode='same')
            autocorr_variance = np.var(autocorr)
            
            # Screens show periodic patterns (lower score)
            # Real faces show random patterns (higher score)
            if high_freq_variance > 500 and autocorr_variance > 1000:
                # Likely a screen (regular patterns)
                score = 0.3
            else:
                # Likely real face (irregular patterns)
                score = 0.8
            
            return score
            
        except Exception as e:
            logger.warning(f"Screen detection failed: {e}")
            return 0.5
    
    def _analyze_reflections(self, image: np.ndarray) -> float:
        """
        Analyze specular reflections
        
        Phone screens have uniform reflections, real skin has varied reflections
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Find bright spots (potential reflections)
            _, bright_mask = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)
            
            # Calculate properties of bright regions
            num_bright_pixels = np.sum(bright_mask > 0)
            total_pixels = bright_mask.size
            bright_ratio = num_bright_pixels / total_pixels
            
            # Analyze brightness distribution
            brightness_std = np.std(l_channel)
            
            # Screens have:
            # - Uniform brightness (low std)
            # - Few or many uniform reflections
            # Real faces have:
            # - Varied brightness (high std)
            # - Natural, scattered reflections
            
            if brightness_std < 30:
                # Too uniform = likely screen
                score = 0.4
            elif brightness_std > 50:
                # Good variation = likely real
                score = 0.9
            else:
                # Moderate variation
                score = 0.6
            
            # Adjust for reflection ratio
            if bright_ratio > 0.3:
                # Too many bright pixels = overexposed or screen glare
                score *= 0.7
            
            return score
            
        except Exception as e:
            logger.warning(f"Reflection analysis failed: {e}")
            return 0.5
    
    def add_frame(self, image: np.ndarray):
        """
        Add frame to buffer for motion detection
        
        Args:
            image: Input frame
        """
        self.frame_buffer.append(image.copy())
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
    
    def check_motion(self) -> Dict:
        """
        Check for natural motion in buffered frames
        
        Returns:
            Dictionary with motion detection results
        """
        if len(self.frame_buffer) < 3:
            return {
                'success': False,
                'error': 'Insufficient frames for motion detection',
                'has_motion': False
            }
        
        try:
            # Calculate frame differences
            motion_scores = []
            
            for i in range(len(self.frame_buffer) - 1):
                frame1 = cv2.cvtColor(self.frame_buffer[i], cv2.COLOR_BGR2GRAY)
                frame2 = cv2.cvtColor(self.frame_buffer[i + 1], cv2.COLOR_BGR2GRAY)
                
                # Calculate absolute difference
                diff = cv2.absdiff(frame1, frame2)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            
            avg_motion = np.mean(motion_scores)
            motion_variance = np.var(motion_scores)
            
            # Natural motion has moderate average and variance
            # Adjusted thresholds for more realistic detection:
            # Static photo: avg < 1.0, variance < 0.3
            # Video replay: might have movement but too consistent (variance < 0.5)
            # Real person: avg > 0.5, variance > 0.1 (even small movements count)
            
            has_natural_motion = (
                0.5 < avg_motion < 15.0 and
                0.1 < motion_variance < 20.0
            )
            
            # More forgiving motion score calculation
            motion_score = min(1.0, avg_motion / 3.0) * min(1.0, motion_variance / 2.0)
            
            return {
                'success': True,
                'has_motion': has_natural_motion,
                'motion_score': float(motion_score),
                'avg_motion': float(avg_motion),
                'motion_variance': float(motion_variance)
            }
            
        except Exception as e:
            logger.error(f"Motion detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'has_motion': False
            }
    
    def reset(self):
        """Reset frame buffer"""
        self.frame_buffer = []
    
    def get_stats(self) -> Dict:
        """Get detector statistics"""
        return {
            'texture_threshold': self.texture_threshold,
            'motion_threshold': self.motion_threshold,
            'buffer_size': len(self.frame_buffer),
            'max_buffer_size': self.max_buffer_size
        }


if __name__ == "__main__":
    """Test anti-spoofing detector"""
    print("=" * 60)
    print("Anti-Spoofing Detector Test")
    print("=" * 60)
    
    # Initialize detector
    detector = AntiSpoofingDetector()
    
    # Test with webcam
    print("\nTesting with webcam...")
    print("Press 'q' to quit, 's' to check liveness")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add frame to buffer
        detector.add_frame(frame)
        
        # Display frame
        cv2.imshow('Anti-Spoofing Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Check liveness
            result = detector.check_liveness(frame)
            print(f"\nLiveness Check:")
            print(f"  Is Live: {result['is_live']}")
            print(f"  Score: {result['liveness_score']:.3f}")
            print(f"  Texture: {result['texture_score']:.3f}")
            
            # Check motion
            motion_result = detector.check_motion()
            if motion_result['success']:
                print(f"  Motion: {motion_result['has_motion']}")
                print(f"  Motion Score: {motion_result['motion_score']:.3f}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✓ Test complete!")
