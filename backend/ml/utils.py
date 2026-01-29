"""
Machine Learning Utility Functions

This module provides utility functions for image preprocessing, validation,
and augmentation used across the ML pipeline.

Author: Face Recognition Team
Date: January 2026
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_image(image_path: str, color_mode: str = 'BGR') -> Optional[np.ndarray]:
    """
    Load image from file path
    
    Args:
        image_path (str): Path to image file
        color_mode (str): 'BGR', 'RGB', or 'GRAY'
    
    Returns:
        Optional[np.ndarray]: Loaded image or None if failed
    """
    if not Path(image_path).exists():
        logger.error(f"Image not found: {image_path}")
        return None
    
    # Read image
    image = cv2.imread(image_path)
    
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return None
    
    # Convert color mode if needed
    if color_mode == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_mode == 'GRAY':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image


def save_image(image: np.ndarray, save_path: str) -> bool:
    """
    Save image to file
    
    Args:
        image (np.ndarray): Image to save
        save_path (str): Path to save the image
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        success = cv2.imwrite(save_path, image)
        
        if success:
            logger.info(f"Image saved to: {save_path}")
        else:
            logger.error(f"Failed to save image to: {save_path}")
        
        return success
    
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return False


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                 maintain_aspect_ratio: bool = False) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image (np.ndarray): Input image
        target_size (Tuple[int, int]): Target size (width, height)
        maintain_aspect_ratio (bool): Whether to maintain aspect ratio
    
    Returns:
        np.ndarray: Resized image
    """
    if maintain_aspect_ratio:
        # Calculate aspect ratio
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas and center the image
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize image
    
    Args:
        image (np.ndarray): Input image
        method (str): Normalization method ('standard', 'minmax', 'mean')
    
    Returns:
        np.ndarray: Normalized image
    """
    image_float = image.astype(np.float32)
    
    if method == 'standard':
        # Scale to [0, 1]
        return image_float / 255.0
    
    elif method == 'minmax':
        # Scale to [-1, 1]
        return (image_float / 127.5) - 1.0
    
    elif method == 'mean':
        # Zero-center with mean and std
        mean = np.mean(image_float)
        std = np.std(image_float)
        return (image_float - mean) / (std + 1e-7)
    
    else:
        logger.warning(f"Unknown normalization method: {method}, using 'standard'")
        return image_float / 255.0


def preprocess_face(face_image: np.ndarray, target_size: Tuple[int, int] = (160, 160),
                   normalize: bool = True) -> np.ndarray:
    """
    Preprocess face image for recognition
    
    Args:
        face_image (np.ndarray): Input face image
        target_size (Tuple[int, int]): Target size for face
        normalize (bool): Whether to normalize pixel values
    
    Returns:
        np.ndarray: Preprocessed face image
    """
    # Resize
    face = cv2.resize(face_image, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize if requested
    if normalize:
        face = normalize_image(face, method='standard')
    
    return face


def enhance_image(image: np.ndarray, method: str = 'histogram') -> np.ndarray:
    """
    Enhance image quality
    
    Args:
        image (np.ndarray): Input image
        method (str): Enhancement method ('histogram', 'clahe', 'gamma')
    
    Returns:
        np.ndarray: Enhanced image
    """
    if method == 'histogram':
        # Histogram equalization
        if len(image.shape) == 2:  # Grayscale
            return cv2.equalizeHist(image)
        else:  # Color
            # Convert to YCrCb and equalize Y channel
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    elif method == 'clahe':
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        if len(image.shape) == 2:  # Grayscale
            return clahe.apply(image)
        else:  # Color
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    elif method == 'gamma':
        # Gamma correction
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    else:
        logger.warning(f"Unknown enhancement method: {method}")
        return image


def validate_image(image: np.ndarray, min_size: Tuple[int, int] = (50, 50)) -> bool:
    """
    Validate if image meets minimum requirements
    
    Args:
        image (np.ndarray): Input image
        min_size (Tuple[int, int]): Minimum size (width, height)
    
    Returns:
        bool: True if valid, False otherwise
    """
    if image is None or image.size == 0:
        logger.warning("Image is None or empty")
        return False
    
    h, w = image.shape[:2]
    min_w, min_h = min_size
    
    if w < min_w or h < min_h:
        logger.warning(f"Image too small: {w}x{h}, minimum: {min_w}x{min_h}")
        return False
    
    return True


def augment_image(image: np.ndarray, augmentation_type: str) -> np.ndarray:
    """
    Apply data augmentation to image
    
    Args:
        image (np.ndarray): Input image
        augmentation_type (str): Type of augmentation
                                ('flip', 'rotate', 'brightness', 'noise')
    
    Returns:
        np.ndarray: Augmented image
    """
    if augmentation_type == 'flip':
        return cv2.flip(image, 1)  # Horizontal flip
    
    elif augmentation_type == 'rotate':
        angle = np.random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    elif augmentation_type == 'brightness':
        factor = np.random.uniform(0.7, 1.3)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif augmentation_type == 'noise':
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        return cv2.add(image, noise)
    
    else:
        logger.warning(f"Unknown augmentation type: {augmentation_type}")
        return image


def calculate_image_quality(image: np.ndarray) -> float:
    """
    Calculate image quality score based on sharpness (Laplacian variance)
    
    Args:
        image (np.ndarray): Input image
    
    Returns:
        float: Quality score (higher is better)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian variance (measure of sharpness)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return float(laplacian_var)


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Check if image is blurry
    
    Args:
        image (np.ndarray): Input image
        threshold (float): Blur threshold (lower = more blurry)
    
    Returns:
        bool: True if blurry, False otherwise
    """
    quality = calculate_image_quality(image)
    return quality < threshold


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("ML UTILITIES TEST")
    print("=" * 60)
    
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\nTest Image Shape:", test_image.shape)
    print("Image Valid:", validate_image(test_image))
    print("Image Quality Score:", calculate_image_quality(test_image))
    print("Is Blurry:", is_blurry(test_image))
    
    # Test normalization
    normalized = normalize_image(test_image)
    print("\nNormalized Image Range:", normalized.min(), "-", normalized.max())
    
    # Test resize
    resized = resize_image(test_image, (224, 224))
    print("Resized Image Shape:", resized.shape)
    
    print("\n✓ Utilities test completed!")
