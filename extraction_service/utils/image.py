"""Image processing utilities."""
import cv2
import numpy as np
from enum import Enum
from typing import Tuple, Dict, Any


class ImageQuality(Enum):
    """Image quality classification for preprocessing routing."""
    STANDARD = "standard"
    DARK_MODE = "dark_mode"
    LOW_CONTRAST = "low_contrast"
    MIXED_LIGHTING = "mixed_lighting"
    BLURRY = "blurry"
    UNKNOWN = "unknown"


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if needed.

    Args:
        img: Input image (BGR or grayscale)

    Returns:
        Grayscale image
    """
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def add_border(img: np.ndarray, size: int = 20, value: int = 0) -> np.ndarray:
    """Add constant border around image.

    Args:
        img: Input image
        size: Border size in pixels
        value: Border pixel value

    Returns:
        Image with border
    """
    return cv2.copyMakeBorder(
        img, size, size, size, size,
        cv2.BORDER_CONSTANT, value=value
    )


def classify_image_quality(img: np.ndarray) -> Tuple[ImageQuality, Dict[str, Any]]:
    """Classify image quality to route preprocessing.

    Analyzes image characteristics to determine the best preprocessing
    strategy.

    Args:
        img: Input image (BGR or grayscale)

    Returns:
        Tuple of (ImageQuality constant, metrics dict)
    """
    if img is None or img.size == 0:
        return ImageQuality.UNKNOWN, {}

    gray = to_grayscale(img)

    # Compute metrics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # Edge sharpness via Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edge_sharpness = laplacian.var()

    # Dark pixel ratio (pixels < 80)
    dark_ratio = np.sum(gray < 80) / gray.size

    # Light pixel ratio (pixels > 200)
    light_ratio = np.sum(gray > 200) / gray.size

    metrics = {
        'mean_intensity': float(mean_intensity),
        'std_intensity': float(std_intensity),
        'edge_sharpness': float(edge_sharpness),
        'dark_ratio': float(dark_ratio),
        'light_ratio': float(light_ratio)
    }

    # Classification logic
    if mean_intensity < 80 and dark_ratio > 0.6:
        return ImageQuality.DARK_MODE, metrics

    if mean_intensity > 220 or (mean_intensity > 180 and std_intensity < 30):
        return ImageQuality.LOW_CONTRAST, metrics

    if std_intensity > 70:
        return ImageQuality.MIXED_LIGHTING, metrics

    if edge_sharpness < 100:
        return ImageQuality.BLURRY, metrics

    return ImageQuality.STANDARD, metrics


def get_preprocessing_params(quality: ImageQuality) -> Dict[str, Any]:
    """Get preprocessing parameters based on image quality.

    Args:
        quality: Image quality classification

    Returns:
        Dict of preprocessing parameters
    """
    params = {
        'threshold_method': 'otsu',
        'clahe_clip': 3.0,
        'confidence_threshold': 0.35,
        'use_hsv': False,
        'use_percentile_threshold': False,
        'apply_sharpening': False,
        'multi_scale': False
    }

    if quality == ImageQuality.DARK_MODE:
        params['use_percentile_threshold'] = True
        params['use_hsv'] = True
        params['clahe_clip'] = 4.0

    elif quality == ImageQuality.LOW_CONTRAST:
        params['threshold_method'] = 'adaptive'
        params['clahe_clip'] = 5.0
        params['confidence_threshold'] = 0.30

    elif quality == ImageQuality.MIXED_LIGHTING:
        params['threshold_method'] = 'adaptive'
        params['multi_scale'] = True

    elif quality == ImageQuality.BLURRY:
        params['apply_sharpening'] = True
        params['clahe_clip'] = 4.0

    return params


def auto_invert(binary: np.ndarray) -> np.ndarray:
    """Invert if background is white (corners are bright).

    Args:
        binary: Binary image

    Returns:
        Possibly inverted image
    """
    h, w = binary.shape
    corner_size = max(5, h // 10)
    corners = [
        binary[:corner_size, :corner_size],
        binary[:corner_size, -corner_size:],
        binary[-corner_size:, :corner_size],
        binary[-corner_size:, -corner_size:]
    ]
    corner_mean = np.mean([np.mean(c) for c in corners])

    if corner_mean > 128:
        return 255 - binary
    return binary


def preprocess_hsv_dark_mode(img: np.ndarray) -> np.ndarray:
    """Preprocess dark mode images using HSV color space.

    For dark backgrounds with light digits, uses the Value channel
    and percentile-based thresholding.

    Args:
        img: BGR input image

    Returns:
        Preprocessed grayscale image (digits white on black)
    """
    if len(img.shape) != 3:
        gray = img
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, _, gray = cv2.split(hsv)

    # Percentile-based thresholding
    p15 = np.percentile(gray, 15)
    p85 = np.percentile(gray, 85)

    if p85 - p15 > 25:
        threshold = (p15 + p85) / 2
        binary = (gray > threshold).astype(np.uint8) * 255
    else:
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, -5
        )

    return binary
