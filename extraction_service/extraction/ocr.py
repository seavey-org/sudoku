"""OCR preprocessing and utilities."""
import cv2
import numpy as np
from typing import Tuple

from ..utils.image import to_grayscale, add_border, auto_invert


def fix_digit_confusion(
    digit: int,
    bbox_w: int,
    bbox_h: int,
    conf: float = 1.0
) -> Tuple[int, float]:
    """Fix common digit confusions based on aspect ratio.

    OCR often misreads '1' as '7'. Use aspect ratio to correct:
    - '1' is narrow (ar < 0.65) - if OCR says '7' but shape is narrow, correct to '1'
    - '7' is wider - only correct 1→7 if very wide (ar > 0.789)

    Args:
        digit: Detected digit
        bbox_w: Bounding box width
        bbox_h: Bounding box height
        conf: Original confidence

    Returns:
        Tuple of (corrected_digit, adjusted_confidence)
    """
    if bbox_h <= 0:
        return digit, conf

    ar = bbox_w / bbox_h

    # Narrow shapes labeled '7' are likely '1'
    if digit == 7 and ar < 0.65:
        return 1, conf * 0.95

    # Only very wide shapes labeled '1' might be '7'
    if digit == 1 and ar > 0.789:
        return 7, conf * 0.95

    return digit, conf


class OCRPreprocessor:
    """Unified preprocessing for OCR operations."""

    @staticmethod
    def prepare_digit(
        crop: np.ndarray,
        scale: int = 2,
        method: str = 'otsu'
    ) -> np.ndarray:
        """Prepare cell crop for digit OCR.

        Args:
            crop: Cell image (grayscale or color)
            scale: Upscaling factor
            method: 'otsu' or 'adaptive'

        Returns:
            Preprocessed binary image with border
        """
        if crop.size == 0:
            return crop

        # Convert to grayscale
        gray = to_grayscale(crop)

        # Scale up
        scaled = cv2.resize(
            gray, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(scaled)

        # Threshold
        if method == 'otsu':
            _, binary = cv2.threshold(
                enhanced, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            binary = cv2.adaptiveThreshold(
                enhanced, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 2
            )

        # Auto-invert based on corner sampling
        binary = auto_invert(binary)

        return add_border(binary)

    @staticmethod
    def prepare_cage_sum(crop: np.ndarray, scale: int = 4) -> np.ndarray:
        """Prepare cage sum region for OCR.

        Args:
            crop: Cage sum region image
            scale: Upscaling factor

        Returns:
            Preprocessed binary image with white border
        """
        if crop.size == 0:
            return crop

        gray = to_grayscale(crop)
        scaled = cv2.resize(
            gray, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(scaled)

        _, binary = cv2.threshold(
            enhanced, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return add_border(binary, value=255)
