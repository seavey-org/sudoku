"""Unit tests for OCR preprocessing functions."""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from extraction.ocr import fix_digit_confusion, OCRPreprocessor


class TestFixDigitConfusion:
    """Tests for fix_digit_confusion function."""

    def test_corrects_7_to_1_when_narrow(self):
        """Test that narrow 7s are corrected to 1s."""
        digit, conf = fix_digit_confusion(7, bbox_w=10, bbox_h=30, conf=0.9)

        assert digit == 1
        assert conf == 0.9 * 0.95

    def test_preserves_7_when_wide(self):
        """Test that wide 7s are preserved."""
        digit, conf = fix_digit_confusion(7, bbox_w=25, bbox_h=30, conf=0.9)

        assert digit == 7
        assert conf == 0.9

    def test_corrects_1_to_7_when_very_wide(self):
        """Test that very wide 1s are corrected to 7s."""
        digit, conf = fix_digit_confusion(1, bbox_w=25, bbox_h=30, conf=0.9)

        assert digit == 7
        assert conf == 0.9 * 0.95

    def test_preserves_1_when_narrow(self):
        """Test that narrow 1s are preserved."""
        digit, conf = fix_digit_confusion(1, bbox_w=10, bbox_h=30, conf=0.9)

        assert digit == 1
        assert conf == 0.9

    def test_preserves_other_digits(self):
        """Test that other digits are not modified."""
        for d in [2, 3, 4, 5, 6, 8, 9]:
            digit, conf = fix_digit_confusion(d, bbox_w=20, bbox_h=30, conf=0.9)
            assert digit == d
            assert conf == 0.9

    def test_handles_zero_height(self):
        """Test handling of zero height bounding box."""
        digit, conf = fix_digit_confusion(7, bbox_w=10, bbox_h=0, conf=0.9)

        assert digit == 7
        assert conf == 0.9

    def test_handles_negative_height(self):
        """Test handling of negative height bounding box."""
        digit, conf = fix_digit_confusion(7, bbox_w=10, bbox_h=-5, conf=0.9)

        assert digit == 7
        assert conf == 0.9


class TestOCRPreprocessor:
    """Tests for OCRPreprocessor class."""

    def test_prepare_digit_returns_correct_shape(self):
        """Test that prepare_digit returns image with border."""
        # Create a simple grayscale cell image
        cell = np.ones((50, 50), dtype=np.uint8) * 200

        result = OCRPreprocessor.prepare_digit(cell, scale=2)

        # Original 50x50, scaled 2x = 100x100, then 20px border = 140x140
        assert result.shape == (140, 140)

    def test_prepare_digit_handles_empty_image(self):
        """Test handling of empty image."""
        empty = np.array([], dtype=np.uint8).reshape(0, 0)

        result = OCRPreprocessor.prepare_digit(empty)

        assert result.size == 0

    def test_prepare_digit_handles_color_image(self):
        """Test conversion of color image."""
        color_cell = np.ones((50, 50, 3), dtype=np.uint8) * 200

        result = OCRPreprocessor.prepare_digit(color_cell, scale=2)

        # Should convert to grayscale and process
        assert len(result.shape) == 2

    def test_prepare_cage_sum_returns_correct_shape(self):
        """Test that prepare_cage_sum returns image with border."""
        cell = np.ones((30, 40), dtype=np.uint8) * 200

        result = OCRPreprocessor.prepare_cage_sum(cell, scale=4)

        # Original 30x40, scaled 4x = 120x160, then 20px border = 160x200
        assert result.shape == (160, 200)

    def test_prepare_cage_sum_handles_empty_image(self):
        """Test handling of empty image."""
        empty = np.array([], dtype=np.uint8).reshape(0, 0)

        result = OCRPreprocessor.prepare_cage_sum(empty)

        assert result.size == 0

    def test_prepare_digit_otsu_method(self):
        """Test OTSU thresholding method."""
        cell = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

        result = OCRPreprocessor.prepare_digit(cell, method='otsu')

        # Result should be binary
        unique = np.unique(result)
        assert len(unique) <= 2

    def test_prepare_digit_adaptive_method(self):
        """Test adaptive thresholding method."""
        cell = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

        result = OCRPreprocessor.prepare_digit(cell, method='adaptive')

        # Result should be binary
        unique = np.unique(result)
        assert len(unique) <= 2
