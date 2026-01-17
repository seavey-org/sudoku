"""Unit tests for image utility functions."""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.image import (
    to_grayscale,
    add_border,
    classify_image_quality,
    ImageQuality,
    auto_invert,
    get_preprocessing_params
)


class TestToGrayscale:
    """Tests for to_grayscale function."""

    def test_converts_color_to_grayscale(self):
        """Test conversion of BGR image to grayscale."""
        # Create a simple 3-channel image
        color_img = np.zeros((10, 10, 3), dtype=np.uint8)
        color_img[:, :, 0] = 100  # Blue
        color_img[:, :, 1] = 150  # Green
        color_img[:, :, 2] = 200  # Red

        result = to_grayscale(color_img)

        assert len(result.shape) == 2
        assert result.shape == (10, 10)

    def test_returns_grayscale_unchanged(self):
        """Test that grayscale image passes through unchanged."""
        gray_img = np.ones((10, 10), dtype=np.uint8) * 128

        result = to_grayscale(gray_img)

        assert len(result.shape) == 2
        np.testing.assert_array_equal(result, gray_img)


class TestAddBorder:
    """Tests for add_border function."""

    def test_adds_correct_size_border(self):
        """Test that border is added with correct size."""
        img = np.ones((10, 10), dtype=np.uint8) * 255
        border_size = 5

        result = add_border(img, size=border_size, value=0)

        assert result.shape == (20, 20)

    def test_border_has_correct_value(self):
        """Test that border pixels have the specified value."""
        img = np.ones((10, 10), dtype=np.uint8) * 255
        border_size = 5
        border_value = 128

        result = add_border(img, size=border_size, value=border_value)

        # Check border pixels
        assert result[0, 0] == border_value
        assert result[0, 5] == border_value
        # Check interior pixels (should be unchanged)
        assert result[5, 5] == 255


class TestClassifyImageQuality:
    """Tests for classify_image_quality function."""

    def test_classifies_dark_mode_image(self):
        """Test classification of dark mode image."""
        # Create a dark image
        dark_img = np.ones((100, 100), dtype=np.uint8) * 30

        quality, metrics = classify_image_quality(dark_img)

        assert quality == ImageQuality.DARK_MODE
        assert metrics['mean_intensity'] < 80
        assert metrics['dark_ratio'] > 0.6

    def test_classifies_standard_image(self):
        """Test classification of standard image."""
        # Create a standard contrast image
        img = np.ones((100, 100), dtype=np.uint8) * 128

        quality, metrics = classify_image_quality(img)

        assert quality == ImageQuality.STANDARD

    def test_classifies_low_contrast_image(self):
        """Test classification of low contrast/washed out image."""
        # Create a very bright image
        bright_img = np.ones((100, 100), dtype=np.uint8) * 230

        quality, metrics = classify_image_quality(bright_img)

        assert quality == ImageQuality.LOW_CONTRAST

    def test_handles_empty_image(self):
        """Test handling of empty image."""
        empty_img = np.array([], dtype=np.uint8)

        quality, metrics = classify_image_quality(empty_img)

        assert quality == ImageQuality.UNKNOWN
        assert metrics == {}

    def test_handles_none_input(self):
        """Test handling of None input."""
        quality, metrics = classify_image_quality(None)

        assert quality == ImageQuality.UNKNOWN
        assert metrics == {}


class TestAutoInvert:
    """Tests for auto_invert function."""

    def test_inverts_white_background(self):
        """Test that white background images are inverted."""
        # Create image with white corners (background)
        img = np.ones((100, 100), dtype=np.uint8) * 255
        img[40:60, 40:60] = 0  # Black content in center

        result = auto_invert(img)

        # Corners should now be dark (inverted)
        assert result[0, 0] == 0

    def test_preserves_dark_background(self):
        """Test that dark background images are not inverted."""
        # Create image with dark corners (background)
        img = np.zeros((100, 100), dtype=np.uint8)
        img[40:60, 40:60] = 255  # White content in center

        result = auto_invert(img)

        # Corners should still be dark
        assert result[0, 0] == 0


class TestGetPreprocessingParams:
    """Tests for get_preprocessing_params function."""

    def test_standard_quality_params(self):
        """Test parameters for standard quality images."""
        params = get_preprocessing_params(ImageQuality.STANDARD)

        assert params['threshold_method'] == 'otsu'
        assert params['clahe_clip'] == 3.0
        assert params['confidence_threshold'] == 0.35

    def test_dark_mode_params(self):
        """Test parameters for dark mode images."""
        params = get_preprocessing_params(ImageQuality.DARK_MODE)

        assert params['use_hsv'] is True
        assert params['use_percentile_threshold'] is True
        assert params['clahe_clip'] == 4.0

    def test_low_contrast_params(self):
        """Test parameters for low contrast images."""
        params = get_preprocessing_params(ImageQuality.LOW_CONTRAST)

        assert params['threshold_method'] == 'adaptive'
        assert params['clahe_clip'] == 5.0
        assert params['confidence_threshold'] == 0.30

    def test_blurry_params(self):
        """Test parameters for blurry images."""
        params = get_preprocessing_params(ImageQuality.BLURRY)

        assert params['apply_sharpening'] is True
        assert params['clahe_clip'] == 4.0
