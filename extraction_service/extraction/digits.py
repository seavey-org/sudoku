"""Digit extraction from sudoku grids."""
import numpy as np
from typing import List

from ..utils.logging_config import get_logger
from ..models_lib.registry import get_model_registry

logger = get_logger(__name__)


def extract_board_digits_cnn(
    warped: np.ndarray,
    size: int = 9,
    verbose: bool = False
) -> List[List[int]]:
    """Extract board digits using the digit CNN classifier.

    Args:
        warped: Warped grid image (1800x1800 for 9x9)
        size: Grid size (6 or 9)
        verbose: Print debug info

    Returns:
        2D list of digits (0 = empty cell)
    """
    registry = get_model_registry()
    classifier = registry.get_digit_cnn()

    if classifier is None:
        if verbose:
            logger.warning("Digit CNN not available, returning zeros")
        return [[0] * size for _ in range(size)]

    h, w = warped.shape[:2]
    cell_h = h // size
    cell_w = w // size

    board = [[0] * size for _ in range(size)]

    for r in range(size):
        for c in range(size):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w
            cell_img = warped[y1:y2, x1:x2]

            # Extract center region (avoid cage sum numbers in corners)
            # Use 0.12 margin to match training data
            margin = int(min(cell_h, cell_w) * 0.12)
            center = cell_img[margin:cell_h - margin, margin:cell_w - margin]

            if center.size == 0:
                continue

            # Predict digit
            digit, conf = classifier.predict(center)

            # Only accept if confidence is high enough and digit is valid
            if conf >= 0.5 and 1 <= digit <= size:
                board[r][c] = digit
                if verbose:
                    logger.debug(f"Cell [{r},{c}]: digit={digit}, conf={conf:.2f}")
            elif verbose and digit != 0:
                logger.debug(f"Cell [{r},{c}]: rejected digit={digit}, conf={conf:.2f}")

    return board
