"""Cage extraction for killer sudoku."""
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

from ..utils.logging_config import get_logger
from ..models_lib.registry import get_model_registry

logger = get_logger(__name__)


def preprocess_cage_sum_for_cnn(cell_img: np.ndarray) -> np.ndarray:
    """Preprocess cage sum cell for CNN - MUST match training preprocessing.

    This applies the exact same preprocessing used in prepare_cage_sum_cnn_data.py:
    1. Extract top-left corner region (where cage sum appears)
    2. Apply CLAHE normalization
    3. Bilateral filter denoising
    4. Deskewing
    5. Resize to 64x64

    Args:
        cell_img: Full cell image (e.g., 200x200)

    Returns:
        Preprocessed 64x64 grayscale image ready for CNN
    """
    h, w = cell_img.shape[:2]

    # Step 1: Extract top-left corner (40% height, 65% width with margins)
    crop_h_ratio = 0.40
    crop_w_ratio = 0.65
    margin_top = 5
    margin_left = 8

    crop_h = int(h * crop_h_ratio)
    crop_w = int(w * crop_w_ratio)

    # Ensure we don't go out of bounds
    crop_h = min(crop_h, h - margin_top)
    crop_w = min(crop_w, w - margin_left)

    cage_sum_region = cell_img[
        margin_top:margin_top + crop_h,
        margin_left:margin_left + crop_w
    ]

    if cage_sum_region.size == 0:
        cage_sum_region = cell_img

    # Step 2: Convert to grayscale
    if len(cage_sum_region.shape) == 3:
        gray = cv2.cvtColor(cage_sum_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = cage_sum_region.copy()

    # Step 3: Deskewing
    coords = np.column_stack(np.where(gray < 200))
    if len(coords) >= 10:
        moments = cv2.moments(255 - gray)
        if moments['mu02'] != 0:
            skew = moments['mu11'] / moments['mu02']
            angle = np.degrees(np.arctan(skew))
            if abs(angle) <= 15:
                gh, gw = gray.shape
                M = cv2.getRotationMatrix2D((gw // 2, gh // 2), angle, 1.0)
                gray = cv2.warpAffine(gray, M, (gw, gh), borderValue=255)

    # Step 4: CLAHE normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    normalized = clahe.apply(gray)

    # Step 5: Bilateral filter for denoising
    normalized = cv2.bilateralFilter(normalized, 5, 50, 50)

    # Step 6: Resize to 64x64
    resized = cv2.resize(normalized, (64, 64), interpolation=cv2.INTER_AREA)

    # Step 7: Final CLAHE enhancement
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe2.apply(resized)

    return enhanced


def apply_tta_augmentations(img: np.ndarray) -> List[np.ndarray]:
    """Apply test-time augmentations to an image.

    Args:
        img: Input image

    Returns:
        List of augmented images including original
    """
    augmented = [img]

    # Horizontal flip
    augmented.append(cv2.flip(img, 1))

    # Small rotations
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rotated)

    # Brightness variations
    for gamma in [0.9, 1.1]:
        adjusted = np.clip(img.astype(np.float32) * gamma, 0, 255).astype(np.uint8)
        augmented.append(adjusted)

    return augmented


def extract_cage_sums_cnn(
    warped: np.ndarray,
    structure: Dict[str, Any],
    size: int = 9,
    verbose: bool = False,
    use_tta: bool = True
) -> Dict[str, int]:
    """Extract cage sums using CNN classifier with optional test-time augmentation.

    Args:
        warped: Warped grid image (1800x1800)
        structure: Structure dict with 'cages' list containing 'id' and 'sum_cell'
        size: Grid size (6 or 9)
        verbose: Print debug info
        use_tta: Use test-time augmentation for more robust predictions

    Returns:
        Dict of cage_id -> detected sum
    """
    registry = get_model_registry()
    model, device, label_mapping = registry.get_cage_sum_cnn()

    if model is None:
        return {}

    import torch

    cage_sums = {}
    h, w = warped.shape[:2]
    cell_h = h // size
    cell_w = w // size

    for cage in structure.get('cages', []):
        if not cage.get('sum_cell'):
            continue

        row, col = cage['sum_cell']
        cage_id = cage['id']

        # Extract cell
        y1 = row * cell_h
        y2 = (row + 1) * cell_h
        x1 = col * cell_w
        x2 = (col + 1) * cell_w
        cell_img = warped[y1:y2, x1:x2]

        if use_tta:
            augmented_cells = apply_tta_augmentations(cell_img)
            all_probs = []

            for aug_cell in augmented_cells:
                preprocessed = preprocess_cage_sum_for_cnn(aug_cell)
                normalized = preprocessed.astype(np.float32) / 255.0
                tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.append(probs)

            avg_probs = torch.mean(torch.stack(all_probs), dim=0)
            conf, pred_idx = torch.max(avg_probs, dim=1)
        else:
            preprocessed = preprocess_cage_sum_for_cnn(cell_img)
            normalized = preprocessed.astype(np.float32) / 255.0
            tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

        pred_sum = label_mapping.get(pred_idx.item(), None)
        confidence = conf.item()

        if pred_sum is not None and 1 <= pred_sum <= 45:
            cage_sums[cage_id] = pred_sum
            if verbose:
                logger.debug(
                    f"CNN cage {cage_id} at [{row},{col}]: {pred_sum} (conf={confidence:.2f})"
                )
        elif verbose:
            logger.debug(f"CNN cage {cage_id} at [{row},{col}]: no valid prediction")

    return cage_sums
