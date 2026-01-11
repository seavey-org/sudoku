#!/usr/bin/env python3
"""
Extract cage sum crops from killer sudoku test images for training.
Saves crops with ground truth labels for CNN training.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path

# Puzzles with invalid ground truth - exclude from training
EXCLUDED_KILLER_PUZZLES = {'23', '24', '26', '27', '28'}

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_warped_grid(image_path):
    """Import get_warped_grid from app.py"""
    from app import get_warped_grid as gwg
    return gwg(image_path)

def get_cell_boundaries(warped, size):
    """Import get_cell_boundaries from app.py"""
    from app import get_cell_boundaries as gcb
    return gcb(warped, size)

def extract_cage_sum_crops(image_path, ground_truth_path, output_dir, size=9):
    """Extract cage sum crops from a killer sudoku image.

    Args:
        image_path: Path to puzzle image
        ground_truth_path: Path to ground truth JSON
        output_dir: Directory to save crops
        size: Grid size (6 or 9)

    Returns:
        List of dicts with crop info: {image, label, puzzle_id, cell_pos}
    """
    # Load ground truth
    with open(ground_truth_path) as f:
        gt = json.load(f)

    cage_map = gt['cage_map']
    cage_sums = gt['cage_sums']

    # Get warped grid
    warped = get_warped_grid(image_path)
    if warped is None:
        print(f"Failed to get warped grid for {image_path}")
        return []

    # Get cell boundaries
    x_bounds, y_bounds = get_cell_boundaries(warped, size)

    # Find anchor cells (top-left cell of each cage)
    anchor_cells = {}  # cage_id -> (r, c)

    for cage_id in cage_sums.keys():
        # Find all cells in this cage
        cells_in_cage = []
        for r in range(size):
            for c in range(size):
                if cage_map[r][c] == cage_id:
                    cells_in_cage.append((r, c))

        if cells_in_cage:
            # Anchor is top-left (min row, then min col)
            anchor = min(cells_in_cage, key=lambda x: (x[0], x[1]))
            anchor_cells[cage_id] = anchor

    # Extract crops
    crops = []
    puzzle_id = os.path.basename(image_path).replace('.png', '').replace('.jpg', '')

    for cage_id, (r, c) in anchor_cells.items():
        cage_sum = cage_sums[cage_id]

        # Get cell bounds
        y1, y2 = y_bounds[r], y_bounds[r + 1]
        x1, x2 = x_bounds[c], x_bounds[c + 1]

        # Extract full cell
        cell_crop = warped[y1:y2, x1:x2].copy()

        if cell_crop.size == 0:
            continue

        # Save crop info
        crop_info = {
            'image': cell_crop,
            'label': cage_sum,
            'puzzle_id': puzzle_id,
            'cage_id': cage_id,
            'cell_pos': (r, c),
            'cell_size': (y2 - y1, x2 - x1)
        }
        crops.append(crop_info)

        # Save individual crop file
        filename = f"{puzzle_id}_cage_{cage_id}_sum_{cage_sum}_r{r}_c{c}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, cell_crop)

    return crops

def main():
    """Extract cage sum training data from all killer sudoku test puzzles."""

    # Setup paths
    test_data_dir = Path(__file__).parent.parent / 'test_data' / 'killer_sudoku'
    output_dir = Path(__file__).parent / 'training_data' / 'cage_sums'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting cage sum crops to: {output_dir}")
    print("=" * 80)

    all_crops = []
    stats = {
        'total_puzzles': 0,
        'total_crops': 0,
        'by_sum': {},
        'failed_puzzles': []
    }

    # Process 9x9 puzzles
    puzzles_9x9 = sorted((test_data_dir / '9x9').glob('*.png'))

    for img_path in puzzles_9x9:
        json_path = img_path.with_suffix('.json')

        if not json_path.exists():
            print(f"Skipping {img_path.name} - no ground truth")
            continue

        # Skip puzzles with invalid ground truth
        puzzle_num = img_path.stem
        if puzzle_num in EXCLUDED_KILLER_PUZZLES:
            print(f"Skipping {img_path.name} - invalid ground truth")
            continue

        print(f"\nProcessing {img_path.name}...")

        try:
            crops = extract_cage_sum_crops(
                str(img_path),
                str(json_path),
                str(output_dir),
                size=9
            )

            all_crops.extend(crops)
            stats['total_puzzles'] += 1
            stats['total_crops'] += len(crops)

            # Count by sum value
            for crop in crops:
                label = crop['label']
                stats['by_sum'][label] = stats['by_sum'].get(label, 0) + 1

            print(f"  Extracted {len(crops)} cage sum crops")

        except Exception as e:
            print(f"  ERROR: {e}")
            stats['failed_puzzles'].append(img_path.name)

    # Process 6x6 puzzles
    if (test_data_dir / '6x6').exists():
        puzzles_6x6 = sorted((test_data_dir / '6x6').glob('*.png'))

        for img_path in puzzles_6x6:
            json_path = img_path.with_suffix('.json')

            if not json_path.exists():
                continue

            print(f"\nProcessing {img_path.name}...")

            try:
                crops = extract_cage_sum_crops(
                    str(img_path),
                    str(json_path),
                    str(output_dir),
                    size=6
                )

                all_crops.extend(crops)
                stats['total_puzzles'] += 1
                stats['total_crops'] += len(crops)

                for crop in crops:
                    label = crop['label']
                    stats['by_sum'][label] = stats['by_sum'].get(label, 0) + 1

                print(f"  Extracted {len(crops)} cage sum crops")

            except Exception as e:
                print(f"  ERROR: {e}")
                stats['failed_puzzles'].append(img_path.name)

    # Print statistics
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total puzzles processed: {stats['total_puzzles']}")
    print(f"Total cage sum crops: {stats['total_crops']}")
    print(f"Failed puzzles: {len(stats['failed_puzzles'])}")

    if stats['failed_puzzles']:
        print(f"  {', '.join(stats['failed_puzzles'])}")

    print(f"\nCrops per sum value (top 20):")
    sorted_sums = sorted(stats['by_sum'].items(), key=lambda x: -x[1])[:20]
    for sum_val, count in sorted_sums:
        print(f"  Sum {sum_val:2d}: {count:3d} crops")

    # Check for problematic sums from puzzles 7 and 21
    print(f"\nProblematic sum values:")
    problem_sums = [2, 8, 28]
    for sum_val in problem_sums:
        count = stats['by_sum'].get(sum_val, 0)
        print(f"  Sum {sum_val:2d}: {count:3d} crops")

    # Save metadata
    metadata_path = output_dir / 'extraction_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump({
            'total_puzzles': stats['total_puzzles'],
            'total_crops': stats['total_crops'],
            'crops_per_sum': stats['by_sum'],
            'failed_puzzles': stats['failed_puzzles']
        }, f, indent=2)

    print(f"\nMetadata saved to: {metadata_path}")
    print(f"Crops saved to: {output_dir}")

    return all_crops, stats

if __name__ == '__main__':
    crops, stats = main()
