#!/usr/bin/env python3
"""Test classic sudoku OCR extraction against expected JSON files."""

import json
import os
import sys
from pathlib import Path
import requests

# Use path relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = str(SCRIPT_DIR / "classic_sudoku" / "9x9")
SERVICE_URL = "http://127.0.0.1:5001/extract-classic"


def load_expected(json_path):
    """Load expected board from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('board', [])


def extract_board(image_path):
    """Extract board from image using the service."""
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'size': '9'}
        resp = requests.post(SERVICE_URL, files=files, data=data, timeout=120)

    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text}"

    result = resp.json()
    if 'error' in result:
        return None, result['error']

    return result.get('board', []), None


def compare_boards(expected, actual):
    """Compare two boards and return differences."""
    if len(expected) != len(actual):
        return [(f"Row count mismatch: expected {len(expected)}, got {len(actual)}", -1, -1)]

    differences = []
    for r in range(len(expected)):
        if len(expected[r]) != len(actual[r]):
            differences.append((f"Col count mismatch in row {r}", r, -1))
            continue
        for c in range(len(expected[r])):
            exp_val = expected[r][c]
            act_val = actual[r][c]
            # Only count as error if expected has a digit but actual is wrong
            # (Missing digits are acceptable, false positives are not)
            if exp_val != 0 and act_val != exp_val:
                differences.append((f"[{r},{c}]: expected {exp_val}, got {act_val}", r, c))
            elif exp_val == 0 and act_val != 0:
                differences.append((f"[{r},{c}]: expected empty, got {act_val} (false positive)", r, c))

    return differences


def count_matches(expected, actual):
    """Count correctly extracted digits."""
    total_expected = 0
    total_correct = 0
    false_positives = 0

    for r in range(len(expected)):
        for c in range(len(expected[r])):
            exp_val = expected[r][c]
            act_val = actual[r][c] if r < len(actual) and c < len(actual[r]) else -1

            if exp_val != 0:
                total_expected += 1
                if act_val == exp_val:
                    total_correct += 1
            elif act_val != 0:
                false_positives += 1

    return total_expected, total_correct, false_positives


def main():
    """Run tests on all images."""
    # Find all test images
    test_files = []
    for f in os.listdir(TEST_DIR):
        if f.endswith(('.png', '.jpeg', '.jpg')):
            name = os.path.splitext(f)[0]
            json_path = os.path.join(TEST_DIR, f"{name}.json")
            if os.path.exists(json_path):
                test_files.append((name, os.path.join(TEST_DIR, f), json_path))

    test_files.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])

    print(f"Testing {len(test_files)} classic sudoku images...\n")

    total_images = 0
    perfect_images = 0
    total_expected_digits = 0
    total_correct_digits = 0
    total_false_positives = 0

    failed_images = []

    for name, img_path, json_path in test_files:
        total_images += 1
        print(f"Testing image {name}...", end=" ", flush=True)

        expected = load_expected(json_path)
        actual, error = extract_board(img_path)

        if error:
            print(f"ERROR: {error}")
            failed_images.append((name, f"Extraction failed: {error}"))
            continue

        expected_count, correct_count, fp_count = count_matches(expected, actual)
        total_expected_digits += expected_count
        total_correct_digits += correct_count
        total_false_positives += fp_count

        differences = compare_boards(expected, actual)

        if not differences:
            print(f"PASS ({correct_count}/{expected_count} digits)")
            perfect_images += 1
        else:
            # Check if it's just missing digits vs wrong digits
            missing = sum(1 for d in differences if "expected empty" not in d[0])
            false_pos = len(differences) - missing
            print(f"ISSUES: {len(differences)} ({missing} wrong, {false_pos} false positives)")
            if len(differences) <= 5:
                for diff, r, c in differences:
                    print(f"    {diff}")
            else:
                for diff, r, c in differences[:3]:
                    print(f"    {diff}")
                print(f"    ... and {len(differences) - 3} more")
            failed_images.append((name, differences))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Images: {perfect_images}/{total_images} perfect ({100*perfect_images/total_images:.1f}%)")
    print(f"Digits: {total_correct_digits}/{total_expected_digits} correct ({100*total_correct_digits/total_expected_digits:.1f}%)")
    print(f"False positives: {total_false_positives}")

    if failed_images:
        print(f"\nFailed images: {[name for name, _ in failed_images]}")

    # Return exit code
    return 0 if perfect_images == total_images else 1


if __name__ == '__main__':
    sys.exit(main())
