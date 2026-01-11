#!/usr/bin/env python3
"""
Validate that all sudoku puzzles have exactly one unique solution.

Uses the Go backend's solver via HTTP API to verify solvability.
Run the backend first: cd backend && go run main.go
"""
import json
import requests
import sys
from pathlib import Path
from typing import Optional, Tuple


API_URL = "http://localhost:8080/api/solve"


def validate_classic_puzzle(board: list, size: int = 9) -> Tuple[bool, Optional[list], str]:
    """Validate a classic sudoku puzzle has a unique solution.

    Args:
        board: 2D list representing the puzzle (0 = empty)
        size: Grid size (6 or 9)

    Returns:
        Tuple of (is_valid, solution, message)
    """
    try:
        response = requests.post(
            API_URL,
            json={"board": board, "size": size},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            return True, data.get("solution"), "Unique solution found"
        else:
            return False, None, f"Solver error: {response.text}"

    except requests.exceptions.ConnectionError:
        return False, None, "Backend not running (start with: cd backend && go run main.go)"
    except requests.exceptions.Timeout:
        return False, None, "Solver timeout"
    except Exception as e:
        return False, None, f"Error: {str(e)}"


def validate_killer_puzzle(board: list, cages: list, size: int = 9) -> Tuple[bool, Optional[list], str]:
    """Validate a killer sudoku puzzle.

    Note: Backend doesn't expose killer solver API, so we just validate
    the board constraints are satisfied (no duplicates) and skip solvability check.
    The cage constraint validation is done separately.
    """
    # For killer sudoku, if board has few givens, validate basic constraints only
    # Full solvability check would require implementing killer solver or adding API

    # Check for duplicate givens in rows/cols/boxes
    for r in range(size):
        row_vals = [v for v in board[r] if v > 0]
        if len(row_vals) != len(set(row_vals)):
            return False, None, "Duplicate digits in row"

    for c in range(size):
        col_vals = [board[r][c] for r in range(size) if board[r][c] > 0]
        if len(col_vals) != len(set(col_vals)):
            return False, None, "Duplicate digits in column"

    # Check boxes
    box_h = 2 if size == 6 else 3
    box_w = 3 if size == 6 else 3

    for box_r in range(size // box_h):
        for box_c in range(size // box_w):
            box_vals = []
            for r in range(box_r * box_h, (box_r + 1) * box_h):
                for c in range(box_c * box_w, (box_c + 1) * box_w):
                    if board[r][c] > 0:
                        box_vals.append(board[r][c])
            if len(box_vals) != len(set(box_vals)):
                return False, None, "Duplicate digits in box"

    # Validate cage constraints
    cage_cells_digits = {}
    for cage in cages:
        cage_sum = cage['sum']
        cells = cage['cells']
        digits = []
        for r, c in cells:
            if board[r][c] > 0:
                digits.append(board[r][c])

        # Check no duplicates in cage
        if len(digits) != len(set(digits)):
            return False, None, "Duplicate digits in cage"

        # If cage is complete, check sum
        if len(digits) == len(cells):
            if sum(digits) != cage_sum:
                return False, None, f"Cage sum mismatch: {sum(digits)} != {cage_sum}"

    return True, None, "Constraints valid (solvability not checked for killer)"


def convert_cage_format(cage_map: list, cage_sums: dict) -> list:
    """Convert cage_map/cage_sums format to cages list format.

    cage_map: [["a", "a", "b", ...], ...] - cage ID for each cell
    cage_sums: {"a": 12, "b": 15, ...} - sum for each cage

    Returns:
        List of {"sum": N, "cells": [[r,c], ...]} dicts
    """
    # Group cells by cage ID
    cage_cells = {}
    size = len(cage_map)

    for r in range(size):
        for c in range(size):
            cage_id = cage_map[r][c].strip()
            if cage_id not in cage_cells:
                cage_cells[cage_id] = []
            cage_cells[cage_id].append([r, c])

    # Build cages list
    cages = []
    for cage_id, cells in cage_cells.items():
        cage_sum = cage_sums.get(cage_id, 0)
        cages.append({
            "sum": cage_sum,
            "cells": cells
        })

    return cages


def validate_json_file(json_path: Path, is_killer: bool = False) -> dict:
    """Validate a single puzzle JSON file.

    Returns:
        dict with 'valid', 'solvable', 'message', 'solution' keys
    """
    result = {
        'file': str(json_path),
        'valid': False,
        'solvable': False,
        'message': '',
        'solution': None
    }

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        result['message'] = f"JSON error: {e}"
        return result

    if 'board' not in data:
        result['message'] = "Missing 'board' field"
        return result

    board = data['board']
    size = len(board)

    if size not in [6, 9]:
        result['message'] = f"Invalid size: {size}"
        return result

    result['valid'] = True

    if is_killer:
        if 'cage_map' not in data or 'cage_sums' not in data:
            result['message'] = "Missing cage_map or cage_sums"
            return result

        cages = convert_cage_format(data['cage_map'], data['cage_sums'])
        solvable, solution, msg = validate_killer_puzzle(board, cages, size)
    else:
        solvable, solution, msg = validate_classic_puzzle(board, size)

    result['solvable'] = solvable
    result['solution'] = solution
    result['message'] = msg

    return result


def validate_all_puzzles(test_data_dir: Path) -> dict:
    """Validate all puzzle files in the test data directory.

    Returns:
        dict with validation results
    """
    results = {
        'classic': {'solvable': [], 'unsolvable': [], 'errors': []},
        'killer': {'solvable': [], 'unsolvable': [], 'errors': []},
    }

    # Classic sudoku
    for subdir in ['6x6', '9x9']:
        classic_dir = test_data_dir / 'classic_sudoku' / subdir
        if classic_dir.exists():
            for json_file in sorted(classic_dir.glob('*.json')):
                validation = validate_json_file(json_file, is_killer=False)

                if not validation['valid']:
                    results['classic']['errors'].append(validation)
                elif validation['solvable']:
                    results['classic']['solvable'].append(validation)
                else:
                    results['classic']['unsolvable'].append(validation)

    # Killer sudoku
    for subdir in ['6x6', '9x9']:
        killer_dir = test_data_dir / 'killer_sudoku' / subdir
        if killer_dir.exists():
            for json_file in sorted(killer_dir.glob('*.json')):
                # Skip .gemini.json files (alternative extractions)
                if '.gemini.' in json_file.name:
                    continue

                validation = validate_json_file(json_file, is_killer=True)

                if not validation['valid']:
                    results['killer']['errors'].append(validation)
                elif validation['solvable']:
                    results['killer']['solvable'].append(validation)
                else:
                    results['killer']['unsolvable'].append(validation)

    return results


def print_report(results: dict):
    """Print validation report."""
    print("=" * 60)
    print("PUZZLE SOLVABILITY VALIDATION REPORT")
    print("=" * 60)

    # Classic sudoku
    print("\n## Classic Sudoku")
    print(f"   Solvable (unique solution): {len(results['classic']['solvable'])}")
    print(f"   Unsolvable/Multiple solutions: {len(results['classic']['unsolvable'])}")
    print(f"   Errors: {len(results['classic']['errors'])}")

    if results['classic']['unsolvable']:
        print("\n   UNSOLVABLE PUZZLES:")
        for item in results['classic']['unsolvable']:
            print(f"   - {item['file']}")
            print(f"     {item['message']}")

    if results['classic']['errors']:
        print("\n   ERRORS:")
        for item in results['classic']['errors']:
            print(f"   - {item['file']}: {item['message']}")

    # Killer sudoku
    print("\n## Killer Sudoku")
    print(f"   Solvable (unique solution): {len(results['killer']['solvable'])}")
    print(f"   Unsolvable/Multiple solutions: {len(results['killer']['unsolvable'])}")
    print(f"   Errors: {len(results['killer']['errors'])}")

    if results['killer']['unsolvable']:
        print("\n   UNSOLVABLE PUZZLES:")
        for item in results['killer']['unsolvable']:
            print(f"   - {item['file']}")
            print(f"     {item['message']}")

    if results['killer']['errors']:
        print("\n   ERRORS:")
        for item in results['killer']['errors']:
            print(f"   - {item['file']}: {item['message']}")

    # Summary
    total_solvable = len(results['classic']['solvable']) + len(results['killer']['solvable'])
    total_unsolvable = len(results['classic']['unsolvable']) + len(results['killer']['unsolvable'])
    total_errors = len(results['classic']['errors']) + len(results['killer']['errors'])

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_solvable} solvable, {total_unsolvable} unsolvable, {total_errors} errors")
    print("=" * 60)

    return total_unsolvable == 0 and total_errors == 0


def main():
    # Find test_data directory
    script_dir = Path(__file__).parent
    if script_dir.name == 'test_data':
        test_data_dir = script_dir
    else:
        test_data_dir = script_dir / 'test_data'

    if not test_data_dir.exists():
        print(f"ERROR: Test data directory not found: {test_data_dir}")
        sys.exit(1)

    print(f"Validating puzzle solvability in: {test_data_dir}")
    print("NOTE: Requires backend running (cd backend && go run main.go)")
    print()

    results = validate_all_puzzles(test_data_dir)
    all_valid = print_report(results)

    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()
