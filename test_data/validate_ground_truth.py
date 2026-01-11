#!/usr/bin/env python3
"""
Validate ground truth JSON files for sudoku test data.

Checks:
- Sudoku constraints (no duplicates in rows, columns, boxes)
- Cage sum totals (405 for 9x9, 126 for 6x6)
- Cage map and cage_sums consistency
- Missing JSON files for images
- Valid JSON structure
"""
import json
from pathlib import Path


def get_box_index(row, col, size):
    """Get box index for a cell."""
    if size == 9:
        return (row // 3) * 3 + (col // 3)
    elif size == 6:
        return (row // 2) * 3 + (col // 3)
    return 0


def validate_sudoku_constraints(board):
    """Check for duplicate digits in rows, columns, and boxes.

    Returns:
        list: List of violation descriptions
    """
    size = len(board)
    violations = []

    # Check rows
    for r, row in enumerate(board):
        seen = {}
        for c, val in enumerate(row):
            if val != 0:
                if val in seen:
                    violations.append(f"Row {r}: duplicate {val} at columns {seen[val]} and {c}")
                else:
                    seen[val] = c

    # Check columns
    for c in range(size):
        seen = {}
        for r in range(size):
            val = board[r][c]
            if val != 0:
                if val in seen:
                    violations.append(f"Column {c}: duplicate {val} at rows {seen[val]} and {r}")
                else:
                    seen[val] = r

    # Check boxes
    if size == 9:
        box_h, box_w = 3, 3
    elif size == 6:
        box_h, box_w = 2, 3
    else:
        return violations

    for box_r in range(size // box_h):
        for box_c in range(size // box_w):
            seen = {}
            for r in range(box_r * box_h, (box_r + 1) * box_h):
                for c in range(box_c * box_w, (box_c + 1) * box_w):
                    val = board[r][c]
                    if val != 0:
                        key = (r, c)
                        if val in seen:
                            violations.append(f"Box ({box_r},{box_c}): duplicate {val} at {seen[val]} and {key}")
                        else:
                            seen[val] = key

    return violations


def validate_cage_sums(cage_map, cage_sums, size):
    """Validate cage sum totals and consistency.

    Returns:
        list: List of issues found
    """
    issues = []
    expected_total = 405 if size == 9 else 126

    # Calculate total of all cage sums
    total = sum(cage_sums.values())
    if total != expected_total:
        issues.append(f"Cage sum total: {total}, expected {expected_total}")

    # Check that all cage IDs in map have sums
    cage_ids_in_map = set()
    for row in cage_map:
        for cell_id in row:
            cage_ids_in_map.add(cell_id.strip())

    cage_ids_in_sums = set(cage_sums.keys())

    missing_sums = cage_ids_in_map - cage_ids_in_sums
    if missing_sums:
        issues.append(f"Cage IDs in map but not in sums: {missing_sums}")

    extra_sums = cage_ids_in_sums - cage_ids_in_map
    if extra_sums:
        issues.append(f"Cage IDs in sums but not in map: {extra_sums}")

    # Check for trailing spaces in cage IDs
    for row in cage_map:
        for cell_id in row:
            if cell_id != cell_id.strip():
                issues.append(f"Cage ID has trailing/leading whitespace: '{cell_id}'")
                break
        else:
            continue
        break

    return issues


def validate_json_file(json_path, is_killer=False):
    """Validate a single JSON ground truth file.

    Returns:
        dict: Validation result with 'valid' bool and 'issues' list
    """
    result = {'valid': True, 'issues': [], 'file': str(json_path)}

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result['valid'] = False
        result['issues'].append(f"Invalid JSON: {e}")
        return result
    except Exception as e:
        result['valid'] = False
        result['issues'].append(f"Read error: {e}")
        return result

    # Check board exists
    if 'board' not in data:
        result['valid'] = False
        result['issues'].append("Missing 'board' field")
        return result

    board = data['board']
    size = len(board)

    # Validate board dimensions
    if size not in [6, 9]:
        result['valid'] = False
        result['issues'].append(f"Invalid board size: {size}x{len(board[0]) if board else '?'}")
        return result

    for i, row in enumerate(board):
        if len(row) != size:
            result['valid'] = False
            result['issues'].append(f"Row {i} has {len(row)} columns, expected {size}")

    # Check sudoku constraints
    violations = validate_sudoku_constraints(board)
    if violations:
        result['valid'] = False
        result['issues'].extend(violations)

    # For killer sudoku, validate cages
    if is_killer:
        if 'cage_map' not in data:
            result['issues'].append("Missing 'cage_map' field (killer sudoku)")
        if 'cage_sums' not in data:
            result['issues'].append("Missing 'cage_sums' field (killer sudoku)")

        if 'cage_map' in data and 'cage_sums' in data:
            cage_issues = validate_cage_sums(data['cage_map'], data['cage_sums'], size)
            if cage_issues:
                result['issues'].extend(cage_issues)
                # Don't mark as invalid for cage sum issues - they might be incomplete

    return result


def find_missing_json_files(directory, image_pattern="*.png"):
    """Find images without corresponding JSON files.

    Returns:
        list: List of image paths missing JSON
    """
    missing = []
    dir_path = Path(directory)

    for img_path in dir_path.glob(image_pattern):
        json_path = img_path.with_suffix('.json')
        if not json_path.exists():
            missing.append(str(img_path))

    return missing


def validate_directory(base_dir):
    """Validate all ground truth files in a test data directory.

    Returns:
        dict: Summary of validation results
    """
    base_path = Path(base_dir)
    results = {
        'classic': {'valid': [], 'invalid': [], 'missing_json': []},
        'killer': {'valid': [], 'invalid': [], 'missing_json': []},
        'summary': {}
    }

    # Classic sudoku directories
    for subdir in ['6x6', '9x9']:
        classic_dir = base_path / 'classic_sudoku' / subdir
        if classic_dir.exists():
            # Check for missing JSON files
            missing = find_missing_json_files(classic_dir)
            results['classic']['missing_json'].extend(missing)

            # Validate existing JSON files
            for json_file in classic_dir.glob('*.json'):
                validation = validate_json_file(json_file, is_killer=False)
                if validation['valid'] and not validation['issues']:
                    results['classic']['valid'].append(validation)
                else:
                    results['classic']['invalid'].append(validation)

    # Killer sudoku directories
    for subdir in ['6x6', '9x9']:
        killer_dir = base_path / 'killer_sudoku' / subdir
        if killer_dir.exists():
            # Check for missing JSON files
            missing = find_missing_json_files(killer_dir)
            results['killer']['missing_json'].extend(missing)

            # Validate existing JSON files
            for json_file in killer_dir.glob('*.json'):
                validation = validate_json_file(json_file, is_killer=True)
                if validation['valid'] and not validation['issues']:
                    results['killer']['valid'].append(validation)
                else:
                    results['killer']['invalid'].append(validation)

    # Generate summary
    results['summary'] = {
        'classic_valid': len(results['classic']['valid']),
        'classic_invalid': len(results['classic']['invalid']),
        'classic_missing': len(results['classic']['missing_json']),
        'killer_valid': len(results['killer']['valid']),
        'killer_invalid': len(results['killer']['invalid']),
        'killer_missing': len(results['killer']['missing_json']),
    }

    return results


def print_report(results):
    """Print a formatted validation report."""
    print("=" * 60)
    print("GROUND TRUTH VALIDATION REPORT")
    print("=" * 60)

    # Classic sudoku
    print("\n## Classic Sudoku")
    print(f"   Valid files: {results['summary']['classic_valid']}")
    print(f"   Invalid files: {results['summary']['classic_invalid']}")
    print(f"   Missing JSON: {results['summary']['classic_missing']}")

    if results['classic']['invalid']:
        print("\n   INVALID FILES:")
        for item in results['classic']['invalid']:
            print(f"   - {item['file']}")
            for issue in item['issues']:
                print(f"     * {issue}")

    if results['classic']['missing_json']:
        print("\n   MISSING JSON FILES:")
        for path in results['classic']['missing_json']:
            print(f"   - {path}")

    # Killer sudoku
    print("\n## Killer Sudoku")
    print(f"   Valid files: {results['summary']['killer_valid']}")
    print(f"   Invalid files: {results['summary']['killer_invalid']}")
    print(f"   Missing JSON: {results['summary']['killer_missing']}")

    if results['killer']['invalid']:
        print("\n   INVALID FILES:")
        for item in results['killer']['invalid']:
            print(f"   - {item['file']}")
            for issue in item['issues']:
                print(f"     * {issue}")

    if results['killer']['missing_json']:
        print("\n   MISSING JSON FILES:")
        for path in results['killer']['missing_json']:
            print(f"   - {path}")

    # Summary
    total_valid = results['summary']['classic_valid'] + results['summary']['killer_valid']
    total_invalid = results['summary']['classic_invalid'] + results['summary']['killer_invalid']
    total_missing = results['summary']['classic_missing'] + results['summary']['killer_missing']

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_valid} valid, {total_invalid} invalid, {total_missing} missing")
    print("=" * 60)

    return total_invalid == 0 and total_missing == 0


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Validate ground truth JSON files')
    parser.add_argument('--warn-only', action='store_true',
                        help='Report issues but exit 0 (for CI with known issues)')
    args = parser.parse_args()

    # Find test_data directory
    script_dir = Path(__file__).parent
    if script_dir.name == 'test_data':
        base_dir = script_dir
    else:
        base_dir = script_dir / 'test_data'

    if not base_dir.exists():
        print(f"ERROR: Test data directory not found: {base_dir}")
        sys.exit(1)

    print(f"Validating test data in: {base_dir}")

    results = validate_directory(base_dir)
    all_valid = print_report(results)

    if args.warn_only:
        sys.exit(0)
    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()
