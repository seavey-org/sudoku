#!/usr/bin/env python3
"""
Validate that all killer sudoku cages are contiguous.
Each cage's cells must be connected (horizontally or vertically adjacent).
"""
import json
from pathlib import Path
from collections import defaultdict


def get_cage_cells(cage_map, cage_id):
    """Get all (row, col) positions for a given cage ID."""
    cells = []
    for row in range(len(cage_map)):
        for col in range(len(cage_map[row])):
            if cage_map[row][col].strip() == cage_id:
                cells.append((row, col))
    return cells


def is_contiguous(cells):
    """Check if a set of cells forms a contiguous region."""
    if not cells:
        return True
    if len(cells) == 1:
        return True
    
    cells_set = set(cells)
    visited = set()
    
    # BFS from first cell
    queue = [cells[0]]
    visited.add(cells[0])
    
    while queue:
        row, col = queue.pop(0)
        # Check 4 neighbors (up, down, left, right)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (row + dr, col + dc)
            if neighbor in cells_set and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited) == len(cells)


def validate_file(json_path):
    """Validate cage contiguity for a single file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'cage_map' not in data:
        return None, "Not a killer sudoku file"
    
    cage_map = data['cage_map']
    
    # Get all unique cage IDs
    cage_ids = set()
    for row in cage_map:
        for cell in row:
            cage_ids.add(cell.strip())
    
    issues = []
    for cage_id in sorted(cage_ids):
        cells = get_cage_cells(cage_map, cage_id)
        if not is_contiguous(cells):
            issues.append(f"Cage '{cage_id}' is not contiguous: {cells}")
    
    return len(issues) == 0, issues


def main():
    script_dir = Path(__file__).parent
    
    print("Validating killer sudoku cage contiguity...")
    print("=" * 60)
    
    all_valid = True
    
    for subdir in ['6x6', '9x9']:
        killer_dir = script_dir / 'killer_sudoku' / subdir
        if not killer_dir.exists():
            continue
        
        print(f"\n## {subdir}")
        
        for json_file in sorted(killer_dir.glob('*.json')):
            if '.gemini.' in json_file.name:
                continue
            
            valid, issues = validate_file(json_file)
            
            if valid is None:
                continue
            elif valid:
                print(f"  {json_file.name}: PASS")
            else:
                print(f"  {json_file.name}: FAIL")
                for issue in issues:
                    print(f"    - {issue}")
                all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("All cages are contiguous!")
    else:
        print("Some cages have contiguity issues.")
    
    return 0 if all_valid else 1


if __name__ == '__main__':
    exit(main())
