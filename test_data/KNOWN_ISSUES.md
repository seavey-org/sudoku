# Known Test Data Issues

This document tracks test data files with known issues that affect their validity for training or testing.

**Current status:** All test data files pass validation. Invalid files have been removed from the repository.

## Validation

Run these scripts to verify test data integrity:

```bash
python test_data/validate_ground_truth.py      # Check JSON structure and constraints
python test_data/validate_cage_contiguity.py   # Check killer sudoku cage shapes
python test_data/validate_solvability.py       # Check puzzles have unique solutions (requires backend)
```
