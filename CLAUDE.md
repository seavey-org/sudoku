# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full-stack Sudoku application supporting classic and killer sudoku variants with image extraction (OCR) capabilities. Three-service architecture: Go backend, Vue 3 frontend, Python extraction microservice.

## Build & Run Commands

### Backend (Go - port 8080)
```bash
cd backend
go build -o sudoku-server .          # Build
go test ./...                        # Run all tests
go test ./sudoku -run TestSolve      # Run single test by name
go run main.go                       # Run server
go run main.go -verbose              # Run with verbose logging
```

### Frontend (Vue 3 - port 5173 dev)
```bash
cd frontend
npm install                          # Install dependencies
npm run dev                          # Development server
npm run build                        # Production build to dist/
```

### Extraction Service (Python/Flask - port 5001)
```bash
cd extraction_service
source venv/bin/activate             # Activate virtualenv
pip install -r requirements.txt      # Install dependencies
python app.py --port 5001            # Run service
```

### Testing Extraction
```bash
# Classic sudoku extraction validation (requires extraction service running)
python test_data/test_classic_extraction.py

# Killer sudoku extraction validation
cd test_data/killer_sudoku && python test_killer_extraction.py
```

### ML Model Training
```bash
cd extraction_service
python train_digit_cnn.py --epochs 20 --batch-size 32
python extract_boundary_training_data.py  # Extract from test images
python train_boundary_classifier.py       # Train boundary detection
```

### Deployment
```bash
./deployment/deploy.sh               # Deploy to production server
```

## Architecture

### Service Communication
```
Browser → Frontend (Vue 3, proxies /api/* to :8080 in dev)
           ↓
        Backend API (Go :8080)
           ├→ /api/puzzle     - Generate puzzle
           ├→ /api/solve      - Solve board
           ├→ /api/complete   - Track completed puzzles
           ├→ /api/upload     - Image upload
           │    └→ Extraction Service (Python :5001)
           │         ├→ /extract         - Killer sudoku
           │         └→ /extract-classic - Classic sudoku
           └→ /api/stats      - Game statistics
```

### Key Components

**backend/sudoku/**
- `sudoku.go` - Core solver with unique solution validation
- `killer_*.go` - Killer sudoku cage logic
- `extraction.go` - Proxy to Python extraction service

**extraction_service/app.py** (3600+ lines)
- `get_warped_grid()` - Perspective correction to 1800x1800
- `detect_grid_lines()` - Projection profile-based line detection
- `is_cage_boundary()` - 9 detection methods for dashed cage lines
- `is_placed_digit()` - Filters pencil marks from placed digits
- `extract_classic_sudoku()` - Classic puzzle extraction pipeline
- `solve_extraction()` - Killer puzzle extraction pipeline
- `extract_with_gemini_api()` - Gemini API fallback when OCR fails
- ML models: CNN for digit recognition, Random Forest for boundary detection

**frontend/src/strategies/**
- 24+ solving algorithms: Naked/Hidden Singles, X-Wing, Swordfish, Y-Wing, Skyscraper, Unique Rectangle, etc.
- Killer-specific: Cage combinations, Innies/Outies

### Test Data Structure
```
test_data/
├── classic_sudoku/
│   ├── 9x9/*.png + *.json    # 14 test images with ground truth
│   └── 6x6/*.png + *.json    # 5 test images
├── killer_sudoku/
│   ├── 9x9/*.png + *.json    # 56 test images with ground truth
│   ├── 6x6/*.png + *.json    # 2 test images
│   ├── local_extractor.py    # Standalone killer extraction
│   └── test_killer_extraction.py  # Test runner for killer extraction
├── test_classic_extraction.py    # Test runner for classic extraction
└── validate_*.py                 # Ground truth validation utilities
```

### JSON Ground Truth Format

**Classic Sudoku:**
```json
{"board": [[0,3,8,...], ...]}  // 0 = empty cell
```

**Killer Sudoku:**
```json
{
  "board": [[6,5,0,...], ...],
  "cage_map": [["a","a","b",...], ...],
  "cage_sums": {"a": 12, "b": 26, ...}
}
```

## OCR Pipeline Notes

- Grid warped to 1800x1800 for consistent cell sizes (200x200 per cell for 9x9)
- Multiple threshold methods combined for robustness
- Regularity enforcement corrects irregular grid detection
- Pencil mark filtering uses: height ratio (40% min), corner clustering, aspect ratio validation
- EasyOCR with GPU acceleration (CUDA)

## Key Constraints

- Puzzles must have exactly one unique solution
- Killer sudoku cage sums must total 405 (9x9) or 126 (6x6)
- OCR confidence threshold: 0.35 for digit acceptance
- Both 6x6 and 9x9 grid sizes supported
- Image upload limit: 10MB

## CI/CD

Push to `main` triggers GitHub Actions pipeline that:
1. Tests backend (`go test ./...`)
2. Builds Go binary and Vue frontend
3. Retrains ML models from test_data images
4. Deploys to production (192.168.86.227)

Manual model retraining available via GitHub Actions "Retrain ML Models" workflow.
