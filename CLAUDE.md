# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full-stack Sudoku application supporting classic and killer sudoku variants with image extraction (OCR) capabilities. Three-service architecture: Go backend (Gin), Vue 3 frontend (TypeScript + Tailwind), Python extraction microservice (FastAPI).

## Build & Run Commands

### Backend (Go/Gin - port 8080)
```bash
cd backend
go build -o sudoku-server .          # Build
go test ./...                        # Run all tests
go test ./sudoku -run TestSolve      # Run single test by name
go run main.go                       # Run server
go run main.go -verbose              # Run with verbose logging
golangci-lint run                    # Run linter
```

### Frontend (Vue 3 + TypeScript - port 5173 dev)
```bash
cd frontend
npm install                          # Install dependencies
npm run dev                          # Development server
npm run build                        # Production build (includes type check)
npm run lint                         # Run ESLint
npm run lint:fix                     # Fix lint issues
```

### Extraction Service (Python/FastAPI - port 5001)
```bash
cd extraction_service
pip install -r requirements.txt      # Install dependencies

# Set Google Cloud credentials (required for GCV OCR)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sudoku/google-cloud-adminSvc.json

python main.py --port 5001           # Run FastAPI service (new)
python app.py --port 5001            # Run legacy Flask service
```

**Note:** Google Cloud Vision credentials are stored in `google-cloud-adminSvc.json` in the repo root. This file is gitignored.

### Docker
```bash
# Local development
docker compose -f docker-compose.yml -f docker-compose.local.yml up --build

# Production (after pushing to GHCR)
IMAGE_TAG=<commit-sha> docker compose up -d
```

### Testing Extraction
```bash
# Classic sudoku extraction validation (requires extraction service running)
python test_data/test_classic_extraction.py

# Killer sudoku extraction validation
python test_data/test_killer_extraction.py
```

### ML Model Training
```bash
cd extraction_service
python train_digit_cnn.py --epochs 20 --batch-size 32
python extract_boundary_training_data.py  # Extract from test images
python train_boundary_classifier.py       # Train boundary detection
python extract_cage_sum_training_data.py  # Extract cage sum crops
python prepare_cage_sum_cnn_data.py       # Prepare training data
python train_cage_sum_cnn_v2.py --epochs 30 --batch-size 32  # Train cage sum CNN
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

**extraction_service/app.py** (4300+ lines)
- `get_warped_grid()` - Perspective correction to 1800x1800
- `detect_grid_lines()` - Projection profile-based line detection
- `is_cage_boundary()` - 9 detection methods for dashed cage lines
- `is_placed_digit()` - Filters pencil marks from placed digits
- `extract_classic_sudoku()` - Classic puzzle extraction pipeline (digits only)
- `extract_killer_sudoku()` - Primary killer extraction (superset of classic + cage boundaries + cage sums)
- `extract_killer_sudoku_ocr()` - Legacy OCR-based killer extraction (fallback)
- `extract_structure_only()` - Extract cage boundaries and structure only
- `extract_board_digits_cnn()` - CNN-based digit extraction (used by both classic and killer)
- `extract_cage_sums_cnn()` - CNN-based cage sum extraction
- `extract_with_gemini_api()` - Gemini API fallback when OCR fails
- ML models in `extraction_service/models/`:
  - `digit_cnn.pth` - PyTorch CNN for digit recognition (0-9)
  - `cage_sum_cnn.pth` - PyTorch CNN for cage sum recognition
  - `boundary_classifier_rf.pkl` - Random Forest for cage boundary detection
  - `boundary_scaler.pkl` - Feature scaler for boundary classifier

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
│   └── 6x6/*.png + *.json    # 2 test images
├── test_classic_extraction.py    # Test runner for classic extraction
├── test_killer_extraction.py     # Test runner for killer extraction
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
