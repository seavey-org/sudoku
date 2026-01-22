#!/usr/bin/env python3
"""
Killer Sudoku Image Extraction Service - FastAPI Version

A FastAPI HTTP service that extracts killer sudoku puzzles from images using
OpenCV for image processing and EasyOCR for text recognition.

This is the FastAPI wrapper around the extraction logic in app.py.
"""
import os
import json
import tempfile
import traceback
import logging
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Sudoku Extraction Service",
    description="Extract sudoku puzzles from images using OCR and ML",
    version="2.0.0"
)

# Import extraction functions from the existing app module
# This preserves all the existing logic without modification
from app import (
    extract_with_improvements,
    extract_killer_sudoku,
    extract_classic_sudoku,
    get_warped_grid,
)
import cv2
import base64


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/extract")
async def extract(
    image: UploadFile = File(...),
    size: str = Form("9"),
    include_candidates: str = Form("false")
):
    """
    Extract killer sudoku from uploaded image.

    Request: multipart/form-data with 'image' file
        - image: file
        - size: 6 or 9
        - include_candidates: "true" or "false" (default: false)
    Response: JSON with board, cage_map, cage_sums, and optionally candidates
    """
    try:
        size_int = int(size)
    except ValueError:
        size_int = 9

    include_candidates_bool = include_candidates.lower() == "true"

    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = extract_with_improvements(
            tmp_path,
            size=size_int,
            include_candidates=include_candidates_bool,
            debug=False,
            use_vertex_fallback=False
        )

        if result is None:
            raise HTTPException(status_code=500, detail="Failed to extract puzzle from image")

        response = {
            "board": result.get("board", [[0] * size_int for _ in range(size_int)]),
            "cage_map": result.get("cage_map", []),
            "cage_sums": result.get("cage_sums", {}),
        }

        # Include candidates if requested
        if include_candidates_bool:
            response["candidates"] = result.get("candidates", [[[] for _ in range(size_int)] for _ in range(size_int)])

        total_sum = sum(response["cage_sums"].values())
        num_cages = len(response["cage_sums"])
        expected_sum = 405 if size_int == 9 else 126

        response["metadata"] = {
            "total_sum": total_sum,
            "expected_sum": expected_sum,
            "num_cages": num_cages,
            "valid": total_sum == expected_sum
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.post("/extract-killer")
async def extract_killer(
    image: UploadFile = File(...),
    size: str = Form("9"),
    verbose: str = Form("")
):
    """
    Extract killer sudoku from uploaded image.

    This endpoint uses the unified extract_killer_sudoku function which:
    1. Warps the grid (same as classic extraction)
    2. Extracts board digits using CNN
    3. Detects cage boundaries using ML classifier
    4. Extracts cage sums using CNN (with OCR fallback)

    Returns format compatible with test_killer_extraction.py:
    - board: 9x9 grid of digits (0 = empty)
    - cages: list of {sum, cells} for each cage
    - cage_sums: dict of cage_id -> sum
    - gameType: 'killer'
    """
    try:
        size_int = int(size)
    except ValueError:
        size_int = 9

    verbose_bool = verbose.lower() == "true"

    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = extract_killer_sudoku(tmp_path, size=size_int, debug=verbose_bool)

        if result is None:
            raise HTTPException(status_code=500, detail="Failed to extract puzzle from image")

        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.post("/extract-cells")
async def extract_cells(
    image: UploadFile = File(...),
    cells: str = Form(...),
    size: str = Form("9")
):
    """
    Extract specific cell images from a sudoku puzzle image.

    Request: multipart/form-data with 'image' file and 'cells' JSON array of [row, col] pairs
    Response: JSON with cell images as base64
    """
    try:
        size_int = int(size)
    except ValueError:
        size_int = 9

    try:
        cells_list = json.loads(cells)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid cells JSON")

    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        warped = get_warped_grid(tmp_path)
        if warped is None:
            raise HTTPException(status_code=500, detail="Failed to extract grid from image")

        cell_h, cell_w = 1800 // size_int, 1800 // size_int
        cell_images = {}

        for cell in cells_list:
            row, col = cell[0], cell[1]
            if 0 <= row < size_int and 0 <= col < size_int:
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                cell_img = warped[y1:y2, x1:x2]

                # Encode as base64 PNG
                _, buffer = cv2.imencode('.png', cell_img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                cell_images[f"{row},{col}"] = img_base64

        return {"cells": cell_images}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.post("/extract-classic")
async def extract_classic(
    image: UploadFile = File(...),
    size: str = Form("9"),
    include_candidates: str = Form("false")
):
    """
    Extract classic sudoku from uploaded image.

    Request: multipart/form-data with 'image' file
        - image: file
        - size: 6 or 9
        - include_candidates: "true" or "false" (default: false)
    Response: JSON with board and optionally candidates
    """
    try:
        size_int = int(size)
    except ValueError:
        size_int = 9

    include_candidates_bool = include_candidates.lower() == "true"

    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = extract_classic_sudoku(
            tmp_path,
            size=size_int,
            include_candidates=include_candidates_bool,
            debug=False
        )

        if result is None:
            raise HTTPException(status_code=500, detail="Failed to extract puzzle from image")

        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001, help='Port to run on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    args = parser.parse_args()

    logger.info(f"Starting extraction service on {args.host}:{args.port}")
    logger.info("Extraction service ready!")

    uvicorn.run(app, host=args.host, port=args.port)
