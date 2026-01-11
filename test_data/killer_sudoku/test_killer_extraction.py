import json
import glob
import re
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Extraction service URL
EXTRACTION_SERVICE_URL = os.environ.get('EXTRACTION_SERVICE_URL', 'http://localhost:5001')

def solve_extraction(puzzle_path, size=9):
    """Call the extraction service to extract killer sudoku from image."""
    with open(puzzle_path, 'rb') as f:
        files = {'image': (os.path.basename(puzzle_path), f, 'image/png')}
        data = {'size': str(size)}
        try:
            response = requests.post(f'{EXTRACTION_SERVICE_URL}/extract', files=files, data=data, timeout=120)
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'{response.status_code} - {response.text}'}
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}

def test_puzzle(puzzle_path):
    """Test a single puzzle and return results."""
    puzzle_num = puzzle_path.split('/')[-1].split('.')[0]
    gt_path = puzzle_path.replace('.png', '.json')

    try:
        with open(gt_path) as f:
            gt = json.load(f)
        gt_sum = sum(gt.get('cage_sums', {}).values())
        if gt_sum != 405:
            return {'num': puzzle_num, 'status': 'SKIP', 'reason': f'GT sum {gt_sum} != 405'}
    except Exception as e:
        return {'num': puzzle_num, 'status': 'SKIP', 'reason': f'No valid GT: {e}'}

    res = solve_extraction(puzzle_path, size=9)
    if not res or 'error' in res:
        error = res.get('error', 'Unknown error') if res else 'No response'
        return {'num': puzzle_num, 'status': 'ERROR', 'reason': error}

    pred_sum = sum(res['cage_sums'].values())
    mismatches = 0
    for r in range(9):
        for c in range(9):
            if res['board'][r][c] != gt['board'][r][c]:
                mismatches += 1

    status = 'PASS' if pred_sum == 405 and mismatches == 0 else 'FAIL'
    return {
        'num': puzzle_num,
        'status': status,
        'pred_sum': pred_sum,
        'mismatches': mismatches
    }

def main():
    # Only process files like 1.png, 2.png, ..., 56.png
    png_files = [f for f in glob.glob('9x9/*.png') if re.match(r'9x9/\d+\.png$', f)]
    png_files = sorted(png_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    results = []

    # Use ThreadPoolExecutor for parallel testing
    # Note: Flask server is single-threaded, so use just 2 workers to avoid overwhelming it
    max_workers = 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(test_puzzle, path): path for path in png_files}

        # Collect results as they complete
        for future in as_completed(future_to_path):
            result = future.result()
            results.append(result)

    # Sort results by puzzle number and print
    results.sort(key=lambda x: int(x['num']))

    passed = 0
    failed = 0
    skipped = 0

    for r in results:
        if r['status'] == 'SKIP':
            print(f"Puzzle {r['num']}: {r['reason']} - skipping")
            skipped += 1
        elif r['status'] == 'ERROR':
            print(f"Puzzle {r['num']}: {r['reason']}")
            failed += 1
        else:
            print(f"Puzzle {r['num']}: Sum={r['pred_sum']}/405, Mismatches={r['mismatches']}, {r['status']}")
            if r['status'] == 'PASS':
                passed += 1
            else:
                failed += 1

    print(f"\n{'='*50}")
    print(f"SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
