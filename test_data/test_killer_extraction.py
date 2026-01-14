import json
import glob
import re
import requests
import os
import sys
import argparse

# Extraction service URL
EXTRACTION_SERVICE_URL = os.environ.get('EXTRACTION_SERVICE_URL', 'http://localhost:5001')

def solve_extraction(puzzle_path, size=9):
    """Call the extraction service to extract killer sudoku from image."""
    with open(puzzle_path, 'rb') as f:
        files = {'image': (os.path.basename(puzzle_path), f, 'image/png')}
        data = {'size': str(size)}
        try:
            response = requests.post(f'{EXTRACTION_SERVICE_URL}/extract-killer', files=files, data=data, timeout=120)
            if response.status_code == 200:
                result = response.json()
                # Convert new format to old format for compatibility
                if 'cages' in result and 'cage_sums' not in result:
                    cage_sums = {}
                    for i, cage in enumerate(result['cages']):
                        cage_id = chr(ord('a') + i) if i < 26 else 'a' + chr(ord('a') + i - 26)
                        cage_sums[cage_id] = cage['sum']
                    result['cage_sums'] = cage_sums
                return result
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
    mismatch_details = []
    for r in range(9):
        for c in range(9):
            if res['board'][r][c] != gt['board'][r][c]:
                mismatches += 1
                mismatch_details.append(f"[{r},{c}]: got {res['board'][r][c]}, expected {gt['board'][r][c]}")

    status = 'PASS' if pred_sum == 405 and mismatches == 0 else 'FAIL'
    return {
        'num': puzzle_num,
        'status': status,
        'pred_sum': pred_sum,
        'mismatches': mismatches,
        'mismatch_details': mismatch_details,
        'gt': gt,
        'result': res
    }

def main():
    parser = argparse.ArgumentParser(description='Test killer sudoku extraction')
    parser.add_argument('--fail-fast', '-f', action='store_true',
                        help='Stop on first failure for quick iteration')
    parser.add_argument('--start', '-s', type=int, default=1,
                        help='Start from puzzle number (default: 1)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed failure info')
    args = parser.parse_args()

    # Only process files like 1.png, 2.png, ..., 56.png
    png_files = [f for f in glob.glob('9x9/*.png') if re.match(r'9x9/\d+\.png$', f)]
    png_files = sorted(png_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    # Filter by start number
    png_files = [f for f in png_files if int(f.split('/')[-1].split('.')[0]) >= args.start]

    passed = 0
    failed = 0
    skipped = 0

    # Process sequentially for fail-fast mode
    for puzzle_path in png_files:
        r = test_puzzle(puzzle_path)

        if r['status'] == 'SKIP':
            print(f"Puzzle {r['num']}: {r['reason']} - skipping")
            skipped += 1
        elif r['status'] == 'ERROR':
            print(f"Puzzle {r['num']}: {r['reason']}")
            failed += 1
            if args.fail_fast:
                print(f"\nFAIL-FAST: Stopping on first failure")
                sys.exit(1)
        else:
            print(f"Puzzle {r['num']}: Sum={r['pred_sum']}/405, Mismatches={r['mismatches']}, {r['status']}")
            if r['status'] == 'PASS':
                passed += 1
            else:
                failed += 1
                if args.verbose and r['mismatches'] > 0:
                    for detail in r['mismatch_details'][:5]:
                        print(f"  {detail}")
                if args.fail_fast:
                    print(f"\nFAIL-FAST: Stopping on puzzle {r['num']}")
                    if args.verbose:
                        # Show cage sum comparison
                        gt = r['gt']
                        result = r['result']
                        result_cage_sums = {}
                        for cage in result['cages']:
                            cells = cage['cells']
                            first_r, first_c = cells[0]['row'], cells[0]['col']
                            gt_cage_id = gt['cage_map'][first_r][first_c]
                            result_cage_sums[gt_cage_id] = cage['sum']

                        print("\nCage sum differences:")
                        for cage_id in sorted(gt['cage_sums'].keys()):
                            gt_val = gt['cage_sums'][cage_id]
                            pred_val = result_cage_sums.get(cage_id, 0)
                            if gt_val != pred_val:
                                print(f"  {cage_id}: predicted {pred_val}, expected {gt_val}")
                    sys.exit(1)

        sys.stdout.flush()

    print(f"\n{'='*50}")
    print(f"SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*50}")

    if failed > 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
