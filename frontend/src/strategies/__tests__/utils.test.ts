import { describe, it, expect } from 'vitest'
import {
    formatCell,
    formatCells,
    getRowCells,
    getColCells,
    getBoxCells,
    getPeers,
    getBoxStart
} from '../utils'

describe('formatCell', () => {
    it('formats cell coordinates correctly', () => {
        expect(formatCell(0, 0)).toBe('R1C1')
        expect(formatCell(8, 8)).toBe('R9C9')
        expect(formatCell(4, 5)).toBe('R5C6')
    })
})

describe('formatCells', () => {
    it('formats multiple cells correctly', () => {
        const cells = [
            { row: 0, col: 0 },
            { row: 1, col: 1 },
            { row: 2, col: 2 }
        ]
        expect(formatCells(cells)).toBe('R1C1, R2C2, R3C3')
    })

    it('handles single cell', () => {
        const cells = [{ row: 0, col: 0 }]
        expect(formatCells(cells)).toBe('R1C1')
    })

    it('handles empty array', () => {
        expect(formatCells([])).toBe('')
    })
})

describe('getRowCells', () => {
    it('returns all cells in a row', () => {
        const row = getRowCells(0, 9)
        expect(row).toHaveLength(9)
        expect(row[0]).toEqual({ row: 0, col: 0 })
        expect(row[8]).toEqual({ row: 0, col: 8 })
    })

    it('works for 6x6 grid', () => {
        const row = getRowCells(2, 6)
        expect(row).toHaveLength(6)
        expect(row[0]).toEqual({ row: 2, col: 0 })
        expect(row[5]).toEqual({ row: 2, col: 5 })
    })
})

describe('getColCells', () => {
    it('returns all cells in a column', () => {
        const col = getColCells(3, 9)
        expect(col).toHaveLength(9)
        expect(col[0]).toEqual({ row: 0, col: 3 })
        expect(col[8]).toEqual({ row: 8, col: 3 })
    })
})

describe('getBoxStart', () => {
    it('returns correct box start for 9x9', () => {
        const start = getBoxStart(4, 4, 3, 3)
        // Box containing (4,4) starts at (3,3)
        expect(start.boxStartRow).toBe(3)
        expect(start.boxStartCol).toBe(3)
    })

    it('returns correct box start for 6x6', () => {
        const start = getBoxStart(1, 4, 2, 3)
        // Box containing (1,4) starts at (0,3)
        expect(start.boxStartRow).toBe(0)
        expect(start.boxStartCol).toBe(3)
    })
})

describe('getBoxCells', () => {
    it('returns all cells in a box given start position', () => {
        const box = getBoxCells(0, 0, 3, 3)
        expect(box).toHaveLength(9)
        // Box starting at (0,0) with 3x3 dimensions
        expect(box).toContainEqual({ row: 0, col: 0 })
        expect(box).toContainEqual({ row: 2, col: 2 })
    })

    it('returns correct box for middle position', () => {
        const box = getBoxCells(3, 3, 3, 3)
        expect(box).toHaveLength(9)
        // Box starting at (3,3) is center in 9x9
        expect(box).toContainEqual({ row: 3, col: 3 })
        expect(box).toContainEqual({ row: 5, col: 5 })
    })
})

describe('getPeers', () => {
    it('returns all peer cells for a given cell', () => {
        const peers = getPeers(4, 4, 9, 3, 3)
        // Peers = row + col + box - duplicates
        // Row: 8 cells, Col: 8 cells, Box: 8 cells (excluding self, with overlaps)
        // Total unique peers = 20
        expect(peers).toHaveLength(20)
        // Should not include the cell itself
        expect(peers).not.toContainEqual({ row: 4, col: 4 })
    })

    it('handles corner cell correctly', () => {
        const peers = getPeers(0, 0, 9, 3, 3)
        expect(peers).toHaveLength(20)
        expect(peers).not.toContainEqual({ row: 0, col: 0 })
    })
})
