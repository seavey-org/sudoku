import { describe, it, expect } from 'vitest'
import { nakedSingle } from '../nakedSingle'
import type { SolveContext } from '../types'

describe('nakedSingle', () => {
    it('finds a naked single when a cell has only one candidate', () => {
        const context: SolveContext = {
            board: [
                [1, 2, 3, 4, 5, 6, 7, 8, null],
                [null, null, null, null, null, null, null, null, null],
                [null, null, null, null, null, null, null, null, null],
                [null, null, null, null, null, null, null, null, null],
                [null, null, null, null, null, null, null, null, null],
                [null, null, null, null, null, null, null, null, null],
                [null, null, null, null, null, null, null, null, null],
                [null, null, null, null, null, null, null, null, null],
                [null, null, null, null, null, null, null, null, null],
            ],
            candidates: [
                [[], [], [], [], [], [], [], [], [9]], // Only 9 is valid for R1C9
                [[2, 3], [1, 3], [1, 2], [5, 6], [4, 6], [4, 5], [8, 9], [7, 9], [7, 8]],
                [[2, 3], [1, 3], [1, 2], [5, 6], [4, 6], [4, 5], [8, 9], [7, 9], [7, 8]],
                [[2, 3], [1, 3], [1, 2], [5, 6], [4, 6], [4, 5], [8, 9], [7, 9], [7, 8]],
                [[2, 3], [1, 3], [1, 2], [5, 6], [4, 6], [4, 5], [8, 9], [7, 9], [7, 8]],
                [[2, 3], [1, 3], [1, 2], [5, 6], [4, 6], [4, 5], [8, 9], [7, 9], [7, 8]],
                [[2, 3], [1, 3], [1, 2], [5, 6], [4, 6], [4, 5], [8, 9], [7, 9], [7, 8]],
                [[2, 3], [1, 3], [1, 2], [5, 6], [4, 6], [4, 5], [8, 9], [7, 9], [7, 8]],
                [[2, 3], [1, 3], [1, 2], [5, 6], [4, 6], [4, 5], [8, 9], [7, 9], [7, 8]],
            ],
            size: 9,
            boxHeight: 3,
            boxWidth: 3,
            gameType: 'standard'
        }

        const result = nakedSingle.find(context)

        expect(result).not.toBeNull()
        expect(result?.strategyName).toBe('Naked Single')
        expect(result?.highlights[0]?.row).toBe(0)
        expect(result?.highlights[0]?.col).toBe(8)
        expect(result?.highlights[0]?.candidates).toContain(9)
    })

    it('returns null when no naked singles exist', () => {
        const context: SolveContext = {
            board: Array(9).fill(null).map(() => Array(9).fill(null)),
            candidates: Array(9).fill(null).map(() =>
                Array(9).fill(null).map(() => [1, 2, 3, 4, 5, 6, 7, 8, 9])
            ),
            size: 9,
            boxHeight: 3,
            boxWidth: 3,
            gameType: 'standard'
        }

        const result = nakedSingle.find(context)

        expect(result).toBeNull()
    })

    it('skips filled cells', () => {
        const context: SolveContext = {
            board: [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [4, 5, 6, 7, 8, 9, 1, 2, 3],
                [7, 8, 9, 1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7, 8, 9, 1],
                [5, 6, 7, 8, 9, 1, 2, 3, 4],
                [8, 9, 1, 2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 8, 9, 1, 2],
                [6, 7, 8, 9, 1, 2, 3, 4, 5],
                [9, 1, 2, 3, 4, 5, 6, 7, 8],
            ],
            candidates: Array(9).fill(null).map(() =>
                Array(9).fill(null).map(() => [])
            ),
            size: 9,
            boxHeight: 3,
            boxWidth: 3,
            gameType: 'standard'
        }

        const result = nakedSingle.find(context)

        expect(result).toBeNull()
    })
})
