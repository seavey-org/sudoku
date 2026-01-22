// Hint result structure returned by strategies
export interface HintResult {
  strategyName: string
  difficulty: number // SE rating for sorting/display
  description: string // User-friendly explanation
  eliminations: Elimination[] // Candidates to eliminate
  highlights: CellHighlight[] // Cells/candidates to highlight in UI
}

// A candidate to be eliminated
export interface Elimination {
  row: number
  col: number
  candidate: number
}

// Highlight information for UI
export interface CellHighlight {
  row: number
  col: number
  type: 'primary' | 'secondary' | 'elimination'
  candidates?: number[] // Specific candidates to highlight within the cell
}

// Strategy interface - all strategies implement this
export interface Strategy {
  name: string
  difficulty: number
  find(context: SolveContext): HintResult | null
}

// Context passed to strategies containing current puzzle state
export interface SolveContext {
  board: (number | null)[][]
  candidates: number[][][]
  size: number
  boxHeight: number
  boxWidth: number
  gameType: 'standard' | 'killer'
  cages?: Cage[]
}

// Cage definition for Killer Sudoku
export interface Cage {
  sum: number
  cells: { row: number; col: number }[]
}

// Cell coordinate helper
export interface Cell {
  row: number
  col: number
}

// Imported puzzle data from share links or storage
export interface ImportedPuzzle {
  board: number[][]
  solution: number[][]
  size: number
  difficulty: string
  gameType?: string
  cages?: Cage[]
}

// History state entry for undo functionality
export interface HistoryState {
  board: (number | null)[][]
  candidates: number[][][]
  eliminatedCandidates: Set<number>[][]
}

// Serialized history state (for localStorage)
// eliminatedCandidates: 2D grid where each cell contains an array of eliminated candidate numbers
export interface SerializedHistoryState {
  board: (number | null)[][]
  candidates: number[][][]
  eliminatedCandidates: number[][][]
}
