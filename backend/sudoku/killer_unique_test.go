package sudoku

import (
	"testing"
)

func TestGenerateKillerUniqueness(t *testing.T) {
	size := 6 // Use 6x6 for speed
	// Test Hard (empty board, pure cage constraints)
	pHard := GenerateKiller("hard", size)

	// Verify it has exactly one solution
	// We need to construct a generator to call solveCountKiller
	gen := Generator{
		N:         size,
		BoxHeight: 2,
		BoxWidth:  3,
		Cages:     pHard.Cages,
		cageMap:   make(map[Point]int),
	}
	for idx, cage := range pHard.Cages {
		for _, cell := range cage.Cells {
			gen.cageMap[cell] = idx
		}
	}

	// Start with the generated board (which might have some clues)
	checkBoard := make(Grid, size)
	for i := range pHard.Board {
		checkBoard[i] = make([]int, size)
		copy(checkBoard[i], pHard.Board[i])
	}

	count := 0
	steps := 0
	gen.solveCountKiller(checkBoard, &count, &steps, 100000)

	if count != 1 {
		t.Errorf("GenerateKiller('hard') produced a puzzle with %d solutions", count)
	}

	// Test Easy (should also be unique, but logic is different)
	pEasy := GenerateKiller("easy", size)
	genEasy := Generator{
		N:         size,
		BoxHeight: 2,
		BoxWidth:  3,
		Cages:     pEasy.Cages,
		cageMap:   make(map[Point]int),
	}
	for idx, cage := range pEasy.Cages {
		for _, cell := range cage.Cells {
			genEasy.cageMap[cell] = idx
		}
	}

	// Copy board
	boardEasy := make(Grid, size)
	for i := range pEasy.Board {
		boardEasy[i] = make([]int, size)
		copy(boardEasy[i], pEasy.Board[i])
	}

	countEasy := 0
	stepsEasy := 0
	genEasy.solveCountKiller(boardEasy, &countEasy, &stepsEasy, 100000)

	if countEasy != 1 {
		t.Errorf("GenerateKiller('easy') produced a puzzle with %d solutions", countEasy)
	}
}
