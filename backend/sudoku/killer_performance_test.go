package sudoku

import (
	"testing"
	"time"
)

func TestGenerateKillerHardPerformance(t *testing.T) {
	done := make(chan bool)
	go func() {
		// Attempt to generate a hard killer sudoku (size 9)
		// This used to hang; ensuring it completes within a reasonable time
		GenerateKiller("hard", 9)
		done <- true
	}()

	select {
	case <-done:
		t.Log("Generation finished")
	case <-time.After(30 * time.Second):
		t.Fatal("Generation timed out (likely hang)")
	}
}
