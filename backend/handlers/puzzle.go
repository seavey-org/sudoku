package handlers

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/codyseavey/sudoku/backend/sudoku"
)

// PuzzleHandler handles puzzle generation requests.
type PuzzleHandler struct{}

// NewPuzzleHandler creates a new puzzle handler.
func NewPuzzleHandler() *PuzzleHandler {
	return &PuzzleHandler{}
}

// ServeHTTP handles GET /api/puzzle requests.
func (h *PuzzleHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	difficulty := r.URL.Query().Get("difficulty")
	sizeParam := r.URL.Query().Get("size")
	gameType := r.URL.Query().Get("gameType")
	size := 9
	if sizeParam == "6" {
		size = 6
	}

	cfIP := r.Header.Get("CF-Connecting-IP")
	sourceIP := r.RemoteAddr
	userAgent := r.UserAgent()
	log.Printf("Generating puzzle: difficulty=%s, size=%d, type=%s, CF-Connecting-IP=%s, SourceIP=%s, UserAgent=%s",
		difficulty, size, gameType, cfIP, sourceIP, userAgent)

	var puzzle sudoku.Puzzle
	if gameType == "killer" {
		puzzle = sudoku.GenerateKiller(difficulty, size)
	} else {
		puzzle = sudoku.Generate(difficulty, size)
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(puzzle); err != nil {
		log.Printf("Error encoding puzzle response: %v", err)
	}
}
