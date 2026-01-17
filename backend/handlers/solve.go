package handlers

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/codyseavey/sudoku/backend/sudoku"
)

// SolveHandler handles puzzle solving requests.
type SolveHandler struct{}

// NewSolveHandler creates a new solve handler.
func NewSolveHandler() *SolveHandler {
	return &SolveHandler{}
}

// SolveRequest represents the request body for solving a puzzle.
type SolveRequest struct {
	Board [][]int `json:"board"`
	Size  int     `json:"size"`
}

// SolveResponse represents the response body with the solution.
type SolveResponse struct {
	Solution [][]int `json:"solution"`
}

// ServeHTTP handles POST /api/solve requests.
func (h *SolveHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req SolveRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	solution, solvable := sudoku.Solve(req.Board, req.Size)
	if !solvable {
		http.Error(w, "Invalid puzzle: Must have exactly one unique solution", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(SolveResponse{Solution: solution}); err != nil {
		log.Printf("Error encoding solve response: %v", err)
	}
}
