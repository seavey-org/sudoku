package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"

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

// SolvePuzzle handles POST /api/solve requests.
func (h *SolveHandler) SolvePuzzle(c *gin.Context) {
	var req SolveRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	solution, solvable := sudoku.Solve(req.Board, req.Size)
	if !solvable {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid puzzle: Must have exactly one unique solution"})
		return
	}

	c.JSON(http.StatusOK, SolveResponse{Solution: solution})
}
