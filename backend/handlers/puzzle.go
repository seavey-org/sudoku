package handlers

import (
	"log"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/codyseavey/sudoku/backend/sudoku"
)

// PuzzleHandler handles puzzle generation requests.
type PuzzleHandler struct{}

// NewPuzzleHandler creates a new puzzle handler.
func NewPuzzleHandler() *PuzzleHandler {
	return &PuzzleHandler{}
}

// GetPuzzle handles GET /api/puzzle requests.
func (h *PuzzleHandler) GetPuzzle(c *gin.Context) {
	difficulty := c.Query("difficulty")
	sizeParam := c.Query("size")
	gameType := c.Query("gameType")
	size := 9
	if sizeParam == "6" {
		size = 6
	}

	cfIP := c.GetHeader("CF-Connecting-IP")
	sourceIP := c.ClientIP()
	userAgent := c.GetHeader("User-Agent")
	log.Printf("Generating puzzle: difficulty=%s, size=%d, type=%s, CF-Connecting-IP=%s, SourceIP=%s, UserAgent=%s",
		difficulty, size, gameType, cfIP, sourceIP, userAgent)

	var puzzle sudoku.Puzzle
	if gameType == "killer" {
		puzzle = sudoku.GenerateKiller(difficulty, size)
	} else {
		puzzle = sudoku.Generate(difficulty, size)
	}

	c.JSON(http.StatusOK, puzzle)
}
