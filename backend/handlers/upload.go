package handlers

import (
	"io"
	"log"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/codyseavey/sudoku/backend/sudoku"
)

const maxUploadSize = 10 << 20 // 10 MB

// UploadHandler handles image upload requests for sudoku extraction.
type UploadHandler struct{}

// NewUploadHandler creates a new upload handler.
func NewUploadHandler() *UploadHandler {
	return &UploadHandler{}
}

// UploadImage handles POST /api/upload requests.
func (h *UploadHandler) UploadImage(c *gin.Context) {
	// Limit upload size
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxUploadSize)

	file, _, err := c.Request.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid file"})
		return
	}
	defer file.Close()

	fileBytes, err := io.ReadAll(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error reading file"})
		return
	}

	gameType := c.PostForm("gameType")
	puzzle, err := sudoku.ExtractSudokuFromImage(fileBytes, gameType)
	if err != nil {
		log.Printf("Extraction Error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to process image: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, puzzle)
}
