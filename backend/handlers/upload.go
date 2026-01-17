package handlers

import (
	"encoding/json"
	"io"
	"log"
	"net/http"

	"github.com/codyseavey/sudoku/backend/sudoku"
)

const maxUploadSize = 10 << 20 // 10 MB

// UploadHandler handles image upload requests for sudoku extraction.
type UploadHandler struct{}

// NewUploadHandler creates a new upload handler.
func NewUploadHandler() *UploadHandler {
	return &UploadHandler{}
}

// ServeHTTP handles POST /api/upload requests.
func (h *UploadHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Limit upload size
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadSize)

	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "Invalid file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	fileBytes, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "Error reading file", http.StatusInternalServerError)
		return
	}

	gameType := r.FormValue("gameType")
	puzzle, err := sudoku.ExtractSudokuFromImage(fileBytes, gameType)
	if err != nil {
		log.Printf("Extraction Error: %v", err)
		http.Error(w, "Failed to process image: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(puzzle); err != nil {
		log.Printf("Error encoding upload response: %v", err)
	}
}
