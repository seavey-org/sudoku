package sudoku

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"sort"
	"time"
)

// ExtractionServiceURL is the URL of the Python extraction service
var ExtractionServiceURL = "http://127.0.0.1:5001"

var VerboseLogging bool

// isExtractionServiceAvailable checks if the Python extraction service is running
func isExtractionServiceAvailable() bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(ExtractionServiceURL + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// ExtractSudokuFromImage extracts a sudoku puzzle from an image using the Python extraction service
func ExtractSudokuFromImage(imageBytes []byte, gameType string) (Puzzle, error) {
	var emptyPuzzle Puzzle

	if !isExtractionServiceAvailable() {
		return emptyPuzzle, fmt.Errorf("extraction service unavailable")
	}

	endpoint := "/extract-classic"
	if gameType == "killer" {
		endpoint = "/extract-killer"
	}

	// Proxy to Python service
	return callExtractionService(imageBytes, endpoint)
}

// ExtractionResponse represents the response from Python extraction service
type ExtractionResponse struct {
	Board    [][]int `json:"board"`
	Cages    []struct {
		Sum   int `json:"sum"`
		Cells []struct {
			Row int `json:"row"`
			Col int `json:"col"`
		} `json:"cells"`
	} `json:"cages"`
	GameType string `json:"gameType"`
	Error    string `json:"error,omitempty"`
}

func callExtractionService(imageBytes []byte, endpoint string) (Puzzle, error) {
	var emptyPuzzle Puzzle

	// Build multipart form
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("image", "puzzle.png")
	if err != nil {
		return emptyPuzzle, fmt.Errorf("failed to create form file: %v", err)
	}
	if _, err := part.Write(imageBytes); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to write image: %v", err)
	}

	if VerboseLogging {
		if err := writer.WriteField("verbose", "true"); err != nil {
			return emptyPuzzle, fmt.Errorf("failed to write verbose field: %v", err)
		}
	}

	if err := writer.Close(); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to close writer: %v", err)
	}

	// Send request to Python service
	client := &http.Client{Timeout: 120 * time.Second}
	req, err := http.NewRequest("POST", ExtractionServiceURL+endpoint, body)
	if err != nil {
		return emptyPuzzle, fmt.Errorf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	if VerboseLogging {
		fmt.Printf("Calling Python extraction service: %s%s\n", ExtractionServiceURL, endpoint)
	}

	resp, err := client.Do(req)
	if err != nil {
		return emptyPuzzle, fmt.Errorf("extraction request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return emptyPuzzle, fmt.Errorf("extraction failed (%d): %s", resp.StatusCode, string(respBody))
	}

	// Parse response
	var result ExtractionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to decode response: %v", err)
	}

	if result.Error != "" {
		return emptyPuzzle, fmt.Errorf("extraction error: %s", result.Error)
	}

	// Convert to Puzzle format
	puzzle := Puzzle{
		Board:    result.Board,
		GameType: result.GameType,
	}

	// Convert cages if present (killer sudoku)
	if len(result.Cages) > 0 {
		for _, cage := range result.Cages {
			var cells []Point
			for _, cell := range cage.Cells {
				cells = append(cells, Point{Row: cell.Row, Col: cell.Col})
			}
			// Sort cells by row then col
			sort.Slice(cells, func(i, j int) bool {
				if cells[i].Row != cells[j].Row {
					return cells[i].Row < cells[j].Row
				}
				return cells[i].Col < cells[j].Col
			})
			puzzle.Cages = append(puzzle.Cages, Cage{
				Sum:   cage.Sum,
				Cells: cells,
			})
		}

		// Sort cages by first cell position
		sort.Slice(puzzle.Cages, func(i, j int) bool {
			c1 := puzzle.Cages[i].Cells[0]
			c2 := puzzle.Cages[j].Cells[0]
			if c1.Row != c2.Row {
				return c1.Row < c2.Row
			}
			return c1.Col < c2.Col
		})
	}

	if VerboseLogging {
		fmt.Printf("Extraction complete: %d digits, %d cages\n", countNonZero(puzzle.Board), len(puzzle.Cages))
	}

	return puzzle, nil
}

func countNonZero(board [][]int) int {
	count := 0
	for _, row := range board {
		for _, val := range row {
			if val != 0 {
				count++
			}
		}
	}
	return count
}
