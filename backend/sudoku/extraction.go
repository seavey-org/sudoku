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

	if gameType == "killer" {
		return extractKillerFromService(imageBytes)
	}
	return extractClassicFromService(imageBytes)
}

// extractClassicFromService extracts a classic sudoku puzzle using the Python extraction service
func extractClassicFromService(imageBytes []byte) (Puzzle, error) {
	var emptyPuzzle Puzzle

	// Create a multipart form request
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// Add the image file
	part, err := writer.CreateFormFile("image", "puzzle.png")
	if err != nil {
		return emptyPuzzle, fmt.Errorf("failed to create form file: %v", err)
	}
	if _, err := part.Write(imageBytes); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to write image data: %v", err)
	}

	// Add size parameter
	if err := writer.WriteField("size", "9"); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to write size field: %v", err)
	}

	if err := writer.Close(); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to close writer: %v", err)
	}

	// Make the request with a timeout
	client := &http.Client{Timeout: 120 * time.Second}
	req, err := http.NewRequest("POST", ExtractionServiceURL+"/extract-classic", body)
	if err != nil {
		return emptyPuzzle, fmt.Errorf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := client.Do(req)
	if err != nil {
		return emptyPuzzle, fmt.Errorf("extraction service error: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return emptyPuzzle, fmt.Errorf("extraction failed: %s", string(respBody))
	}

	// Parse the response
	type ClassicExtractionResponse struct {
		Board Grid `json:"board"`
	}

	var extractResp ClassicExtractionResponse
	if err := json.NewDecoder(resp.Body).Decode(&extractResp); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to parse extraction response: %v", err)
	}

	puzzle := Puzzle{
		Board:    extractResp.Board,
		GameType: "standard",
	}

	if VerboseLogging {
		fmt.Printf("DEBUG: Classic extraction successful\n")
	}

	return puzzle, nil
}

// extractKillerFromService extracts a killer sudoku puzzle using the Python extraction service
func extractKillerFromService(imageBytes []byte) (Puzzle, error) {
	var emptyPuzzle Puzzle

	// Create a multipart form request
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// Add the image file
	part, err := writer.CreateFormFile("image", "puzzle.png")
	if err != nil {
		return emptyPuzzle, fmt.Errorf("failed to create form file: %v", err)
	}
	if _, err := part.Write(imageBytes); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to write image data: %v", err)
	}

	// Add size parameter
	if err := writer.WriteField("size", "9"); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to write size field: %v", err)
	}

	if err := writer.Close(); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to close writer: %v", err)
	}

	// Make the request with a timeout
	client := &http.Client{Timeout: 120 * time.Second}
	req, err := http.NewRequest("POST", ExtractionServiceURL+"/extract", body)
	if err != nil {
		return emptyPuzzle, fmt.Errorf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := client.Do(req)
	if err != nil {
		return emptyPuzzle, fmt.Errorf("extraction service error: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return emptyPuzzle, fmt.Errorf("extraction failed: %s", string(respBody))
	}

	// Parse the response
	type KillerExtractionResponse struct {
		Board    Grid                   `json:"board"`
		CageMap  [][]string             `json:"cage_map"`
		CageSums map[string]int         `json:"cage_sums"`
		Metadata map[string]interface{} `json:"metadata"`
	}

	var extractResp KillerExtractionResponse
	if err := json.NewDecoder(resp.Body).Decode(&extractResp); err != nil {
		return emptyPuzzle, fmt.Errorf("failed to parse extraction response: %v", err)
	}

	// Log metadata if verbose
	if VerboseLogging && extractResp.Metadata != nil {
		if valid, ok := extractResp.Metadata["valid"].(bool); ok && !valid {
			totalSum := 0
			if ts, ok := extractResp.Metadata["total_sum"].(float64); ok {
				totalSum = int(ts)
			}
			fmt.Printf("DEBUG: Extraction reported invalid (sum=%d, expected=405)\n", totalSum)
		}
		if fallback, ok := extractResp.Metadata["fallback_used"].(string); ok {
			fmt.Printf("DEBUG: Fallback used: %s\n", fallback)
		}
	}

	// Convert to Puzzle format
	puzzle := Puzzle{
		Board:    extractResp.Board,
		GameType: "killer",
	}

	// Build cages from cage_map and cage_sums
	if len(extractResp.CageMap) == 9 && len(extractResp.CageSums) > 0 {
		cageCells := make(map[string][]Point)

		for r := 0; r < 9; r++ {
			if len(extractResp.CageMap[r]) != 9 {
				return emptyPuzzle, fmt.Errorf("invalid cage_map row size at row %d", r)
			}
			for c := 0; c < 9; c++ {
				id := extractResp.CageMap[r][c]
				cageCells[id] = append(cageCells[id], Point{Row: r, Col: c})
			}
		}

		for id, sum := range extractResp.CageSums {
			if cells, ok := cageCells[id]; ok {
				sort.Slice(cells, func(i, j int) bool {
					if cells[i].Row != cells[j].Row {
						return cells[i].Row < cells[j].Row
					}
					return cells[i].Col < cells[j].Col
				})

				puzzle.Cages = append(puzzle.Cages, Cage{
					Sum:   sum,
					Cells: cells,
				})
			}
		}

		// Sort cages by the position of their first cell
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
		totalSum := 0
		for _, cage := range puzzle.Cages {
			totalSum += cage.Sum
		}
		fmt.Printf("DEBUG: Killer extraction successful. Cages=%d, TotalSum=%d\n", len(puzzle.Cages), totalSum)
	}

	return puzzle, nil
}
