package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/codyseavey/sudoku/backend/sudoku"
)

func main() {
	mux := http.NewServeMux()

	// API Endpoint
	mux.HandleFunc("/api/puzzle", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		difficulty := r.URL.Query().Get("difficulty")
		puzzle := sudoku.Generate(difficulty)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(puzzle)
	})

	// Static File Serving
	// Check for "dist" (production) first, then fallback to "../frontend/dist" (local dev)
	frontendDir := "dist"
	if _, err := os.Stat(frontendDir); os.IsNotExist(err) {
		frontendDir = "../frontend/dist"
		if _, err := os.Stat(frontendDir); os.IsNotExist(err) {
			log.Printf("Warning: Frontend directory not found. Checked 'dist' and '../frontend/dist'.")
		}
	}

	log.Printf("Serving frontend from: %s", frontendDir)
	fs := http.FileServer(http.Dir(frontendDir))
	mux.Handle("/", fs)

	port := "8080"
	fmt.Printf("Server starting on port %s...\n", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		fmt.Printf("Error starting server: %v\n", err)
		log.Fatal(err)
	}
	fmt.Println("Server stopped unexpectedly.")
}
