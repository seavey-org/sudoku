package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"

	"github.com/codyseavey/sudoku/backend/config"
	"github.com/codyseavey/sudoku/backend/handlers"
	"github.com/codyseavey/sudoku/backend/storage"
	"github.com/codyseavey/sudoku/backend/sudoku"
)

func main() {
	verbose := flag.Bool("verbose", false, "Enable verbose logging")
	flag.Parse()

	cfg := config.Load()

	if *verbose || cfg.Verbose {
		sudoku.VerboseLogging = true
		log.Println("Verbose logging enabled")
	}

	log.Println("Starting Sudoku Server...")

	// Initialize storage
	statsStore := storage.NewStatsStore(cfg.StatsFile)

	// Initialize rate limiter (60 requests per minute per IP)
	rateLimiter := handlers.NewRateLimiter(cfg.RateLimitRate, cfg.RateLimitBurst)

	// Initialize handlers
	puzzleHandler := handlers.NewPuzzleHandler()
	solveHandler := handlers.NewSolveHandler()
	statsHandler := handlers.NewStatsHandler(statsStore)
	completeHandler := handlers.NewCompleteHandler(statsStore)
	uploadHandler := handlers.NewUploadHandler()

	mux := http.NewServeMux()

	// API routes with rate limiting
	mux.Handle("/api/puzzle", rateLimiter.Middleware(puzzleHandler))
	mux.Handle("/api/solve", rateLimiter.Middleware(solveHandler))
	mux.Handle("/api/stats", rateLimiter.Middleware(statsHandler))
	mux.Handle("/api/complete", rateLimiter.Middleware(completeHandler))
	mux.Handle("/api/upload", rateLimiter.Middleware(uploadHandler))

	// Static file serving
	frontendDir := cfg.ResolveFrontendDir()
	if frontendDir == "" {
		log.Printf("Warning: Frontend directory not found. Checked 'dist' and '../frontend/dist'.")
	} else {
		log.Printf("Serving frontend from: %s", frontendDir)
		fs := http.FileServer(http.Dir(frontendDir))
		mux.Handle("/", fs)
	}

	fmt.Printf("Server starting on port %s...\n", cfg.Port)
	if err := http.ListenAndServe(":"+cfg.Port, mux); err != nil {
		log.Fatalf("Error starting server: %v", err)
	}
}
