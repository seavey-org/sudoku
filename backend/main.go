package main

import (
	"flag"
	"log"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"

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
	} else {
		gin.SetMode(gin.ReleaseMode)
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

	r := gin.New()

	// Add recovery and logging middleware
	r.Use(gin.Recovery())
	if *verbose || cfg.Verbose {
		r.Use(gin.Logger())
	}

	// Configure CORS
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type"},
		AllowCredentials: false,
	}))

	// API routes with rate limiting
	api := r.Group("/api")
	api.Use(rateLimiter.Middleware())
	{
		api.GET("/puzzle", puzzleHandler.GetPuzzle)
		api.POST("/solve", solveHandler.SolvePuzzle)
		api.GET("/stats", statsHandler.GetStats)
		api.POST("/complete", completeHandler.RecordCompletion)
		api.POST("/upload", uploadHandler.UploadImage)
	}

	// Static file serving
	frontendDir := cfg.ResolveFrontendDir()
	if frontendDir == "" {
		log.Printf("Warning: Frontend directory not found. Checked 'dist' and '../frontend/dist'.")
	} else {
		log.Printf("Serving frontend from: %s", frontendDir)
		r.Static("/assets", frontendDir+"/assets")
		r.StaticFile("/", frontendDir+"/index.html")
		r.NoRoute(func(c *gin.Context) {
			c.File(frontendDir + "/index.html")
		})
	}

	log.Printf("Server starting on port %s...\n", cfg.Port)
	if err := r.Run(":" + cfg.Port); err != nil {
		log.Fatalf("Error starting server: %v", err)
	}
}
