package handlers

import (
	"log"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/codyseavey/sudoku/backend/storage"
)

// StatsHandler handles stats retrieval requests.
type StatsHandler struct {
	store *storage.StatsStore
}

// NewStatsHandler creates a new stats handler.
func NewStatsHandler(store *storage.StatsStore) *StatsHandler {
	return &StatsHandler{store: store}
}

// GetStats handles GET /api/stats requests.
func (h *StatsHandler) GetStats(c *gin.Context) {
	stats := h.store.Get()
	c.JSON(http.StatusOK, stats)
}

// CompleteHandler handles puzzle completion requests.
type CompleteHandler struct {
	store *storage.StatsStore
}

// NewCompleteHandler creates a new complete handler.
func NewCompleteHandler(store *storage.StatsStore) *CompleteHandler {
	return &CompleteHandler{store: store}
}

// CompleteRequest represents the request body for recording a completion.
type CompleteRequest struct {
	Difficulty string `json:"difficulty"`
	GameType   string `json:"gameType"`
	Size       int    `json:"size"`
}

// RecordCompletion handles POST /api/complete requests.
func (h *CompleteHandler) RecordCompletion(c *gin.Context) {
	var req CompleteRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	cfIP := c.GetHeader("CF-Connecting-IP")
	sourceIP := c.ClientIP()
	userAgent := c.GetHeader("User-Agent")
	log.Printf("Puzzle Completed: difficulty=%s, size=%d, type=%s, CF-Connecting-IP=%s, SourceIP=%s, UserAgent=%s",
		req.Difficulty, req.Size, req.GameType, cfIP, sourceIP, userAgent)

	h.store.RecordCompletion(req.GameType, req.Size, req.Difficulty)

	c.Status(http.StatusOK)
}
