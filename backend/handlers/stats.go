package handlers

import (
	"encoding/json"
	"log"
	"net/http"

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

// ServeHTTP handles GET /api/stats requests.
func (h *StatsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	stats := h.store.Get()

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(stats); err != nil {
		log.Printf("Error encoding stats response: %v", err)
	}
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

// ServeHTTP handles POST /api/complete requests.
func (h *CompleteHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CompleteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	cfIP := r.Header.Get("CF-Connecting-IP")
	sourceIP := r.RemoteAddr
	userAgent := r.UserAgent()
	log.Printf("Puzzle Completed: difficulty=%s, size=%d, type=%s, CF-Connecting-IP=%s, SourceIP=%s, UserAgent=%s",
		req.Difficulty, req.Size, req.GameType, cfIP, sourceIP, userAgent)

	h.store.RecordCompletion(req.GameType, req.Size, req.Difficulty)

	w.WriteHeader(http.StatusOK)
}
