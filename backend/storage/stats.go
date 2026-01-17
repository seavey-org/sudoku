package storage

import (
	"encoding/json"
	"log"
	"os"
	"strconv"
	"sync"
)

// Stats holds puzzle completion statistics.
type Stats struct {
	TotalSolved int                                  `json:"totalSolved"`
	Details     map[string]map[string]map[string]int `json:"details"`
}

// StatsStore manages persistence and thread-safe access to stats.
type StatsStore struct {
	stats    Stats
	mu       sync.RWMutex
	filepath string
}

// NewStatsStore creates a new stats store with the given file path.
func NewStatsStore(filepath string) *StatsStore {
	s := &StatsStore{
		filepath: filepath,
		stats: Stats{
			Details: make(map[string]map[string]map[string]int),
		},
	}
	s.load()
	return s
}

// load reads stats from disk.
func (s *StatsStore) load() {
	file, err := os.Open(s.filepath)
	if err != nil {
		if !os.IsNotExist(err) {
			log.Printf("Error opening stats file: %v", err)
		}
		return
	}
	defer file.Close()

	if err := json.NewDecoder(file).Decode(&s.stats); err != nil {
		log.Printf("Error decoding stats file: %v", err)
	}

	if s.stats.Details == nil {
		s.stats.Details = make(map[string]map[string]map[string]int)
	}
}

// save writes stats to disk.
func (s *StatsStore) save() {
	file, err := os.Create(s.filepath)
	if err != nil {
		log.Printf("Error creating stats file: %v", err)
		return
	}
	defer file.Close()

	if err := json.NewEncoder(file).Encode(&s.stats); err != nil {
		log.Printf("Error encoding stats file: %v", err)
	}
}

// Get returns a copy of the current stats.
func (s *StatsStore) Get() Stats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Return a shallow copy
	result := Stats{
		TotalSolved: s.stats.TotalSolved,
		Details:     s.stats.Details,
	}
	return result
}

// RecordCompletion records a puzzle completion.
func (s *StatsStore) RecordCompletion(gameType string, size int, difficulty string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.stats.TotalSolved++

	if gameType == "" {
		gameType = "standard"
	}

	if s.stats.Details[gameType] == nil {
		s.stats.Details[gameType] = make(map[string]map[string]int)
	}

	sizeStr := strconv.Itoa(size)
	if s.stats.Details[gameType][sizeStr] == nil {
		s.stats.Details[gameType][sizeStr] = make(map[string]int)
	}

	s.stats.Details[gameType][sizeStr][difficulty]++

	s.save()
}
