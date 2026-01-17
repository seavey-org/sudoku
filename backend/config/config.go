package config

import (
	"os"
	"strconv"
	"time"
)

// Config holds application configuration.
type Config struct {
	Port           string
	StatsFile      string
	FrontendDir    string
	RateLimitRate  int
	RateLimitBurst time.Duration
	Verbose        bool
}

// Load creates a Config from environment variables with defaults.
func Load() *Config {
	return &Config{
		Port:           getEnv("SUDOKU_PORT", "8080"),
		StatsFile:      getEnv("SUDOKU_STATS_FILE", "stats.json"),
		FrontendDir:    getEnv("SUDOKU_FRONTEND_DIR", ""),
		RateLimitRate:  getEnvInt("SUDOKU_RATE_LIMIT", 60),
		RateLimitBurst: time.Minute,
		Verbose:        getEnvBool("SUDOKU_VERBOSE", false),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if i, err := strconv.Atoi(value); err == nil {
			return i
		}
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if b, err := strconv.ParseBool(value); err == nil {
			return b
		}
	}
	return defaultValue
}

// ResolveFrontendDir finds the frontend directory.
// It checks the configured path, then "dist", then "../frontend/dist".
func (c *Config) ResolveFrontendDir() string {
	candidates := []string{c.FrontendDir, "dist", "../frontend/dist"}
	for _, dir := range candidates {
		if dir == "" {
			continue
		}
		if _, err := os.Stat(dir); err == nil {
			return dir
		}
	}
	return ""
}
