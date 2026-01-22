package handlers

import (
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
)

// RateLimiter implements a simple token bucket rate limiter.
type RateLimiter struct {
	mu       sync.Mutex
	clients  map[string]*clientBucket
	rate     int           // requests per interval
	interval time.Duration // time interval
	cleanup  time.Duration // cleanup interval for old entries
}

type clientBucket struct {
	tokens    int
	lastCheck time.Time
}

// NewRateLimiter creates a new rate limiter.
// rate is the number of requests allowed per interval.
func NewRateLimiter(rate int, interval time.Duration) *RateLimiter {
	rl := &RateLimiter{
		clients:  make(map[string]*clientBucket),
		rate:     rate,
		interval: interval,
		cleanup:  5 * time.Minute,
	}
	go rl.cleanupLoop()
	return rl
}

// cleanupLoop periodically removes stale client entries.
func (rl *RateLimiter) cleanupLoop() {
	ticker := time.NewTicker(rl.cleanup)
	for range ticker.C {
		rl.mu.Lock()
		cutoff := time.Now().Add(-rl.cleanup)
		for ip, bucket := range rl.clients {
			if bucket.lastCheck.Before(cutoff) {
				delete(rl.clients, ip)
			}
		}
		rl.mu.Unlock()
	}
}

// Allow checks if a request from the given IP should be allowed.
func (rl *RateLimiter) Allow(ip string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	bucket, exists := rl.clients[ip]

	if !exists {
		rl.clients[ip] = &clientBucket{
			tokens:    rl.rate - 1,
			lastCheck: now,
		}
		return true
	}

	// Refill tokens based on elapsed time
	elapsed := now.Sub(bucket.lastCheck)
	tokensToAdd := int(elapsed/rl.interval) * rl.rate
	bucket.tokens += tokensToAdd
	if bucket.tokens > rl.rate {
		bucket.tokens = rl.rate
	}
	bucket.lastCheck = now

	if bucket.tokens > 0 {
		bucket.tokens--
		return true
	}

	return false
}

// getClientIP extracts the client IP from Gin context.
func getClientIP(c *gin.Context) string {
	if cfIP := c.GetHeader("CF-Connecting-IP"); cfIP != "" {
		return cfIP
	}
	return c.ClientIP()
}

// Middleware returns a Gin middleware for rate limiting.
func (rl *RateLimiter) Middleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		ip := getClientIP(c)
		if !rl.Allow(ip) {
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{"error": "Too many requests"})
			return
		}
		c.Next()
	}
}
