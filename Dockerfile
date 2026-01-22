# Combined Frontend + Backend Dockerfile
# Builds Vue.js frontend and Go backend into a single image

# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ .
RUN npm run build

# Stage 2: Build backend
FROM golang:1.24-alpine AS backend-builder

WORKDIR /app

COPY backend/go.mod backend/go.sum ./
RUN go mod download

COPY backend/ .
RUN CGO_ENABLED=0 GOOS=linux go build -o server .

# Stage 3: Production image
FROM alpine:3.21

WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache ca-certificates

# Copy backend binary
COPY --from=backend-builder /app/server .

# Copy frontend build
COPY --from=frontend-builder /frontend/dist ./frontend/dist

# Create data directory
RUN mkdir -p /app/data

# Environment variables
ENV SUDOKU_PORT=8080
ENV SUDOKU_STATS_FILE=/app/data/stats.json
ENV SUDOKU_FRONTEND_DIR=/app/frontend/dist

EXPOSE 8080

CMD ["./server"]
