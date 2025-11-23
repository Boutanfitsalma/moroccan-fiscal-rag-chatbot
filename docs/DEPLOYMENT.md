# 🚀 Deployment Guide

This guide covers all deployment scenarios: local development, production, and troubleshooting.

---

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (11+), Windows 10+ with WSL2
- **CPU**: 4 cores minimum (8 recommended)
- **RAM**: 8GB minimum (16GB recommended for concurrent users)
- **Storage**: 10GB free space (models + vectors + documents)
- **GPU**: Optional (CUDA 11.8+ for 3× faster indexing)

### Software Dependencies
```bash
# Docker & Docker Compose
docker --version  # ≥20.10
docker-compose --version  # ≥1.29

# Python (for local development)
python --version  # ≥3.10
```

---

## Quick Start (5 minutes)

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/moroccan-fiscal-rag-chatbot.git
cd moroccan-fiscal-rag-chatbot
```

### 2. Configure Environment
```bash
# Copy template
cp .env.template .env

# Edit configuration
nano .env
```

**Required variables**:
```env
# LLM Provider (choose one)
OPENROUTER_API_KEY=sk-or-v1-xxxxx  # From https://openrouter.ai/keys
# OR
OLLAMA_BASE_URL=http://localhost:11434  # For local Ollama

# Vector Database
COLLECTION_NAME=fiscal_docs
CHROMA_DB_PATH=/app/chroma

# Embedding Model
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
```

### 3. Build & Launch
```bash
# Build Docker images
docker-compose build

# Start services (detached mode)
docker-compose up -d

# Verify containers
docker ps
```

**Expected output**:
```
CONTAINER ID   IMAGE                    STATUS         PORTS
abc123def456   moroccan-fiscal-rag-api  Up 2 minutes   0.0.0.0:8000->8000/tcp
789ghi012jkl   ghcr.io/open-webui       Up 2 minutes   0.0.0.0:3000->8080/tcp
```

### 4. Index Documents
```bash
# Run one-time indexing
docker exec -it moroccan-fiscal-rag-api python run_indexing.py

# Monitor progress
docker logs -f moroccan-fiscal-rag-api
```

**Expected duration**: ~12 minutes (3,500 chunks)

### 5. Access Interface
- **OpenWebUI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## Configuration Options

### Environment Variables (.env)
```env
# ========================================
# LLM PROVIDER CONFIGURATION
# ========================================

# OpenRouter (cloud inference)
OPENROUTER_API_KEY=sk-or-v1-xxxxx
DEFAULT_MODEL=deepseek/deepseek-chat

# Ollama (local inference)
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.1:8b

# ========================================
# EMBEDDING MODEL
# ========================================
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
EMBEDDING_DEVICE=cpu  # or 'cuda' if GPU available

# ========================================
# VECTOR DATABASE
# ========================================
COLLECTION_NAME=fiscal_docs
CHROMA_DB_PATH=/app/chroma
DISTANCE_METRIC=cosine  # or 'euclidean', 'l2'

# ========================================
# RETRIEVAL PARAMETERS
# ========================================
TOP_K_RETRIEVAL=25  # Initial candidates
TOP_N_FINAL=15      # After re-ranking
USE_RERANKER=true   # CrossEncoder re-ranking
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# ========================================
# CHUNKING PARAMETERS
# ========================================
CHUNK_SIZE=512      # Tokens per chunk
CHUNK_OVERLAP=50    # Token overlap between chunks

# ========================================
# GENERATION PARAMETERS
# ========================================
LLM_TEMPERATURE=0.1   # Low for factual consistency
LLM_MAX_TOKENS=1000   # Max response length
LLM_TOP_P=0.9         # Nucleus sampling

# ========================================
# API CONFIGURATION
# ========================================
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4             # Uvicorn workers (1 per 4GB RAM)
LOG_LEVEL=info        # debug, info, warning, error

# ========================================
# SECURITY
# ========================================
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
API_KEY_REQUIRED=false  # Set true for production

# ========================================
# PERFORMANCE
# ========================================
CACHE_EMBEDDINGS=true
CACHE_TTL=3600        # Seconds (1 hour)
MAX_CONCURRENT_REQUESTS=50
```

---

## Production Deployment

### 1. Hardware Recommendations

#### Small Deployment (1-10 users)
```
CPU: 4 cores
RAM: 8 GB
Storage: 20 GB SSD
Network: 100 Mbps
```

#### Medium Deployment (10-50 users)
```
CPU: 8 cores
RAM: 16 GB
Storage: 50 GB SSD
Network: 1 Gbps
```

#### Large Deployment (50+ users)
```
CPU: 16 cores
RAM: 32 GB
Storage: 100 GB NVMe SSD
Network: 10 Gbps
GPU: NVIDIA T4 (optional, for faster indexing)
```

### 2. Production docker-compose.yml
```yaml
version: '3.8'

services:
  rag-api:
    image: ghcr.io/YOUR_ORG/moroccan-fiscal-rag-api:latest
    restart: always
    ports:
      - "8000:8000"
    volumes:
      - ./vector_db:/app/chroma:rw
      - ./data:/app/data:ro
      - ./logs:/app/logs:rw
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - LOG_LEVEL=warning
      - WORKERS=8  # Scale based on CPU cores
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
        reservations:
          cpus: '4'
          memory: 8G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    restart: always
    ports:
      - "3000:8080"
    volumes:
      - open-webui-data:/app/backend/data
    environment:
      - OPENAI_API_BASE_URL=http://rag-api:8000/v1
      - ENABLE_SIGNUP=false  # Disable public registration
      - WEBUI_AUTH=true
    depends_on:
      rag-api:
        condition: service_healthy

  nginx:
    image: nginx:alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - open-webui

  prometheus:
    image: prom/prometheus:latest
    restart: always
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    restart: always
    ports:
      - "3001:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    depends_on:
      - prometheus

volumes:
  open-webui-data:
  prometheus-data:
  grafana-data:
```

