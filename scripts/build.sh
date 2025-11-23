#!/bin/bash
# build.sh - Docker Build Script with Progress Tracking

echo "🚀 Starting Moroccan Fiscal RAG Docker Build..."
echo "================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Show build progress with timestamps
echo "⏰ Build started at: $(date)"
echo ""

# Build with progress output and no cache for clean build
echo "🔨 Building Docker image with progress tracking..."
docker build --progress=plain --no-cache -t moroccan-fiscal-rag . 2>&1 | while IFS= read -r line; do
    echo "[$(date '+%H:%M:%S')] $line"
done

# Check if build was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ Docker build completed successfully!"
    echo "⏰ Build finished at: $(date)"
    echo ""
    echo "🚢 To start the application, run:"
    echo "   docker-compose up -d"
    echo ""
    echo "🌐 The API will be available at: http://localhost:8000"
    echo "📋 API documentation: http://localhost:8000/docs"
else
    echo ""
    echo "❌ Docker build failed!"
    echo "⏰ Build failed at: $(date)"
    exit 1
fi