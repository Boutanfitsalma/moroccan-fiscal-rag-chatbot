FROM python:3.11-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set proper UTF-8 locale and encoding
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8

WORKDIR /app

# Install minimal system dependencies for OpenCV and OCR
RUN echo "Installing system dependencies..." && \
    apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-ara \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/* && \
    echo "System dependencies installed successfully!"

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with progress tracking
RUN echo "Starting Python package installation..." && \
    pip install --no-cache-dir --upgrade pip && \
    echo "Installing packages with progress tracking..." && \
    pip install --no-cache-dir --progress-bar on -r requirements.txt && \
    echo "Python dependencies installed successfully!"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/vector_db

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["python", "main_api.py"]