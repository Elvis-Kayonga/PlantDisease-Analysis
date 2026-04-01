FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    nodejs \
    npm \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

RUN mkdir -p models uploaded_data logs

# Expose ports
EXPOSE 8000

# Build the React frontend
RUN cd frontend && npm install && npm run build

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run unified FastAPI and React UI
CMD ["python", "main.py"]
