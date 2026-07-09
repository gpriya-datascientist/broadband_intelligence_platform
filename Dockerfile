# Broadband Intelligence Platform — API Container
# Build: docker build -t broadband-api .
# Run:   docker run -p 8000:8000 broadband-api

FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and trained artifacts
COPY config/       ./config/
COPY src/          ./src/
COPY api/          ./api/
COPY data/models/  ./data/models/
COPY data/processed/ ./data/processed/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
