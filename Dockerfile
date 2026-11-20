# ---- Build & Run Python Environment ----
FROM python:3.11-slim

# --- Environment ---
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies (layer-cached) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application source + pre-scraped dataset ---
COPY . .

# Ensure data directory exists with correct permissions before switching user
RUN mkdir -p /app/data /app/static

# --- Non-root user (Hugging Face Spaces & container best practices) ---
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# --- Port ---
EXPOSE 7860

# --- Health check ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# --- Start server (production: no --reload) ---
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
