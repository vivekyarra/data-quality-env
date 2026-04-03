# ── DataQualityEnv — Hugging Face Spaces Dockerfile ──────────────────────────
#
# HF Spaces runs containers as a non-root user on port 7860.
# Build:  docker build -t data-quality-env .
# Run:    docker run -p 7860:7860 data-quality-env
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (required by HF Spaces)
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install Python dependencies first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HF Spaces runs as uid 1000
RUN chown -R appuser:appuser /app
USER appuser

# HF Spaces default port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
