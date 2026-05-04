# ── Stage 1: build deps + download the embedding model ──────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download sentence-transformer model so cold starts are instant
ENV HF_HOME=/build/hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


# ── Stage 2: lean runtime image ──────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder (no gcc/g++ in final image)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-downloaded HuggingFace model
COPY --from=builder /build/hf_cache /app/hf_cache

# Copy application source
COPY . .

# Non-root user for security
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/data/uploads /app/data/chroma_db \
    && chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/hf_cache \
    # Default data paths (overridden by GCS volume mount in Cloud Run)
    UPLOAD_DIR=/app/data/uploads \
    CHROMA_PATH=/app/data/chroma_db \
    PORT=8080

EXPOSE 8080

# Single worker — Cloud Run scales horizontally; multiple workers
# would open competing ChromaDB connections on the same volume.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
