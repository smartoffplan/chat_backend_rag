# ── Stage 1: install deps + pre-download the embedding model ─────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Build tools needed by some Python packages (removed in final stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install with extended timeout — sentence-transformers pulls large wheels
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --timeout=300 -r requirements.txt

# Pre-download the model into the image so startup is instant
# Uses a local cache dir instead of /root to make copying easier
ENV HF_HOME=/build/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/build/hf_cache
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('Downloading all-MiniLM-L6-v2 ...'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Model downloaded.')"


# ── Stage 2: lean production image ───────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy only installed packages — no build tools in final image
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-downloaded model cache
COPY --from=builder /build/hf_cache /app/hf_cache

# Copy application source (excluded files are listed in .dockerignore)
COPY . .

# Create data dirs and a non-root user
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/data/uploads /app/data/chroma_db \
    && chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/hf_cache \
    UPLOAD_DIR=/app/data/uploads \
    CHROMA_PATH=/app/data/chroma_db \
    PORT=8080

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout-keep-alive", "65"]
