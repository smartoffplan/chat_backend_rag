# ── Stage 1: build deps + pre-download model ─────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch FIRST — prevents pip from pulling the 2 GB CUDA build
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.2.2 \
        --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (sentence-transformers will reuse torch above)
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Pre-download the embedding model into the image — zero network at runtime
ENV HF_HOME=/build/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/build/hf_cache
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('Downloading model...'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Done.')"


# ── Stage 2: lean runtime image ──────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Only copy installed packages — no build tools in final image
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-baked model cache
COPY --from=builder /build/hf_cache /app/hf_cache

# Copy application source (.dockerignore excludes .env, uploads/, chroma_db/, etc.)
COPY . .

# Non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/uploads /app/data/chroma_db && \
    chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/hf_cache \
    UPLOAD_DIR=/app/data/uploads \
    CHROMA_PATH=/app/data/chroma_db \
    PORT=8080

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
