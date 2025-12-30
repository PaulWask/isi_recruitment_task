# =============================================================================
# Knowledge Base RAG - Production Dockerfile
# =============================================================================
# Multi-stage build for optimal image size and security
#
# OPTIMIZATION: Uses CPU-only PyTorch (via UV_EXTRA_INDEX_URL)
# - Default PyTorch with CUDA: ~8GB image
# - CPU-only PyTorch: ~2.5GB image
# Since we use Ollama for LLM (external service), we don't need GPU in container.
#
# Build: docker build -t knowledge-base-rag .
# Run:   docker run -p 8501:8501 knowledge-base-rag
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build stage with UV for dependency resolution
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock* ./

# Create virtual environment and install dependencies
# Use CPU-only PyTorch index to avoid downloading CUDA (~6GB savings)
# --index-strategy unsafe-best-match: allows getting packages from any index (needed for requests compatibility)
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
RUN uv sync --no-dev \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match

# -----------------------------------------------------------------------------
# Stage 2: Runtime stage (minimal)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser pyproject.toml ./

# Create directories for data (will be mounted as volumes)
RUN mkdir -p /app/domaindata /app/qdrant_db /app/.cache \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Pre-download all models and data during build (not at runtime!)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV NLTK_DATA=/app/.cache/nltk_data

# 1. Download embedding model (~80MB)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# 2. Download cross-encoder reranker model (~80MB) 
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# 3. Download NLTK data required by LlamaIndex
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/app/.cache/nltk_data', quiet=True); nltk.download('averaged_perceptron_tagger_eng', download_dir='/app/.cache/nltk_data', quiet=True); nltk.download('stopwords', download_dir='/app/.cache/nltk_data', quiet=True)"

# 4. Compile Python bytecode for faster startup
RUN python -m compileall -q /app/src

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
# Disable file watcher for stable caching (critical for Docker)
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_RUNNER_FAST_RERUNS=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose Streamlit port
EXPOSE 8501

# Default command: run Streamlit app
CMD ["python", "-m", "streamlit", "run", "src/knowledge_base_rag/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]

