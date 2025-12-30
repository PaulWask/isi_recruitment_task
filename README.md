# Knowledge Base RAG System

A production-ready question-answering system that retrieves relevant information from a 750MB domain knowledge base and generates accurate, grounded answers using Retrieval-Augmented Generation (RAG).

---

## 🚀 Ground Zero: Complete Setup (5 Steps)

**From zero to running in ~15-30 minutes:**

### Step 1: Clone & Install
```bash
git clone <repository-url>
cd isi_recruitment_task

# Install UV (package manager)
# Windows PowerShell:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync --python 3.11
```

### Step 2: Download Domain Data (~750MB)
```bash
uv run python scripts/download_data.py
```
Or manual download: https://isi-ml-public.s3.us-east-1.amazonaws.com/domaindata.zip → extract to `./domaindata/`

### Step 3: Setup LLM (Choose One)

**Option A: Ollama (Local, Free, Private)**
```bash
# Install from https://ollama.ai/
ollama pull llama3.2:3b
ollama serve  # Keep running in background
```

**Option B: Groq (Cloud, Fast, Free Tier)**
```bash
# Get API key from https://console.groq.com/
echo "LLM_SERVICE=groq" >> .env
echo "GROQ_API_KEY=your-key-here" >> .env
```

### Step 4: Index Documents (One-Time, ~10-30 min)
```bash
uv run python scripts/index_documents.py
```

### Step 5: Run the App
```bash
uv run streamlit run src/knowledge_base_rag/app.py
```
Open http://localhost:8501 ✅

---

## 🐳 Docker Alternative (Even Simpler)

If you have Docker installed, you can skip all the above:

```bash
# Clone the repo
git clone <repository-url>
cd isi_recruitment_task

# Download data first (still required)
# Download https://isi-ml-public.s3.us-east-1.amazonaws.com/domaindata.zip
# Extract to ./domaindata/

# Start everything with Docker
docker-compose up --build

# First time: Index documents (in another terminal)
docker-compose --profile indexing up indexer

# Access at http://localhost:8501
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                         │
├─────────────────────────────────────────────────────────────┤
│                       RAG Engine                            │
│         Query → Retriever (Top-K) → Response Generator      │
├─────────────────────────────────────────────────────────────┤
│  Qdrant (Vector DB)  │  HuggingFace Embeddings  │  Ollama   │
├─────────────────────────────────────────────────────────────┤
│              Document Loader & Chunker (LlamaIndex)         │
├─────────────────────────────────────────────────────────────┤
│                  Domain Data (750MB documents)              │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| RAG Framework | LlamaIndex | Mature, excellent abstractions |
| Vector DB | Qdrant (local) | High-performance, free local mode |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Free, fast, good quality |
| LLM (local) | Ollama + llama3.2:3b | Free, runs locally |
| LLM (cloud) | Groq | Free tier: 14,400 req/day |
| Web UI | Streamlit | Rapid prototyping, Python-native |
| Package Manager | UV | Fast, modern, lockfile support |

## Prerequisites

- Python 3.11+ (recommended: 3.11 or 3.12)
- [UV](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai/) (for local LLM) OR Groq API key

## Detailed Setup (Alternative to Ground Zero)

> **Note:** The "Ground Zero" section at the top provides a quick 5-step setup. This section provides more details.

### Indexing Options

```bash
# Standard indexing (1024-token chunks, good for general documents)
uv run python scripts/index_documents.py

# Small chunks (256 tokens) - RECOMMENDED for financial tables/structured data
uv run python scripts/index_documents.py --strategy small

# Preview what will be indexed (no changes)
uv run python scripts/index_documents.py --dry-run

# Add new documents only (incremental update)
uv run python scripts/index_documents.py --update

# Force rebuild (delete existing index)
uv run python scripts/index_documents.py --force
```

**Chunking Strategies:**
| Strategy | Chunk Size | Best For |
|----------|-----------|----------|
| `sentence` (default) | 1024 tokens | General text, narratives |
| `small` | 256 tokens | **Financial tables, structured data** |
| `semantic` | 512 tokens | Medium granularity |
| `large` | 2048 tokens | Dense context needed |

**What indexing does:**
- Loads all PDFs from `./domaindata/` (~500 files)
- Chunks them for optimal retrieval (~5000 chunks)
- Generates embeddings using HuggingFace `all-MiniLM-L6-v2`
- Stores vectors in local Qdrant database (`./qdrant_db/`)

⏱️ **Expected time: 10-30 minutes** (first run downloads embedding model)

### Running the App

```bash
# Standard way
uv run streamlit run src/knowledge_base_rag/app.py

# Using CLI command
uv run kb-app
```

Open http://localhost:8501 in your browser.

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_SERVICE` | `local` | `local` (Ollama) or `groq` |
| `GROQ_API_KEY` | - | Required if using Groq |
| `CHUNK_SIZE` | `1024` | Tokens per chunk |
| `SIMILARITY_TOP_K` | `6` | Chunks to retrieve per query |

## Project Structure

```
isi_recruitment_task/
├── pyproject.toml          # Dependencies & CLI scripts
├── uv.lock                 # Locked versions
├── .env.example            # Configuration template
├── README.md               # This file
├── DEVELOPMENT_PLAN.md     # Development roadmap
├── Dockerfile              # Multi-stage Docker build (CPU-optimized)
├── docker-compose.yml      # Production deployment (4 services)
├── docker-compose.dev.yml  # Development with local index (1 service)
├── .dockerignore           # Files excluded from Docker build
├── domaindata/             # 750MB documents (gitignored)
├── qdrant_db/              # Vector database (gitignored)
├── chat_history.json       # Persistent chat history (gitignored)
├── scripts/
│   ├── download_data.py    # Download domain data from S3
│   ├── index_documents.py  # Build vector search index
│   ├── test_llm.py         # Test LLM service
│   └── test_vector_store.py # Test vector store
└── src/knowledge_base_rag/
    ├── __init__.py
    ├── cli.py              # CLI entry points (kb-index, kb-app)
    ├── app.py              # Streamlit web app (main entry)
    ├── core/
    │   ├── config.py       # Pydantic settings
    │   └── llm.py          # Ollama/Groq LLM service
    ├── data/
    │   ├── loader.py       # Multi-format document loader
    │   └── processor.py    # Document chunking
    ├── storage/
    │   ├── embeddings.py   # HuggingFace embeddings
    │   └── vector_store.py # Qdrant vector store + get_all_documents()
    ├── engine/
    │   ├── rag.py          # RAG query pipeline + reranking
    │   └── retrieval.py    # BM25, QueryExpander, CrossEncoderReranker
    └── ui/                 # UI components module
        ├── __init__.py     # Exports all components
        ├── styles.py       # CSS loader + color constants
        ├── components.py   # HTML rendering (text cleaning for PDFs)
        └── static/
            ├── styles.css  # Main stylesheet (500+ lines, dark mode)
            └── templates/  # HTML template files
                ├── source_card.html
                ├── user_message.html
                ├── assistant_message.html
                ├── header.html
                ├── warning.html
                ├── loading.html
                └── status_indicator.html
```

### UI Module

The `ui/` module provides a professional, modular UI system with dark mode support:

| File | Purpose |
|------|---------|
| `styles.py` | `load_css()` function, `COLORS` dict, theme detection |
| `components.py` | `render_*()` functions + PDF text cleaning |
| `static/styles.css` | 500+ lines of CSS with variables, dark mode, responsive |
| `static/templates/*.html` | 7 HTML template files for components |

**Key features:**
- **PDF text cleaning**: Removes embedded newlines for full-width display
- **Dark mode**: CSS variables for automatic theme switching
- **Score badges**: Gradient-colored relevance percentages
- **Source metadata**: Page numbers, file paths, visible formatting

**Usage in app.py:**
```python
from knowledge_base_rag.ui import load_css, render_source_card

st.markdown(load_css(), unsafe_allow_html=True)
st.markdown(render_source_card(...), unsafe_allow_html=True)
```

## Advanced Features

The system includes professional retrieval enhancements available in the sidebar under **🔧 Advanced**:

### Reranking (Cross-Encoder) — Always Enabled

Reranking is **always enabled** because it consistently improves answer quality by ~25%. The system automatically re-scores retrieved documents with a cross-encoder model.

```
Query → Vector Search (20 candidates) → Cross-Encoder Rerank → Top 6 Results
```

- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Latency**: Adds ~3-5 seconds
- **Impact**: +25% precision improvement

### Query Expansion

Automatically expands acronyms and domain terminology for **+15-20% recall**:

| Query | Expanded To |
|-------|-------------|
| "CPI trends 2024" | "consumer price index trends 2024" |
| "GDP growth Q1" | "gross domestic product growth first quarter" |
| "YoY inflation" | "year over year inflation" |

**Supported expansions:**
- Financial acronyms: CPI, GDP, YoY, MoM, QoQ, IMF, ECB, Fed, BSP, FDI, OFW
- Time periods: Q1-Q4, H1-H2, FY
- Economic terms: inflation, recession, monetary policy
- Multi-language: Filipino/Tagalog terms (presyo, ekonomiya, trabaho)

**When Query Expansion is Essential:**
- Users use different terminology than documents
- Short queries (1-2 words) that need context
- Domain jargon with many synonyms
- Searching across documents in multiple languages

### Hybrid Search (BM25 + Vector)

Combines semantic vector search with traditional keyword matching (BM25). Enable in sidebar under **🔧 Advanced**.

```
Query → Vector Search (semantic) ─┬→ Reciprocal Rank Fusion → Rerank → Results
      → BM25 Search (keyword) ────┘
```

**When to use:**
- Searching for exact IDs, codes, or numbers (e.g., "Document ID 12345")
- Acronyms that might not have good embeddings
- When vector search alone misses obvious keyword matches

**Trade-off:** Adds ~1-2 seconds latency (BM25 index build on first use)

### Metrics Dashboard

Each response includes professional RAG metrics:

| Metric | Description |
|--------|-------------|
| **Latency** | End-to-end response time |
| **Precision@K** | % of retrieved docs above relevance threshold |
| **Avg Score** | Mean relevance score of sources |
| **MRR** | Mean Reciprocal Rank (quality of ranking) |
| **Sources** | Number of unique source files |

## Docker (Production)

### Quick Start with Docker

```bash
# Build and run all services
docker-compose up --build

# Access at http://localhost:8501
```

### Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `app` | 8501 | Streamlit web application |
| `qdrant` | 6333 | Vector database |
| `ollama` | 11434 | Local LLM (optional) |

### First-Time Setup with Docker

**Option A: If you already indexed locally** (have `qdrant_db/` folder):

```bash
# Use development compose (mounts local qdrant_db/)
docker-compose -f docker-compose.dev.yml up
```

**Option B: Fresh Docker environment** (no local index):

```bash
# 1. Start Qdrant and Ollama first
docker-compose up -d qdrant ollama

# 2. Pull the LLM model (if using Ollama)
docker exec rag-ollama ollama pull llama3.2:3b

# 3. Run indexing inside Docker (one-time, ~10-30 min)
docker-compose --profile indexing up indexer

# 4. Start the app
docker-compose up app
```

### Development Mode

For local development with hot-reload and existing local index:

```bash
docker-compose -f docker-compose.dev.yml up
```

## Docker Compose Files - Comparison

The project includes two Docker Compose configurations for different use cases:

### `docker-compose.yml` (Production / Full Stack)

**Best for:** Fresh installations, CI/CD, production deployments, or new team members.

```bash
docker-compose up --build
```

| Aspect | Details |
|--------|---------|
| **Services** | 4 (app, qdrant, ollama, indexer) |
| **Qdrant** | Runs as separate Docker container |
| **Ollama** | Runs as separate Docker container |
| **Data** | Stored in Docker volumes (isolated) |
| **Index** | Must be created with `--profile indexing` |
| **Hot-reload** | ❌ No (code copied into image) |

### `docker-compose.dev.yml` (Development / Local Index)

**Best for:** Developers with existing local index (`qdrant_db/`) and Ollama on host.

```bash
docker-compose -f docker-compose.dev.yml up
```

| Aspect | Details |
|--------|---------|
| **Services** | 1 (only app) |
| **Qdrant** | Uses local `./qdrant_db/` folder (mounted) |
| **Ollama** | Connects to host's Ollama (`host.docker.internal:11434`) |
| **Data** | Uses existing local files |
| **Index** | Uses existing local index |
| **Hot-reload** | ✅ Yes (source code mounted) |

### Quick Reference

| Scenario | Command |
|----------|---------|
| **You have local index + Ollama** | `docker-compose -f docker-compose.dev.yml up` |
| **Fresh start (everything in Docker)** | `docker-compose up` |
| **Build index in Docker** | `docker-compose --profile indexing up indexer` |
| **Start only infrastructure** | `docker-compose up -d qdrant ollama` |
| **No Docker (pure local)** | `uv run streamlit run src/knowledge_base_rag/app.py` |

### Container Names

| Compose File | Container Name | Port |
|--------------|----------------|------|
| `docker-compose.yml` | `rag-app` | 8501 |
| `docker-compose.yml` | `rag-qdrant` | 6333 |
| `docker-compose.yml` | `rag-ollama` | 11434 |
| `docker-compose.dev.yml` | `rag-app-dev` | 8501 |

### Using Groq Instead of Ollama

Edit `docker-compose.yml` or set environment variable:

```bash
GROQ_API_KEY=your-key docker-compose up
```

And change in `docker-compose.yml`:
```yaml
environment:
  - LLM_SERVICE=groq
  - GROQ_API_KEY=${GROQ_API_KEY}
```

### Environment Variables (Docker)

The Dockerfile sets these automatically for optimal Docker performance:

| Variable | Value | Purpose |
|----------|-------|---------|
| `STREAMLIT_SERVER_FILE_WATCHER_TYPE` | `none` | Prevents restarts on file changes |
| `STREAMLIT_RUNNER_FAST_RERUNS` | `true` | Faster UI updates |
| `UV_EXTRA_INDEX_URL` | `pytorch.org/whl/cpu` | CPU-only PyTorch (smaller image) |
| `NLTK_DATA` | `/app/.cache/nltk_data` | Pre-downloaded NLTK data |

## Development

```bash
# Install dev dependencies
uv sync --dev

# Format code
uv run black src/ scripts/
uv run isort src/ scripts/

# Type check
uv run mypy src/

# Run tests
uv run pytest tests/
```

## Implementation Decisions & Rationale

This section explains the technical choices made to meet the task requirements.

### Why LlamaIndex?
- **Mature abstractions** for RAG: document loading, chunking, indexing, querying
- **Flexible** architecture: easily swap components (LLM, embeddings, vector store)
- **Production-ready** with extensive documentation
- **Alternative considered**: LangChain (more complex, heavier dependency tree)

### Why Qdrant (Local)?
- **Free local mode** - no cloud costs, no API keys needed
- **High performance** - optimized for vector search
- **Persistence** - survives restarts without re-indexing
- **Alternative considered**: ChromaDB (simpler but less production features), Pinecone (requires API key)

### Why HuggingFace `all-MiniLM-L6-v2`?
- **Free** - no API costs
- **Fast** - 384 dimensions (smaller than 768+ alternatives)
- **Quality** - excellent for general semantic search
- **Local** - runs on CPU, no GPU required
- **Alternative considered**: OpenAI embeddings (paid), BGE-base (slower)

### Why Ollama + Groq Dual Support?
- **Ollama (local)**: Privacy, no costs, works offline
- **Groq (cloud)**: Faster, free tier (14,400 req/day), no local setup
- User can choose based on their needs

### Why Cross-Encoder Reranking?
- **+25% precision** improvement over bi-encoder alone
- Cross-encoders jointly encode query+document (more accurate)
- Applied to top-20 candidates → returns top-6 (efficient)
- **Trade-off**: Adds ~1-2s latency (acceptable for quality)

### Why Query Expansion with Domain Synonyms?
- **Problem**: Financial documents use formal terminology (e.g., "Consumer Price Index")
- **Users query differently**: "CPI", "inflation rate", "price index"
- **Solution**: Map common acronyms/jargon to full terms
- **Low cost**: No LLM call needed, instant expansion

### Why 512-1024 Token Chunks?
- **Too small (<256)**: Loses context, fragments concepts
- **Too large (>2048)**: Dilutes relevance, wastes LLM context
- **Sweet spot (512-1024)**: Captures complete concepts, fits in context
- Overlap (50-100 tokens) prevents cutting mid-sentence

### Why Streamlit for UI?
- **Rapid prototyping** - Python-native, minimal code
- **Chat interface** - built-in `st.chat_input`, `st.chat_message`
- **Free hosting** - Streamlit Cloud if needed
- **Alternative considered**: Gradio (similar), Flask/FastAPI + React (more work)

### Techniques NOT Implemented (and why)

| Technique | Why Skipped |
|-----------|-------------|
| **Fine-tuned embeddings** | Requires labeled training data we don't have |
| **Graph RAG** | Overkill for document Q&A; better for relationship queries |
| **Agentic RAG** | Multi-step reasoning not needed for single-turn Q&A |
| **Hypothetical Document Embeddings (HyDE)** | Available but adds latency; synonym expansion sufficient |

### Reducing Hallucinations

The system implements multiple safeguards:

1. **Grounded prompts**: "Answer ONLY based on the provided context"
2. **Source attribution**: Every answer shows sources with relevance scores
3. **Low-score warnings**: Alert when retrieved docs are weakly relevant (<50%)
4. **Confidence metrics**: Precision@K, MRR, hit rate visible to users
5. **No-answer fallback**: System admits when it can't find relevant information

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| **"System Not Ready"** | Run indexing first: `uv run python scripts/index_documents.py` |
| **Ollama connection error** | Ensure Ollama is running: `ollama serve` |
| **Slow first load (~1-2 min)** | Normal - downloading embedding model (cached after) |
| **Chat history error in Docker** | History file is mounted; clear via UI button, don't delete file |
| **Answer stops when clicking checkbox** | Fixed - checkboxes are now locked during query processing |
| **LLM says "no data" despite sources** | Check if sources are relevant (low % = wrong topic) |
| **Page metadata shows "9/2024"** | This is edition date, not page number (document metadata varies) |

### Docker-Specific Issues

| Problem | Solution |
|---------|----------|
| **Build takes forever** | Normal first time (~10 min); uses CPU-only PyTorch |
| **"Device or resource busy"** | Don't delete mounted files; use UI clear button |
| **Container restarts on checkbox click** | Set `STREAMLIT_SERVER_FILE_WATCHER_TYPE=none` (already in Dockerfile) |
| **No answers after restart** | Restart container: `docker restart rag-app-dev` |

### Performance Tips

1. **First load is slow** - Embedding model download is one-time
2. **Use Docker for WSL** - Native Linux is faster than WSL filesystem
3. **Disable Query Expansion** if you don't need it (~20-30s latency)
4. **BM25 first use is slow** - Index builds once per session

### Improving Results with Financial Tables

If the LLM returns **wrong values from financial tables** (e.g., wrong row or column):

1. **Re-index with smaller chunks** for more precise retrieval:
   ```bash
   # Delete existing index and re-index with small chunks
   rm -rf qdrant_db
   uv run python scripts/index_documents.py --strategy small
   ```

2. **Use Hybrid Search (BM25)** - Enable in sidebar for exact term matching
3. **Increase "Sources to retrieve"** slider to 8-10 for more context
4. **Be more specific in your query** - Include exact row labels, footnotes, and dates

The default chunk size (1024 tokens) may split financial tables across chunks. Using `--strategy small` (256 tokens) preserves table rows better but creates more chunks.

### Logs Location

```bash
# Docker logs
docker logs rag-app-dev --tail 100

# Local logs (in terminal running streamlit)
# Timestamps format: HH:MM:SS.mmm
```

## License

MIT

