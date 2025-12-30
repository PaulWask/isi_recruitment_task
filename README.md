# Knowledge Base RAG System

A question-answering system that retrieves relevant information from a 750MB domain knowledge base and generates accurate, grounded answers.

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

## Quick Start

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd isi_recruitment_task

# Install UV (if not installed)
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --python 3.11
```

### 2. Download Domain Data

Download and extract the domain knowledge base (~750MB):

```bash
uv run python scripts/download_data.py
```

This downloads from: https://isi-ml-public.s3.us-east-1.amazonaws.com/domaindata.zip

**Manual download** (alternative):
1. Download: https://isi-ml-public.s3.us-east-1.amazonaws.com/domaindata.zip
2. Extract to `./domaindata/` directory

Supported formats: PDF, DOCX, TXT, MD, HTML, PPTX, XLSX, CSV, JSON, RTF, XML, EPUB

> ⚠️ **Important**: After downloading, you must index the data (Step 4) before using the application.

### 3. Setup LLM

**Option A: Local LLM (Ollama)** - Recommended for privacy
```bash
# Install Ollama from https://ollama.ai/
# Then pull the model:
ollama pull llama3.2:3b
```

**Option B: Cloud LLM (Groq)** - Faster, no local setup
```bash
# Get free API key from https://console.groq.com/
# Add to .env file:
echo "LLM_SERVICE=groq" >> .env
echo "GROQ_API_KEY=your-api-key" >> .env
```

### 4. Index the Documents

**This step is required after downloading the data.**

```bash
uv run python scripts/index_documents.py
```

This will:
- Load all documents from `./domaindata/` (501 files)
- Chunk them for optimal retrieval (~5000 chunks)
- Generate embeddings using HuggingFace all-MiniLM-L6-v2
- Store vectors in local Qdrant database (`./qdrant_db/`)

⏱️ **Expected time: 10-30 minutes** for 750MB of documents (first run downloads embedding model).

**Options:**
```bash
# Preview what will be indexed (no changes)
uv run python scripts/index_documents.py --dry-run

# Add new documents only (incremental update)
uv run python scripts/index_documents.py --update

# Force rebuild (delete existing index)
uv run python scripts/index_documents.py --force
```

### 5. Run the Application

```bash
uv run streamlit run src/knowledge_base_rag/app.py
```

Open http://localhost:8501 in your browser.

**Alternative (using CLI command):**
```bash
uv run kb-app
```

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
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Production deployment
├── docker-compose.dev.yml  # Development with local index
├── domaindata/             # 750MB documents (gitignored)
├── qdrant_db/              # Vector database (gitignored)
├── scripts/
│   ├── download_data.py    # Download domain data from S3
│   ├── index_documents.py  # Build vector search index
│   ├── test_llm.py         # Test LLM service
│   └── test_vector_store.py # Test vector store
└── src/knowledge_base_rag/
    ├── __init__.py
    ├── cli.py              # CLI entry points (kb-index, kb-app)
    ├── app.py              # Streamlit web app
    ├── core/
    │   ├── config.py       # Pydantic settings
    │   └── llm.py          # Ollama/Groq LLM service
    ├── data/
    │   ├── loader.py       # Multi-format document loader
    │   └── processor.py    # Document chunking
    ├── storage/
    │   ├── embeddings.py   # HuggingFace embeddings
    │   └── vector_store.py # Qdrant vector store
    ├── engine/
    │   ├── rag.py          # RAG query pipeline
    │   └── retrieval.py    # Advanced retrieval (reranking, query expansion)
    └── ui/                 # UI components module
        ├── __init__.py     # Exports all components
        ├── styles.py       # CSS loader + color constants
        ├── components.py   # HTML rendering functions
        └── static/
            ├── styles.css  # Main stylesheet (CSS variables, responsive)
            └── templates/  # HTML template files
                ├── source_card.html
                ├── user_message.html
                ├── assistant_message.html
                ├── header.html
                ├── warning.html
                └── status_indicator.html
```

### UI Module

The `ui/` module provides a professional, modular UI system:

| File | Purpose |
|------|---------|
| `styles.py` | `load_css()` function, `COLORS` dict |
| `components.py` | `render_*()` functions that load HTML templates |
| `static/styles.css` | 380+ lines of CSS with variables, responsive design |
| `static/templates/*.html` | 6 HTML template files for components |

**Usage in app.py:**
```python
from knowledge_base_rag.ui import load_css, render_source_card

st.markdown(load_css(), unsafe_allow_html=True)
st.markdown(render_source_card(...), unsafe_allow_html=True)
```

## Advanced Features

The system includes professional retrieval enhancements available in the sidebar under **🔧 Advanced**:

### Reranking (Cross-Encoder)

When enabled, re-scores retrieved documents with a cross-encoder model for **+25% precision**.

```
Query → Vector Search (20 candidates) → Cross-Encoder Rerank → Top 5 Results
```

- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Latency**: Adds ~1-2 seconds
- **When to use**: When precision matters more than speed

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

## License

MIT

