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

The domain data (~750MB) will be downloaded automatically:

```bash
uv run python scripts/download_data.py
```

This downloads from: https://isi-ml-public.s3.us-east-1.amazonaws.com/domaindata.zip

**Manual download** (alternative):
1. Download: https://isi-ml-public.s3.us-east-1.amazonaws.com/domaindata.zip
2. Extract to `./domaindata/` directory

Supported formats: PDF, DOCX, TXT, MD, HTML, PPTX, XLSX

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

### 4. Build the Index

```bash
uv run python scripts/index_documents.py
```
This will:
- Load all documents from `./domaindata/`
- Chunk them for optimal retrieval
- Create embeddings and store in Qdrant vector database

⏱️ **Expected time: 10-30 minutes** for 750MB of documents.

### 5. Run the Application

```bash
uv run streamlit run src/knowledge_base_rag/ui/app.py
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
├── pyproject.toml         # Dependencies
├── uv.lock                # Locked versions
├── .env.example           # Configuration template
├── domaindata/            # 750MB documents (gitignored)
├── qdrant_db/             # Vector database (gitignored)
├── storage/               # Index cache (gitignored)
├── scripts/
│   ├── download_data.py   # Download domain data
│   └── index_documents.py # Build search index
└── src/knowledge_base_rag/
    ├── core/              # Configuration, LLM service
    ├── data/              # Document loading, chunking
    ├── storage/           # Embeddings, vector store
    ├── engine/            # RAG query engine
    └── ui/                # Streamlit web app
```

## Docker (Production)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:8501
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

## License

MIT

