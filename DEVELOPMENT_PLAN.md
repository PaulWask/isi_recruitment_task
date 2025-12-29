# Knowledge Base RAG - Development Plan

## 🎯 Project Goal
Build a knowledge base aware question answering system that retrieves relevant information from 750MB of domain documents and generates accurate, grounded answers.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                            │
│              (Question input + Response display)                │
├─────────────────────────────────────────────────────────────────┤
│                        RAG Engine                               │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │    Query     │→ │   Retriever  │→ │  Response Generator   │ │
│  │  Processor   │  │   (Top-K)    │  │  (LLM + Context)      │ │
│  └──────────────┘  └──────────────┘  └───────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Vector Store    │   Embeddings      │        LLM              │
│  (Qdrant local)  │ (HuggingFace)     │ (Ollama/Groq)          │
├─────────────────────────────────────────────────────────────────┤
│              Document Loader & Chunker (LlamaIndex)             │
├─────────────────────────────────────────────────────────────────┤
│                  Domain Data (AWS S3 → local)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack & Justification

| Component | Technology | Why This Choice |
|-----------|------------|-----------------|
| **Framework** | LlamaIndex | Mature RAG framework, excellent abstractions, document loaders built-in |
| **Vector DB** | Qdrant (local) | High-performance, free local mode, easy Docker deployment |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Free, local, 384 dims, good quality/speed tradeoff |
| **LLM (local)** | Ollama + llama3.2:3b | Free, runs locally, 3B params fits most machines |
| **LLM (cloud)** | Groq | Free tier (14,400 req/day), fastest inference |
| **Web UI** | Streamlit | Rapid prototyping, Python-native, beautiful out of box |
| **Package Manager** | UV | Fast, modern, lockfile support |

---

## 📁 Project Structure

```
isi_recruitment_task/
├── pyproject.toml           # Dependencies
├── uv.lock                  # Locked dependencies
├── .env.example             # Environment template
├── .gitignore               # Excludes domaindata/, qdrant_db/, storage/
├── docker-compose.yml       # Local deployment
├── Dockerfile               # App container
│
├── domaindata/              # 750MB documents (from S3, gitignored)
├── qdrant_db/               # Vector DB storage (gitignored)
├── storage/                 # Index cache (gitignored)
│
├── scripts/
│   ├── download_data.py     # Download from S3
│   ├── index_documents.py   # Build vector index
│   └── run_app.py           # Launch Streamlit
│
├── src/knowledge_base_rag/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Settings with pydantic
│   │   └── llm.py           # LLM service (Ollama/Groq)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py        # Document loading
│   │   └── processor.py     # Chunking
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── embeddings.py    # Embedding service
│   │   └── vector_store.py  # Qdrant management
│   ├── engine/
│   │   ├── __init__.py
│   │   └── rag.py           # RAG query engine
│   └── ui/
│       ├── __init__.py
│       └── app.py           # Streamlit app
│
└── tests/
    ├── test_loader.py
    ├── test_processor.py
    └── test_rag.py
```

---

## 📋 Commit-by-Commit Development Plan

### Phase 1: Foundation (Commits 1-3)

#### Commit 1: Project Setup
**Files:** `.gitignore`, `.env.example`, `pyproject.toml`
**Validation before commit:**
```bash
uv sync                    # Verify dependencies install
uv run python -c "import llama_index; print('OK')"
```

#### Commit 2: Configuration Module
**Files:** `src/knowledge_base_rag/core/config.py`, `core/__init__.py`
**Validation:**
```bash
uv run python -c "from knowledge_base_rag.core.config import settings; print(settings)"
```

#### Commit 3: Data Download Script
**Files:** `scripts/download_data.py`
**Validation:**
```bash
uv run python scripts/download_data.py --help
```

---

### Phase 2: Data Pipeline (Commits 4-5)

#### Commit 4: Document Loader
**Files:** `src/knowledge_base_rag/data/loader.py`, `data/__init__.py`
**Validation:**
```bash
# Create test file first
echo "Test document content" > domaindata/test.txt
uv run python -c "
from knowledge_base_rag.data.loader import DocumentLoader
loader = DocumentLoader()
print(loader.count_documents())
"
```

#### Commit 5: Document Processor (Chunking)
**Files:** `src/knowledge_base_rag/data/processor.py`
**Validation:**
```bash
uv run python -c "
from knowledge_base_rag.data.loader import DocumentLoader
from knowledge_base_rag.data.processor import DocumentProcessor
docs = DocumentLoader().load_documents()
chunks = DocumentProcessor().process_documents(docs)
print(f'Created {len(chunks)} chunks')
"
```

---

### Phase 3: Storage Layer (Commits 6-7)

#### Commit 6: Embedding Service
**Files:** `src/knowledge_base_rag/storage/embeddings.py`, `storage/__init__.py`
**Validation:**
```bash
uv run python -c "
from knowledge_base_rag.storage.embeddings import EmbeddingService
svc = EmbeddingService()
emb = svc.get_embedding('Hello world')
print(f'Embedding dimension: {len(emb)}')
"
```

#### Commit 7: Vector Store (Qdrant)
**Files:** `src/knowledge_base_rag/storage/vector_store.py`
**Validation:**
```bash
uv run python -c "
from knowledge_base_rag.storage.vector_store import VectorStoreManager
mgr = VectorStoreManager()
print(f'Collection exists: {mgr.collection_exists()}')
"
```

---

### Phase 4: LLM & Engine (Commits 8-9)

#### Commit 8: LLM Service
**Files:** `src/knowledge_base_rag/core/llm.py`
**Validation:**
```bash
# Requires Ollama running: ollama run llama3.2:3b
uv run python -c "
from knowledge_base_rag.core.llm import LLMService
svc = LLMService()
print(f'LLM available: {svc.is_available()}')
"
```

#### Commit 9: RAG Engine
**Files:** `src/knowledge_base_rag/engine/rag.py`, `engine/__init__.py`
**Validation:**
```bash
uv run python -c "
from knowledge_base_rag.engine.rag import RAGEngine
engine = RAGEngine()
print('RAG Engine initialized')
"
```

---

### Phase 5: Indexing & UI (Commits 10-11)

#### Commit 10: Indexing Script
**Files:** `scripts/index_documents.py`
**Validation:**
```bash
uv run python scripts/index_documents.py --help
# Full test (with domaindata):
uv run python scripts/index_documents.py
```

#### Commit 11: Streamlit UI
**Files:** `src/knowledge_base_rag/ui/app.py`, `scripts/run_app.py`
**Validation:**
```bash
uv run streamlit run src/knowledge_base_rag/ui/app.py
# Open http://localhost:8501 and test a query
```

---

### Phase 6: Optimization & Docker (Commits 12-14)

#### Commit 12: Performance Optimization
- Batch indexing for large datasets
- Caching for repeated queries
- Memory-efficient document loading

#### Commit 13: Docker Setup
**Files:** `Dockerfile`, `docker-compose.yml`
**Validation:**
```bash
docker-compose build
docker-compose up -d
# Test at http://localhost:8501
```

#### Commit 14: Final Documentation
**Files:** `README.md` (update with full instructions)

---

## 🚀 How to Run (After Implementation)

### Option 1: Local Development
```bash
# 1. Install dependencies
uv sync

# 2. Download domain data from S3
uv run python scripts/download_data.py

# 3. Start Ollama (if using local LLM)
ollama run llama3.2:3b

# 4. Build the index (one-time, ~10-30 min for 750MB)
uv run python scripts/index_documents.py

# 5. Run the app
uv run streamlit run src/knowledge_base_rag/ui/app.py
```

### Option 2: Docker (Production-like)
```bash
# 1. Download domain data first
uv run python scripts/download_data.py

# 2. Build and run everything
docker-compose up --build

# 3. Access at http://localhost:8501
```

---

## ⚙️ Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_DB` | `local` | `local` or `qdrant_cloud` |
| `LLM_SERVICE` | `local` | `local` (Ollama) or `groq` |
| `GROQ_API_KEY` | - | Required if using Groq |
| `CHUNK_SIZE` | `1024` | Tokens per chunk |
| `SIMILARITY_TOP_K` | `6` | Number of chunks to retrieve |

---

## 🔍 RAG Parameters Explained

- **chunk_size=1024**: Balances context preservation with retrieval precision
- **chunk_overlap=128**: Prevents losing context at chunk boundaries  
- **similarity_top_k=6**: Enough context without overwhelming the LLM
- **similarity_cutoff=0.7**: Filters out low-relevance noise

---

## 📊 Expected Performance

| Metric | Estimate |
|--------|----------|
| Indexing time (750MB) | 10-30 minutes |
| Query latency | 2-5 seconds |
| Vector DB size | ~500MB |
| RAM usage | 4-8 GB |

---

## ✅ Pre-Commit Checklist

Before each commit, run:
```bash
# 1. Format code
uv run black src/ scripts/ tests/
uv run isort src/ scripts/ tests/

# 2. Type check
uv run mypy src/ --ignore-missing-imports

# 3. Run tests
uv run pytest tests/ -v

# 4. Verify imports work
uv run python -c "from knowledge_base_rag import __version__; print(__version__)"
```

---

Ready to start? Let's begin with **Commit 1: Project Setup**!


