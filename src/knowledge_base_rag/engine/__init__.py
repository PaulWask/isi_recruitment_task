"""RAG Engine - Query pipeline connecting retrieval with generation.

Components:
- RAGEngine: Main query engine with metrics and warnings
- Reranker: Cross-encoder reranking for better precision
- HybridRetriever: BM25 + Vector fusion for better recall
- RetrievalResult: Structured retrieval result with scores

Usage:
    from knowledge_base_rag.engine import RAGEngine
    
    # Basic usage
    engine = RAGEngine()
    response = engine.query("What is...?")
    
    # With reranking for better precision (+25%)
    engine = RAGEngine(enable_reranking=True)
    response = engine.query("What is...?")
"""

from knowledge_base_rag.engine.rag import RAGEngine, RAGResponse, RAGMetrics
from knowledge_base_rag.engine.retrieval import (
    Reranker,
    HybridRetriever,
    BM25Index,
    RetrievalResult,
    QueryExpander,
    MultiQueryRetriever,
    DOMAIN_SYNONYMS,
)

__all__ = [
    # Core RAG
    "RAGEngine",
    "RAGResponse",
    "RAGMetrics",
    # Retrieval enhancements
    "Reranker",
    "HybridRetriever",
    "BM25Index",
    "RetrievalResult",
    # Query expansion
    "QueryExpander",
    "MultiQueryRetriever",
    "DOMAIN_SYNONYMS",
]
