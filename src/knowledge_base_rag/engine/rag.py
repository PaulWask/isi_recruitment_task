"""RAG Engine - Retrieval Augmented Generation pipeline.

Design decisions:
- Combines vector retrieval with LLM generation for grounded answers
- Configurable retrieval parameters (top_k, similarity threshold)
- Source attribution in responses for transparency
- Streaming support for responsive UI

RAG Pipeline:
    1. User asks a question
    2. Question → Embedding → Vector search (retrieve top-k chunks)
    3. Retrieved chunks + Question → LLM prompt
    4. LLM generates answer grounded in retrieved context
    5. Return answer + source citations

Why RAG over pure LLM:
- Grounded: Answers based on your actual documents
- Accurate: Reduces hallucination by providing context
- Traceable: Can cite sources for every answer
- Up-to-date: No need to retrain model for new documents

Retrieval parameters:
- similarity_top_k=6: Retrieve 6 most relevant chunks
- This provides ~6K tokens of context (6 × 1024)
- Balances context richness vs. noise
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode

from knowledge_base_rag.core.config import settings
from knowledge_base_rag.core.llm import get_llm
from knowledge_base_rag.storage.embeddings import get_embed_model
from knowledge_base_rag.storage.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Structured response from RAG engine.
    
    Attributes:
        answer: Generated answer text
        sources: List of source documents used
        confidence: Optional confidence score
        query: Original query
    """
    answer: str
    sources: list[dict] = field(default_factory=list)
    confidence: Optional[float] = None
    query: str = ""
    
    def __str__(self) -> str:
        return self.answer
    
    def get_sources_text(self) -> str:
        """Get formatted sources for display."""
        if not self.sources:
            return "No sources found."
        
        lines = ["Sources:"]
        for i, src in enumerate(self.sources, 1):
            source_name = src.get("source", "Unknown")
            score = src.get("score", 0)
            lines.append(f"  {i}. {source_name} (relevance: {score:.2f})")
        
        return "\n".join(lines)


class RAGEngine:
    """RAG Engine for question answering over the knowledge base.
    
    Orchestrates the full RAG pipeline:
    - Vector retrieval from Qdrant
    - Response generation with LLM
    - Source attribution
    
    Example:
        # Initialize
        engine = RAGEngine()
        
        # Query
        response = engine.query("What are the symptoms of diabetes?")
        print(response.answer)
        print(response.get_sources_text())
        
        # With custom parameters
        response = engine.query(
            "Explain the treatment options",
            similarity_top_k=10,
        )
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        similarity_top_k: Optional[int] = None,
    ):
        """Initialize the RAG engine.
        
        Args:
            collection_name: Qdrant collection name. Defaults to config.
            llm: LLM for generation. Defaults to configured LLM.
            embed_model: Embedding model. Defaults to configured model.
            similarity_top_k: Number of chunks to retrieve. Defaults to config.
        """
        self.collection_name = collection_name or settings.collection_name
        self.similarity_top_k = similarity_top_k or settings.similarity_top_k
        
        # Lazy initialization
        self._llm = llm
        self._embed_model = embed_model
        self._vector_store_manager: Optional[VectorStoreManager] = None
        self._index: Optional[VectorStoreIndex] = None
        self._query_engine = None
        
        logger.info(
            f"RAGEngine initialized: collection={self.collection_name}, "
            f"top_k={self.similarity_top_k}"
        )
    
    @property
    def llm(self) -> LLM:
        """Get or create the LLM."""
        if self._llm is None:
            self._llm = get_llm()
        return self._llm
    
    @property
    def embed_model(self) -> BaseEmbedding:
        """Get or create the embedding model."""
        if self._embed_model is None:
            self._embed_model = get_embed_model()
        return self._embed_model
    
    @property
    def vector_store_manager(self) -> VectorStoreManager:
        """Get or create the vector store manager."""
        if self._vector_store_manager is None:
            self._vector_store_manager = VectorStoreManager(
                collection_name=self.collection_name,
                embed_model=self.embed_model,
            )
        return self._vector_store_manager
    
    @property
    def index(self) -> VectorStoreIndex:
        """Get or load the vector index."""
        if self._index is None:
            self._index = self.vector_store_manager.load_index()
            if self._index is None:
                raise ValueError(
                    f"No index found for collection '{self.collection_name}'.\n"
                    "Run indexing first: uv run python scripts/index_documents.py"
                )
        return self._index
    
    def _create_query_engine(
        self,
        similarity_top_k: Optional[int] = None,
    ) -> RetrieverQueryEngine:
        """Create a query engine with specified parameters.
        
        Args:
            similarity_top_k: Override default top_k for this query.
            
        Returns:
            Configured query engine.
        """
        top_k = similarity_top_k or self.similarity_top_k
        
        # Create retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
        )
        
        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode=ResponseMode.COMPACT,  # Compact context into single prompt
        )
        
        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        
        return query_engine
    
    def query(
        self,
        question: str,
        similarity_top_k: Optional[int] = None,
    ) -> RAGResponse:
        """Query the knowledge base and generate an answer.
        
        Args:
            question: User's question.
            similarity_top_k: Override default number of chunks to retrieve.
            
        Returns:
            RAGResponse with answer and sources.
        """
        logger.info(f"Query: {question[:100]}...")
        
        # Create query engine
        query_engine = self._create_query_engine(similarity_top_k)
        
        # Execute query
        response = query_engine.query(question)
        
        # Extract sources
        sources = []
        for node in response.source_nodes:
            source_info = {
                "source": node.metadata.get("source", node.metadata.get("file_name", "Unknown")),
                "score": node.score if hasattr(node, "score") and node.score else 0.0,
                "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                "metadata": node.metadata,
            }
            sources.append(source_info)
        
        # Build response
        rag_response = RAGResponse(
            answer=str(response),
            sources=sources,
            query=question,
        )
        
        logger.info(f"Generated answer with {len(sources)} sources")
        
        return rag_response
    
    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Retrieve relevant chunks without generating an answer.
        
        Useful for debugging or showing relevant documents.
        
        Args:
            question: Query text.
            top_k: Number of chunks to retrieve.
            
        Returns:
            List of retrieved chunks with metadata.
        """
        top_k = top_k or self.similarity_top_k
        
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
        )
        
        nodes = retriever.retrieve(question)
        
        results = []
        for node in nodes:
            results.append({
                "text": node.text,
                "score": node.score if hasattr(node, "score") else 0.0,
                "source": node.metadata.get("source", "Unknown"),
                "metadata": node.metadata,
            })
        
        return results
    
    def is_ready(self) -> bool:
        """Check if the RAG engine is ready for queries.
        
        Returns:
            True if index exists and is accessible.
        """
        try:
            return self.vector_store_manager.collection_exists()
        except Exception:
            return False
    
    def get_stats(self) -> dict:
        """Get RAG engine statistics.
        
        Returns:
            Dictionary with engine stats.
        """
        vs_stats = self.vector_store_manager.get_stats()
        
        return {
            "ready": self.is_ready(),
            "collection_name": self.collection_name,
            "similarity_top_k": self.similarity_top_k,
            "llm_model": settings.llm_model if settings.llm_service == "local" else settings.groq_model,
            "llm_service": settings.llm_service,
            "embed_model": settings.embed_model,
            "vectors_count": vs_stats.get("vectors_count", 0),
        }


def create_rag_engine(
    collection_name: Optional[str] = None,
    **kwargs,
) -> RAGEngine:
    """Factory function to create a RAG engine.
    
    Args:
        collection_name: Optional collection name override.
        **kwargs: Additional arguments passed to RAGEngine.
        
    Returns:
        Configured RAGEngine instance.
    
    Example:
        from knowledge_base_rag.engine import create_rag_engine
        
        engine = create_rag_engine()
        response = engine.query("What is...?")
    """
    return RAGEngine(collection_name=collection_name, **kwargs)


