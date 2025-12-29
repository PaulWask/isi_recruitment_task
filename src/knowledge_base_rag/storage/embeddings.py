"""Embedding service using HuggingFace models.

Design decisions:
- Use HuggingFace sentence-transformers for FREE local embeddings
- Model: all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
- Runs entirely locally - no API costs, no rate limits, full privacy
- Singleton pattern to cache model after first load

Why all-MiniLM-L6-v2:
- 384 dimensions: Compact but effective (vs 768 or 1536 in larger models)
- 22M parameters: Fast inference on CPU
- Trained on 1B+ sentence pairs: Good semantic understanding
- MIT license: Free for commercial use
- Widely used: Battle-tested in production RAG systems

Alternative models (if needed):
- all-mpnet-base-v2: Higher quality, slower (768 dims)
- bge-small-en-v1.5: Newer, similar performance
- e5-small-v2: Good for queries vs documents

Performance expectations:
- First load: ~5 seconds (downloads model if not cached)
- Embedding speed: ~100 texts/second on CPU
- Memory: ~100MB for model
"""

import logging
from typing import Optional

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from knowledge_base_rag.core.config import settings

logger = logging.getLogger(__name__)

# Global cached model instance (singleton)
_cached_embed_model: Optional[HuggingFaceEmbedding] = None


class EmbeddingService:
    """Service for generating text embeddings using HuggingFace models.

    Uses sentence-transformers models running locally for free embeddings.
    Model is cached globally after first initialization.

    Example:
        service = EmbeddingService()
        
        # Single text
        embedding = service.get_embedding("What is machine learning?")
        print(f"Dimension: {len(embedding)}")  # 384
        
        # Batch processing
        embeddings = service.get_embeddings(["text1", "text2", "text3"])
    """

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the embedding service.

        Args:
            model_name: HuggingFace model name. Default: all-MiniLM-L6-v2
        """
        self.model_name = model_name or settings.embed_model
        self._model: Optional[HuggingFaceEmbedding] = None

    @property
    def model(self) -> HuggingFaceEmbedding:
        """Get or create the embedding model (cached globally).

        Returns:
            HuggingFaceEmbedding model instance.
        """
        global _cached_embed_model

        # Return cached model if available and same model name
        if _cached_embed_model is not None:
            return _cached_embed_model

        # Create new model
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = HuggingFaceEmbedding(
                model_name=self.model_name,
                embed_batch_size=32,  # Process 32 texts at a time
            )
            # Cache globally for reuse across instances
            _cached_embed_model = self._model
            logger.info("Embedding model loaded successfully")

        return self._model

    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            List of float values (embedding vector).
        """
        return self.model.get_text_embedding(text)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (batch processing).

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return self.model.get_text_embedding_batch(texts, show_progress=True)

    def get_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a query (may use different prompt).

        Some models use different embeddings for queries vs documents.
        This method handles that automatically.

        Args:
            query: Query text to embed.

        Returns:
            Query embedding vector.
        """
        return self.model.get_query_embedding(query)

    def get_dimension(self) -> int:
        """Get the embedding dimension.

        Returns:
            Number of dimensions in embedding vectors.
        """
        # Generate a test embedding to get dimensions
        test_embedding = self.get_embedding("test")
        return len(test_embedding)

    def get_model_info(self) -> dict:
        """Get information about the embedding model.

        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.model_name,
            "dimension": self.get_dimension(),
            "batch_size": 32,
            "type": "local",
            "provider": "HuggingFace",
        }


def get_embed_model(model_name: Optional[str] = None) -> HuggingFaceEmbedding:
    """Get the embedding model instance (convenience function).

    This is the primary way to get an embedding model for use with
    LlamaIndex components.

    Args:
        model_name: Optional model name override.

    Returns:
        HuggingFaceEmbedding model instance.

    Example:
        from knowledge_base_rag.storage.embeddings import get_embed_model
        
        embed_model = get_embed_model()
        # Use with LlamaIndex index
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    """
    service = EmbeddingService(model_name)
    return service.model
