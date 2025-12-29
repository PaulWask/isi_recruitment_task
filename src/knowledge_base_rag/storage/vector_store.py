"""Vector store management using Qdrant.

Design decisions:
- Qdrant chosen for high performance and flexible deployment options
- Local mode: Zero-config, runs in-process, data persisted to disk
- Cloud mode: Free tier (1GB) available for production deployments
- Automatic collection creation with proper vector configuration

Why Qdrant over alternatives:
1. vs Pinecone: Qdrant has free local mode, Pinecone requires cloud
2. vs Chroma: Qdrant is more performant at scale (750MB+ data)
3. vs Milvus: Qdrant is lighter weight, easier to deploy
4. vs FAISS: Qdrant provides persistence, filtering, and metadata

Deployment options:
- Local (default): Data stored in ./qdrant_db/, runs in-process
- Docker: docker run -p 6333:6333 qdrant/qdrant
- Cloud: Free tier at https://cloud.qdrant.io/

Performance characteristics:
- Indexing: ~1000 vectors/second
- Search: <10ms for top-k retrieval
- Storage: ~1KB per vector (384 dims + metadata)
"""

import logging
from pathlib import Path
from typing import Optional

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from knowledge_base_rag.core.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton for Qdrant client to prevent concurrent access issues
# Local file-based Qdrant doesn't support multiple connections
_qdrant_client_singleton: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Get singleton Qdrant client to avoid concurrent access issues.
    
    Local file-based Qdrant only allows one client connection at a time.
    This singleton ensures we reuse the same client instance.
    """
    global _qdrant_client_singleton
    
    if _qdrant_client_singleton is None:
        if settings.vector_db == "qdrant_cloud":
            if not settings.qdrant_cloud_url:
                raise ValueError("QDRANT_CLOUD_URL is required for cloud mode")
            
            if settings.qdrant_cloud_api_key:
                _qdrant_client_singleton = QdrantClient(
                    url=settings.qdrant_cloud_url,
                    api_key=settings.qdrant_cloud_api_key,
                )
            else:
                # Self-hosted Qdrant (Docker) - no API key needed
                _qdrant_client_singleton = QdrantClient(url=settings.qdrant_cloud_url)
            logger.info(f"Connected to Qdrant server: {settings.qdrant_cloud_url}")
        else:
            # Local file-based client
            qdrant_path = settings.get_absolute_qdrant_path()
            logger.info(f"Using local Qdrant at: {qdrant_path}")
            logger.info(f"Qdrant path exists: {qdrant_path.exists()}")
            
            if not qdrant_path.exists():
                qdrant_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created qdrant_db directory: {qdrant_path}")
            
            _qdrant_client_singleton = QdrantClient(path=str(qdrant_path))
    
    return _qdrant_client_singleton


class VectorStoreManager:
    """Manage Qdrant vector store for document embeddings.

    Handles collection lifecycle, indexing, and retrieval operations.
    Supports both local (in-process) and cloud deployments.

    Example:
        from knowledge_base_rag.storage.vector_store import VectorStoreManager
        from knowledge_base_rag.storage.embeddings import get_embed_model

        # Initialize
        manager = VectorStoreManager(embed_model=get_embed_model())

        # Create index from documents
        index = manager.create_index(documents)

        # Or load existing index
        index = manager.load_index()

        # Query
        query_engine = index.as_query_engine()
        response = query_engine.query("What is...")
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embed_model: Optional[BaseEmbedding] = None,
    ):
        """Initialize the vector store manager.

        Args:
            collection_name: Name for the vector collection.
            embed_model: Embedding model for vectorization.
        """
        self.collection_name = collection_name or settings.collection_name
        self.embed_model = embed_model
        self._client: Optional[QdrantClient] = None
        self._vector_store: Optional[QdrantVectorStore] = None

        logger.info(f"VectorStoreManager initialized: collection={self.collection_name}")

    @property
    def client(self) -> QdrantClient:
        """Get the Qdrant client (uses module singleton for local mode)."""
        if self._client is None:
            self._client = get_qdrant_client()
        return self._client

    @property
    def vector_store(self) -> QdrantVectorStore:
        """Get or create the vector store."""
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
            )
        return self._vector_store

    # =========================================================================
    # Collection Management
    # =========================================================================

    def collection_exists(self) -> bool:
        """Check if the collection exists.

        Returns:
            True if collection exists, False otherwise.
        """
        try:
            collections = self.client.get_collections()
            return any(
                c.name == self.collection_name
                for c in collections.collections
            )
        except Exception as e:
            logger.error(f"Error checking collection: {e}")
            return False

    def get_collection_info(self) -> Optional[dict]:
        """Get detailed information about the collection.

        Returns:
            Dictionary with collection stats or None if not exists.
        """
        if not self.collection_exists():
            return None

        try:
            info = self.client.get_collection(self.collection_name)

            # Extract vector size safely (API varies between versions)
            vector_size = None
            if hasattr(info.config, 'params') and hasattr(info.config.params, 'vectors'):
                vectors_config = info.config.params.vectors
                if hasattr(vectors_config, 'size'):
                    vector_size = getattr(vectors_config, 'size', None)
                elif isinstance(vectors_config, dict):
                    first_config = next(iter(vectors_config.values()), None)
                    if first_config is not None:
                        vector_size = getattr(first_config, 'size', None)

            # Handle different Qdrant client versions
            vectors_count = getattr(info, 'vectors_count', None)
            if vectors_count is None:
                vectors_count = getattr(info, 'points_count', 0)

            points_count = getattr(info, 'points_count', vectors_count)

            status = "unknown"
            if hasattr(info, 'status'):
                status = info.status.value if hasattr(info.status, 'value') else str(info.status)

            return {
                "name": self.collection_name,
                "vectors_count": vectors_count,
                "points_count": points_count,
                "status": status,
                "vector_size": vector_size,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None

    def delete_collection(self) -> bool:
        """Delete the collection if it exists.

        Returns:
            True if deleted successfully, False otherwise.
        """
        if not self.collection_exists():
            logger.info(f"Collection '{self.collection_name}' does not exist")
            return False

        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")

            # Reset cached vector store
            self._vector_store = None
            return True

        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

    # =========================================================================
    # Indexing Operations
    # =========================================================================

    def create_index(
        self,
        documents: list[Document],
        show_progress: bool = True,
    ) -> VectorStoreIndex:
        """Create a new index from documents.

        This will delete any existing collection and build fresh.

        Args:
            documents: Documents to index (should be pre-chunked).
            show_progress: Show indexing progress bar.

        Returns:
            VectorStoreIndex for querying.

        Raises:
            ValueError: If no documents or embedding model not set.
        """
        if self.embed_model is None:
            raise ValueError(
                "Embedding model required for indexing. "
                "Pass embed_model to VectorStoreManager constructor."
            )

        if not documents:
            raise ValueError("No documents provided for indexing.")

        logger.info(f"Creating index with {len(documents)} documents...")

        # Delete existing collection for clean rebuild
        if self.collection_exists():
            self.delete_collection()

        # Create storage context with our vector store
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
        )

        # Build the index (this embeds and stores all documents)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=show_progress,
        )

        # Log stats
        info = self.get_collection_info()
        if info:
            logger.info(
                f"Index created: {info['vectors_count']} vectors, "
                f"dimension={info['vector_size']}"
            )

        return index

    def load_index(self) -> Optional[VectorStoreIndex]:
        """Load an existing index from the vector store.

        Returns:
            VectorStoreIndex if collection exists, None otherwise.

        Raises:
            ValueError: If embedding model not set.
        """
        if self.embed_model is None:
            raise ValueError(
                "Embedding model required for loading index. "
                "Pass embed_model to VectorStoreManager constructor."
            )

        if not self.collection_exists():
            logger.warning(f"Collection '{self.collection_name}' does not exist")
            return None

        logger.info(f"Loading index from collection: {self.collection_name}")

        index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )

        info = self.get_collection_info()
        if info:
            logger.info(f"Loaded index: {info['vectors_count']} vectors")

        return index

    def get_or_create_index(
        self,
        documents: Optional[list[Document]] = None,
    ) -> VectorStoreIndex:
        """Get existing index or create new one.

        Args:
            documents: Documents to index if collection doesn't exist.

        Returns:
            VectorStoreIndex for querying.

        Raises:
            ValueError: If no index exists and no documents provided.
        """
        # Try to load existing
        index = self.load_index()
        if index is not None:
            return index

        # Create new if documents provided
        if documents:
            return self.create_index(documents)

        raise ValueError(
            "No existing index found and no documents provided.\n"
            "Either run indexing first or provide documents."
        )

    def add_documents(
        self,
        documents: list[Document],
        show_progress: bool = True,
    ) -> VectorStoreIndex:
        """Add documents to an existing index (incremental update).

        Unlike create_index, this does NOT delete the existing collection.
        New documents are added alongside existing ones.

        Args:
            documents: New documents to add (should be pre-chunked).
            show_progress: Show indexing progress bar.

        Returns:
            VectorStoreIndex for querying.

        Raises:
            ValueError: If no documents or embedding model not set.
        """
        if self.embed_model is None:
            raise ValueError(
                "Embedding model required for indexing. "
                "Pass embed_model to VectorStoreManager constructor."
            )

        if not documents:
            raise ValueError("No documents provided for indexing.")

        if not self.collection_exists():
            # No existing index, create new one
            logger.info("No existing index, creating new one")
            return self.create_index(documents, show_progress)

        logger.info(f"Adding {len(documents)} documents to existing index...")

        # Load existing index
        index = self.load_index()
        if index is None:
            return self.create_index(documents, show_progress)

        # Insert new documents into existing index
        for doc in documents:
            index.insert(doc, show_progress=show_progress)

        # Log stats
        info = self.get_collection_info()
        if info:
            logger.info(
                f"Index updated: now {info['vectors_count']} vectors"
            )

        return index

    # =========================================================================
    # Search Operations
    # =========================================================================

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        """Direct vector search (low-level API).

        For most use cases, use index.as_query_engine() instead.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score (0-1).

        Returns:
            List of search results with scores and metadata.
        """
        if not self.collection_exists():
            logger.warning("Collection does not exist")
            return []

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
            )

            return [
                {
                    "id": str(r.id),
                    "score": r.score,
                    "payload": r.payload,
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_stats(self) -> dict:
        """Get vector store statistics.

        Returns:
            Dictionary with storage stats.
        """
        info = self.get_collection_info()

        if info is None:
            return {
                "exists": False,
                "collection_name": self.collection_name,
                "mode": "cloud" if settings.vector_db == "qdrant_cloud" else "local",
            }

        return {
            "exists": True,
            "collection_name": self.collection_name,
            "vectors_count": info.get("vectors_count", 0),
            "points_count": info.get("points_count", 0),
            "vector_size": info.get("vector_size"),
            "status": info.get("status", "unknown"),
            "mode": "cloud" if settings.vector_db == "qdrant_cloud" else "local",
        }

    def health_check(self) -> bool:
        """Check if vector store is healthy and has indexed data.

        Returns:
            True if healthy AND collection exists with vectors, False otherwise.
        """
        try:
            # Check if we can connect
            self.client.get_collections()
            
            # Also check if our collection exists and has data
            if not self.collection_exists():
                logger.warning(f"Collection '{self.collection_name}' does not exist")
                return False
            
            # Check if collection has vectors
            info = self.get_collection_info()
            if info:
                vectors_count = info.get("vectors_count", 0)
                if vectors_count == 0:
                    logger.warning(f"Collection '{self.collection_name}' is empty")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False


