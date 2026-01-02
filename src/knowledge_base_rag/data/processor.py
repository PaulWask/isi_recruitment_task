"""Document processor for chunking and text preparation.

Design decisions:
- Use SentenceSplitter for semantic-aware chunking
- All chunking settings are centralized in config.py (Single Source of Truth)
- Preserve all metadata through chunking for source attribution

Chunking Strategies Available:
1. SENTENCE (default): SentenceSplitter - respects sentences + enforces size limits
2. SEMANTIC: Groups semantically similar sentences (slower, more accurate)
3. SMALL: Smaller chunks for precise retrieval (recommended for financial tables)
4. LARGE: Larger chunks for more context

Why SentenceSplitter (SENTENCE) is default:
1. Fixed-size: Breaks mid-sentence, loses semantic meaning
2. Pure sentence: Inconsistent sizes, some sentences too short/long
3. SentenceSplitter: Best of both - respects sentences + enforces size limits
4. Semantic chunking: More accurate but slower (requires embeddings)

All chunk sizes are configurable via .env file or environment variables.
See config.py for the single source of truth for all settings.
"""

import logging
from typing import Optional, Literal

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# Import from centralized config (Single Source of Truth)
from knowledge_base_rag.core.config import settings, ChunkingStrategy

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk documents for vector indexing.

    Uses semantic-aware chunking that respects sentence boundaries
    while enforcing size limits for optimal retrieval performance.
    
    All chunking settings are loaded from config.py (Single Source of Truth).
    Settings can be overridden via .env file or environment variables.

    Example:
        # Default strategy from config (settings.default_chunking_strategy)
        processor = DocumentProcessor()
        
        # Small chunks for precise retrieval (recommended for financial tables)
        processor = DocumentProcessor(strategy="small")
        
        # Custom sizes (overrides strategy preset)
        processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: Literal["sentence", "semantic", "small", "large"] | None = None,
    ):
        """Initialize the document processor.

        Args:
            chunk_size: Target size per chunk in tokens. Overrides strategy preset.
            chunk_overlap: Overlap between chunks in tokens. Overrides strategy preset.
            strategy: Chunking strategy preset. If None, uses settings.default_chunking_strategy.
                Options:
                - "sentence": General text (chunk_size from CHUNK_SIZE_SENTENCE env var)
                - "semantic": Semantic grouping (chunk_size from CHUNK_SIZE_SEMANTIC env var)
                - "small": Financial tables (chunk_size from CHUNK_SIZE_SMALL env var)
                - "large": Dense context (chunk_size from CHUNK_SIZE_LARGE env var)
        """
        # Use default strategy from config if not specified
        self.strategy = strategy or settings.default_chunking_strategy
        
        # Get preset values from centralized config (Single Source of Truth)
        preset = settings.get_chunking_preset(self.strategy)
        
        # Allow explicit overrides (for programmatic use)
        self.chunk_size = chunk_size if chunk_size is not None else preset["chunk_size"]
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else preset["chunk_overlap"]

        # SentenceSplitter respects sentence boundaries when possible
        # Falls back to character splitting only when necessary
        self._splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator="\n\n",
            # Secondary regex for sentence-like boundaries
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
        )

        logger.info(
            f"DocumentProcessor initialized: strategy={self.strategy}, "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

    def process_documents(
        self,
        documents: list[Document],
        show_progress: bool = True,
    ) -> list[Document]:
        """Chunk documents for vector indexing.

        Args:
            documents: List of documents to process.
            show_progress: Show progress during processing.

        Returns:
            List of chunked documents with preserved metadata.
        """
        if not documents:
            logger.warning("No documents to process")
            return []

        logger.info(f"Processing {len(documents)} documents...")

        # Get nodes (chunks) from documents
        nodes = self._splitter.get_nodes_from_documents(
            documents,
            show_progress=show_progress,
        )

        # Convert nodes back to Documents with enriched metadata
        chunks: list[Document] = []
        for i, node in enumerate(nodes):
            # Build metadata dict
            meta = node.metadata.copy() if node.metadata else {}
            meta["chunk_index"] = i
            meta["chunk_size"] = len(node.get_content())
            
            # Create document from node content
            chunk = Document(
                text=node.get_content(),
                extra_info=meta,
                doc_id=node.node_id,
            )

            chunks.append(chunk)

        # Log statistics
        stats = self.get_chunk_stats(chunks)
        logger.info(
            f"Created {stats['total_chunks']} chunks "
            f"(avg size: {stats['avg_size']:.0f} chars)"
        )

        return chunks

    def get_chunk_stats(self, chunks: list[Document]) -> dict:
        """Get statistics about processed chunks.

        Args:
            chunks: List of chunked documents.

        Returns:
            Dictionary with chunk statistics.
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_size": 0,
                "min_size": 0,
                "max_size": 0,
                "total_chars": 0,
            }

        sizes = [len(chunk.get_content()) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_size": sum(sizes) / len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "total_chars": sum(sizes),
        }

    def estimate_chunks(self, documents: list[Document]) -> int:
        """Estimate number of chunks that will be created.

        Useful for progress estimation before actual processing.

        Args:
            documents: Documents to estimate.

        Returns:
            Estimated number of chunks.
        """
        total_chars = sum(len(doc.get_content()) for doc in documents)
        # Rough estimate: ~4 chars per token, account for overlap
        effective_chunk_size = self.chunk_size * 4 * 0.9  # 90% due to overlap
        return max(1, int(total_chars / effective_chunk_size))
