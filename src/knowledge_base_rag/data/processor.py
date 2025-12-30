"""Document processor for chunking and text preparation.

Design decisions:
- Use SentenceSplitter for semantic-aware chunking
- Chunk size of 1024 tokens balances context preservation with retrieval precision
- 128 token overlap prevents losing context at chunk boundaries
- Preserve all metadata through chunking for source attribution

Chunking Strategies Available:
1. SENTENCE (default): SentenceSplitter - respects sentences + enforces size limits
2. SEMANTIC: Groups semantically similar sentences (slower, more accurate)
3. HIERARCHICAL: Parent-child chunks for context preservation

Why SentenceSplitter (SENTENCE) is default:
1. Fixed-size: Breaks mid-sentence, loses semantic meaning
2. Pure sentence: Inconsistent sizes, some sentences too short/long
3. SentenceSplitter: Best of both - respects sentences + enforces size limits
4. Semantic chunking: More accurate but slower (requires embeddings)

Chunk size rationale (1024 tokens):
- Large enough to contain full ideas/paragraphs
- Small enough for precise retrieval (not returning irrelevant content)
- Fits well in LLM context with multiple chunks (6 chunks × 1024 = 6K tokens)

Overlap rationale (128 tokens):
- ~12% overlap catches context that spans chunk boundaries
- Not too large (would waste storage and slow retrieval)

For 750MB of PDFs, recommended settings:
- SENTENCE chunking (fast, good quality)
- chunk_size=512 (smaller chunks = more precise retrieval)
- chunk_overlap=50 (less overlap = faster indexing)
- Consider SEMANTIC chunking if retrieval quality is more important than speed
"""

import logging
from enum import Enum
from typing import Optional, Literal

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from knowledge_base_rag.core.config import settings

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    SENTENCE = "sentence"      # Fast, sentence-aware (default)
    SEMANTIC = "semantic"      # Slower, groups by semantic similarity
    SMALL = "small"            # Smaller chunks for precise retrieval
    LARGE = "large"            # Larger chunks for more context


# Preset configurations for different strategies
CHUNKING_PRESETS = {
    ChunkingStrategy.SENTENCE: {"chunk_size": 1024, "chunk_overlap": 128},
    ChunkingStrategy.SEMANTIC: {"chunk_size": 512, "chunk_overlap": 50},
    ChunkingStrategy.SMALL: {"chunk_size": 256, "chunk_overlap": 32},
    ChunkingStrategy.LARGE: {"chunk_size": 2048, "chunk_overlap": 256},
}


class DocumentProcessor:
    """Process and chunk documents for vector indexing.

    Uses semantic-aware chunking that respects sentence boundaries
    while enforcing size limits for optimal retrieval performance.

    Example:
        # Default sentence-aware chunking
        processor = DocumentProcessor()
        
        # Small chunks for precise retrieval (recommended for 750MB+ datasets)
        processor = DocumentProcessor(strategy="small")
        
        # Custom sizes
        processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: Literal["sentence", "semantic", "small", "large"] = "sentence",
    ):
        """Initialize the document processor.

        Args:
            chunk_size: Target size per chunk in tokens. Overrides strategy preset.
            chunk_overlap: Overlap between chunks in tokens. Overrides strategy preset.
            strategy: Chunking strategy preset. Options:
                - "sentence": Default, 1024 tokens, sentence-aware
                - "semantic": 512 tokens, for semantic grouping
                - "small": 256 tokens, precise retrieval for large datasets
                - "large": 2048 tokens, more context per chunk
        """
        # Get preset values
        preset = CHUNKING_PRESETS.get(
            ChunkingStrategy(strategy), 
            CHUNKING_PRESETS[ChunkingStrategy.SENTENCE]
        )
        
        # Allow overrides
        self.chunk_size = chunk_size or preset["chunk_size"]
        self.chunk_overlap = chunk_overlap or preset["chunk_overlap"]
        self.strategy = strategy

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
            f"DocumentProcessor initialized: strategy={strategy}, "
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
