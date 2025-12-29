"""Document processor for chunking and text preparation.

Design decisions:
- Use SentenceSplitter for semantic-aware chunking
- Chunk size of 1024 tokens balances context preservation with retrieval precision
- 128 token overlap prevents losing context at chunk boundaries
- Preserve all metadata through chunking for source attribution

Why SentenceSplitter over alternatives:
1. Fixed-size: Breaks mid-sentence, loses semantic meaning
2. Pure sentence: Inconsistent sizes, some sentences too short/long
3. SentenceSplitter: Best of both - respects sentences + enforces size limits
4. Semantic chunking: Too slow (requires embeddings for each split decision)

Chunk size rationale (1024 tokens):
- Large enough to contain full ideas/paragraphs
- Small enough for precise retrieval (not returning irrelevant content)
- Fits well in LLM context with multiple chunks (6 chunks × 1024 = 6K tokens)

Overlap rationale (128 tokens):
- ~12% overlap catches context that spans chunk boundaries
- Not too large (would waste storage and slow retrieval)
"""

import logging
from typing import Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from knowledge_base_rag.core.config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk documents for vector indexing.

    Uses semantic-aware chunking that respects sentence boundaries
    while enforcing size limits for optimal retrieval performance.

    Example:
        processor = DocumentProcessor(chunk_size=1024, chunk_overlap=128)
        chunks = processor.process_documents(documents)
        print(f"Created {len(chunks)} chunks")
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """Initialize the document processor.

        Args:
            chunk_size: Target size per chunk in tokens. Default: 1024
            chunk_overlap: Overlap between chunks in tokens. Default: 128
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

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
            f"DocumentProcessor initialized: "
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
            # Create document from node content
            chunk = Document(
                text=node.get_content(),
                metadata=node.metadata.copy() if node.metadata else {},
                doc_id=node.node_id,
            )

            # Add chunk-specific metadata
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(node.get_content())

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
