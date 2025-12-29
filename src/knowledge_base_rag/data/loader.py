"""Document loader for various file types.

Design decisions:
- Use LlamaIndex SimpleDirectoryReader for robust multi-format support
- Supports PDF, DOCX, TXT, MD, HTML, PPTX, XLSX out of the box
- Add metadata (filename, path, type) for source attribution in answers
- Handle large directories (750MB) with progress tracking

Why LlamaIndex SimpleDirectoryReader:
- Handles 15+ file formats automatically
- Extracts text from PDFs, Word docs, etc. without complex setup
- Provides consistent Document objects with metadata
- Built-in progress tracking for large directories
"""

import logging
from pathlib import Path
from typing import Optional

from llama_index.core import Document, SimpleDirectoryReader

from knowledge_base_rag.core.config import settings

logger = logging.getLogger(__name__)

# Supported file extensions with descriptions
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "PDF documents",
    ".docx": "Word documents",
    ".doc": "Legacy Word documents",
    ".txt": "Plain text files",
    ".md": "Markdown files",
    ".html": "HTML files",
    ".htm": "HTML files",
    ".csv": "CSV files",
    ".json": "JSON files",
    ".xml": "XML files",
    ".pptx": "PowerPoint presentations",
    ".ppt": "Legacy PowerPoint",
    ".xlsx": "Excel spreadsheets",
    ".xls": "Legacy Excel files",
    ".epub": "EPUB books",
    ".rtf": "Rich Text Format",
}


class DocumentLoader:
    """Load documents from the knowledge base directory.

    Handles multiple file formats and extracts metadata for source attribution.
    Designed for large document collections (750MB+).

    Example:
        loader = DocumentLoader()
        print(loader.count_documents())  # {'pdf': 100, '.txt': 50}
        documents = loader.load_documents()
    """

    def __init__(
        self,
        knowledge_base_dir: Optional[Path] = None,
        recursive: bool = True,
    ):
        """Initialize the document loader.

        Args:
            knowledge_base_dir: Directory containing documents. Defaults to config.
            recursive: Search subdirectories. Defaults to True.
        """
        self.knowledge_base_dir = Path(
            knowledge_base_dir or settings.knowledge_base_dir
        )
        self.recursive = recursive

    def count_documents(self) -> dict[str, int]:
        """Count documents by file type.

        Returns:
            Dictionary mapping extensions to counts.
            Example: {'.pdf': 100, '.txt': 50, '.docx': 25}
        """
        if not self.knowledge_base_dir.exists():
            logger.warning(f"Directory not found: {self.knowledge_base_dir}")
            return {}

        counts: dict[str, int] = {}
        pattern = "**/*" if self.recursive else "*"

        for file_path in self.knowledge_base_dir.glob(pattern):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    counts[ext] = counts.get(ext, 0) + 1

        return counts

    def get_total_size_mb(self) -> float:
        """Get total size of supported documents in MB.

        Returns:
            Total size in megabytes.
        """
        if not self.knowledge_base_dir.exists():
            return 0.0

        total_bytes = 0
        pattern = "**/*" if self.recursive else "*"

        for file_path in self.knowledge_base_dir.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                total_bytes += file_path.stat().st_size

        return total_bytes / (1024 * 1024)

    def load_documents(
        self,
        show_progress: bool = True,
    ) -> list[Document]:
        """Load all documents from the knowledge base.

        Args:
            show_progress: Show loading progress bar.

        Returns:
            List of Document objects with metadata.

        Raises:
            FileNotFoundError: If knowledge base directory doesn't exist.
        """
        if not self.knowledge_base_dir.exists():
            raise FileNotFoundError(
                f"Knowledge base directory not found: {self.knowledge_base_dir}\n"
                f"Run: uv run python scripts/download_data.py"
            )

        logger.info(f"Loading documents from: {self.knowledge_base_dir}")

        # Count files first
        doc_counts = self.count_documents()
        total_files = sum(doc_counts.values())

        if total_files == 0:
            logger.warning("No supported documents found in knowledge base")
            return []

        logger.info(f"Found {total_files} documents ({self.get_total_size_mb():.1f} MB):")
        for ext, count in sorted(doc_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {ext}: {count} files")

        # Load using LlamaIndex SimpleDirectoryReader
        try:
            reader = SimpleDirectoryReader(
                input_dir=str(self.knowledge_base_dir),
                recursive=self.recursive,
                required_exts=list(SUPPORTED_EXTENSIONS.keys()),
                filename_as_id=True,
            )

            documents = reader.load_data(show_progress=show_progress)

            # Enrich metadata for source attribution
            for doc in documents:
                self._enrich_metadata(doc)

            logger.info(f"Loaded {len(documents)} document segments")
            return documents

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

    def _enrich_metadata(self, doc: Document) -> None:
        """Add metadata for source attribution in answers.

        Args:
            doc: Document to enrich with metadata.
        """
        if doc.metadata is None:
            doc.metadata = {}

        # Extract file info
        file_path = doc.metadata.get("file_path", doc.doc_id or "")
        if file_path:
            path = Path(file_path)
            doc.metadata["file_name"] = path.name
            doc.metadata["file_type"] = path.suffix.lower()

            # Relative path for cleaner display in UI
            try:
                rel_path = path.relative_to(self.knowledge_base_dir)
                doc.metadata["source"] = str(rel_path)
            except ValueError:
                doc.metadata["source"] = path.name

        # Mark as from knowledge base
        doc.metadata["source_type"] = "knowledge_base"

    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of extensions like ['.pdf', '.docx', '.txt']
        """
        return list(SUPPORTED_EXTENSIONS.keys())

    def get_file_list(self) -> list[Path]:
        """Get list of all supported files in the knowledge base.

        Returns:
            List of file paths.
        """
        if not self.knowledge_base_dir.exists():
            return []

        files = []
        pattern = "**/*" if self.recursive else "*"

        for file_path in self.knowledge_base_dir.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(file_path)

        return sorted(files)

    def load_file(self, file_path: Path) -> Optional[Document]:
        """Load a single file and return as Document.

        Args:
            file_path: Path to the file to load.

        Returns:
            Document object with metadata, or None if loading fails.
        """
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file type: {ext}")
            return None

        try:
            # Use SimpleDirectoryReader for single file
            reader = SimpleDirectoryReader(
                input_files=[str(file_path)],
                filename_as_id=True,
            )
            docs = reader.load_data(show_progress=False)

            if docs:
                doc = docs[0]
                self._enrich_metadata(doc)
                self._stats["successful"] += 1
                
                # Track by extension
                self._stats["by_extension"][ext] = self._stats["by_extension"].get(ext, 0) + 1
                
                return doc

        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            self._stats["failed"] += 1

        return None

    def load_all(self, show_progress: bool = True) -> list[Document]:
        """Load all documents (alias for load_documents with stats tracking).

        Args:
            show_progress: Show loading progress bar.

        Returns:
            List of Document objects.
        """
        self._reset_stats()
        docs = self.load_documents(show_progress=show_progress)
        
        # Update stats
        self._stats["successful"] = len(docs)
        for doc in docs:
            ext = doc.metadata.get("file_type", "")
            if ext:
                self._stats["by_extension"][ext] = self._stats["by_extension"].get(ext, 0) + 1
        
        return docs

    def _reset_stats(self) -> None:
        """Reset loading statistics."""
        self._stats = {
            "successful": 0,
            "failed": 0,
            "by_extension": {},
        }

    def get_stats(self) -> dict:
        """Get loading statistics from the last load operation.

        Returns:
            Dictionary with 'successful', 'failed', and 'by_extension' counts.
        """
        if not hasattr(self, "_stats"):
            self._reset_stats()
        return self._stats
