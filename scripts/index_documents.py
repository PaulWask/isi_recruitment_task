#!/usr/bin/env python3
"""Index documents into the vector store.

This script processes all documents in the knowledge base directory,
chunks them, generates embeddings, and stores them in Qdrant.

Usage:
    uv run python scripts/index_documents.py            # First time indexing
    uv run python scripts/index_documents.py --update   # Add new documents only
    uv run python scripts/index_documents.py --force    # Rebuild from scratch
    uv run python scripts/index_documents.py --dry-run  # Preview only

Or via CLI command:
    uv run kb-index
    uv run kb-index --update
    uv run kb-index --force
"""

# =============================================================================
# NLTK DATA SETUP - MUST BE FIRST, BEFORE ANY LLAMAINDEX IMPORTS!
# LlamaIndex checks NLTK_DATA env var when imported.
# =============================================================================
import os
import sys
from pathlib import Path

# Configure NLTK before anything else
_nltk_data_dir = Path(__file__).parent.parent / ".nltk_data"
_nltk_data_dir.mkdir(exist_ok=True)
os.environ["NLTK_DATA"] = str(_nltk_data_dir.resolve())

# =============================================================================
# STANDARD IMPORTS
# =============================================================================
import argparse
import logging
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_base_rag.core.config import settings
from knowledge_base_rag.data.loader import DocumentLoader
from knowledge_base_rag.data.processor import DocumentProcessor
from knowledge_base_rag.storage.embeddings import get_embed_model
from knowledge_base_rag.storage.vector_store import VectorStoreManager


def ensure_nltk_data():
    """Pre-download NLTK data to avoid runtime downloads.
    
    Downloads all NLTK packages required by LlamaIndex:
    - punkt_tab: Tokenizer for sentence splitting
    - averaged_perceptron_tagger: POS tagging (sometimes needed)
    
    Data is stored in .nltk_data/ in the project root for portability.
    """
    import nltk
    
    # Ensure NLTK knows about our data directory
    nltk_data_dir = _nltk_data_dir
    if str(nltk_data_dir.resolve()) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_data_dir.resolve()))
    
    # Packages required by LlamaIndex for text processing
    required_packages = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    
    print(f"📁 NLTK data directory: {nltk_data_dir.resolve()}")
    
    for data_path, package_name in required_packages:
        try:
            nltk.data.find(data_path)
            print(f"  ✅ {package_name}: already downloaded")
        except LookupError:
            print(f"  📦 {package_name}: downloading...")
            try:
                nltk.download(package_name, download_dir=str(nltk_data_dir.resolve()), quiet=True)
                print(f"  ✅ {package_name}: downloaded successfully")
            except Exception as e:
                print(f"  ⚠️ {package_name}: download failed ({e}), will retry at runtime")


# Configure logging with professional timestamp format (HH:MM:SS.mmm)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_size(bytes_size: float) -> str:
    """Format bytes into human-readable size."""
    size = float(bytes_size)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def print_banner(mode: str = "full"):
    """Print the indexing banner."""
    print()
    print("=" * 60)
    if mode == "update":
        print("📚 Knowledge Base Incremental Update")
    else:
        print("📚 Knowledge Base Indexing Pipeline")
    print("=" * 60)
    print()


def print_config():
    """Print current configuration."""
    print("📋 Configuration:")
    print(f"   • Knowledge base: {settings.knowledge_base_dir}")
    print(f"   • Collection: {settings.collection_name}")
    print(f"   • Chunk size: {settings.chunk_size} chars")
    print(f"   • Chunk overlap: {settings.chunk_overlap} chars")
    print(f"   • Embedding model: {settings.embed_model}")
    print(f"   • Vector DB: {settings.vector_db}")
    print()


def check_knowledge_base(kb_path: Path) -> tuple[bool, int, int]:
    """Check if knowledge base exists and count files.
    
    Returns:
        Tuple of (exists, file_count, total_size_bytes)
    """
    if not kb_path.exists():
        return False, 0, 0
    
    file_count = 0
    total_size = 0
    
    for file_path in kb_path.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith("."):
            file_count += 1
            total_size += file_path.stat().st_size
    
    return True, file_count, total_size


def get_indexed_files(vector_manager: VectorStoreManager) -> set[str]:
    """Get set of file paths already indexed in the vector store.
    
    Args:
        vector_manager: Vector store manager instance.
        
    Returns:
        Set of source file paths that are already indexed.
    """
    indexed_files: set[str] = set()
    
    if not vector_manager.collection_exists():
        return indexed_files
    
    try:
        # Scroll through all points to get unique sources
        # This uses Qdrant's scroll API for efficiency
        client = vector_manager.client
        collection_name = vector_manager.collection_name
        
        offset = None
        batch_size = 100
        
        while True:
            results = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            
            points, next_offset = results
            
            for point in points:
                if point.payload:
                    # Try different metadata keys for source
                    source = (
                        point.payload.get("source") or 
                        point.payload.get("file_path") or
                        point.payload.get("file_name") or
                        ""
                    )
                    if source:
                        indexed_files.add(source)
            
            if next_offset is None:
                break
            offset = next_offset
        
        logger.info(f"Found {len(indexed_files)} unique files in index")
        
    except Exception as e:
        logger.warning(f"Could not retrieve indexed files: {e}")
    
    return indexed_files


def get_all_files(kb_path: Path) -> dict[str, Path]:
    """Get all files in knowledge base with their relative paths.
    
    Args:
        kb_path: Path to knowledge base directory.
        
    Returns:
        Dictionary mapping relative path strings to full Path objects.
    """
    files = {}
    for file_path in kb_path.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith("."):
            # Use relative path as key for comparison
            rel_path = str(file_path.relative_to(kb_path))
            files[rel_path] = file_path
    return files


def run_update(
    dry_run: bool = False,
    batch_size: int = 50,
) -> bool:
    """Run incremental update - add only new documents.
    
    Args:
        dry_run: Preview only, don't actually index.
        batch_size: Number of documents to process at once.
        
    Returns:
        True if successful, False otherwise.
    """
    start_time = time.time()
    
    print_banner(mode="update")
    print_config()
    
    # Check knowledge base
    kb_path = Path(settings.knowledge_base_dir)
    exists, file_count, total_size = check_knowledge_base(kb_path)
    
    if not exists:
        print(f"❌ Knowledge base not found: {kb_path}")
        print()
        print("   To download the knowledge base, run:")
        print("   uv run python scripts/download_data.py")
        print()
        return False
    
    print(f"📁 Found {file_count} files ({format_size(total_size)})")
    print()
    
    # Initialize components
    print("🔧 Initializing components...")
    
    try:
        # Load embedding model
        print("   • Loading embedding model...")
        embed_model = get_embed_model()
        print(f"     ✅ {embed_model.model_name}")
        
        # Initialize vector store manager
        print("   • Connecting to vector store...")
        vector_manager = VectorStoreManager(
            collection_name=settings.collection_name,
            embed_model=embed_model,
        )
        
        # Check if collection exists
        if not vector_manager.collection_exists():
            print("     ⚠️  No existing index found")
            print()
            print("   Run without --update for first-time indexing:")
            print("   uv run python scripts/index_documents.py")
            print()
            return False
        
        stats = vector_manager.get_stats()
        current_vectors = stats.get("vectors_count", 0)
        print(f"     ✅ Current index: {current_vectors} vectors")
        
        print()
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        logger.exception("Initialization error")
        return False
    
    # Find new files
    print("🔍 Finding new documents...")
    
    try:
        # Get already indexed files
        indexed_files = get_indexed_files(vector_manager)
        print(f"   • Already indexed: {len(indexed_files)} files")
        
        # Get all files in knowledge base
        all_files = get_all_files(kb_path)
        print(f"   • Total in knowledge base: {len(all_files)} files")
        
        # Find new files (not in index)
        new_file_paths = []
        for rel_path, full_path in all_files.items():
            # Check various formats the source might be stored as
            if (rel_path not in indexed_files and 
                str(full_path) not in indexed_files and
                full_path.name not in indexed_files):
                new_file_paths.append(full_path)
        
        print(f"   • New files to index: {len(new_file_paths)}")
        print()
        
        if not new_file_paths:
            print("✅ Index is up to date - no new documents found")
            print()
            return True
        
        # Show some new files
        print("   📄 New files:")
        for path in new_file_paths[:5]:
            print(f"      • {path.name}")
        if len(new_file_paths) > 5:
            print(f"      ... and {len(new_file_paths) - 5} more")
        print()
        
    except Exception as e:
        print(f"❌ Failed to find new documents: {e}")
        logger.exception("Error finding new documents")
        return False
    
    if dry_run:
        print("🔍 DRY RUN - No changes will be made")
        print(f"   Would add {len(new_file_paths)} new documents")
        print()
        return True
    
    # Step 1: Load new documents only
    print("📄 Step 1/3: Loading new documents...")
    load_start = time.time()
    
    try:
        loader = DocumentLoader(kb_path)
        # Load only the new files
        documents = []
        for file_path in new_file_paths:
            try:
                doc = loader.load_file(file_path)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        load_time = time.time() - load_start
        print(f"   ✅ Loaded {len(documents)} new documents in {format_time(load_time)}")
        print()
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        logger.exception("Loading error")
        return False
    
    if not documents:
        print("❌ No new documents were loaded")
        return False
    
    # Step 2: Process/chunk documents
    print("✂️  Step 2/3: Chunking new documents...")
    chunk_start = time.time()
    
    try:
        processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = processor.process_documents(documents, show_progress=True)
        
        chunk_time = time.time() - chunk_start
        print(f"   ✅ Created {len(chunks)} chunks in {format_time(chunk_time)}")
        print()
        
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        logger.exception("Chunking error")
        return False
    
    # Step 3: Add to existing index
    print("🔨 Step 3/3: Adding to vector index...")
    print(f"   Embedding {len(chunks)} chunks...")
    index_start = time.time()
    
    try:
        # Add documents to existing index (don't delete)
        index = vector_manager.add_documents(chunks, show_progress=True)
        
        index_time = time.time() - index_start
        print(f"   ✅ Documents added in {format_time(index_time)}")
        
        # Show final stats
        final_stats = vector_manager.get_stats()
        new_vectors = final_stats.get("vectors_count", 0)
        added_vectors = new_vectors - current_vectors
        print(f"   • Vectors added: {added_vectors}")
        print(f"   • Total vectors: {new_vectors}")
        print()
        
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        logger.exception("Indexing error")
        return False
    
    # Summary
    total_time = time.time() - start_time
    print("=" * 60)
    print("✅ Update Complete!")
    print("=" * 60)
    print()
    print(f"   📊 Summary:")
    print(f"   • New documents added: {len(documents)}")
    print(f"   • New chunks created: {len(chunks)}")
    print(f"   • Total vectors now: {new_vectors}")
    print(f"   • Time: {format_time(total_time)}")
    print()
    
    return True


def run_indexing(
    force: bool = False,
    dry_run: bool = False,
    batch_size: int = 50,
) -> bool:
    """Run the full indexing pipeline.
    
    Args:
        force: Force rebuild even if index exists.
        dry_run: Preview only, don't actually index.
        batch_size: Number of documents to process at once.
        
    Returns:
        True if successful, False otherwise.
    """
    start_time = time.time()
    
    print_banner()
    print_config()
    
    # Check knowledge base
    kb_path = Path(settings.knowledge_base_dir)
    exists, file_count, total_size = check_knowledge_base(kb_path)
    
    if not exists:
        print(f"❌ Knowledge base not found: {kb_path}")
        print()
        print("   To download the knowledge base, run:")
        print("   uv run python scripts/download_data.py")
        print()
        return False
    
    print(f"📁 Found {file_count} files ({format_size(total_size)})")
    print()
    
    if dry_run:
        print("🔍 DRY RUN - No changes will be made")
        print()
    
    # Initialize components
    print("🔧 Initializing components...")
    
    try:
        # Load embedding model (this downloads if needed)
        print("   • Loading embedding model...")
        embed_model = get_embed_model()
        print(f"     ✅ {embed_model.model_name}")
        
        # Initialize vector store manager
        print("   • Connecting to vector store...")
        vector_manager = VectorStoreManager(
            collection_name=settings.collection_name,
            embed_model=embed_model,
        )
        
        # Check if collection exists
        collection_exists = vector_manager.collection_exists()
        if collection_exists:
            stats = vector_manager.get_stats()
            vectors = stats.get("vectors_count", 0)
            print(f"     ⚠️  Collection exists with {vectors} vectors")
            
            if not force and not dry_run:
                print()
                print("   Collection already exists. Options:")
                print("   • Use --force to rebuild from scratch")
                print("   • Use --update to add new documents only")
                print()
                return True
        else:
            print("     ✅ Vector store ready (new collection)")
        
        print()
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        logger.exception("Initialization error")
        return False
    
    if dry_run:
        print("✅ Dry run complete - all checks passed")
        print()
        return True
    
    # Step 1: Load documents
    print("📄 Step 1/3: Loading documents...")
    load_start = time.time()
    
    try:
        loader = DocumentLoader(kb_path)
        documents = loader.load_all(show_progress=True)
        
        load_time = time.time() - load_start
        print(f"   ✅ Loaded {len(documents)} documents in {format_time(load_time)}")
        
        # Show stats
        stats = loader.get_stats()
        print(f"   • Successful: {stats['successful']}")
        if stats['failed'] > 0:
            print(f"   • Failed: {stats['failed']}")
        
        # Show file type breakdown
        if stats.get('by_extension'):
            print("   • By type:", end=" ")
            types = [f"{ext}: {count}" for ext, count in 
                     sorted(stats['by_extension'].items(), key=lambda x: -x[1])[:5]]
            print(", ".join(types))
        
        print()
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        logger.exception("Loading error")
        return False
    
    if not documents:
        print("❌ No documents were loaded")
        return False
    
    # Step 2: Process/chunk documents
    print("✂️  Step 2/3: Chunking documents...")
    chunk_start = time.time()
    
    try:
        processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = processor.process_documents(documents, show_progress=True)
        
        chunk_time = time.time() - chunk_start
        print(f"   ✅ Created {len(chunks)} chunks in {format_time(chunk_time)}")
        
        # Show chunk stats
        chunk_stats = processor.get_chunk_stats(chunks)
        print(f"   • Avg chunk size: {chunk_stats['avg_size']:.0f} chars")
        print(f"   • Min/Max: {chunk_stats['min_size']}/{chunk_stats['max_size']} chars")
        print()
        
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        logger.exception("Chunking error")
        return False
    
    # Step 3: Create index (embed and store)
    print("🔨 Step 3/3: Creating vector index...")
    print(f"   Embedding {len(chunks)} chunks (this may take a while)...")
    index_start = time.time()
    
    try:
        index = vector_manager.create_index(
            documents=chunks,
            show_progress=True,
        )
        
        index_time = time.time() - index_start
        print(f"   ✅ Index created in {format_time(index_time)}")
        
        # Show final stats
        final_stats = vector_manager.get_stats()
        print(f"   • Vectors stored: {final_stats.get('vectors_count', 0)}")
        print(f"   • Vector dimension: {final_stats.get('vector_size', 'N/A')}")
        print()
        
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        logger.exception("Indexing error")
        return False
    
    # Summary
    total_time = time.time() - start_time
    print("=" * 60)
    print("✅ Indexing Complete!")
    print("=" * 60)
    print()
    print(f"   📊 Summary:")
    print(f"   • Documents processed: {len(documents)}")
    print(f"   • Chunks created: {len(chunks)}")
    print(f"   • Vectors stored: {final_stats.get('vectors_count', 0)}")
    print(f"   • Total time: {format_time(total_time)}")
    print()
    print("   🚀 Ready to query! Run:")
    print("   uv run streamlit run src/knowledge_base_rag/app.py")
    print()
    
    return True


def main():
    """Main entry point."""
    # Pre-download NLTK data (one-time, speeds up subsequent app starts)
    ensure_nltk_data()
    
    parser = argparse.ArgumentParser(
        description="Index documents into the vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/index_documents.py            # First time indexing
  python scripts/index_documents.py --update   # Add new documents only
  python scripts/index_documents.py --force    # Force complete rebuild
  python scripts/index_documents.py --dry-run  # Preview only
        """,
    )
    
    parser.add_argument(
        "--update", "-u",
        action="store_true",
        help="Incremental update - add only new documents",
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force rebuild of the index (delete existing)",
    )
    
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview only, don't actually index",
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.update and args.force:
        print("Error: Cannot use --update and --force together")
        print("  --update: Add new documents to existing index")
        print("  --force:  Delete and rebuild entire index")
        sys.exit(1)
    
    try:
        if args.update:
            success = run_update(
                dry_run=args.dry_run,
                batch_size=args.batch_size,
            )
        else:
            success = run_indexing(
                force=args.force,
                dry_run=args.dry_run,
                batch_size=args.batch_size,
            )
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Indexing interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()
