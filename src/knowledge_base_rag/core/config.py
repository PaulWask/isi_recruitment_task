"""Configuration settings for the RAG system.

Design decisions:
- pydantic-settings for type-safe configuration with .env file support
- Support both local (free) and cloud (free tier) services
- Optimized defaults for 750MB knowledge base
- Single source of truth: all settings in one place, used by all modules
"""

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class FreeTierService(str, Enum):
    """Available free-tier services."""

    LOCAL = "local"
    GROQ = "groq"  # Free: 14,400 requests/day
    QDRANT_CLOUD = "qdrant_cloud"  # Free: 1GB storage


class ChunkingStrategy(str, Enum):
    """Available chunking strategies for document processing."""
    SENTENCE = "sentence"   # Fast, sentence-aware (default)
    SEMANTIC = "semantic"   # Slower, groups by semantic similarity
    SMALL = "small"         # Smaller chunks for precise retrieval (financial tables)
    LARGE = "large"         # Larger chunks for more context


class RAGSettings(BaseSettings):
    """RAG system configuration with .env file support.
    
    All settings can be overridden via environment variables or .env file.
    Environment variable names are uppercase versions of the field names.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # =========================================================================
    # PATHS
    # =========================================================================
    knowledge_base_dir: Path = Path("./domaindata")
    storage_dir: Path = Path("./storage")
    qdrant_path: Path = Path("./qdrant_db")

    def get_absolute_qdrant_path(self) -> Path:
        """Get absolute path to qdrant_db, resolving relative paths."""
        if self.qdrant_path.is_absolute():
            return self.qdrant_path
        return self.qdrant_path.resolve()

    def get_absolute_knowledge_base_dir(self) -> Path:
        """Get absolute path to knowledge base directory."""
        if self.knowledge_base_dir.is_absolute():
            return self.knowledge_base_dir
        return self.knowledge_base_dir.resolve()

    # =========================================================================
    # VECTOR DATABASE
    # =========================================================================
    vector_db: Literal["local", "qdrant_cloud"] = "local"
    qdrant_cloud_url: str = ""
    qdrant_cloud_api_key: str = ""
    collection_name: str = "knowledge_base"

    # =========================================================================
    # LLM SERVICE
    # =========================================================================
    llm_service: Literal["local", "groq"] = "local"
    groq_api_key: str = ""
    llm_model: str = "llama3.2:3b"  # Ollama model
    groq_model: str = "llama-3.1-70b-versatile"  # Groq model
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: float = 120.0

    # =========================================================================
    # CHUNKING CONFIGURATION (Single Source of Truth)
    # =========================================================================
    # Default strategy - can be overridden with --strategy flag or env var
    default_chunking_strategy: str = "small"  # "sentence", "semantic", "small", "large"
    
    # Strategy presets (all configurable via env vars)
    # SENTENCE strategy: general text, narratives
    chunk_size_sentence: int = 1024
    chunk_overlap_sentence: int = 128
    
    # SEMANTIC strategy: medium granularity
    chunk_size_semantic: int = 512
    chunk_overlap_semantic: int = 50
    
    # SMALL strategy: financial tables, structured data (RECOMMENDED)
    chunk_size_small: int = 256
    chunk_overlap_small: int = 32
    
    # LARGE strategy: dense context needed
    chunk_size_large: int = 2048
    chunk_overlap_large: int = 256

    def get_chunking_preset(self, strategy: str | None = None) -> dict[str, int]:
        """Get chunk_size and chunk_overlap for a given strategy.
        
        Args:
            strategy: One of "sentence", "semantic", "small", "large".
                     If None, uses default_chunking_strategy.
        
        Returns:
            dict with "chunk_size" and "chunk_overlap" keys.
        """
        strategy = strategy or self.default_chunking_strategy
        
        presets = {
            "sentence": {
                "chunk_size": self.chunk_size_sentence,
                "chunk_overlap": self.chunk_overlap_sentence,
            },
            "semantic": {
                "chunk_size": self.chunk_size_semantic,
                "chunk_overlap": self.chunk_overlap_semantic,
            },
            "small": {
                "chunk_size": self.chunk_size_small,
                "chunk_overlap": self.chunk_overlap_small,
            },
            "large": {
                "chunk_size": self.chunk_size_large,
                "chunk_overlap": self.chunk_overlap_large,
            },
        }
        
        return presets.get(strategy, presets["small"])

    # =========================================================================
    # RETRIEVAL SETTINGS
    # =========================================================================
    # With 256-token chunks, we need more chunks to get equivalent context
    similarity_top_k: int = 10  # 10 x 256 = ~2560 tokens context
    similarity_cutoff: float = 0.7  # Filters low-relevance noise

    # =========================================================================
    # EMBEDDING MODEL
    # =========================================================================
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # =========================================================================
    # AWS S3 SETTINGS (for data download)
    # =========================================================================
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "eu-central-1"
    s3_bucket: str = ""
    s3_domaindata_key: str = "domaindata.zip"

    # =========================================================================
    # PROCESSING
    # =========================================================================
    batch_size: int = 100
    show_progress: bool = True

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================
    def validate_llm_config(self) -> None:
        """Validate LLM configuration."""
        if self.llm_service == "groq" and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY required when using Groq")

    def validate_vector_db_config(self) -> None:
        """Validate vector DB configuration."""
        if self.vector_db == "qdrant_cloud":
            if not self.qdrant_cloud_url:
                raise ValueError("QDRANT_CLOUD_URL required for Qdrant Cloud")
            if not self.qdrant_cloud_api_key:
                raise ValueError("QDRANT_CLOUD_API_KEY required for Qdrant Cloud")

    def validate_s3_config(self) -> None:
        """Validate S3 configuration."""
        if not self.s3_bucket:
            raise ValueError("S3_BUCKET required for data download")

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)


# Global settings instance - import this in other modules
settings = RAGSettings()
