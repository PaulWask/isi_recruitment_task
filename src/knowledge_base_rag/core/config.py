"""Configuration settings for the RAG system.

Design decisions:
- pydantic-settings for type-safe configuration with .env file support
- Support both local (free) and cloud (free tier) services
- Optimized defaults for 750MB knowledge base
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


class RAGSettings(BaseSettings):
    """RAG system configuration with .env file support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Paths (relative to project root or absolute)
    knowledge_base_dir: Path = Path("./domaindata")
    storage_dir: Path = Path("./storage")
    qdrant_path: Path = Path("./qdrant_db")

    def get_absolute_qdrant_path(self) -> Path:
        """Get absolute path to qdrant_db, resolving relative paths."""
        if self.qdrant_path.is_absolute():
            return self.qdrant_path
        # Resolve relative to current working directory
        return self.qdrant_path.resolve()

    def get_absolute_knowledge_base_dir(self) -> Path:
        """Get absolute path to knowledge base directory."""
        if self.knowledge_base_dir.is_absolute():
            return self.knowledge_base_dir
        return self.knowledge_base_dir.resolve()

    # Vector DB switching
    vector_db: Literal["local", "qdrant_cloud"] = "local"
    qdrant_cloud_url: str = ""
    qdrant_cloud_api_key: str = ""
    collection_name: str = "knowledge_base"

    # LLM switching
    llm_service: Literal["local", "groq"] = "local"
    groq_api_key: str = ""

    # Chunking defaults (can be overridden with --strategy in indexing)
    chunk_size: int = 1024  # Default strategy; use --strategy small for 256
    chunk_overlap: int = 128  # Default; small strategy uses 32
    
    # Retrieval settings - optimized for small chunks (256 tokens)
    # With 256-token chunks, we need more chunks to get equivalent context
    similarity_top_k: int = 10  # 10 x 256 = ~2560 tokens context
    similarity_cutoff: float = 0.7  # Filters low-relevance noise

    # Models
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama3.2:3b"  # Ollama model
    groq_model: str = "llama-3.1-70b-versatile"  # Groq model

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: float = 120.0

    # AWS S3 settings
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "eu-central-1"
    s3_bucket: str = ""
    s3_domaindata_key: str = "domaindata.zip"

    # Processing
    batch_size: int = 100
    show_progress: bool = True

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


settings = RAGSettings()
