"""Streamlit Web Application for Knowledge Base RAG.

A professional chat interface for querying the domain knowledge base.
Features:
- Chat-style Q&A interface
- Source attribution with expandable details
- Confidence indicators
- Persistent session history
- System status dashboard

Run with:
    uv run streamlit run src/knowledge_base_rag/app.py
    # or
    uv run kb-app
"""

# =============================================================================
# FAST IMPORTS ONLY - Heavy imports are lazy-loaded to show loading screen first
# =============================================================================
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

# ============================================================================
# NLTK Configuration - MUST be set before any LlamaIndex imports
# LlamaIndex uses its own NLTK cache path, we need to override it
# ============================================================================
# Check multiple possible NLTK data locations (Docker vs local)
_nltk_paths = [
    Path("/app/.cache/nltk_data"),  # Docker location (pre-downloaded)
    Path(__file__).parent.parent.parent / ".nltk_data",  # Local dev location
]
for _nltk_data_dir in _nltk_paths:
    if _nltk_data_dir.exists():
        os.environ["NLTK_DATA"] = str(_nltk_data_dir.absolute())
        try:
            import nltk
            if str(_nltk_data_dir.absolute()) not in nltk.data.path:
                nltk.data.path.insert(0, str(_nltk_data_dir.absolute()))
        except ImportError:
            pass
        break  # Use first valid path

import streamlit as st

# Light imports only (no LlamaIndex/NLTK trigger)
from knowledge_base_rag.core.config import settings
from knowledge_base_rag.ui import (
    load_css,
    render_source_card,
    render_user_message,
    render_assistant_message,
    render_header as render_header_html,
)

# Type hints only (not imported at runtime)
if TYPE_CHECKING:
    from knowledge_base_rag.engine.rag import RAGEngine, RAGResponse
    from knowledge_base_rag.storage.vector_store import VectorStoreManager

# Chat history persistence file
CHAT_HISTORY_FILE = Path("chat_history.json")

# Configure logging with professional timestamp format (HH:MM:SS.mmm)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Knowledge Base Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS from external file (src/knowledge_base_rag/ui/static/styles.css)
st.markdown(load_css(), unsafe_allow_html=True)


# =============================================================================
# Cached Resources - Using session_state for reliability
# Note: @st.cache_resource can fail in Docker/WSL environments
# =============================================================================

def get_cached_rag_engine():
    """Get RAG engine from session state (initialized once per session)."""
    if "_rag_engine" not in st.session_state:
        from knowledge_base_rag.engine.rag import RAGEngine
        logger.info("🚀 Initializing RAG engine...")
        st.session_state._rag_engine = RAGEngine()
    return st.session_state._rag_engine


def get_cached_embed_model():
    """Get embedding model from session state (initialized once per session)."""
    if "_embed_model" not in st.session_state:
        from knowledge_base_rag.storage.embeddings import get_embed_model
        logger.info("🚀 Loading embedding model...")
        st.session_state._embed_model = get_embed_model()
    return st.session_state._embed_model


def get_cached_vector_store_stats() -> dict:
    """Get vector store stats from session state (initialized once per session)."""
    if "_vs_stats" not in st.session_state:
        from knowledge_base_rag.storage.vector_store import VectorStoreManager
        logger.info("🚀 Checking vector store...")
        try:
            embed_model = get_cached_embed_model()
            manager = VectorStoreManager(embed_model=embed_model)
            stats = manager.get_stats()
            stats["_cached"] = True
            st.session_state._vs_stats = stats
        except Exception as e:
            logger.error(f"Vector store check failed: {e}")
            st.session_state._vs_stats = {"exists": False, "vectors_count": 0, "_cached": True}
    return st.session_state._vs_stats


def check_ollama_status() -> bool:
    """Check Ollama status (cached in session state with TTL)."""
    now = time.time()
    cache_key = "_ollama_status"
    cache_time_key = "_ollama_status_time"
    
    # Cache for 60 seconds
    if cache_key in st.session_state:
        last_check = st.session_state.get(cache_time_key, 0)
        if now - last_check < 60:
            return st.session_state[cache_key]
    
    from knowledge_base_rag.core.llm import is_ollama_available
    result = is_ollama_available()
    st.session_state[cache_key] = result
    st.session_state[cache_time_key] = now
    return result


def check_system_ready() -> tuple[bool, dict]:
    """Quick check if system is ready (uses cached resources)."""
    stats = get_cached_vector_store_stats()
    exists = stats.get("exists", False)
    vectors = stats.get("vectors_count", 0)
    return exists and vectors > 0, stats


# =============================================================================
# Chat History Persistence
# =============================================================================

def save_chat_history():
    """Save chat history to file and invalidate cache."""
    try:
        history = {
            "messages": st.session_state.messages,
            "saved_at": datetime.now().isoformat(),
        }
        CHAT_HISTORY_FILE.write_text(json.dumps(history, indent=2, default=str))
        # Invalidate cache so next load gets fresh data
        _load_chat_history_from_file.clear()
        logger.info(f"Chat history saved: {len(st.session_state.messages)} messages")
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")


@st.cache_data(ttl=5, show_spinner=False)  # Cache for 5 seconds to avoid disk reads on rapid refreshes
def _load_chat_history_from_file() -> list:
    """Load chat history from file (cached)."""
    try:
        if CHAT_HISTORY_FILE.exists():
            content = CHAT_HISTORY_FILE.read_text().strip()
            if not content:
                return []
            data = json.loads(content)
            # Handle both formats: {"messages": [...]} or [...]
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("messages", [])
            return []
    except Exception as e:
        logger.error(f"Failed to load chat history: {e}")
    return []


def load_chat_history():
    """Load chat history - uses cached version when possible."""
    return _load_chat_history_from_file()


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        # Try to load from file first
        st.session_state.messages = load_chat_history()
    
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
    
    if "engine_ready" not in st.session_state:
        st.session_state.engine_ready = False
    
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    
    if "initialized" not in st.session_state:
        # First load - always show startup screen
        st.session_state.initialized = False
    
    if "is_processing" not in st.session_state:
        # Flag to prevent sidebar changes from interrupting queries
        st.session_state.is_processing = False
    
    if "stop_requested" not in st.session_state:
        # Flag to signal query cancellation
        st.session_state.stop_requested = False
    
    if "query_to_process" not in st.session_state:
        # Query waiting to be processed (after rerun for frozen UI)
        st.session_state.query_to_process = None
    
    # Initialize advanced retrieval settings
    if "setting_query_expansion" not in st.session_state:
        st.session_state.setting_query_expansion = False
    if "setting_hybrid_search" not in st.session_state:
        st.session_state.setting_hybrid_search = False
    if "setting_show_sources" not in st.session_state:
        st.session_state.setting_show_sources = True
    if "setting_top_k" not in st.session_state:
        st.session_state.setting_top_k = 10  # Default for small chunks (256 tokens)


def do_initialization():
    """Perform initialization synchronously (no UI updates during init).
    
    This avoids Streamlit's client-server synchronization delays.
    All resources are cached in st.session_state for reliability.
    """
    logger.info("🎬 Initializing all components...")
    
    try:
        # Initialize all resources (cached in session_state)
        vs_stats = get_cached_vector_store_stats()
        st.session_state.rag_engine = get_cached_rag_engine()
        
        # Check readiness
        vs_exists = vs_stats.get("exists", False)
        vectors = vs_stats.get("vectors_count", 0)
        st.session_state.engine_ready = vs_exists and vectors > 0
        st.session_state._vs_count = vectors  # Store for frozen sidebar display
        
        logger.info(f"✅ Initialization complete: {vectors} vectors, ready={st.session_state.engine_ready}")
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        logger.exception(e)
        st.session_state.engine_ready = False
    
    # Mark as initialized (persists for this session)
    st.session_state.initialized = True


def get_rag_engine():
    """Get RAG engine (uses cached version for speed)."""
    if st.session_state.rag_engine is None:
        try:
            # Use cached engine for faster loads
            st.session_state.rag_engine = get_cached_rag_engine()
            # Quick system check
            is_ready, stats = check_system_ready()
            st.session_state.engine_ready = is_ready
        except Exception as e:
            logger.error(f"Failed to get RAG engine: {e}")
            st.session_state.engine_ready = False
    
    return st.session_state.rag_engine


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render the sidebar with system status and settings."""
    with st.sidebar:
        st.markdown("## ⚙️ System Status")
        
        # Check services (using CACHED status - no repeated DB/HTTP calls)
        col1, col2 = st.columns(2)
        
        with col1:
            # Vector store status (CACHED)
            vs_stats = get_cached_vector_store_stats()
            vs_healthy = vs_stats.get("exists", False) and vs_stats.get("vectors_count", 0) > 0
            
            if vs_healthy:
                st.markdown("🟢 **Vector Store**")
                vectors = vs_stats.get("vectors_count", 0)
                st.caption(f"{vectors:,} vectors")
            else:
                st.markdown("🔴 **Vector Store**")
                if vs_stats.get("exists", False):
                    st.caption("Empty collection")
                else:
                    st.caption("Not indexed")
        
        with col2:
            # LLM status (CACHED for 30 seconds)
            if settings.llm_service == "local":
                llm_available = check_ollama_status()  # Cached!
                if llm_available:
                    st.markdown("🟢 **LLM (Ollama)**")
                    st.caption(settings.llm_model)
                else:
                    st.markdown("🔴 **LLM (Ollama)**")
                    st.caption("Not running")
            else:
                has_key = bool(settings.groq_api_key)
                if has_key:
                    st.markdown("🟢 **LLM (Groq)**")
                    st.caption(settings.groq_model)
                else:
                    st.markdown("🔴 **LLM (Groq)**")
                    st.caption("No API key")
        
        st.divider()
        
        # Configuration
        st.markdown("## 🎛️ Settings")
        
        top_k = st.slider(
            "Sources to retrieve",
            min_value=1,
            max_value=15,
            value=settings.similarity_top_k,
            help="Number of document chunks to retrieve for each query"
        )
        
        show_sources = st.checkbox("Show sources", value=st.session_state.setting_show_sources)
        st.session_state.setting_show_sources = show_sources
        
        # Advanced settings - Reranking is ALWAYS ON (best practice)
        use_reranking = True  # Always enabled - proven to give best results
        
        # Initialize session state for settings (prevents rerun interruption)
        if "setting_query_expansion" not in st.session_state:
            st.session_state.setting_query_expansion = False
        if "setting_hybrid_search" not in st.session_state:
            st.session_state.setting_hybrid_search = False
        
        # Check if a query is being processed
        is_processing = st.session_state.get("is_processing", False)
        
        # Make Advanced Retrieval header more visible
        st.markdown("### 🔧 Advanced Retrieval")
        with st.expander("Show Options", expanded=False):
            # Show reranking status (not toggleable)
            st.success("✅ **Reranking: Always Enabled** — Cross-encoder for best accuracy")
            st.caption("Adds ~3-5s latency but improves precision by ~25%")
            
            st.divider()
            st.markdown("##### 🧪 Optional Enhancements")
            st.caption("⚠️ These may increase latency significantly. Use only when needed.")
            
            # CRITICAL: During processing, show STATIC display instead of interactive widgets
            # This prevents any click from triggering a rerun that would interrupt the query
            if is_processing:
                st.warning("⏳ Query in progress... please wait")
                # Show current settings as static text (no interactive widgets!)
                exp_status = "✅ Enabled" if st.session_state.setting_query_expansion else "⬜ Disabled"
                hyb_status = "✅ Enabled" if st.session_state.setting_hybrid_search else "⬜ Disabled"
                st.markdown(f"**Query Expansion:** {exp_status}")
                st.markdown(f"**Hybrid Search:** {hyb_status}")
                # Use stored values
                use_query_expansion = st.session_state.setting_query_expansion
                use_hybrid_search = st.session_state.setting_hybrid_search
            else:
                # NOT processing - show interactive checkboxes
                use_query_expansion = st.checkbox(
                    "Query Expansion",
                    value=st.session_state.setting_query_expansion,
                    key="checkbox_query_expansion",
                    help="Expands acronyms. Adds ~20-30s latency (runs 3 queries)."
                )
                st.session_state.setting_query_expansion = use_query_expansion
                
                if use_query_expansion:
                    st.warning("⚠️ Adds ~20-30s latency (3x queries)")
                    st.caption("Expanding: CPI→Consumer Price Index, GDP, YoY, etc.")
                else:
                    st.caption("💡 Only use for: Short queries with acronyms (CPI, GDP, YoY)")
                
                use_hybrid_search = st.checkbox(
                    "Hybrid Search (BM25)",
                    value=st.session_state.setting_hybrid_search,
                    key="checkbox_hybrid_search",
                    help="Adds keyword matching. Useful for exact terms/IDs."
                )
                st.session_state.setting_hybrid_search = use_hybrid_search
                
                if use_hybrid_search:
                    st.info("🔀 BM25 + Vector fusion (60/40)")
                else:
                    st.caption("💡 Only use for: Document IDs, specific numbers, exact terms")
            
            # Quick guide (always show)
            st.divider()
            with st.popover("📖 When to use these?"):
                st.markdown("""
                **Query Expansion** - Use when:
                - Query contains acronyms (CPI, GDP, YoY)
                - Very short queries (1-2 words)
                - ⚠️ Adds ~20-30s latency
                
                **Hybrid Search** - Use when:
                - Looking for document IDs (Circular 1234)
                - Specific numbers/dates
                - Exact phrase matching
                
                **For most queries: Just use default settings!**
                """)
        
        st.divider()
        
        # Session info
        st.markdown("## 📊 Session")
        # Count queries from messages (more reliable than session counter)
        query_count = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric("Queries", query_count)
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.is_processing = False
            st.session_state.stop_requested = False
            st.session_state.query_to_process = None
            st.session_state.total_queries = 0
            # Clear the saved history file (truncate instead of delete for Docker volumes)
            try:
                CHAT_HISTORY_FILE.write_text("[]")
            except Exception:
                pass  # Ignore if file doesn't exist or can't be written
            st.rerun()
        
        st.divider()
        
        # Help
        with st.expander("ℹ️ Help"):
            st.markdown("""
            **How to use:**
            1. Type your question in the chat
            2. The system searches the knowledge base
            3. An answer is generated from relevant documents
            
            **Tips:**
            - Be specific in your questions
            - Check the sources for verification
            - Low relevance scores may indicate uncertainty
            """)
        
        return top_k, show_sources, use_reranking, use_query_expansion, use_hybrid_search


def render_header():
    """Render the main header using UI component."""
    st.markdown(render_header_html(), unsafe_allow_html=True)


def render_message(role: str, content: str, sources: list | None = None, show_sources: bool = True):
    """Render a chat message with optional sources using UI components."""
    if role == "user":
        st.markdown(render_user_message(content), unsafe_allow_html=True)
    else:
        st.markdown(render_assistant_message(content), unsafe_allow_html=True)
        
        # Show sources if available
        if sources and show_sources:
            with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                for src in sources:
                    score = src.get("score", 0)
                    metadata = src.get("metadata", {})
                    
                    # Extract source info from metadata
                    source_path = metadata.get("source", src.get("source", "Unknown"))
                    file_name = metadata.get("file_name", source_path.split("/")[-1] if "/" in source_path else source_path)
                    page_label = metadata.get("page_label")
                    file_type = metadata.get("file_type", "")
                    
                    # Show more text (2500 chars) for better context visibility
                    full_text = src.get("text", "")
                    if len(full_text) > 2500:
                        text_preview = full_text[:2500] + "..."
                    else:
                        text_preview = full_text
                    
                    # Use UI component for source card
                    st.markdown(
                        render_source_card(
                            score=score,
                            file_name=file_name,
                            source_path=source_path,
                            text_preview=text_preview,
                            page_label=page_label,
                            file_type=file_type,
                        ),
                        unsafe_allow_html=True
                    )


def render_metrics_summary(metrics) -> None:
    """Render a compact metrics summary below the response.
    
    Args:
        metrics: RAGMetrics object with performance data
    """
    latency = metrics.e2e_ms
    precision = metrics.precision_at_k
    avg_score = metrics.avg_score
    sources = metrics.source_coverage
    
    # Format latency
    if latency >= 1000:
        latency_str = f"{latency/1000:.2f}s"
    else:
        latency_str = f"{latency:.0f}ms"
    
    # Color-code precision
    if precision >= 0.7:
        precision_color = "#48bb78"  # green
    elif precision >= 0.4:
        precision_color = "#ecc94b"  # yellow
    else:
        precision_color = "#f56565"  # red
    
    # Query expansion indicator - always show status
    expansion_enabled = hasattr(metrics, 'expansion_enabled') and metrics.expansion_enabled
    if expansion_enabled:
        queries_used = getattr(metrics, 'queries_used', 1)
        expansion_str = f'<span style="color: #48bb78;">✓ Expansion ({queries_used}q)</span>'
    else:
        expansion_str = ''
    
    # Hybrid search indicator - always show status
    hybrid_enabled = hasattr(metrics, 'hybrid_enabled') and metrics.hybrid_enabled
    if hybrid_enabled:
        hybrid_str = '<span style="color: #48bb78;">✓ Hybrid</span>'
    else:
        hybrid_str = ''
    
    # Render compact metrics bar
    st.markdown(f"""
    <div style="display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.8rem; color: #a0aec0; margin-top: 0.5rem; padding: 0.5rem; background: rgba(102,126,234,0.05); border-radius: 0.5rem;">
        <span>⏱️ <strong>{latency_str}</strong></span>
        <span>📊 Precision: <strong style="color: {precision_color}">{precision:.0%}</strong></span>
        <span>📈 Avg Score: <strong>{avg_score:.0%}</strong></span>
        <span>📁 {sources} source{'s' if sources != 1 else ''}</span>
        <span>🎯 MRR: <strong>{metrics.mrr:.2f}</strong></span>
        {expansion_str}
        {hybrid_str}
    </div>
    """, unsafe_allow_html=True)
    
    # Optional: Detailed metrics expander
    with st.expander("📈 Detailed Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Latency**")
            st.caption(f"TTFR: {metrics.ttfr_ms:.0f}ms")
            st.caption(f"E2E: {metrics.e2e_ms:.0f}ms")
        with col2:
            st.markdown("**Retrieval Quality**")
            st.caption(f"Precision@K: {metrics.precision_at_k:.1%}")
            st.caption(f"MRR: {metrics.mrr:.3f}")
            st.caption(f"Hit Rate: {metrics.hit_rate:.0%}")
        with col3:
            st.markdown("**Scores**")
            st.caption(f"Avg: {metrics.avg_score:.1%}")
            st.caption(f"Max: {metrics.max_score:.1%}")
            st.caption(f"Min: {metrics.min_score:.1%}")
            st.caption(f"Above threshold: {metrics.above_threshold}/{metrics.total_retrieved}")


def render_chat_history(show_sources: bool):
    """Render the chat history with latency info, metrics, and warnings."""
    for msg in st.session_state.messages:
        render_message(
            msg["role"], 
            msg["content"],
            msg.get("sources"),
            show_sources
        )
        # Show metrics for assistant messages
        if msg["role"] == "assistant":
            # Show warning if present
            warning = msg.get("warning")
            if warning:
                st.warning(warning)
            
            metrics_dict = msg.get("metrics", {})
            if metrics_dict:
                # Reconstruct metrics display from stored dict
                latency_data = metrics_dict.get("latency", {})
                retrieval_data = metrics_dict.get("retrieval", {})
                scores_data = metrics_dict.get("scores", {})
                coverage_data = metrics_dict.get("coverage", {})
                
                latency = latency_data.get("e2e_ms", msg.get("latency_ms", 0))
                precision = retrieval_data.get("precision_at_k", 0)
                avg_score = scores_data.get("avg", 0)
                sources = coverage_data.get("unique_sources", 0)
                mrr = retrieval_data.get("mrr", 0)
                
                # Format latency
                if latency >= 1000:
                    latency_str = f"{latency/1000:.2f}s"
                else:
                    latency_str = f"{latency:.0f}ms"
                
                # Color-code precision
                if precision >= 0.7:
                    precision_color = "#48bb78"
                elif precision >= 0.4:
                    precision_color = "#ecc94b"
                else:
                    precision_color = "#f56565"
                
                # Query expansion indicator - only show when enabled
                expansion_data = metrics_dict.get("expansion", {})
                expansion_enabled = expansion_data.get("enabled", False)
                if expansion_enabled:
                    queries_used = expansion_data.get("queries_used", 1)
                    expansion_str = f'<span style="color: #48bb78;">✓ Expansion ({queries_used}q)</span>'
                else:
                    expansion_str = ''
                
                # Hybrid search indicator - only show when enabled
                hybrid_data = metrics_dict.get("hybrid", {})
                hybrid_enabled = hybrid_data.get("enabled", False)
                if hybrid_enabled:
                    hybrid_str = '<span style="color: #48bb78;">✓ Hybrid</span>'
                else:
                    hybrid_str = ''
                
                st.markdown(f"""
                <div style="display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.8rem; color: #a0aec0; margin-top: 0.5rem; padding: 0.5rem; background: rgba(102,126,234,0.05); border-radius: 0.5rem;">
                    <span>⏱️ <strong>{latency_str}</strong></span>
                    <span>📊 P@K: <strong style="color: {precision_color}">{precision:.0%}</strong></span>
                    <span>📈 Score: <strong>{avg_score:.0%}</strong></span>
                    <span>📁 {sources} src</span>
                    <span>🎯 MRR: {mrr:.2f}</span>
                    {expansion_str}
                    {hybrid_str}
                </div>
                """, unsafe_allow_html=True)
            elif "latency_ms" in msg:
                # Fallback for old messages without full metrics
                latency = msg["latency_ms"]
                if latency >= 1000:
                    st.caption(f"⏱️ {latency/1000:.2f}s")
                else:
                    st.caption(f"⏱️ {latency:.0f}ms")


def render_not_ready_warning():
    """Render warning when system is not ready."""
    st.warning("""
    ⚠️ **System Not Ready**
    
    The knowledge base has not been indexed yet. Please run:
    
    ```bash
    uv run python scripts/index_documents.py
    ```
    
    This will process your documents and create the search index.
    """)
    
    # Check what's missing
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Checklist")
        
        # Check domaindata
        from pathlib import Path
        data_exists = Path(settings.knowledge_base_dir).exists()
        st.checkbox("Domain data downloaded", value=data_exists, disabled=True)
        
        # Check index (use cached stats)
        vs_stats = get_cached_vector_store_stats()
        index_exists = vs_stats.get("exists", False) and vs_stats.get("vectors_count", 0) > 0
        st.checkbox("Index created", value=index_exists, disabled=True)
        
        # Check LLM (use cached status)
        if settings.llm_service == "local":
            llm_ready = check_ollama_status()
        else:
            llm_ready = bool(settings.groq_api_key)
        st.checkbox("LLM available", value=llm_ready, disabled=True)
    
    with col2:
        st.markdown("### Quick Setup")
        st.code("""
# 1. Download data
uv run python scripts/download_data.py

# 2. Index documents
uv run python scripts/index_documents.py

# 3. Start Ollama (if using local LLM)
ollama serve
ollama pull llama3.2:3b
        """)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    # Initialize session state first
    init_session_state()
    
    # =========================================================================
    # CRITICAL: If processing a query, show FROZEN UI and do the processing
    # This happens AFTER st.rerun() was called, ensuring no interactive widgets
    # =========================================================================
    if st.session_state.get("is_processing", False) and st.session_state.get("query_to_process"):
        # Show frozen sidebar (NO interactive widgets) - mirrors normal sidebar structure
        with st.sidebar:
            # System Status (same as normal)
            st.markdown("## ⚙️ System Status")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🟢 **Vector Store**")
                st.caption(f"{st.session_state.get('_vs_count', 0):,} vectors")
            with col2:
                st.markdown("🟢 **LLM (Ollama)**")
                st.caption("llama3.2:3b")
            
            st.divider()
            
            # Settings header (same as normal)
            st.markdown("## 🎯 Settings")
            st.markdown(f"**Sources to retrieve:** {st.session_state.get('setting_top_k', 10)}")
            st.markdown(f"**Show sources:** {'Yes' if st.session_state.get('setting_show_sources', True) else 'No'}")
            
            st.divider()
            
            # Advanced Retrieval - locked during processing
            st.markdown("### 🔧 Advanced Retrieval")
            st.info("🔍 **Processing query...**")
            st.markdown(f"• Query Expansion: **{'On' if st.session_state.get('setting_query_expansion') else 'Off'}**")
            st.markdown(f"• Hybrid Search: **{'On' if st.session_state.get('setting_hybrid_search') else 'Off'}**")
            
            st.divider()
            
            # Stop button in sidebar during processing
            if st.button("🛑 Stop Query", type="primary", use_container_width=True, key="stop_sidebar"):
                st.session_state.stop_requested = True
                st.session_state.is_processing = False
                st.session_state.query_to_process = None
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ Query cancelled by user.",
                    "sources": [],
                })
                save_chat_history()  # Save before rerun to persist
                st.rerun()
        
        # Show header
        st.markdown("## 📚 Knowledge Base Q&A")
        
        # Show chat history
        show_sources = st.session_state.get("setting_show_sources", True)
        render_chat_history(show_sources)
        
        # ==========================================================================
        # STREAMING GENERATION WITH STOP BUTTON AT BOTTOM
        # ==========================================================================
        
        # Get the query to process
        query = st.session_state.query_to_process
        st.session_state.query_to_process = None  # Clear immediately
        
        # Check if stop was requested before starting
        if st.session_state.get("stop_requested", False):
            st.session_state.stop_requested = False
            st.session_state.is_processing = False
            st.rerun()
        
        # Placeholders for streaming UI
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        
        # ==========================================================================
        # CHAT INPUT AREA (STOP BUTTON) - Always at bottom during processing
        # ==========================================================================
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns([6, 1])
        with col1:
            st.text_input(
                "Query",
                value="⏳ Generating response...",
                disabled=True,
                label_visibility="collapsed",
                key="frozen_input"
            )
        with col2:
            stop_clicked = st.button("🛑 Stop", type="primary", use_container_width=True, key="stop_frozen")
            if stop_clicked:
                st.session_state.stop_requested = True
        
        # Start streaming
        status_placeholder.info("🔍 Searching knowledge base...")
        
        engine = get_rag_engine()
        if engine:
            current_top_k = st.session_state.get("setting_top_k", 10)
            current_query_expansion = st.session_state.get("setting_query_expansion", False)
            current_hybrid_search = st.session_state.get("setting_hybrid_search", False)
            
            def check_stop():
                return st.session_state.get("stop_requested", False)
            
            try:
                if check_stop():
                    raise InterruptedError("Query cancelled")
                
                stream = engine.query_stream(
                    query,
                    similarity_top_k=current_top_k,
                    use_query_expansion=current_query_expansion,
                    use_hybrid_search=current_hybrid_search,
                    stop_check=check_stop,
                )
                
                sources = []
                full_answer = ""
                metrics_data = {}
                
                for chunk in stream:
                    chunk_type = chunk.get("type")
                    
                    if chunk_type == "sources":
                        sources = chunk.get("data", [])
                        retrieval_ms = chunk.get("retrieval_ms", 0)
                        status_placeholder.success(f"✅ Found {len(sources)} sources ({retrieval_ms:.0f}ms)")
                        
                    elif chunk_type == "token":
                        token = chunk.get("data", "")
                        full_answer += token
                        answer_placeholder.markdown(full_answer + "▌")
                        
                    elif chunk_type == "complete":
                        full_answer = chunk.get("data", full_answer)
                        metrics_data = chunk.get("metrics", {})
                        answer_placeholder.markdown(full_answer)
                        status_placeholder.empty()
                        
                    elif chunk_type == "stopped":
                        full_answer = chunk.get("data", "") or "⚠️ Query cancelled."
                        answer_placeholder.markdown(full_answer)
                        status_placeholder.empty()
                        break
                        
                    elif chunk_type == "error":
                        full_answer = f"❌ Error: {chunk.get('data', 'Unknown')}"
                        answer_placeholder.markdown(full_answer)
                        status_placeholder.empty()
                        break
                
                # Save response
                e2e_ms = metrics_data.get("e2e_ms", 0) if isinstance(metrics_data, dict) else 0
                retrieval_ms = metrics_data.get("retrieval_ms", 0) if isinstance(metrics_data, dict) else 0
                source_scores = [s["score"] for s in sources if isinstance(s, dict) and "score" in s]
                avg_score = sum(source_scores) / max(len(source_scores), 1)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_answer,
                    "sources": sources,
                    "latency_ms": e2e_ms,
                    "metrics": {
                        "latency": {"e2e_ms": e2e_ms, "ttfr_ms": retrieval_ms},
                        "retrieval": {"precision_at_k": 1.0 if sources else 0, "mrr": 1.0, "hit_rate": 1.0 if sources else 0, "avg_score": avg_score},
                        "counts": {"above_threshold": len(sources), "total_retrieved": len(sources)},
                        "expansion": {"enabled": current_query_expansion, "queries_used": 1},
                        "hybrid": {"enabled": current_hybrid_search},
                    },
                    "warning": None,
                })
                st.session_state.total_queries += 1
                save_chat_history()
                
            except InterruptedError:
                logger.info("Query cancelled by user")
                    
            except Exception as e:
                logger.exception("Query failed")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {str(e)}",
                    "sources": [],
                })
            finally:
                st.session_state.is_processing = False
                st.session_state.stop_requested = False
        
        # Rerun to show normal UI
        st.rerun()
    
    # Show loading indicator if not initialized yet
    loading_placeholder = st.empty()
    
    if not st.session_state.initialized:
        # First load - show loading message with animated spinner
        loading_placeholder.markdown("""
        <style>
            @keyframes kb-spin { to { transform: rotate(360deg); } }
            @keyframes kb-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        </style>
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:50vh;">
            <div style="font-size:3rem; margin-bottom:1rem;">📚</div>
            <h2 style="color:#667eea; margin:0;">Loading Knowledge Base...</h2>
            <p style="color:#888; margin-top:0.5rem; animation: kb-pulse 1.5s ease-in-out infinite;">First load may take a moment...</p>
            <div style="width:40px; height:40px; border:4px solid rgba(102,126,234,0.2); border-top-color:#667eea; border-radius:50%; margin-top:1.5rem; animation: kb-spin 1s linear infinite;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Do initialization
        logger.info("📦 Initializing session...")
        do_initialization()
    
    # Clear loading placeholder now that we're ready
    loading_placeholder.empty()
    
    # Sidebar
    top_k, show_sources, use_reranking, use_query_expansion, use_hybrid_search = render_sidebar()
    
    # Header
    render_header()
    
    # Get RAG engine (with all enhancement settings)
    engine = get_rag_engine()
    if engine:
        engine.enable_reranking = use_reranking
        engine.enable_query_expansion = use_query_expansion
        engine.enable_hybrid_search = use_hybrid_search
    
    # Check if system is ready
    if not st.session_state.engine_ready or engine is None:
        render_not_ready_warning()
        return
    
    # Chat history
    render_chat_history(show_sources)
    
    # ==========================================================================
    # QUERY PROCESSING
    # ==========================================================================
    # Simple flow: Only process new queries from chat_input.
    # During processing, input is disabled to prevent duplicate submissions.
    # ==========================================================================
    
    # ==========================================================================
    # CUSTOM CHAT INPUT WITH SEND/STOP BUTTON
    # ==========================================================================
    is_processing = st.session_state.get("is_processing", False)
    
    # Initialize query input in session state
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        if is_processing:
            # Show disabled input during processing
            st.text_input(
                "Query",
                value="⏳ Processing your query...",
                disabled=True,
                label_visibility="collapsed",
                key="query_disabled"
            )
        else:
            # Normal text input
            query_text = st.text_input(
                "Query",
                value="",
                placeholder="Ask a question about your documents...",
                label_visibility="collapsed",
                key="query_input_field"
            )
    
    with col2:
        if is_processing:
            # STOP button (red)
            if st.button("🛑 Stop", type="primary", use_container_width=True, key="stop_main"):
                st.session_state.stop_requested = True
                st.session_state.is_processing = False
                st.session_state.query_to_process = None
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ Query cancelled by user.",
                    "sources": [],
                })
                save_chat_history()
                st.rerun()
        else:
            # SEND button (normal)
            send_clicked = st.button("➤ Send", type="secondary", use_container_width=True, key="send_main")
            
            # Handle send (button click or Enter key via form would be better, but this works)
            if send_clicked and query_text and query_text.strip():
                new_prompt = query_text.strip()
                # Reset stop flag
                st.session_state.stop_requested = False
                # Store query and settings
                st.session_state.query_to_process = new_prompt
                st.session_state.is_processing = True
                st.session_state.setting_top_k = top_k
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": new_prompt,
                })
                save_chat_history()
                st.rerun()


# Streamlit runs the entire script on each interaction
# Call main() directly (not inside __name__ guard)
main()

