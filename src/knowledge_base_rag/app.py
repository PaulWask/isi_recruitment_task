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
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

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

# Configure logging
logging.basicConfig(level=logging.INFO)
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
# Cached Resources (persist across page refreshes AND sessions)
# LAZY IMPORTS: Heavy modules loaded only when first needed
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_cached_rag_engine():
    """Get cached RAG engine - initialized ONCE per server lifetime."""
    # Lazy import to avoid blocking UI
    from knowledge_base_rag.engine.rag import RAGEngine
    logger.info("🚀 Initializing RAG engine (first time only)...")
    return RAGEngine()


@st.cache_resource(show_spinner=False)
def get_cached_embed_model():
    """Get cached embedding model - loaded ONCE per server lifetime."""
    # Lazy import
    from knowledge_base_rag.storage.embeddings import get_embed_model
    logger.info("🚀 Loading embedding model (first time only)...")
    return get_embed_model()


@st.cache_resource(show_spinner=False)
def get_cached_vector_store_stats() -> dict:
    """Get cached vector store stats - checked ONCE per server lifetime."""
    # Lazy import
    from knowledge_base_rag.storage.vector_store import VectorStoreManager
    logger.info("🚀 Checking vector store (first time only)...")
    try:
        embed_model = get_cached_embed_model()
        manager = VectorStoreManager(embed_model=embed_model)
        stats = manager.get_stats()
        stats["_cached"] = True
        return stats
    except Exception as e:
        logger.error(f"Vector store check failed: {e}")
        return {"exists": False, "vectors_count": 0, "_cached": True}


@st.cache_data(ttl=30, show_spinner=False)  # Cache for 30 seconds
def check_ollama_status() -> bool:
    """Check Ollama status - cached for 30 seconds to avoid spam."""
    # Lazy import
    from knowledge_base_rag.core.llm import is_ollama_available
    return is_ollama_available()


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
    """Save chat history to file."""
    try:
        history = {
            "messages": st.session_state.messages,
            "saved_at": datetime.now().isoformat(),
        }
        CHAT_HISTORY_FILE.write_text(json.dumps(history, indent=2, default=str))
        logger.info(f"Chat history saved: {len(st.session_state.messages)} messages")
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")


def load_chat_history():
    """Load chat history from file."""
    try:
        if CHAT_HISTORY_FILE.exists():
            data = json.loads(CHAT_HISTORY_FILE.read_text())
            return data.get("messages", [])
    except Exception as e:
        logger.error(f"Failed to load chat history: {e}")
    return []


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


def show_startup_screen():
    """Show a loading screen during first initialization with real progress.
    
    Uses cached resources for faster subsequent loads.
    """
    logger.info("🎬 Rendering loading screen...")
    
    # Loading screen with INLINE styles (works in both light/dark mode)
    # This renders IMMEDIATELY before any heavy imports
    loading_html = st.empty()
    loading_html.markdown("""
    <style>
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
    </style>
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:60vh; text-align:center;">
        <div style="background:rgba(30, 30, 50, 0.95); border:1px solid rgba(102,126,234,0.3); padding:2rem; border-radius:1rem; box-shadow:0 8px 32px rgba(0,0,0,0.3), 0 0 30px rgba(102,126,234,0.15); max-width:400px; width:90%;">
            <div style="font-size:4rem; margin-bottom:1rem; animation:float 3s ease-in-out infinite;">📚</div>
            <h1 style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; font-size:1.8rem; font-weight:700; margin:0 0 0.5rem 0;">Knowledge Base Q&A</h1>
            <p style="color:#a1a1aa; font-size:1rem; margin:0 0 1.5rem 0;">Initializing system...</p>
            <div style="width:40px; height:40px; border:3px solid rgba(102,126,234,0.3); border-top-color:#667eea; border-radius:50%; margin:0 auto; animation:spin 1s linear infinite;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress containers - centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        progress_bar = st.progress(0)
        status_text = st.empty()
        details_text = st.empty()
    
    def update_status(step: str, progress: float, detail: str = "", delay: float = 0.15):
        """Update loading status with visible delay."""
        progress_bar.progress(progress)
        status_text.markdown(f"<p style='text-align:center; color:#667eea; font-weight:600;'>🔄 {step}</p>", unsafe_allow_html=True)
        if detail:
            details_text.markdown(f"<p style='text-align:center; color:#a1a1aa; font-size:0.85rem;'>{detail}</p>", unsafe_allow_html=True)
        time.sleep(delay)  # Small delay so user can see progress
    
    try:
        # Step 1: Configuration
        update_status("Loading configuration...", 0.1, f"LLM: {settings.llm_service}")
        
        # Step 2: Animate progress while loading (gives visual feedback)
        update_status("Initializing embedding model...", 0.25, "Loading all-MiniLM-L6-v2...")
        
        # Step 3: Load embedding model + check vector store (CACHED)
        update_status("Connecting to vector store...", 0.4, "Checking Qdrant...")
        vs_stats = get_cached_vector_store_stats()
        vs_exists = vs_stats.get("exists", False)
        vectors = vs_stats.get("vectors_count", 0)
        
        if vs_exists and vectors > 0:
            update_status("Vector store connected", 0.55, f"Found {vectors:,} vectors")
        elif vs_exists:
            update_status("Vector store empty", 0.55, "⚠️ Run indexing first")
        else:
            update_status("No index found", 0.55, "⚠️ Run indexing first")
        
        # Step 4: LLM check (CACHED for 30 seconds)
        update_status("Checking language model...", 0.7, f"Service: {settings.llm_service}")
        if settings.llm_service == "local":
            llm_available = check_ollama_status()  # Cached!
            if llm_available:
                update_status("LLM connected", 0.8, f"Ollama ({settings.llm_model})")
            else:
                update_status("LLM offline", 0.8, "⚠️ Start Ollama to enable queries")
        else:
            has_key = bool(settings.groq_api_key)
            if has_key:
                update_status("LLM configured", 0.8, f"Groq ({settings.groq_model})")
            else:
                update_status("LLM not configured", 0.8, "⚠️ Set GROQ_API_KEY")
        
        # Step 5: Initialize RAG engine (CACHED)
        update_status("Initializing RAG engine...", 0.9, "Building query pipeline...")
        st.session_state.rag_engine = get_cached_rag_engine()
        
        # Engine is ready if we have vectors
        st.session_state.engine_ready = vs_exists and vectors > 0
        
        # Final status
        progress_bar.progress(1.0)
        if st.session_state.engine_ready:
            status_text.markdown("<p style='text-align:center; color:#48bb78; font-weight:600;'>✅ Ready!</p>", unsafe_allow_html=True)
            details_text.markdown("<p style='text-align:center; color:#a1a1aa; font-size:0.85rem;'>All systems initialized</p>", unsafe_allow_html=True)
        else:
            status_text.markdown("<p style='text-align:center; color:#ecc94b; font-weight:600;'>⚠️ Setup Required</p>", unsafe_allow_html=True)
            details_text.markdown("<p style='text-align:center; color:#a1a1aa; font-size:0.85rem;'>Run indexing to get started</p>", unsafe_allow_html=True)
        
        time.sleep(0.3)  # Brief pause to show final status
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.exception(e)
        status_text.error(f"Initialization error: {e}")
        # Still mark as initialized so user can see the "not ready" warning with instructions
        st.session_state.engine_ready = False
        time.sleep(2)
    
    # Mark as initialized and set cache flag for fast subsequent loads
    st.session_state.initialized = True
    st.session_state._resources_loaded = True
    st.rerun()


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
        
        show_sources = st.checkbox("Show sources", value=True)
        
        st.divider()
        
        # Session info
        st.markdown("## 📊 Session")
        # Count queries from messages (more reliable than session counter)
        query_count = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric("Queries", query_count)
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            # Also delete the saved history file
            if CHAT_HISTORY_FILE.exists():
                CHAT_HISTORY_FILE.unlink()
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
        
        return top_k, show_sources


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
                    
                    # Show more text (1000 chars) with ellipsis if truncated
                    full_text = src.get("text", "")
                    if len(full_text) > 1000:
                        text_preview = full_text[:1000] + "..."
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
    
    # Render compact metrics bar
    st.markdown(f"""
    <div style="display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.8rem; color: #a0aec0; margin-top: 0.5rem; padding: 0.5rem; background: rgba(102,126,234,0.05); border-radius: 0.5rem;">
        <span>⏱️ <strong>{latency_str}</strong></span>
        <span>📊 Precision: <strong style="color: {precision_color}">{precision:.0%}</strong></span>
        <span>📈 Avg Score: <strong>{avg_score:.0%}</strong></span>
        <span>📁 {sources} source{'s' if sources != 1 else ''}</span>
        <span>🎯 MRR: <strong>{metrics.mrr:.2f}</strong></span>
    </div>
    """, unsafe_allow_html=True)


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
                
                st.markdown(f"""
                <div style="display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.8rem; color: #a0aec0; margin-top: 0.5rem; padding: 0.5rem; background: rgba(102,126,234,0.05); border-radius: 0.5rem;">
                    <span>⏱️ <strong>{latency_str}</strong></span>
                    <span>📊 P@K: <strong style="color: {precision_color}">{precision:.0%}</strong></span>
                    <span>📈 Score: <strong>{avg_score:.0%}</strong></span>
                    <span>📁 {sources} src</span>
                    <span>🎯 MRR: {mrr:.2f}</span>
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
    init_session_state()
    
    # Show startup screen only on FIRST load (not cached yet)
    if not st.session_state.initialized:
        logger.info("📦 First load - showing startup screen...")
        show_startup_screen()
        return
    
    # Sidebar
    top_k, show_sources = render_sidebar()
    
    # Header
    render_header()
    
    # Get RAG engine
    engine = get_rag_engine()
    
    # Check if system is ready
    if not st.session_state.engine_ready or engine is None:
        render_not_ready_warning()
        return
    
    # Chat history
    render_chat_history(show_sources)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })
        
        # Display user message
        render_message("user", prompt)
        
        # Generate response
        with st.spinner("Searching knowledge base..."):
            try:
                response = engine.query(prompt, similarity_top_k=top_k)
                metrics = response.metrics
                
                # Prepare metrics for storage
                metrics_dict = metrics.to_dict() if metrics else {}
                
                # Add assistant message with metrics
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "latency_ms": metrics.e2e_ms if metrics else 0,
                    "metrics": metrics_dict,
                    "warning": response.warning,
                })
                
                st.session_state.total_queries += 1
                
                # Display response
                render_message("assistant", response.answer, response.sources, show_sources)
                
                # Show warning if present (low relevance, etc.)
                if response.warning:
                    st.warning(response.warning)
                
                # Show detailed metrics
                if metrics:
                    render_metrics_summary(metrics)
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                logger.exception("Query failed")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ {error_msg}",
                    "sources": [],
                })
        
        # Save chat history to file (persists across refreshes)
        save_chat_history()
        
        # Rerun to update sidebar with new query count
        st.rerun()


# Streamlit runs the entire script on each interaction
# Call main() directly (not inside __name__ guard)
main()

