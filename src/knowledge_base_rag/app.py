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

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from knowledge_base_rag.core.config import settings
from knowledge_base_rag.core.llm import get_llm, is_ollama_available
from knowledge_base_rag.engine.rag import RAGEngine, RAGResponse
from knowledge_base_rag.storage.vector_store import VectorStoreManager
from knowledge_base_rag.storage.embeddings import get_embed_model
from knowledge_base_rag.ui import (
    load_css,
    render_source_card,
    render_user_message,
    render_assistant_message,
    render_header as render_header_html,
)

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
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_cached_rag_engine() -> RAGEngine:
    """Get cached RAG engine - initialized ONCE per server lifetime."""
    logger.info("🚀 Initializing RAG engine (first time only)...")
    return RAGEngine()


@st.cache_resource(show_spinner=False)
def get_cached_embed_model():
    """Get cached embedding model - loaded ONCE per server lifetime."""
    logger.info("🚀 Loading embedding model (first time only)...")
    return get_embed_model()


@st.cache_resource(show_spinner=False)
def get_cached_vector_store_stats() -> dict:
    """Get cached vector store stats - checked ONCE per server lifetime."""
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
    return is_ollama_available()


def check_system_ready() -> tuple[bool, dict]:
    """Quick check if system is ready (uses cached resources)."""
    stats = get_cached_vector_store_stats()
    exists = stats.get("exists", False)
    vectors = stats.get("vectors_count", 0)
    return exists and vectors > 0, stats


def are_resources_cached() -> bool:
    """Check if heavy resources are already cached (instant check)."""
    try:
        # This is a quick check - if cache exists, it returns instantly
        # We check if the cache key exists without triggering full computation
        return get_cached_vector_store_stats().get("_cached", False)
    except Exception:
        return False


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
        # Skip startup screen if resources are already cached (fast path)
        if are_resources_cached():
            logger.info("⚡ Resources already cached - skipping startup screen")
            st.session_state.initialized = True
            st.session_state.rag_engine = get_cached_rag_engine()
            is_ready, _ = check_system_ready()
            st.session_state.engine_ready = is_ready
        else:
            st.session_state.initialized = False


def show_startup_screen():
    """Show a loading screen during first initialization with real progress.
    
    Uses cached resources for faster subsequent loads.
    """
    # Loading screen with INLINE styles (works in both light/dark mode)
    st.markdown("""
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
    
    def update_status(step: str, progress: float, detail: str = ""):
        """Update loading status."""
        progress_bar.progress(progress)
        status_text.markdown(f"<p style='text-align:center; color:#667eea; font-weight:600;'>🔄 {step}</p>", unsafe_allow_html=True)
        if detail:
            details_text.markdown(f"<p style='text-align:center; color:#a1a1aa; font-size:0.85rem;'>{detail}</p>", unsafe_allow_html=True)
    
    try:
        # Step 1: Configuration (instant)
        update_status("Loading configuration...", 0.2, f"LLM: {settings.llm_service}")
        
        # Step 2 & 3: Load embedding model + check vector store (CACHED)
        update_status("Loading resources...", 0.5, "Embedding model + Vector store...")
        vs_stats = get_cached_vector_store_stats()
        vs_exists = vs_stats.get("exists", False)
        vectors = vs_stats.get("vectors_count", 0)
        
        if vs_exists and vectors > 0:
            update_status("Vector store ready!", 0.65, f"✅ {vectors:,} vectors indexed")
        elif vs_exists:
            update_status("Vector store empty", 0.65, "⚠️ Run indexing first")
        else:
            update_status("No index found", 0.65, "⚠️ Run indexing first")
        
        # Step 4: LLM check (CACHED for 30 seconds)
        update_status("Checking language model...", 0.75, f"Service: {settings.llm_service}")
        if settings.llm_service == "local":
            llm_available = check_ollama_status()  # Cached!
            if llm_available:
                update_status("LLM ready!", 0.85, f"✅ Ollama ({settings.llm_model})")
            else:
                update_status("LLM offline", 0.85, "⚠️ Start Ollama to enable queries")
        else:
            has_key = bool(settings.groq_api_key)
            if has_key:
                update_status("LLM ready!", 0.85, f"✅ Groq ({settings.groq_model})")
            else:
                update_status("LLM not configured", 0.85, "⚠️ Set GROQ_API_KEY")
        
        # Step 5: Initialize RAG engine (CACHED)
        update_status("Preparing RAG engine...", 0.95, "Almost ready...")
        st.session_state.rag_engine = get_cached_rag_engine()
        
        # Engine is ready if we have vectors
        st.session_state.engine_ready = vs_exists and vectors > 0
        
        # Done!
        if st.session_state.engine_ready:
            update_status("Ready!", 1.0, "✅ All systems initialized")
        else:
            update_status("Setup required", 1.0, "⚠️ Run indexing to get started")
        time.sleep(0.5)
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.exception(e)
        status_text.error(f"Initialization error: {e}")
        # Still mark as initialized so user can see the "not ready" warning with instructions
        st.session_state.engine_ready = False
        time.sleep(2)
    
    # Mark as initialized and refresh
    st.session_state.initialized = True
    st.rerun()


def get_rag_engine() -> RAGEngine | None:
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
                    text_preview = src.get("text", "")[:300]
                    
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


def render_chat_history(show_sources: bool):
    """Render the chat history with latency info."""
    for msg in st.session_state.messages:
        render_message(
            msg["role"], 
            msg["content"],
            msg.get("sources"),
            show_sources
        )
        # Show latency for assistant messages
        if msg["role"] == "assistant" and "latency_ms" in msg:
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
        
        # Check index
        try:
            embed_model = get_embed_model()
            manager = VectorStoreManager(embed_model=embed_model)
            index_exists = manager.collection_exists()
        except Exception:
            index_exists = False
        st.checkbox("Index created", value=index_exists, disabled=True)
        
        # Check LLM
        if settings.llm_service == "local":
            llm_ready = is_ollama_available()
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
                start_time = time.time()
                response = engine.query(prompt, similarity_top_k=top_k)
                latency = (time.time() - start_time) * 1000
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "latency_ms": latency,
                })
                
                st.session_state.total_queries += 1
                
                # Display response
                render_message("assistant", response.answer, response.sources, show_sources)
                
                # Show latency in appropriate unit
                if latency >= 1000:
                    st.caption(f"⏱️ {latency/1000:.2f}s")
                else:
                    st.caption(f"⏱️ {latency:.0f}ms")
                
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

