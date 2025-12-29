"""Streamlit Web Application for Knowledge Base RAG.

A professional chat interface for querying the domain knowledge base.
Features:
- Chat-style Q&A interface
- Source attribution with expandable details
- Confidence indicators
- Session history
- System status dashboard

Run with:
    uv run streamlit run src/knowledge_base_rag/app.py
    # or
    uv run kb-app
"""

import logging
import time
from datetime import datetime

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
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
    
    if "engine_ready" not in st.session_state:
        st.session_state.engine_ready = False
    
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0


def get_rag_engine() -> RAGEngine:
    """Get or create RAG engine (cached in session)."""
    if st.session_state.rag_engine is None:
        try:
            st.session_state.rag_engine = RAGEngine()
            st.session_state.engine_ready = st.session_state.rag_engine.is_ready()
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            st.session_state.engine_ready = False
    
    return st.session_state.rag_engine


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render the sidebar with system status and settings."""
    with st.sidebar:
        st.markdown("## ⚙️ System Status")
        
        # Check services
        col1, col2 = st.columns(2)
        
        with col1:
            # Vector store status
            try:
                embed_model = get_embed_model()
                manager = VectorStoreManager(embed_model=embed_model)
                vs_healthy = manager.health_check()
                vs_stats = manager.get_stats() if vs_healthy else {}
            except Exception:
                vs_healthy = False
                vs_stats = {}
            
            if vs_healthy:
                st.markdown("🟢 **Vector Store**")
                vectors = vs_stats.get("vectors_count", 0)
                st.caption(f"{vectors:,} vectors")
            else:
                st.markdown("🔴 **Vector Store**")
                st.caption("Not indexed")
        
        with col2:
            # LLM status
            if settings.llm_service == "local":
                llm_available = is_ollama_available()
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
        st.metric("Queries", st.session_state.total_queries)
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
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
    """Render the chat history."""
    for msg in st.session_state.messages:
        render_message(
            msg["role"], 
            msg["content"],
            msg.get("sources"),
            show_sources
        )


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
    
    # Sidebar
    top_k, show_sources = render_sidebar()
    
    # Header
    render_header()
    
    # Get RAG engine
    engine = get_rag_engine()
    
    # Check if system is ready
    if not st.session_state.engine_ready:
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
                
                # Show latency
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


if __name__ == "__main__":
    main()

