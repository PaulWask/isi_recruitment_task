"""UI components for the Knowledge Base RAG application.

A professional, modular UI system with separation of concerns:

Structure:
    ui/
    ├── __init__.py              # This file - exports all UI components
    ├── styles.py                # CSS loader and color constants
    ├── components.py            # HTML component rendering functions
    └── static/
        ├── styles.css           # External CSS stylesheet (362 lines)
        └── templates/           # HTML template files
            ├── source_card.html
            ├── user_message.html
            ├── assistant_message.html
            ├── header.html
            ├── warning.html
            └── status_indicator.html

Usage:
    from knowledge_base_rag.ui import load_css, render_source_card
    
    # Load CSS once at app start
    st.markdown(load_css(), unsafe_allow_html=True)
    
    # Render components
    st.markdown(render_source_card(...), unsafe_allow_html=True)
"""

from knowledge_base_rag.ui.styles import load_css, COLORS
from knowledge_base_rag.ui.components import (
    render_source_card,
    render_user_message,
    render_assistant_message,
    render_header,
    render_not_ready_warning,
    render_status_indicator,
)

__all__ = [
    # Styles
    "load_css",
    "COLORS",
    # Components
    "render_source_card",
    "render_user_message",
    "render_assistant_message",
    "render_header",
    "render_not_ready_warning",
    "render_status_indicator",
]
