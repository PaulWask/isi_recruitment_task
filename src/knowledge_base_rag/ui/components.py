"""UI components for the Knowledge Base RAG application.

Reusable components that load HTML from template files.
Each component returns a string that can be rendered with st.markdown().

Template files are located in: ui/static/templates/
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional
import html


# =============================================================================
# Template Configuration
# =============================================================================
_TEMPLATES_DIR = Path(__file__).parent / "static" / "templates"


@lru_cache(maxsize=10)
def _load_template(name: str) -> str:
    """Load an HTML template file (cached).
    
    Args:
        name: Template filename (e.g., 'source_card.html')
        
    Returns:
        Template content as string.
        
    Raises:
        FileNotFoundError: If template doesn't exist.
    """
    template_path = _TEMPLATES_DIR / name
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


# =============================================================================
# Utility Functions
# =============================================================================
def escape_html(text: str) -> str:
    """Safely escape HTML entities in user content."""
    return html.escape(str(text))


# =============================================================================
# UI Components
# =============================================================================
def render_source_card(
    score: float,
    file_name: str,
    source_path: str,
    text_preview: str,
    page_label: Optional[str] = None,
    file_type: Optional[str] = None,
) -> str:
    """Render a source citation card.
    
    Args:
        score: Relevance score (0-1)
        file_name: Display name of the file
        source_path: Full path to the source file
        text_preview: Preview of the matched text
        page_label: Optional page number
        file_type: Optional file type (PDF, DOCX, etc.)
    
    Returns:
        HTML string for the source card
    """
    # Determine score class
    if score >= 0.7:
        score_class = "score-high"
    elif score >= 0.4:
        score_class = "score-medium"
    else:
        score_class = "score-low"
    
    # Build location string
    location_parts = []
    if page_label:
        location_parts.append(f"Page {escape_html(str(page_label))}")
    if file_type:
        location_parts.append(escape_html(file_type.upper()))
    location_str = " • ".join(location_parts)
    
    # Only show path details if it adds information (path differs from filename)
    escaped_path = escape_html(source_path)
    escaped_filename = escape_html(file_name)
    
    # Show path only if it contains more than just the filename
    if escaped_path and escaped_path != escaped_filename and not escaped_path.endswith(f"/{escaped_filename}"):
        path_section = f'''<details>
        <summary>📁 Show full path</summary>
        <div class="source-path">{escaped_path}</div>
    </details>'''
    else:
        path_section = ""
    
    # Load and fill template
    template = _load_template("source_card.html")
    return template.format(
        score_percent=f"{score:.0%}",
        score_class=score_class,
        location_str=location_str,
        file_name=escaped_filename,
        text_preview=escape_html(text_preview),
        path_section=path_section,
    )


def render_user_message(content: str) -> str:
    """Render a user chat message.
    
    Args:
        content: Message content
    
    Returns:
        HTML string for the user message
    """
    template = _load_template("user_message.html")
    return template.format(content=escape_html(content))


def render_assistant_message(content: str) -> str:
    """Render an assistant chat message.
    
    Args:
        content: Message content (may contain markdown)
    
    Returns:
        HTML string for the assistant message
    """
    template = _load_template("assistant_message.html")
    # Note: We don't escape here as content may contain formatted text
    return template.format(content=content)


def render_header() -> str:
    """Render the application header.
    
    Returns:
        HTML string for the header
    """
    return _load_template("header.html")


def render_not_ready_warning(indexing_command: str = "uv run python scripts/index_documents.py") -> str:
    """Render warning when system is not ready.
    
    Args:
        indexing_command: Command to run for indexing
    
    Returns:
        HTML string for the warning
    """
    template = _load_template("warning.html")
    return template.format(command=escape_html(indexing_command))


def render_status_indicator(is_online: bool, label: str, details: str = "") -> str:
    """Render a status indicator with dot.
    
    Args:
        is_online: Whether the service is online
        label: Status label
        details: Optional details text
    
    Returns:
        HTML string for the status indicator
    """
    status_class = "online" if is_online else "offline"
    details_html = f'<span class="status-details"> — {escape_html(details)}</span>' if details else ""
    
    template = _load_template("status_indicator.html")
    return template.format(
        status_class=status_class,
        label=escape_html(label),
        details_html=details_html,
    )
