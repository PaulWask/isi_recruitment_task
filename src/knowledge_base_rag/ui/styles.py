"""CSS styles loader for the Knowledge Base RAG application.

Loads CSS from external file for proper separation of concerns.
The actual styles are in: ui/static/styles.css
"""

from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================
_STATIC_DIR = Path(__file__).parent / "static"
_CSS_FILE = _STATIC_DIR / "styles.css"


# =============================================================================
# CSS Loader
# =============================================================================
def load_css() -> str:
    """Load CSS from external file.
    
    Returns:
        CSS wrapped in <style> tags for Streamlit injection.
        
    Raises:
        FileNotFoundError: If CSS file doesn't exist.
    """
    if not _CSS_FILE.exists():
        raise FileNotFoundError(f"CSS file not found: {_CSS_FILE}")
    
    css_content = _CSS_FILE.read_text(encoding="utf-8")
    return f"<style>\n{css_content}\n</style>"


# =============================================================================
# Color Palette (for programmatic use in Python)
# =============================================================================
# These match the CSS variables in styles.css
COLORS = {
    # Primary colors
    "primary": "#2c3e50",
    "primary_light": "#3498db",
    "primary_dark": "#1a252f",
    
    # Background colors
    "bg_user": "#e3f2fd",
    "bg_assistant": "#f8f9fa",
    "bg_card": "#ffffff",
    "bg_warning": "#fff3cd",
    
    # Border colors
    "border_light": "#dee2e6",
    "border_medium": "#adb5bd",
    
    # Text colors
    "text_primary": "#212529",
    "text_secondary": "#6c757d",
    "text_muted": "#495057",
    
    # Status colors
    "score_high": "#28a745",
    "score_medium": "#ffc107",
    "score_low": "#dc3545",
}
