"""CLI entry points for the Knowledge Base RAG system.

These functions are registered as console scripts in pyproject.toml:
- kb-index: Run the document indexing pipeline
- kb-app: Start the Streamlit web application
"""

import subprocess
import sys
from pathlib import Path


def index_main():
    """Entry point for kb-index command.
    
    Runs the indexing script to process documents and create the vector index.
    
    Usage:
        uv run kb-index
        uv run kb-index --force
        uv run kb-index --dry-run
    """
    script_path = Path(__file__).parent.parent.parent / "scripts" / "index_documents.py"
    
    if not script_path.exists():
        print(f"Error: Indexing script not found at {script_path}")
        sys.exit(1)
    
    # Pass all arguments to the script
    result = subprocess.run(
        [sys.executable, str(script_path)] + sys.argv[1:],
        cwd=Path(__file__).parent.parent.parent,
    )
    sys.exit(result.returncode)


def app_main():
    """Entry point for kb-app command.
    
    Starts the Streamlit web application for querying the knowledge base.
    
    Usage:
        uv run kb-app
        uv run kb-app --server.port 8080
    """
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        print("The app will be created in Commit 11: Streamlit UI")
        sys.exit(1)
    
    # Run streamlit with the app
    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)] + sys.argv[1:],
        cwd=Path(__file__).parent.parent.parent,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    # Default to showing help
    print("Knowledge Base RAG CLI")
    print()
    print("Available commands:")
    print("  kb-index  - Index documents into vector store")
    print("  kb-app    - Start the web application")
    print()
    print("Run 'uv run kb-index --help' for more information")


