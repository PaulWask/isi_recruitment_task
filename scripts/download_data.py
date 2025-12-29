#!/usr/bin/env python3
"""Download domain data from public URL.

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --url <custom-url>
"""

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import httpx

from knowledge_base_rag.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default public URL for domain data
DEFAULT_DATA_URL = "https://isi-ml-public.s3.us-east-1.amazonaws.com/domaindata.zip"


def download_and_extract(
    url: str,
    output_dir: Path,
) -> Path:
    """Download and extract data from URL.

    Args:
        url: URL to download from
        output_dir: Directory to extract files to

    Returns:
        Path to the extracted directory
    """
    logger.info(f"Downloading from: {url}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "domaindata.zip"

    # Download with progress
    with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        logger.info(f"File size: {total_size / (1024*1024):.1f} MB")

        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\rDownloading: {pct:.1f}% ({downloaded // (1024*1024)} MB)", end="", flush=True)

        print()  # New line after progress

    logger.info(f"Downloaded to: {zip_path}")

    # Extract zip file
    logger.info(f"Extracting to: {output_dir}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()
        total_files = len(file_list)

        for i, file in enumerate(file_list, 1):
            zip_ref.extract(file, output_dir)
            if i % 100 == 0 or i == total_files:
                print(f"\rExtracting: {i}/{total_files}", end="", flush=True)

        print()  # New line

    logger.info(f"Extracted {total_files} files")

    # Clean up zip file
    zip_path.unlink()
    logger.info("Cleaned up zip file")

    return output_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download domain data for the Knowledge Base RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    # Download from default URL
    uv run python scripts/download_data.py

    # Download from custom URL
    uv run python scripts/download_data.py --url https://example.com/data.zip

Default URL:
    {DEFAULT_DATA_URL}
        """,
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_DATA_URL,
        help="URL to download data from",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.knowledge_base_dir,
        help="Output directory (default: ./domaindata)",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip download if output directory already has files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if files exist",
    )

    args = parser.parse_args()

    # Check if data already exists
    if not args.force and args.output.exists():
        existing_files = list(args.output.iterdir())
        if existing_files:
            if args.skip_if_exists:
                logger.info(f"Data already exists at {args.output} ({len(existing_files)} items), skipping")
                return 0
            else:
                logger.warning(f"Data already exists at {args.output} ({len(existing_files)} items)")
                logger.warning("Use --force to overwrite or --skip-if-exists to skip")
                return 0

    try:
        download_and_extract(
            url=args.url,
            output_dir=args.output,
        )
        logger.info("✅ Download complete!")
        logger.info(f"Domain data is now available at: {args.output}")
        return 0

    except httpx.HTTPError as e:
        logger.error(f"HTTP error: {e}")
        return 1
    except zipfile.BadZipFile:
        logger.error("Downloaded file is not a valid zip file")
        return 1
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
