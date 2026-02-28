"""
text_converter.py — Convert downloaded attachments to text files.

This standalone utility converts all PDF and DOCX files in a docket's
comment_attachments directory to corresponding .txt files. This allows for
faster reprocessing and makes text content easily accessible without repeated
conversions.

Usage:
    python downloader/text_converter.py CMS-2025-0050
    python downloader/text_converter.py CMS-2025-0050 --attachments-dir CMS-2025-0050/comment_attachments
    python downloader/text_converter.py CMS-2025-0050 --force  # Reconvert even if .txt exists

This module is also called automatically when using the --convert-text flag
with the downloader:
    python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --convert-text
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

# Text extraction libraries
try:
    from pypdf import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Text extraction functions ─────────────────────────────────────────────────

def extract_text_from_pdf(filepath: Path) -> str:
    """Extract text from a PDF file."""
    if not HAS_PDF:
        logger.warning(f"pypdf not available, skipping PDF extraction: {filepath}")
        return ""

    try:
        reader = PdfReader(str(filepath))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                # Fix: Collapse single newlines within paragraphs into spaces
                # This handles PDFs where each word is on its own line
                text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
                # Collapse multiple spaces
                text = re.sub(r' +', ' ', text)
                # Preserve paragraph breaks (double newlines)
                text = re.sub(r'\n\n+', '\n\n', text)
                text_parts.append(text)
        return "\n".join(text_parts)
    except Exception as e:
        logger.warning(f"Failed to extract text from PDF {filepath}: {e}")
        return ""


def extract_text_from_docx(filepath: Path) -> str:
    """Extract text from a DOCX file."""
    if not HAS_DOCX:
        logger.warning(f"python-docx not available, skipping DOCX extraction: {filepath}")
        return ""

    try:
        doc = Document(str(filepath))
        text_parts = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        return "\n".join(text_parts)
    except Exception as e:
        logger.warning(f"Failed to extract text from DOCX {filepath}: {e}")
        return ""


def extract_text_from_file(filepath: Path) -> str:
    """Extract text from a file based on its extension."""
    suffix = filepath.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(filepath)
    elif suffix in (".docx", ".doc"):
        return extract_text_from_docx(filepath)
    else:
        logger.debug(f"Unsupported file type for text extraction: {filepath}")
        return ""


# ── Docket conversion ─────────────────────────────────────────────────────────

def convert_docket_to_text(
    docket_id: str,
    attachments_dir: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Convert all attachments in a docket to text files.

    This function:
    1. Finds the attachments directory: {docket_id}/comment_attachments/
    2. Iterates through all document subdirectories
    3. For each .pdf or .docx file, creates a corresponding .txt file
    4. Skips conversion if .txt already exists (unless force=True)

    Parameters
    ----------
    docket_id : str
        Docket identifier (e.g., "CMS-2025-0050")
    attachments_dir : str, optional
        Directory containing attachments (default: {docket_id}/comment_attachments/)
    force : bool
        If True, reconvert even when .txt files already exist (default: False)

    Returns
    -------
    dict
        Statistics about the conversion:
        - total_files: Total number of source files found
        - converted: Number of files newly converted
        - skipped: Number of files skipped (already had .txt)
        - failed: Number of conversion failures
        - document_count: Number of document directories processed
    """
    if attachments_dir:
        docket_path = Path(attachments_dir)
    else:
        docket_path = Path(docket_id) / "comment_attachments"

    if not docket_path.exists():
        raise FileNotFoundError(
            f"Attachments directory not found: {docket_path}\n"
            f"Make sure you've downloaded attachments for {docket_id} first."
        )

    logger.info(f"Converting attachments to text for docket: {docket_id}")
    logger.info(f"Attachments directory: {docket_path}")

    stats = {
        "total_files": 0,
        "converted": 0,
        "skipped": 0,
        "failed": 0,
        "document_count": 0,
    }

    # Find all document subdirectories (e.g., CMS-2025-0050-0004)
    document_dirs = sorted([d for d in docket_path.iterdir() if d.is_dir()])

    if not document_dirs:
        logger.warning(f"No document directories found in {docket_path}")
        return stats

    stats["document_count"] = len(document_dirs)
    logger.info(f"Found {len(document_dirs)} document directories")

    # Process each document directory
    for doc_dir in document_dirs:
        doc_id = doc_dir.name

        # Find all convertible files (.pdf, .docx, .doc)
        convertible_files = []
        for pattern in ["*.pdf", "*.docx", "*.doc"]:
            convertible_files.extend(doc_dir.glob(pattern))

        if not convertible_files:
            continue

        stats["total_files"] += len(convertible_files)

        # Convert each file
        for source_file in convertible_files:
            txt_file = source_file.with_suffix(".txt")

            # Skip if .txt already exists and not forcing
            if txt_file.exists() and not force:
                logger.debug(f"  Skipping {source_file.name} (txt exists)")
                stats["skipped"] += 1
                continue

            # Extract text
            logger.info(f"  Converting {doc_id}/{source_file.name}")
            text = extract_text_from_file(source_file)

            if text.strip():
                # Save to .txt file
                try:
                    txt_file.write_text(text, encoding="utf-8")
                    stats["converted"] += 1
                    logger.info(f"    → Created {txt_file.name} ({len(text)} chars)")
                except Exception as e:
                    logger.error(f"    Failed to write {txt_file.name}: {e}")
                    stats["failed"] += 1
            else:
                logger.warning(f"    No text extracted from {source_file.name}")
                stats["failed"] += 1

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Docket: {docket_id}")
    logger.info(f"Documents processed: {stats['document_count']}")
    logger.info(f"Total source files: {stats['total_files']}")
    logger.info(f"  Newly converted: {stats['converted']}")
    logger.info(f"  Skipped (already existed): {stats['skipped']}")
    logger.info(f"  Failed: {stats['failed']}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert docket attachments to text files"
    )
    parser.add_argument(
        "docket_id",
        help="Docket identifier (e.g., CMS-2025-0050)"
    )
    parser.add_argument(
        "--attachments-dir",
        default=None,
        help="Directory containing attachments (default: {docket_id}/comment_attachments/)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconvert even if .txt files already exist"
    )

    args = parser.parse_args()

    try:
        stats = convert_docket_to_text(
            docket_id=args.docket_id,
            attachments_dir=args.attachments_dir,
            force=args.force,
        )

        # Return non-zero exit code if there were failures
        if stats["failed"] > 0:
            logger.warning(f"\nCompleted with {stats['failed']} failures")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
