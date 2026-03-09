"""
text_converter.py — Convert downloaded attachments to text files.

This standalone utility converts all PDF and DOCX files in a docket's
comment_attachments directory to corresponding .txt files. This allows for
faster reprocessing and makes text content easily accessible without repeated
conversions.

By default, PDFs that appear to be presentation slides (PowerPoint, Keynote,
etc.) are skipped, since their text extraction is typically fragmented and
incoherent.  Detection uses PDF metadata (creator/producer fields) and average
word count per page.  Use --include-presentations to convert them anyway.

Usage:
    python downloader/text_converter.py CMS-2025-0050
    python downloader/text_converter.py CMS-2025-0050 --attachments-dir CMS-2025-0050/comment_attachments
    python downloader/text_converter.py CMS-2025-0050 --force  # Reconvert even if .txt exists
    python downloader/text_converter.py CMS-2025-0050 --include-presentations  # Include slide PDFs

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


# ── Presentation detection constants ─────────────────────────────────────────

# PDF creator/producer strings that indicate presentation software.
# Matched case-insensitively as substrings of the metadata fields.
PRESENTATION_CREATOR_KEYWORDS = [
    "powerpoint",
    "impress",       # LibreOffice Impress
    "keynote",
    "google slides",
    "prezi",
]

# Minimum average words per page for a PDF to be considered narrative text.
# Presentation slides typically average 10–75 words/slide; narrative documents
# typically average 200–500 words/page.
MIN_WORDS_PER_PAGE = 100

# Number of pages to sample when computing the word-density signal.
# Sampling keeps classification fast for large PDFs.
SAMPLE_PAGE_COUNT = 10


# ── PDF readability classifier ────────────────────────────────────────────────

def classify_pdf_readability(filepath: Path) -> dict[str, Any]:
    """
    Classify a PDF as either narrative text or a presentation/graphics-heavy file.

    Uses two signals in priority order:

    1. **Metadata** – If the PDF's Creator or Producer field contains a known
       presentation-software keyword (PowerPoint, Impress, Keynote, etc.) the
       file is immediately classified as a presentation.

    2. **Word density** – The average word count per page across the first
       SAMPLE_PAGE_COUNT pages must reach MIN_WORDS_PER_PAGE.  Slide decks
       typically have far fewer words per page than narrative documents.

    Note: a "short text block" heuristic was intentionally omitted.  pypdf
    extracts text line-by-line from the PDF's internal structure, so even
    dense narrative prose produces many short fragments.  Word density is a
    more reliable signal.

    Parameters
    ----------
    filepath : Path
        Path to the PDF file to classify.

    Returns
    -------
    dict with keys:
        is_narrative (bool)       – True if the PDF looks like readable text.
        reason (str)              – Human-readable reason for the verdict.
        creator (str)             – Value of the PDF Creator metadata field.
        producer (str)            – Value of the PDF Producer metadata field.
        avg_words_per_page (float)– Average word count across sampled pages.
    """
    result: dict[str, Any] = {
        "is_narrative": True,
        "reason": "ok",
        "creator": "",
        "producer": "",
        "avg_words_per_page": 0.0,
    }

    if not HAS_PDF:
        # Can't classify without pypdf; assume narrative so we still attempt
        # extraction (which will also fail, and be caught elsewhere).
        return result

    try:
        reader = PdfReader(str(filepath))

        # ── Signal 1: metadata ────────────────────────────────────────────
        meta = reader.metadata or {}
        creator = str(meta.get("/Creator", "") or "").strip()
        producer = str(meta.get("/Producer", "") or "").strip()
        result["creator"] = creator
        result["producer"] = producer

        combined_meta = (creator + " " + producer).lower()
        for keyword in PRESENTATION_CREATOR_KEYWORDS:
            if keyword in combined_meta:
                result["is_narrative"] = False
                result["reason"] = f"presentation_metadata (matched '{keyword}' in creator/producer)"
                return result

        # ── Signal 2: sample page word density ───────────────────────────
        pages = reader.pages
        sample = pages[: min(SAMPLE_PAGE_COUNT, len(pages))]

        total_words = 0
        for page in sample:
            raw = page.extract_text() or ""
            total_words += len(raw.split())

        num_pages = len(sample)
        avg_words = total_words / num_pages if num_pages else 0.0
        result["avg_words_per_page"] = round(avg_words, 1)

        if avg_words < MIN_WORDS_PER_PAGE:
            result["is_narrative"] = False
            result["reason"] = (
                f"low_density ({avg_words:.0f} words/page avg, "
                f"threshold {MIN_WORDS_PER_PAGE})"
            )
            return result

    except Exception as e:
        logger.debug(f"classify_pdf_readability error for {filepath}: {e}")
        # On any error, fall through and let the normal extractor handle it.

    return result


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
    skip_presentations: bool = True,
) -> dict[str, Any]:
    """
    Convert all attachments in a docket to text files.

    This function:
    1. Finds the attachments directory: {docket_id}/comment_attachments/
    2. Iterates through all document subdirectories
    3. For each .pdf or .docx file, creates a corresponding .txt file
    4. Skips conversion if .txt already exists (unless force=True)
    5. By default, skips PDF files that appear to be presentations/slides

    Parameters
    ----------
    docket_id : str
        Docket identifier (e.g., "CMS-2025-0050")
    attachments_dir : str, optional
        Directory containing attachments (default: {docket_id}/comment_attachments/)
    force : bool
        If True, reconvert even when .txt files already exist (default: False)
    skip_presentations : bool
        If True (default), skip PDFs that appear to be presentation slides
        or other graphics-heavy documents with little narrative text.

    Returns
    -------
    dict
        Statistics about the conversion:
        - total_files: Total number of source files found
        - converted: Number of files newly converted
        - skipped: Number of files skipped (already had .txt)
        - presentations_skipped: PDFs skipped due to presentation detection
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
    if skip_presentations:
        logger.info(
            "Presentation filtering: ON  "
            "(use --include-presentations to convert slide-style PDFs)"
        )
    else:
        logger.info("Presentation filtering: OFF  (all PDFs will be converted)")

    stats = {
        "total_files": 0,
        "converted": 0,
        "skipped": 0,
        "presentations_skipped": 0,
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
                logger.info(f"  Skipping {doc_id}/{source_file.name} (already converted)")
                stats["skipped"] += 1
                continue

            # For PDFs, run the readability classifier before extracting
            if source_file.suffix.lower() == ".pdf" and skip_presentations:
                classification = classify_pdf_readability(source_file)
                if not classification["is_narrative"]:
                    logger.warning(
                        f"  SKIPPED (presentation) {doc_id}/{source_file.name}"
                        f" — {classification['reason']}"
                        + (
                            f" | creator: '{classification['creator']}'"
                            if classification["creator"]
                            else ""
                        )
                    )
                    stats["presentations_skipped"] += 1
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
    logger.info(f"  Newly converted:          {stats['converted']}")
    logger.info(f"  Presentations skipped:    {stats['presentations_skipped']}")
    logger.info(f"  Skipped (already existed):{stats['skipped']}")
    logger.info(f"  Failed:                   {stats['failed']}")

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
    parser.add_argument(
        "--include-presentations",
        action="store_true",
        help=(
            "Convert PDFs that appear to be presentation slides or other "
            "graphics-heavy documents (skipped by default because their "
            "text extraction is typically fragmented and incoherent)"
        ),
    )

    args = parser.parse_args()

    try:
        stats = convert_docket_to_text(
            docket_id=args.docket_id,
            attachments_dir=args.attachments_dir,
            force=args.force,
            skip_presentations=not args.include_presentations,
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
