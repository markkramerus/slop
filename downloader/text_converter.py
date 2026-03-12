"""
text_converter.py — Convert downloaded comment attachments to text files.

This utility reads a ``attachment_classification.csv`` produced by the AI
classifier (``classify_attachments_ai.py``) and converts **only** the PDF
and DOCX files that were classified as ``comment`` to corresponding ``.txt``
files.

Workflow
--------
1. ``download_attachments.py`` downloads all PDFs for a docket.
2. ``classify_attachments_ai.py`` classifies each PDF → ``attachment_classification.csv``.
3. **This script** reads the CSV and converts only *comment* PDFs to text.

Usage:
    python downloader/text_converter.py CMS-2025-0050
    python downloader/text_converter.py CMS-2025-0050 --classification-csv CMS-2025-0050/attachment_classification.csv
    python downloader/text_converter.py CMS-2025-0050 --force  # Reconvert even if .txt exists

This module is also called automatically when using the --convert-text flag
with the downloader:
    python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --convert-text
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

# Text extraction libraries
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    from pdf2image import convert_from_path
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Classification CSV reader ─────────────────────────────────────────────────

def load_comment_paths_from_csv(csv_path: Path, base_dir: Path | None = None) -> set[Path]:
    """Read a attachment_classification.csv and return the set of attachment paths
    whose ``ai_label`` is ``comment``.

    Parameters
    ----------
    csv_path : Path
        Path to the classification CSV file.
    base_dir : Path, optional
        Base directory to resolve relative attachment paths against.
        If not provided, paths in the CSV are resolved relative to the
        current working directory.  Typically this should be the
        ``comment_attachments`` directory so that a CSV entry like
        ``CMS-2025-0050-0004/attachment_1.pdf`` resolves correctly.

    Returns a set of :class:`Path` objects (resolved to absolute paths for
    reliable comparison).
    """
    comment_paths: set[Path] = set()
    if not csv_path.exists():
        logger.warning(f"Classification CSV not found: {csv_path}")
        return comment_paths

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("ai_label") or "").strip().lower()
            att_path = (row.get("attachment_path") or "").strip()
            if label == "comment" and att_path:
                p = Path(att_path)
                if base_dir and not p.is_absolute():
                    p = base_dir / p
                comment_paths.add(p.resolve())

    logger.info(f"Loaded {len(comment_paths)} comment paths from {csv_path}")
    return comment_paths


# ── Text extraction functions ─────────────────────────────────────────────────

def _is_garbage_text(text: str) -> bool:
    """Return True if extracted PDF text looks like encoding garbage.

    Some PDFs use custom font encodings that defeat text-based extractors,
    producing output full of ``(cid:N)`` tokens or text where most characters
    are non-alphabetic.  This check catches both patterns.
    """
    if not text or len(text.strip()) < 50:
        return True  # too little text to judge

    # Explicit (cid:...) markers left by PDF text extractors
    cid_count = text.count("(cid:")
    if cid_count > 10:
        return True

    # Ratio of normal letters (a-z, A-Z, spaces) to total characters.
    # Readable English text is typically >70 %; garbage is usually <40 %.
    alpha_space = sum(1 for ch in text if ch.isalpha() or ch == " ")
    ratio = alpha_space / len(text)
    return ratio < 0.40


def _extract_pdf_via_ocr(filepath: Path) -> str:
    """Extract text from a PDF using OCR (pdf2image + pytesseract).

    This is slower than native text extraction but handles PDFs with
    custom font encodings that defeat pdfplumber and similar libraries.
    """
    if not HAS_OCR:
        logger.warning(
            f"OCR libraries (pdf2image, pytesseract) not available; "
            f"cannot OCR-fallback for {filepath}"
        )
        return ""

    try:
        images = convert_from_path(str(filepath), dpi=300)
        text_parts = []
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img)
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        logger.warning(f"OCR extraction failed for {filepath}: {e}")
        return ""


def extract_text_from_pdf(filepath: Path) -> str:
    """Extract text from a PDF file.

    Uses pdfplumber for text extraction (handles character spacing well).
    If the result looks like encoding garbage, falls back to OCR via
    pdf2image + pytesseract.
    """
    text = ""

    # ── Attempt 1: pdfplumber (fast, good character grouping) ────────────
    if HAS_PDF:
        try:
            with pdfplumber.open(str(filepath)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                text = "\n\n".join(text_parts)
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed for {filepath}: {e}")
            text = ""

    # ── Check quality; OCR fallback if garbage ───────────────────────────
    if _is_garbage_text(text):
        if text:
            logger.info(f"    pdfplumber output looks like garbage, trying OCR fallback...")
        text = _extract_pdf_via_ocr(filepath)

    return text


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
    classification_csv: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Convert comment attachments in a docket to text files.

    Reads ``attachment_classification.csv`` to determine which PDFs the AI
    classified as *comment*, then converts only those files.

    Parameters
    ----------
    docket_id : str
        Docket identifier (e.g., "CMS-2025-0050").
    attachments_dir : str, optional
        Directory containing attachments
        (default: ``{docket_id}/comment_attachments/``).
    classification_csv : str, optional
        Path to the AI classification CSV
        (default: ``{docket_id}/comment_attachments/attachment_classification.csv``).
    force : bool
        If True, reconvert even when .txt files already exist (default: False).

    Returns
    -------
    dict
        Statistics about the conversion:
        - total_comment_files: PDFs classified as comment in the CSV
        - converted: Number of files newly converted
        - skipped: Number of files skipped (already had .txt)
        - failed: Number of conversion failures
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

    if classification_csv:
        csv_path = Path(classification_csv)
    else:
        csv_path = docket_path / "attachment_classification.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Classification CSV not found: {csv_path}\n"
            f"Run the AI classifier first:\n"
            f"  python downloader/classify_attachments_ai.py "
            f"{docket_path} --output {csv_path}"
        )

    logger.info(f"Converting comment attachments to text for docket: {docket_id}")
    logger.info(f"Attachments directory: {docket_path}")
    logger.info(f"Classification CSV: {csv_path}")

    # Load the set of PDF paths classified as "comment".
    # Pass docket_path as base_dir so relative paths in the CSV
    # (e.g. "CMS-2025-0050-0004/attachment_1.pdf") resolve correctly
    # against the comment_attachments directory.
    comment_paths = load_comment_paths_from_csv(csv_path, base_dir=docket_path)

    stats: dict[str, Any] = {
        "total_comment_files": len(comment_paths),
        "converted": 0,
        "skipped": 0,
        "failed": 0,
    }

    if not comment_paths:
        logger.warning("No files classified as 'comment' — nothing to convert.")
        return stats

    # Process each comment file.
    for source_path in sorted(comment_paths):
        source_file = Path(source_path)

        if not source_file.exists():
            logger.warning(f"  File listed in CSV not found on disk: {source_file}")
            stats["failed"] += 1
            continue

        txt_file = source_file.with_suffix(".txt")

        # If .txt already exists and we're not forcing, skip.
        if txt_file.exists() and not force:
            logger.info(f"  Skipping {source_file.name} (already converted)")
            stats["skipped"] += 1
            continue

        # Extract text.
        logger.info(f"  Converting {source_file}")
        text = extract_text_from_file(source_file)

        if text.strip():
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

    # Print summary.
    logger.info("\n" + "=" * 60)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Docket: {docket_id}")
    logger.info(f"Comment files in CSV:       {stats['total_comment_files']}")
    logger.info(f"  Newly converted:          {stats['converted']}")
    logger.info(f"  Skipped (already existed): {stats['skipped']}")
    logger.info(f"  Failed:                   {stats['failed']}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert comment-classified docket attachments to text files"
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
        "--classification-csv",
        default=None,
        help=(
            "Path to the AI classification CSV "
            "(default: {docket_id}/attachment_classification.csv)"
        ),
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
            classification_csv=args.classification_csv,
            force=args.force,
        )

        # Return non-zero exit code if there were failures.
        if stats["failed"] > 0:
            logger.warning(f"\nCompleted with {stats['failed']} failures")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
