"""
stylometry_utils.py — Utility functions for stylometry analysis.

This module contains utility functions extracted from the original syncom/ingestion.py
to make the stylometry application completely independent of syncom.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
logger = logging.getLogger(__name__)

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and strip column names for fuzzy matching."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, col_name: str) -> str | None:
    if col_name in df.columns:
        return col_name
    else:
        return None


# ── Archetype classification ──────────────────────────────────────────────────

ARCHETYPE_KEYWORDS: dict[str, list[str]] = {
    "government": [
        "department", "agency", "bureau", "administration", "office of", "congressional",
        "county", "city of", "state of", "federal", "municipal", "government",
    ],
    "advocacy_group": [
        "association", "coalition", "alliance", "network", "federation",
        "foundation", "institute", "center for", "advocates", "council",
        "society", "union",
    ],
    "industry": [
        "industry", "vendor", "inc", "llc", "corp", "ltd", "co.", "group", "solutions",
        "systems", "services", "technologies", "hospital", "health system",
        "medical center", "clinic", "hospice"
    ],
    "academic": [
        "academic", "university", "college", "school of", "professor", "phd",
        "research", "lab",
    ],
    "individual": [
        "consumer", "individual", 
    ],
    "professional": [
        "provider", "lawyer", "doctor", "dr", "nurse", "attorney", "therapist","md",
    ]
}


def classify_archetype(org: str, name: str, category: str) -> str:
    """Heuristically classify a submitter into one of five archetypes."""
    combined = f"{org} {category}".lower()
    for archetype, keywords in ARCHETYPE_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return archetype
    # Short comments with no org → individual
    if name:
        return "individual_consumer"
    else:
        return "unknown"


# ── Linguistic fingerprinting ─────────────────────────────────────────────────

def _sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def fingerprint(text: str) -> dict[str, float]:
    """Compute lightweight linguistic metrics for one comment."""
    words = text.split()
    sentences = _sentences(text) or [""]
    word_lengths = [len(w) for w in words] if words else [0]
    sent_lengths = [len(s.split()) for s in sentences]

    # Simple error proxies
    typo_like = sum(1 for w in words if len(w) > 2 and w == w.lower() and
                    re.search(r'[^a-z\'-]', w)) / max(len(words), 1)
    bullet_ratio = text.count("\n") / max(len(sentences), 1)
    first_person = sum(1 for w in words if w.lower() in
                       {"i", "me", "my", "we", "our", "us"}) / max(len(words), 1)
    citation_like = len(re.findall(r'\b\d{4}\b|\bcfr\b|\busc\b|\bfr\b', text.lower()))

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "mean_sentence_len": float(np.mean(sent_lengths)),
        "std_sentence_len": float(np.std(sent_lengths)),
        "mean_word_len": float(np.mean(word_lengths)),
        "first_person_ratio": first_person,
        "bullet_ratio": bullet_ratio,
        "citation_count": citation_like,
        "typo_proxy": typo_like,
    }


# ── Text extraction from attachments ──────────────────────────────────────────

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
                # Replace single newlines (not followed by another newline) with spaces
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


def get_attachment_text(document_id: str, docket_id: str, attachments_dir: str | None = None) -> str:
    """
    Extract and combine text from all attachments for a given document.
    
    Looks for attachments in: {docket_id}/comment_attachments/{document_id}/*.{pdf,docx}
    
    The attachments_dir parameter can override the default path. When provided,
    attachments are looked up at: {attachments_dir}/{document_id}/
    When not provided, the default path is: {docket_id}/comment_attachments/{document_id}/
    
    If a .txt file exists for a source file, it will be used instead of converting.
    For example, if attachment_1.txt exists, it will be used instead of extracting
    from attachment_1.pdf.
    """
    if attachments_dir:
        attachment_dir = Path(attachments_dir) / document_id
    else:
        attachment_dir = Path(docket_id) / "comment_attachments" / document_id
    
    if not attachment_dir.exists():
        logger.debug(f"Attachment directory not found: {attachment_dir} (doc_id={document_id}, docket={docket_id})")
        return ""
    
    text_parts = []
    attachment_files = sorted(attachment_dir.glob("*"))
    
    # Track which files we've already processed to avoid duplicates
    # (e.g., don't process both attachment_1.pdf and attachment_1.txt)
    processed_basenames = set()
    
    for filepath in attachment_files:
        if not filepath.is_file():
            continue
        
        # Get the base name without extension
        basename = filepath.stem
        
        # Skip if we've already processed this file
        if basename in processed_basenames:
            continue
        
        # Check if a .txt version exists
        txt_file = filepath.with_suffix(".txt")
        
        if txt_file.exists() and txt_file != filepath:
            # Use the pre-converted .txt file
            try:
                text = txt_file.read_text(encoding="utf-8")
                if text.strip():
                    text_parts.append(text)
                    processed_basenames.add(basename)
                    logger.debug(f"Using pre-converted text from {txt_file.name} for {document_id}")
            except Exception as e:
                logger.warning(f"Failed to read {txt_file}: {e}")
                # Fall through to try extracting from source file
        
        # If no .txt file was successfully loaded, extract from source
        if basename not in processed_basenames:
            text = extract_text_from_file(filepath)
            if text.strip():
                text_parts.append(text)
                processed_basenames.add(basename)
    
    if text_parts:
        logger.info(f"Loaded text from {len(text_parts)} attachment(s) for {document_id}")
    
    return "\n\n".join(text_parts)
