"""
stylometry_utils.py — Utility functions for stylometry analysis.

This module contains utility functions extracted from the original syncom/ingestion.py
to make the stylometry application completely independent of syncom.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
import numpy as np
import pandas as pd

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
#
# Five archetypes — all organization archetypes embed the word "organization":
#
#   individual              — Organization Name field is empty (mandatory rule)
#   organization            — non-empty org, no specific sub-type match (catch-all)
#   government_organization — government agencies
#   advocacy_organization   — associations, coalitions, foundations, etc.
#   academic_organization   — universities, research institutions
#
# Detecting "is this an org?" downstream: "organization" in archetype
# Detecting "is this an individual?":     archetype == "individual"

# Keywords used to sub-classify organizations (applied to org name text only).
# Only consulted when Organization Name is non-empty.
ARCHETYPE_KEYWORDS: dict[str, list[str]] = {
    "government_organization": [
        "department", "agency", "bureau", "administration", "office of", "congressional",
        "county", "city of", "state of", "federal", "municipal", "government",
    ],
    "advocacy_organization": [
        "association", "coalition", "alliance", "network", "federation",
        "foundation", "institute", "center for", "advocates", "council",
        "society", "union",
    ],
    "academic_organization": [
        "university", "college", "school of",
        "research", "lab",
    ],
}

# ── Regulations.gov Category → org archetype mapping ─────────────────────────
# Only consulted when Organization Name is non-empty (org comments only).
# Maps known regulations.gov Category field prefixes/values to org archetypes.
# More specific entries should come first.
CATEGORY_TO_ARCHETYPE: list[tuple[str, str]] = [
    # Government
    ("Federal Government", "government_organization"),
    ("State Government", "government_organization"),
    ("Other Government", "government_organization"),
    ("Government - Federal", "government_organization"),
    ("Government - State", "government_organization"),
    ("Government - Local", "government_organization"),
    ("Government - Other", "government_organization"),
    ("Government", "government_organization"),
    ("Congressional", "government_organization"),
    # Advocacy / associations
    ("Health Care Professional/Association", "advocacy_organization"),
    ("Health Care Provider/Association", "advocacy_organization"),
    ("Health Plan or Association", "advocacy_organization"),
    ("Association", "advocacy_organization"),
    ("Device Association", "advocacy_organization"),
    # Industry / provider org (generic organization)
    ("Health Care Industry", "organization"),
    ("Private Industry", "organization"),
    ("Device Industry", "organization"),
    ("Drug Industry", "organization"),
    ("Hospital", "organization"),
    ("Home Health Facility", "organization"),
    ("Hospice", "organization"),
    ("Ambulatory Surgical Center", "organization"),
    ("Long-term Care", "organization"),
    ("Other Health Care Provider", "organization"),
    # Professionals submitting as individuals — these categories map to
    # individual when org is blank; when org is non-empty, treat as organization.
    ("Physician", "organization"),
    ("Nurse", "organization"),
    ("Pharmacist", "organization"),
    ("Other Health Care Professional", "organization"),
    ("Occupational Therapist", "organization"),
    ("Physical Therapist", "organization"),
    ("Physician Assistant", "organization"),
    ("Radiologist", "organization"),
    ("Social Worker", "organization"),
    ("Attorney/Law Firm", "organization"),
    # Academic
    ("Academic", "academic_organization"),
]


def classify_org_archetype_from_category(category: str) -> str | None:
    """
    Map a regulations.gov Category field value to an org archetype.

    Only called when the Organization Name field is non-empty (i.e. the
    submitter is known to be an org).  Returns the archetype string if a
    match is found, or None if the category is blank or unrecognized
    (caller should fall back to keyword / default logic).
    """
    cat = category.strip()
    if not cat:
        return None
    cat_lower = cat.lower()
    for prefix, archetype in CATEGORY_TO_ARCHETYPE:
        if cat_lower.startswith(prefix.lower()):
            return archetype
    return None


def classify_archetype(org: str, name: str, category: str) -> str:
    """
    Classify a submitter into one of five archetypes.

    **Mandatory rule**: the Organization Name field is the gold standard for
    the individual/organization split.

      - Empty org  →  ``individual``  (no further checks)
      - Non-empty org  →  one of the four org archetypes below

    For non-empty orgs, classification priority:
      1. regulations.gov ``category`` field lookup (org-only archetypes).
      2. Keyword match on the organization name text.
      3. Default catch-all: ``organization``.

    All org archetypes embed the word ``organization`` so downstream code
    can detect any org with: ``"organization" in archetype``.
    """
    # Mandatory rule: empty org = individual (no exceptions)
    if not org.strip():
        return "individual"

    # Non-empty org: try category lookup first (sparse but authoritative)
    archetype = classify_org_archetype_from_category(category)
    if archetype is not None:
        return archetype

    # Keyword match on org name text only
    org_lower = org.lower()
    for kw_archetype, keywords in ARCHETYPE_KEYWORDS.items():
        if any(kw in org_lower for kw in keywords):
            return kw_archetype

    # Default catch-all: unclassified organization
    return "organization"


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


# ── Attachment classification helpers ─────────────────────────────────────────

_classification_cache: dict[str, pd.DataFrame] = {}


def load_attachment_classification(attachments_dir: str) -> pd.DataFrame | None:
    """
    Load and cache the attachment_classification.csv for a given attachments directory.

    Returns a DataFrame with columns including document_id, attachment_filename,
    and ai_label, or None if the file doesn't exist.
    """
    cache_key = str(attachments_dir)
    if cache_key in _classification_cache:
        return _classification_cache[cache_key]

    csv_path = Path(attachments_dir) / "attachment_classification.csv"
    if not csv_path.exists():
        logger.warning(
            f"attachment_classification.csv not found at {csv_path}. "
            "Falling back to reading all .txt attachments."
        )
        _classification_cache[cache_key] = None
        return None

    try:
        df = pd.read_csv(csv_path, dtype=str, low_memory=False).fillna("")
        # Normalise key columns
        df.columns = [c.strip().lower() for c in df.columns]
        logger.info(
            f"Loaded attachment classification: {len(df)} entries from {csv_path}"
        )
        _classification_cache[cache_key] = df
        return df
    except Exception as e:
        logger.warning(f"Failed to load attachment_classification.csv: {e}")
        _classification_cache[cache_key] = None
        return None


def _get_comment_attachment_filenames(
    document_id: str, classification_df: pd.DataFrame
) -> list[str]:
    """
    Return the list of attachment filenames (e.g. 'attachment_1.pdf') that are
    classified as 'comment' for the given document_id.

    Results are sorted by attachment number so attachment_1 comes first.
    """
    mask = (
        (classification_df["document_id"] == document_id)
        & (classification_df["ai_label"].str.strip().str.lower() == "comment")
    )
    rows = classification_df.loc[mask, "attachment_filename"].tolist()

    # Sort by attachment number
    att_re = re.compile(r"attachment_(\d+)", re.IGNORECASE)

    def sort_key(fname: str) -> int:
        m = att_re.search(fname)
        return int(m.group(1)) if m else 999

    return sorted(rows, key=sort_key)


# ── Text extraction from attachments ──────────────────────────────────────────

def get_attachment_text(document_id: str, docket_id: str, attachments_dir: str | None = None) -> str:
    """
    Load pre-converted text from attachments classified as **comments** for a
    given document.

    This function reads the ``attachment_classification.csv`` in the
    *attachments_dir* to decide which attachments are substantive comments
    (``ai_label == "comment"``).  Only those attachments are included.  When
    a single submission has multiple comment-classified attachments their text
    is concatenated (separated by a blank line).

    PDF and DOCX conversion is handled upstream by the text_downloader
    (downloader/text_converter.py), which writes attachment_N.txt files
    alongside the originals.  This function reads those pre-converted files.

    The attachments_dir parameter can override the default path. When provided,
    attachments are looked up at: {attachments_dir}/{document_id}/
    When not provided, the default path is: {docket_id}/comment_attachments/{document_id}/

    Fallback (no classification CSV):
      - If attachment_classification.csv is missing, falls back to using the
        lowest-numbered attachment_N.txt (legacy behaviour).
    """
    if attachments_dir:
        att_base = Path(attachments_dir)
    else:
        att_base = Path(docket_id) / "comment_attachments"

    attachment_dir = att_base / document_id

    if not attachment_dir.exists():
        logger.debug(f"Attachment directory not found: {attachment_dir} (doc_id={document_id}, docket={docket_id})")
        return ""

    # ── Try classification-based selection first ──────────────────────────
    classification_df = load_attachment_classification(str(att_base))

    if classification_df is not None:
        comment_filenames = _get_comment_attachment_filenames(document_id, classification_df)

        if not comment_filenames:
            logger.debug(
                f"No attachments classified as 'comment' for {document_id}. Skipping."
            )
            return ""

        # Read the .txt version of each comment-classified attachment
        collected_texts: list[str] = []
        for fname in comment_filenames:
            # Derive .txt filename from the original (e.g. attachment_1.pdf → attachment_1.txt)
            stem = Path(fname).stem  # e.g. "attachment_1"
            txt_path = attachment_dir / f"{stem}.txt"
            if not txt_path.exists():
                logger.debug(
                    f"Pre-converted text not found for {fname} ({txt_path}). "
                    "Run downloader/text_converter.py first."
                )
                continue
            try:
                text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    logger.info(
                        f"Loaded comment attachment text: {txt_path.name} for {document_id}"
                    )
                    collected_texts.append(text)
            except Exception as e:
                logger.warning(f"Failed to read {txt_path}: {e}")

        if collected_texts:
            return "\n\n".join(collected_texts)
        return ""

    # ── Fallback: no classification CSV → legacy behaviour (lowest N) ─────
    logger.debug(
        f"No classification CSV available; falling back to lowest-N attachment for {document_id}"
    )
    attachment_re = re.compile(r"^attachment_(\d+)$", re.IGNORECASE)

    txt_candidates: list[tuple[int, Path]] = []
    for p in attachment_dir.glob("*.txt"):
        m = attachment_re.match(p.stem)
        if m:
            try:
                txt_candidates.append((int(m.group(1)), p))
            except ValueError:
                pass

    if not txt_candidates:
        logger.debug(
            f"No pre-converted .txt attachment found for {document_id}. "
            "Run downloader/text_converter.py to convert attachments first."
        )
        return ""

    chosen_txt = min(txt_candidates, key=lambda t: t[0])[1]
    try:
        text = chosen_txt.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            logger.info(f"Loaded text from primary attachment {chosen_txt.name} for {document_id}")
            return text
    except Exception as e:
        logger.warning(f"Failed to read {chosen_txt}: {e}")

    return ""
