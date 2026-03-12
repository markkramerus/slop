"""Compatibility wrapper for the attachment AI classifier.

The original task request referred to an "attachment_comment_classifier".
Implementation lives in :mod:`downloader.attachment_ai_classifier`.

This file re-exports the public API so callers can import either module name.
"""

from __future__ import annotations

from .attachment_ai_classifier import (  # noqa: F401
    AttachmentItem,
    ClassifierConfig,
    CSV_COLUMNS,
    PROMPT_VERSION,
    classify_attachment_tree,
    classify_pdf_via_ai,
    iter_pdf_attachments,
    load_classifier_config,
    reparse_csv,
)
