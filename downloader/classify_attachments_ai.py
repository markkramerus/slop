#!/usr/bin/env python3
"""classify_attachments_ai.py

Walk a comment_attachments directory tree and classify each PDF attachment as
"comment" vs "not_comment" using an AI endpoint.

The AI endpoint is configured by environment variables (loaded from .env if
present):
  - SLOP_CLASSIFER_API_BASE_URL
  - SLOP_CLASSIFER_API_KEY
  - SLOP_CLASSIFER_MODEL

Output is a CSV (default name: attachment_classification.csv) with one row per
attachment containing the AI classification result.

Example:
  python downloader/classify_attachments_ai.py CMS-2025-0050/comment_attachments \
    --output CMS-2025-0050/attachment_classification.csv \
    --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AI-classify PDFs in a comment_attachments tree as comment vs not-comment",
    )
    p.add_argument(
        "attachments_root",
        help=(
            "Root directory containing per-document subdirectories of PDFs "
            "(e.g. CMS-2025-0050/comment_attachments)."
        ),
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="CSV_PATH",
        help="Output CSV path (will be created or appended).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Reclassify even if attachment_path already exists in the CSV.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only classify N attachments (debug).",
    )
    p.add_argument(
        "--request-format",
        default="content_parts",
        choices=["content_parts", "attachments_field", "multipart_form"],
        help=(
            "How to send the PDF to the endpoint. "
            "Default uses OpenAI-style messages[].content parts with rendered page images. "
            "Use attachments_field or multipart_form if your gateway expects a different format."
        ),
    )
    p.add_argument(
        "--max-mb",
        type=float,
        default=20.0,
        metavar="MB",
        help="Skip AI classification for PDFs larger than this many MB (default 20).",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Sleep between requests (rate limiting).",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=2,
        metavar="N",
        help=(
            "When using content_parts, render the first N PDF pages as images and send those "
            "(default 2)."
        ),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        metavar="DPI",
        help="DPI for PDF->image rendering when using content_parts (default 150).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=10,
        metavar="N",
        help="If tqdm is unavailable, print a status line every N PDFs (default 10).",
    )
    p.add_argument(
        "--reparse",
        action="store_true",
        help=(
            "Instead of classifying new PDFs, re-read the --output CSV and "
            "re-parse all ai_raw values with the improved parser.  Recovers "
            "label/confidence/doc_type/rationale from previously-uncertain rows "
            "without re-calling the AI.  Requires --output to point to an "
            "existing CSV."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out = Path(args.output)

    # ── Reparse mode: re-read existing CSV, no AI calls ────────────────
    if args.reparse:
        try:
            from downloader.attachment_ai_classifier import reparse_csv
        except Exception:
            from attachment_ai_classifier import reparse_csv

        stats = reparse_csv(
            input_csv=out,
            output_csv=out,
            verbose=not args.quiet,
        )
        print("\nReparse complete:")
        print(json.dumps(stats, indent=2))
        return 0

    # ── Normal classification mode ─────────────────────────────────────
    # Local import so `--help` works even if deps aren't installed.
    try:
        from downloader.attachment_ai_classifier import classify_attachment_tree
    except Exception:
        # Support running as `python downloader/classify_attachments_ai.py ...`
        from attachment_ai_classifier import classify_attachment_tree

    stats = classify_attachment_tree(
        attachments_root=args.attachments_root,
        output_csv=out,
        force=args.force,
        limit=args.limit,
        request_format=args.request_format,
        max_mb=args.max_mb,
        sleep_s=args.sleep,
        max_pages=args.max_pages,
        dpi=args.dpi,
        verbose=not args.quiet,
        progress_every=args.progress_every,
    )

    print("\nClassification complete:")
    print(json.dumps(stats, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
