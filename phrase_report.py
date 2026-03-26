"""
phrase_report.py — Standalone phrasal-repetition analyser.

Reads a ♔-delimited PSV file (or a synthetic .txt file, which is first
converted to PSV format), scans every comment for repeated distinctive
n-gram phrases using the syncom phrase-check engine, and writes a
human-readable Markdown report.

This is purely an analysis tool — no rewriting, no LLM calls.  It runs
entirely offline and is fast even for thousands of comments.

Usage
-----
    python phrase_report.py <input_file>             # .psv or .txt
    python phrase_report.py <input_file> -o out.md
    python phrase_report.py <input_file> --min-count 3

Output
------
  <input_stem>_phrase_report.md   — Markdown table of repeated phrases

If the input is a synthetic .txt file it is translated to PSV first via the
same function used by the shuffler pipeline.  The intermediate .psv file is
written next to the original .txt (e.g. foo.txt → foo.psv) and kept.

The phrase-check engine uses two passes:
  1. N-gram analysis (default 4–8 words): flags phrases shared by
     ≥ min_count comments.
  2. Watch-list scan: flags known short LLM-favourite anchor phrases
     (e.g. "last spring", "who is responsible") that are too short for
     the n-gram pass.

No environment variables are required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from shuffler.translate_to_psv_format import translate_synthetic_to_psv
from syncom.phrase_check import run_phrase_check_on_psv


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phrase_report",
        description=(
            "Scan a PSV file of comments for repeated distinctive phrases "
            "and write a Markdown report."
        ),
    )
    parser.add_argument(
        "input_file",
        help=(
            "Path to a ♔-delimited .psv file, or a synthetic .txt file "
            "(which will be converted to .psv automatically)."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Output Markdown report path.  Defaults to "
            "<input_stem>_phrase_report.md in the same directory as the "
            "input file."
        ),
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=4,
        metavar="N",
        help="Minimum n-gram length to consider (default: 4).",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=8,
        metavar="N",
        help="Maximum n-gram length to consider (default: 8).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Minimum number of comments a phrase must appear in to be "
            "reported (default: 2).  Use 3+ for a shorter, more actionable "
            "report on large batches."
        ),
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # ── Step 0: Convert .txt → .psv if needed ─────────────────────────────────
    original_input_path = input_path
    if input_path.suffix.lower() == ".txt":
        psv_path = input_path.with_suffix(".psv")
        if not args.quiet:
            print(f"[phrase-report] Translating synthetic .txt → .psv …")
            print(f"                Input  : {input_path}")
            print(f"                Output : {psv_path}")
        n = translate_synthetic_to_psv(str(input_path), str(psv_path))
        if not args.quiet:
            print(f"[phrase-report] Translation complete — {n} records written.\n")
        input_path = psv_path
    elif input_path.suffix.lower() != ".psv":
        print(
            f"Warning: unexpected file extension '{input_path.suffix}'. "
            "Expected .psv or .txt.  Attempting to read as PSV.",
            file=sys.stderr,
        )

    # ── Step 1: Determine output path ─────────────────────────────────────────
    if args.output:
        output_path = str(Path(args.output))
    else:
        # Base output on the *original* input file (before any .txt→.psv step)
        stem = original_input_path.stem
        output_path = str(original_input_path.with_name(f"{stem}_phrase_report.md"))

    if not args.quiet:
        print(f"[phrase-report] Input   : {input_path}")
        print(f"[phrase-report] Output  : {output_path}")
        print(f"[phrase-report] N-gram  : {args.min_n}–{args.max_n} words")
        print(f"[phrase-report] Min hits: {args.min_count} comments")
        print()

    # ── Step 2: Run phrase check ───────────────────────────────────────────────
    try:
        report_path = run_phrase_check_on_psv(
            psv_path=str(input_path),
            output_path=output_path,
            min_n=args.min_n,
            max_n=args.max_n,
            min_count=args.min_count,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"Error during phrase check: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not args.quiet:
        print(f"\n[phrase-report] Report written → {report_path}")


if __name__ == "__main__":
    main()
