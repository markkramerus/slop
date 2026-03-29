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


# ── Rule-text helpers ─────────────────────────────────────────────────────────

def _load_text_file(path: Path) -> str | None:
    """Load a text file with encoding fallback.  Returns None on failure."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _find_rule_text(psv_path: Path, rule_arg: str | None) -> str | None:
    """
    Locate and load the rule/RFI text file.

    Resolution order
    ----------------
    1. If ``--rule`` was supplied and points to a file, load that file.
    2. If ``--rule`` points to a directory, load the first ``.txt`` file inside it.
    3. Auto-detect: walk up from the PSV file directory looking for a sibling
       ``rule/`` folder (checks up to 3 levels up), then load the first
       ``.txt`` file found there.
    4. If nothing is found, return ``None`` (classification is skipped).
    """
    # ── Explicit --rule argument ───────────────────────────────────────────
    if rule_arg:
        p = Path(rule_arg)
        if p.is_file():
            return _load_text_file(p)
        if p.is_dir():
            txts = sorted(p.glob("*.txt"))
            if txts:
                return _load_text_file(txts[0])
        print(
            f"[phrase-report] Warning: --rule path not found or empty: {rule_arg}",
            file=sys.stderr,
        )
        return None

    # ── Auto-detect from docket directory structure ────────────────────────
    parent = psv_path.parent
    for _ in range(3):
        rule_dir = parent / "rule"
        if rule_dir.is_dir():
            txts = sorted(rule_dir.glob("*.txt"))
            if txts:
                return _load_text_file(txts[0])
        parent = parent.parent

    return None


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
        "--rule",
        default=None,
        metavar="PATH",
        help=(
            "Path to the rule/RFI text file (or directory containing one). "
            "When provided, phrases are classified as 'slopical' (generic "
            "AI-generation signals) or 'topical' (domain-specific vocabulary) "
            "using TF-IDF scoring against the rule text.  If omitted, "
            "phrase_report auto-detects a 'rule/' folder adjacent to the "
            "input file's docket root."
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

    # ── Step 2: Locate rule text ───────────────────────────────────────────────
    rule_text = _find_rule_text(input_path, getattr(args, "rule", None))

    if not args.quiet:
        print(f"[phrase-report] Input   : {input_path}")
        print(f"[phrase-report] Output  : {output_path}")
        print(f"[phrase-report] N-gram  : {args.min_n}–{args.max_n} words")
        print(f"[phrase-report] Min hits: {args.min_count} comments")
        if rule_text:
            print(f"[phrase-report] Rule    : {len(rule_text):,} chars — TF-IDF classification enabled")
        else:
            print("[phrase-report] Rule    : not found — phrases will not be classified")
        print()

    # ── Step 3: Run phrase check ───────────────────────────────────────────────
    try:
        report_path = run_phrase_check_on_psv(
            psv_path=str(input_path),
            output_path=output_path,
            min_n=args.min_n,
            max_n=args.max_n,
            min_count=args.min_count,
            verbose=not args.quiet,
            rule_text=rule_text,
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
