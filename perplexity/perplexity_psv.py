"""
perplexity_psv.py — Batch perplexity scorer for PSV comment files.

Reads a ♔-delimited PSV file (or a synthetic .txt file, which is first
converted to PSV format — same behaviour as judge.py), calls
get_sentence_perplexities() for every comment, and writes a CSV with the
median and standard deviation of per-sentence perplexity scores.

Usage
-----
    python perplexity/perplexity_psv.py <input_file>  --api-key YOUR_KEY
    python perplexity/perplexity_psv.py <input_file>  -o scores.csv
    python perplexity/perplexity_psv.py <input_file>  --model devstral --parallel 10

Output
------
  <input_stem>_perplexity.csv   with columns:
      UID               — Document ID from the PSV
      median_perplexity — median of per-sentence perplexity scores
      stdev_perplexity  — sample standard deviation (empty if < 2 sentences scored)

Environment variables
---------------------
  MITRE_API_KEY    — API key for the MITRE LLM endpoint (alternative to --api-key)

Dependencies
------------
  pip install httpx python-dotenv
  (perplexity.py must be importable — run from the repo root, or add it to PYTHONPATH)
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure both the repo root and this script's directory are on sys.path.
#
# When invoked as `python perplexity/perplexity_psv.py` Python places the
# script's own directory (perplexity/) at sys.path[0], which means neither
# the repo root (needed for `shuffler`) nor the perplexity folder itself
# (needed for `perplexity.py`) is guaranteed to be present.
_HERE = Path(__file__).parent          # …/perplexity/
_ROOT = _HERE.parent                   # …/slop-github/  (repo root)

for _p in (_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Import PSV reader from the shuffler package (same as judge.py)
from shuffler.psv_io import read_psv
from shuffler.translate_to_psv_format import translate_synthetic_to_psv

# Import the core perplexity function
from perplexity import get_sentence_perplexities, DEFAULT_MODEL, DEFAULT_PARALLEL


# ── Output column names ───────────────────────────────────────────────────────

_OUT_FIELDNAMES = ["UID", "median_perplexity", "stdev_perplexity"]


# ── Core scoring logic ────────────────────────────────────────────────────────

def _score_comment(
    uid: str,
    comment_text: str,
    api_key: str,
    model: str,
    parallel: int,
    index: int,
    total: int,
) -> dict:
    """
    Score a single comment and return a result dict with UID, median, and stdev.
    """
    print(f"[{index}/{total}] Scoring {uid} …", flush=True)

    if not comment_text.strip():
        print(f"  → empty comment, skipping.", flush=True)
        return {"UID": uid, "median_perplexity": "", "stdev_perplexity": ""}

    try:
        sentence_results = get_sentence_perplexities(
            document=comment_text,
            api_key=api_key,
            model=model,
            parallel=parallel,
        )
    except Exception as exc:
        print(f"  → ERROR: {exc}", file=sys.stderr, flush=True)
        return {"UID": uid, "median_perplexity": "", "stdev_perplexity": ""}

    # Collect valid (non-None) perplexity scores
    scores = [
        r["perplexity"]
        for r in sentence_results
        if r["perplexity"] is not None
    ]

    # Report any errors so they are visible
    errors = [r for r in sentence_results if r.get("error")]
    if errors:
        print(f"  → {len(errors)} sentence(s) had errors:", flush=True)
        for e in errors[:3]:  # show at most 3 so output stays readable
            print(f"     • {e['error'][:200]}", flush=True)
        if len(errors) > 3:
            print(f"     … (and {len(errors) - 3} more)", flush=True)

    if not scores:
        print(f"  → no valid scores returned.", flush=True)
        return {"UID": uid, "median_perplexity": "", "stdev_perplexity": ""}

    median = statistics.median(scores)
    stdev  = statistics.stdev(scores) if len(scores) >= 2 else ""

    print(
        f"  → {len(scores)} sentences scored | "
        f"median={median:.2f}"
        + (f" | stdev={stdev:.2f}" if stdev != "" else ""),
        flush=True,
    )

    return {
        "UID": uid,
        "median_perplexity": round(median, 4),
        "stdev_perplexity":  round(stdev, 4) if stdev != "" else "",
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="perplexity_psv",
        description=(
            "Score every comment in a PSV file with the perplexity analyser "
            "and write a CSV of UID, median_perplexity, stdev_perplexity."
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
            "Output CSV path.  Defaults to <input_stem>_perplexity.csv "
            "in the same directory as the input file."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "MITRE API key.  Falls back to the MITRE_API_KEY environment variable."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model identifier (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL,
        metavar="N",
        help=f"Parallel API calls per comment (default: {DEFAULT_PARALLEL}).",
    )

    args = parser.parse_args()

    # ── Resolve API key ────────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("MITRE_API_KEY", "")
    if not api_key:
        print(
            "Error: no API key provided.  Use --api-key or set MITRE_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Resolve input file ─────────────────────────────────────────────────────
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    original_input_path = input_path

    # Convert .txt → .psv if needed (mirrors judge.py behaviour)
    if input_path.suffix.lower() == ".txt":
        psv_path = input_path.with_suffix(".psv")
        print(f"[perplexity_psv] Translating synthetic .txt → .psv …")
        print(f"                 Input  : {input_path}")
        print(f"                 Output : {psv_path}")
        n = translate_synthetic_to_psv(str(input_path), str(psv_path))
        print(f"[perplexity_psv] Translation complete — {n} records written.\n")
        input_path = psv_path
    elif input_path.suffix.lower() != ".psv":
        print(
            f"Warning: unexpected file extension '{input_path.suffix}'. "
            "Expected .psv or .txt.  Attempting to read as PSV.",
            file=sys.stderr,
        )

    # ── Read PSV ───────────────────────────────────────────────────────────────
    print(f"[perplexity_psv] Reading PSV: {input_path}")
    rows, fieldnames = read_psv(str(input_path))
    print(f"[perplexity_psv] {len(rows)} comments loaded.\n")

    if not rows:
        print("No comments found in input file.  Nothing to score.", file=sys.stderr)
        sys.exit(0)

    # ── Determine output path ──────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        stem = original_input_path.stem
        output_path = original_input_path.with_name(f"{stem}_perplexity.csv")

    print(f"[perplexity_psv] Model      : {args.model}")
    print(f"[perplexity_psv] Parallel   : {args.parallel} calls/comment")
    print(f"[perplexity_psv] Output     : {output_path}")
    print()

    # ── Score each comment ─────────────────────────────────────────────────────
    total = len(rows)
    results = []

    for i, row in enumerate(rows):
        uid          = row.get("Document ID", f"ROW-{i + 1:04d}")
        comment_text = row.get("Comment", "")

        result = _score_comment(
            uid=uid,
            comment_text=comment_text,
            api_key=api_key,
            model=args.model,
            parallel=args.parallel,
            index=i + 1,
            total=total,
        )
        results.append(result)

    # ── Write output CSV ───────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_OUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    # ── Summary ────────────────────────────────────────────────────────────────
    scored  = sum(1 for r in results if r["median_perplexity"] != "")
    skipped = total - scored

    print()
    print("=" * 60)
    print("Perplexity scoring complete!")
    print(f"  Total comments : {total}")
    print(f"  Scored         : {scored}")
    if skipped:
        print(f"  Skipped/errors : {skipped}")
    print(f"  Output         : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
