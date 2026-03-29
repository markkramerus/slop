"""
judge.py — Standalone AI-vs-human evaluator.

Reads a ♔-delimited PSV file (or a synthetic .txt file, which is first
converted to PSV format), calls the LLM judge on every comment concurrently,
and writes the results to a CSV.

Usage
-----
    python judge.py <input_file>           # .psv or .txt
    python judge.py <input_file> -o out.csv
    python judge.py <input_file> --verbose

Output
------
  <input_stem>_judged.csv   with columns: UID, human_author_probability, reason

If the input is a synthetic .txt file it is translated to PSV first via the
same function used by the shuffler pipeline.  The intermediate .psv file is
written next to the original .txt (e.g. foo.txt → foo.psv) and kept for
reference.

Environment variables (same as the rewriter pipeline)
------------------------------------------------------
  JUDGE_API_BASE_URL   default: https://api.openai.com/v1
  JUDGE_API_KEY        (required)
  JUDGE_CHAT_MODEL     default: gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from shuffler.psv_io import read_psv
from shuffler.translate_to_psv_format import translate_synthetic_to_psv
from syncom.rewriter import RewriterConfig, judge_comment_async


# ── Output column names ────────────────────────────────────────────────────────

_OUT_FIELDNAMES = ["UID", "human_author_probability", "reason"]

# Maximum number of concurrent judge calls.  Keep this modest to stay within
# rate limits; increase if your API tier allows it.
_MAX_CONCURRENCY = 10


# ── Core async logic ──────────────────────────────────────────────────────────

async def _judge_row(
    uid: str,
    comment_text: str,
    config: RewriterConfig,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
    verbose: bool,
) -> dict:
    """Judge a single comment and return a result dict."""
    async with semaphore:
        if verbose:
            print(f"  [{index}/{total}] Judging {uid} …", flush=True)
        try:
            verdict = await judge_comment_async(comment_text, config)
            score = verdict.human_author_probability
            reason = verdict.reasons
        except Exception as exc:
            print(f"  [{index}/{total}] ERROR judging {uid}: {exc}", file=sys.stderr)
            score = -1
            reason = f"Error: {exc}"

        if verbose:
            label = "HUMAN" if score > 50 else "AI" if score >= 0 else "ERROR"
            print(f"  [{index}/{total}] {uid} → {score}/100  ({label})", flush=True)
        else:
            print(f"[{index}/{total}] {uid} → {score}/100", flush=True)

        return {
            "UID": uid,
            "human_author_probability": score,
            "reason": reason,
        }


async def _run_async(
    rows: list[dict],
    config: RewriterConfig,
    verbose: bool,
    concurrency: int = _MAX_CONCURRENCY,
) -> list[dict]:
    """Judge all rows concurrently (bounded by *concurrency*)."""
    semaphore = asyncio.Semaphore(concurrency)
    total = len(rows)

    tasks = [
        _judge_row(
            uid=row.get("Document ID", f"ROW-{i+1:04d}"),
            comment_text=row.get("Comment", ""),
            config=config,
            semaphore=semaphore,
            index=i + 1,
            total=total,
            verbose=verbose,
        )
        for i, row in enumerate(rows)
    ]

    # gather preserves order
    results = await asyncio.gather(*tasks)
    return list(results)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="judge",
        description=(
            "Evaluate every comment in a PSV file with the LLM judge and "
            "write a CSV of UID, human_author_probability, reason."
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
            "Output CSV path.  Defaults to <input_stem>_judged.csv "
            "in the same directory as the input file."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print extra details for each comment.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=_MAX_CONCURRENCY,
        metavar="N",
        help=f"Maximum concurrent judge calls (default: {_MAX_CONCURRENCY}).",
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
        print(f"[judge] Translating synthetic .txt → .psv …")
        print(f"        Input  : {input_path}")
        print(f"        Output : {psv_path}")
        n = translate_synthetic_to_psv(str(input_path), str(psv_path))
        print(f"[judge] Translation complete — {n} records written.\n")
        input_path = psv_path
    elif input_path.suffix.lower() != ".psv":
        print(
            f"Warning: unexpected file extension '{input_path.suffix}'. "
            "Expected .psv or .txt.  Attempting to read as PSV.",
            file=sys.stderr,
        )

    # ── Step 1: Read PSV ───────────────────────────────────────────────────────
    print(f"[judge] Reading PSV: {input_path}")
    rows, fieldnames = read_psv(str(input_path))
    print(f"[judge] {len(rows)} comments loaded.\n")

    if not rows:
        print("No comments found in input file. Nothing to judge.", file=sys.stderr)
        sys.exit(0)

    # ── Step 2: Determine output path ─────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        # Base output on the *original* input file (before any .txt→.psv step)
        stem = original_input_path.stem
        output_path = original_input_path.with_name(f"{stem}_judged.csv")

    # ── Step 3: Build config and validate ─────────────────────────────────────
    config = RewriterConfig()
    try:
        # Only the judge key is needed; the rewrite key may be absent
        if not config.judge_api_key:
            raise ValueError("JUDGE_API_KEY environment variable is not set.")
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[judge] Judge model : {config.judge_model}")
    print(f"[judge] Concurrency : {args.concurrency}")
    print(f"[judge] Output      : {output_path}")
    print()

    # ── Step 4: Judge concurrently ─────────────────────────────────────────────
    results = asyncio.run(
        _run_async(rows, config, verbose=args.verbose, concurrency=args.concurrency)
    )

    # ── Step 5: Write output CSV ───────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_OUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    # ── Summary ────────────────────────────────────────────────────────────────
    passed  = sum(1 for r in results if isinstance(r["human_author_probability"], int) and r["human_author_probability"] > 50)
    failed  = sum(1 for r in results if isinstance(r["human_author_probability"], int) and 0 <= r["human_author_probability"] <= 50)
    errors  = sum(1 for r in results if r["human_author_probability"] == -1)
    total   = len(results)

    print()
    print("=" * 60)
    print("Judge complete!")
    print(f"  Total comments : {total}")
    print(f"  Likely human   : {passed}  (score > 50)")
    print(f"  Likely AI      : {failed}  (score 0–50)")
    if errors:
        print(f"  Errors         : {errors}")
    print(f"  Output         : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
