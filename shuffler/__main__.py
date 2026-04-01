#!/usr/bin/env python3
"""
shuffler package CLI — python -m shuffler <subcommand>

Subcommands
-----------
    preprocess <docket_id>
        Build the merged PSV from a docket's CSV + attachment text.
        Reads  {docket_id}/comments/{docket_id}.csv
               {docket_id}/comment_attachments/
        Writes {docket_id}/comments/{docket_id}.psv

        The resulting PSV is required by stylometry_analyzer before running
        voice-group analysis.

Example
-------
    python -m shuffler preprocess HHS-ONC-2026-0001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shuffler.shuffler import preprocess_real_comments


# ── subcommand: preprocess ────────────────────────────────────────────────────

def cmd_preprocess(args: argparse.Namespace) -> int:
    docket_id = args.docket_id
    csv_path = args.csv or str(
        Path(docket_id) / "comments" / f"{docket_id}.csv"
    )
    attachments_dir = args.attachments_dir or str(
        Path(docket_id) / "comment_attachments"
    )
    output_path = args.output or str(
        Path(docket_id) / "comments" / f"{docket_id}.psv"
    )

    if not Path(csv_path).exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    print(f"Input CSV:    {csv_path}", file=sys.stderr)
    print(f"Attachments:  {attachments_dir}", file=sys.stderr)
    print(f"Output PSV:   {output_path}", file=sys.stderr)

    try:
        stats = preprocess_real_comments(
            real_file=csv_path,
            attachments_dir=attachments_dir,
            output_file=output_path,
            verbose=True,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    print(
        f"\nDone.  PSV written to: {output_path}",
        file=sys.stderr,
    )
    if isinstance(stats, dict):
        total = stats.get("total_rows", "?")
        subst = stats.get("rows_substituted", "?")
        print(
            f"       {total} rows total, {subst} with attachment text substituted.",
            file=sys.stderr,
        )
    return 0


# ── argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m shuffler",
        description="shuffler utilities for building and combining comment PSV files.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser(
        "preprocess",
        help="Build the merged PSV (CSV comment text + attachment text) for a docket",
        description=(
            "Reads {docket_id}/comments/{docket_id}.csv and merges attachment "
            "text from {docket_id}/comment_attachments/ into the Comment field.  "
            "Writes the result to {docket_id}/comments/{docket_id}.psv.  "
            "The PSV is the mandatory input for stylometry_analyzer."
        ),
    )
    pp.add_argument(
        "docket_id",
        metavar="DOCKET_ID",
        help="Docket identifier, e.g. HHS-ONC-2026-0001",
    )
    pp.add_argument(
        "--csv",
        metavar="PATH",
        default=None,
        help="Override input CSV path (default: {docket_id}/comments/{docket_id}.csv)",
    )
    pp.add_argument(
        "--attachments-dir",
        metavar="DIR",
        default=None,
        help="Override attachments directory (default: {docket_id}/comment_attachments)",
    )
    pp.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Override output PSV path (default: {docket_id}/comments/{docket_id}.psv)",
    )

    return p


# ── entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "preprocess":
        return cmd_preprocess(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
