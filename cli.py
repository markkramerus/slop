#!/usr/bin/env python3
"""
cli.py — Command-line interface for the slop synthetic comment generator.

Usage
-----
    python cli.py \\
        --docket-csv  CMS-2025-0050-0031.csv \\
        --rule-text   proposed_rule.txt \\
        --vector      2 \\
        --objective   "oppose the proposed reduction of Medicare Advantage quality bonus payments" \\
        --volume      50 \\
        --output      synthetic_comments.csv

Environment variables (or .env file):
    SLOP_API_BASE_URL   Base URL for the OpenAI-compatible API
    SLOP_API_KEY        API key
    SLOP_CHAT_MODEL     Chat/completion model name
    SLOP_EMBED_MODEL    Embeddings model name

Optional flags let you tune cost vs. quality:
    --no-relevance-check    Skip LLM topical-relevance QC
    --no-argument-check     Skip LLM argument-presence QC
    --no-embedding-check    Skip embedding-based deduplication
    --include-failed-qc     Write QC-failed rows to the CSV (flagged)
    --seed                  Random seed (default 42)
    --similarity-threshold  Cosine similarity ceiling for dedup (default 0.92)
    --max-retries           Retries per comment slot on QC failure (default 3)
    --comment-period-days   Simulated comment period length in days (default 60)
    --api-base-url          Override SLOP_API_BASE_URL
    --api-key               Override SLOP_API_KEY
    --chat-model            Override SLOP_CHAT_MODEL
    --embed-model           Override SLOP_EMBED_MODEL
    --docket-id             Override docket ID in output (inferred from CSV name)
    --quiet                 Suppress progress output

Vector descriptions:
    1  Semantic Variance  — Same argument, maximally varied surface forms
    2  Persona Mimicry    — Engineered stakeholder consensus across diverse personas
    3  Citation Flooding  — Arguments loaded with plausible-sounding citations
    4  Dilution / Noise   — High-volume, low-substance vague agreement
"""

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="slop",
        description=(
            "Generate synthetic regulatory public comments for comment-spam "
            "detection research."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required inputs
    req = p.add_argument_group("required arguments")
    req.add_argument(
        "--docket-csv",
        required=True,
        metavar="PATH",
        help="Path to a Regulations.gov CSV from a previous (or same-topic) docket.",
    )
    req.add_argument(
        "--rule-text",
        required=True,
        metavar="PATH_OR_TEXT",
        help=(
            "Path to a file containing the proposed rule text, OR the rule text "
            "itself as a string."
        ),
    )
    req.add_argument(
        "--vector",
        required=True,
        type=int,
        choices=[1, 2, 3, 4],
        metavar="{1,2,3,4}",
        help=(
            "Attack vector: 1=Semantic Variance, 2=Persona Mimicry, "
            "3=Citation Flooding, 4=Dilution/Noise."
        ),
    )
    req.add_argument(
        "--objective",
        required=True,
        metavar="TEXT",
        help="The position to advance or oppose (free-text string).",
    )
    req.add_argument(
        "--volume",
        required=True,
        type=int,
        metavar="N",
        help="Number of accepted synthetic comments to produce.",
    )
    req.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Destination CSV file path.",
    )

    # API configuration
    api = p.add_argument_group("API configuration")
    api.add_argument("--api-base-url", metavar="URL", default=None,
                     help="OpenAI-compatible API base URL (overrides SLOP_API_BASE_URL).")
    api.add_argument("--api-key", metavar="KEY", default=None,
                     help="API key (overrides SLOP_API_KEY).")
    api.add_argument("--chat-model", metavar="MODEL", default=None,
                     help="Chat model name (overrides SLOP_CHAT_MODEL).")
    api.add_argument("--embed-model", metavar="MODEL", default=None,
                     help="Embeddings model name (overrides SLOP_EMBED_MODEL).")

    # QC options
    qc = p.add_argument_group("quality control")
    qc.add_argument("--no-relevance-check", action="store_true",
                    help="Skip LLM topical-relevance check.")
    qc.add_argument("--no-argument-check", action="store_true",
                    help="Skip LLM argument-presence check.")
    qc.add_argument("--no-embedding-check", action="store_true",
                    help="Skip embedding-based near-duplicate check.")
    qc.add_argument("--include-failed-qc", action="store_true",
                    help="Include QC-failed rows in the output CSV (flagged).")
    qc.add_argument("--similarity-threshold", type=float, default=0.92, metavar="FLOAT",
                    help="Cosine similarity ceiling for dedup (default 0.92).")
    qc.add_argument("--max-retries", type=int, default=3, metavar="N",
                    help="Generation retries per comment slot on QC failure (default 3).")

    # Generation options
    gen = p.add_argument_group("generation options")
    gen.add_argument("--seed", type=int, default=42,
                     help="Random seed (default 42).")
    gen.add_argument("--comment-period-days", type=int, default=60, metavar="N",
                     help="Simulated comment period length in days (default 60).")
    gen.add_argument("--docket-id", metavar="ID", default="",
                     help="Override docket ID in output (inferred from CSV name if omitted).")

    # Verbosity
    p.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    return p


def resolve_rule_text(path_or_text: str) -> str:
    """If the argument looks like a file path that exists, read it; otherwise use as-is."""
    if os.path.exists(path_or_text):
        with open(path_or_text, "r", encoding="utf-8") as f:
            return f.read()
    return path_or_text


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Import here so the module is usable without installed deps when just
    # running --help
    from slop.config import Config
    from slop.pipeline import run

    # Build config — CLI flags take precedence over env vars
    config = Config()
    if args.api_base_url:
        config.api_base_url = args.api_base_url
    if args.api_key:
        config.api_key = args.api_key
    if args.chat_model:
        config.chat_model = args.chat_model
    if args.embed_model:
        config.embed_model = args.embed_model

    try:
        config.validate()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rule_text = resolve_rule_text(args.rule_text)

    try:
        result = run(
            docket_csv=args.docket_csv,
            rule_text=rule_text,
            vector=args.vector,
            objective=args.objective,
            volume=args.volume,
            output_path=args.output,
            config=config,
            seed=args.seed,
            similarity_threshold=args.similarity_threshold,
            max_retries=args.max_retries,
            comment_period_days=args.comment_period_days,
            include_failed_qc=args.include_failed_qc,
            skip_relevance_check=args.no_relevance_check,
            skip_argument_check=args.no_argument_check,
            skip_embedding_check=args.no_embedding_check,
            docket_id=args.docket_id,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    if not args.quiet:
        print(result.summary())

    return 0 if result.total_accepted > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
