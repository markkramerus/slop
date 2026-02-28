#!/usr/bin/env python3
"""
cli.py — Command-line interface for the slop synthetic comment generator.

Usage
-----
    python cli.py \\
        --docket-id   CMS-2025-0050 \\
        --rule-text   CMS-2025-0050/rule/proposed_rule.txt \\
        --vector      2 \\
        --objective   "oppose the proposed reduction of Medicare Advantage quality bonus payments" \\
        --volume      50 \\
        --output      CMS-2025-0050/synthetic_comments/comments.txt

Prerequisites:
    Run stylometry_analyzer.py first to generate voice profiles for the docket.
    This creates {docket_id}/stylometry/ with index.json and voice skill .md files.

Environment variables (or .env file):
    SLOP_API_BASE_URL        Base URL for the chat/completion API
    SLOP_API_KEY             API key for chat/completion
    SLOP_CHAT_MODEL          Chat/completion model name

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
        "--docket-id",
        required=True,
        metavar="ID",
        help=(
            "Docket identifier (e.g., 'CMS-2025-0050'). "
            "The tool looks for stylometry data in {docket_id}/stylometry/. "
            "Run stylometry_analyzer.py first to generate voice profiles."
        ),
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
        required=False,
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        metavar="{1,2,3,4}",
        help=(
            "Attack vector: 1=Semantic Variance, 2=Persona Mimicry, "
            "3=Citation Flooding, 4=Dilution/Noise. "
            "Required unless --campaign-plan is provided (which distributes across vectors)."
        ),
    )
    req.add_argument(
        "--objective",
        required=False,
        default=None,
        metavar="TEXT",
        help=(
            "The position to advance or oppose (free-text string). "
            "Required unless --campaign-plan is provided (which includes the objective)."
        ),
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
        help="Destination file path (use .txt extension for ♔ delimited format).",
    )

    # Campaign plan
    campaign = p.add_argument_group("campaign plan")
    campaign.add_argument(
        "--campaign-plan",
        default=None,
        metavar="PATH",
        help=(
            "Path to a campaign_plan.json file (produced by campaign/planner.py). "
            "When provided, --objective and --vector are read from the plan. "
            "--vector can still be used to override the plan's vector mix."
        ),
    )

    # API configuration
    api = p.add_argument_group("API configuration")
    api.add_argument("--api-base-url", metavar="URL", default=None,
                     help="Chat API base URL (overrides SLOP_API_BASE_URL).")
    api.add_argument("--api-key", metavar="KEY", default=None,
                     help="Chat API key (overrides SLOP_API_KEY).")
    api.add_argument("--chat-model", metavar="MODEL", default=None,
                     help="Chat model name (overrides SLOP_CHAT_MODEL).")
    
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
    gen.add_argument("--max-concurrent", type=int, default=10, metavar="N",
                     help="Max concurrent API requests (async mode, default 10).")
    gen.add_argument("--no-async", action="store_true",
                     help="Disable async parallelization (slower but more predictable).")

    # Verbosity
    p.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    return p


def resolve_rule_text(path_or_text: str) -> str:
    """If the argument looks like a file path that exists, read it; otherwise use as-is."""
    if os.path.exists(path_or_text):
        # Try multiple encodings to handle various file formats
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(path_or_text, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # If all else fails, read with UTF-8 and replace problematic characters
        with open(path_or_text, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return path_or_text


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Import here so the module is usable without installed deps when just
    # running --help
    from config import Config
    from syncom.pipeline import run, run_async, run_campaign, run_campaign_async

    # Validate argument combinations
    use_campaign = args.campaign_plan is not None
    if not use_campaign:
        if args.objective is None:
            print("Error: --objective is required unless --campaign-plan is provided.", file=sys.stderr)
            return 1
        if args.vector is None:
            print("Error: --vector is required unless --campaign-plan is provided.", file=sys.stderr)
            return 1

    # Build config — CLI flags take precedence over env vars
    config = Config()
    if args.api_base_url:
        config.api_base_url = args.api_base_url
    if args.api_key:
        config.api_key = args.api_key
    if args.chat_model:
        config.chat_model = args.chat_model
    if args.embed_api_base_url:
        config.embed_api_base_url = args.embed_api_base_url
    if args.embed_api_key:
        config.embed_api_key = args.embed_api_key
    if args.embed_model:
        config.embed_model = args.embed_model

    try:
        config.validate()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rule_text = resolve_rule_text(args.rule_text)

    try:
        if use_campaign:
            # ── Campaign-plan mode ────────────────────────────────────────
            if not args.quiet:
                print(f"Using campaign plan: {args.campaign_plan}", file=sys.stderr)

            common_kwargs = dict(
                docket_id=args.docket_id,
                rule_text=rule_text,
                campaign_plan_path=args.campaign_plan,
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
                verbose=not args.quiet,
                vector_override=args.vector,  # None if not specified → use plan's mix
            )

            if args.no_async:
                result = run_campaign(**common_kwargs)
            else:
                result = run_campaign_async(
                    **common_kwargs,
                    max_concurrent=args.max_concurrent,
                )

        else:
            # ── Direct mode (original behavior) ──────────────────────────
            if args.no_async:
                result = run(
                    docket_id=args.docket_id,
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
                    verbose=not args.quiet,
                )
            else:
                result = run_async(
                    docket_id=args.docket_id,
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
                    verbose=not args.quiet,
                    max_concurrent=args.max_concurrent,
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
