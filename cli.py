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


# ── Shuffle subcommand ─────────────────────────────────────────────────────────

def build_shuffle_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="slop shuffle",
        description=(
            "Shuffler phase: (0) pre-process the real CMS CSV by substituting "
            "attachment text where it is the bulk of the comment, then "
            "(1) translate synthetic comments to CMS format, then "
            "(2) randomly interleave them with the pre-processed real comments "
            "and produce a key that labels every row as real or synthetic.  "
            "Attachment URLs are cleared in the final combined.csv so no row "
            "reveals whether it came from a real submission."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Convention-based defaults (when --docket-id is provided)
---------------------------------------------------------
  --attachments-dir   {docket_id}/comment_attachments
  --preprocessed-output {docket_id}/shuffled_comments/preprocessed_real.csv
  --syncom-output     {docket_id}/synthetic_comments/synthetic.txt
  --translated-output {docket_id}/shuffled_comments/synthetic_cms.csv
  --real-comments     {docket_id}/comments/{docket_id}.csv
  --combined-output   {docket_id}/shuffled_comments/combined.csv

Examples
--------
  # Simplest form — all paths derived from docket ID:
  python cli.py shuffle --docket-id CMS-2025-0050

  # Full pipeline (translate + shuffle) with explicit paths:
  python cli.py shuffle \\
      --syncom-output  CMS-2025-0050/synthetic_comments/synthetic.txt \\
      --translated-output CMS-2025-0050/shuffled_comments/synthetic_cms.csv \\
      --real-comments  CMS-2025-0050/comments/CMS-2025-0050.csv \\
      --combined-output CMS-2025-0050/shuffled_comments/combined.csv

  # Skip translation (provide an already-translated CMS CSV):
  python cli.py shuffle --docket-id CMS-2025-0050 --skip-translation

  # Skip pre-processing (no attachment substitution):
  python cli.py shuffle --docket-id CMS-2025-0050 --skip-preprocess

Key file
--------
  A companion key CSV is written automatically next to the combined output
  (e.g., combined_key.csv).  Use --key-output to override its path.
  The key has four columns: row_number, uid, original_document_id, type (real | synthetic).
""",
    )

    p.add_argument(
        "--docket-id",
        default=None,
        metavar="ID",
        help=(
            "Docket identifier (e.g., 'CMS-2025-0050'). When provided, all "
            "file paths default to conventional locations inside the docket "
            "directory; any explicit path argument overrides the default."
        ),
    )
    p.add_argument(
        "--attachments-dir",
        default=None,
        metavar="PATH",
        help=(
            "Path to the directory containing per-comment attachment "
            "subdirectories (e.g. 'CMS-2025-0050/comment_attachments'). "
            "Default: {docket_id}/comment_attachments"
        ),
    )
    p.add_argument(
        "--preprocessed-output",
        default=None,
        metavar="PATH",
        help=(
            "Path where the pre-processed real comments CSV will be saved. "
            "Default: {docket_id}/shuffled_comments/preprocessed_real.csv"
        ),
    )
    p.add_argument(
        "--skip-preprocess",
        action="store_true",
        help=(
            "Skip the attachment pre-processing step.  The raw real comments "
            "CSV will be used directly for shuffling."
        ),
    )
    p.add_argument(
        "--syncom-output",
        metavar="PATH",
        default=None,
        help=(
            "Path to the ♔-delimited syncom output file produced by the "
            "generate phase.  Required unless --skip-translation is set. "
            "Default: {docket_id}/synthetic_comments/synthetic.txt"
        ),
    )
    p.add_argument(
        "--translated-output",
        default=None,
        metavar="PATH",
        help=(
            "Path where the translated synthetic CMS CSV will be saved.  "
            "If --skip-translation is set this file must already exist. "
            "Default: {docket_id}/shuffled_comments/synthetic_cms.csv"
        ),
    )
    p.add_argument(
        "--real-comments",
        default=None,
        metavar="PATH",
        help=(
            "Path to the real CMS comments file (comma-delimited CSV). "
            "Default: {docket_id}/comments/{docket_id}.csv"
        ),
    )
    p.add_argument(
        "--combined-output",
        default=None,
        metavar="PATH",
        help=(
            "Path where the combined (shuffled) CMS CSV will be saved. "
            "Default: {docket_id}/shuffled_comments/combined.csv"
        ),
    )
    p.add_argument(
        "--key-output",
        default=None,
        metavar="PATH",
        help=(
            "Path where the key CSV will be saved.  "
            "Defaults to <combined-output-stem>_key.csv in the same directory."
        ),
    )
    p.add_argument(
        "--skip-translation",
        action="store_true",
        help=(
            "Skip the translation step and use --translated-output directly "
            "as the synthetic CMS CSV.  Useful when translation was run "
            "previously and the file already exists."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed for reproducible shuffling (default 42).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )

    return p


def run_shuffle(argv: list[str] | None = None) -> int:
    """Entry point for the 'shuffle' subcommand."""
    parser = build_shuffle_parser()
    args = parser.parse_args(argv)

    from shuffler.shuffler import preprocess_real_comments, translate_syncom_to_cms, shuffle_comments

    verbose = not args.quiet

    # ── Resolve defaults from docket-id ────────────────────────────────────
    docket_id = args.docket_id

    attachments_dir      = args.attachments_dir
    preprocessed_output  = args.preprocessed_output
    syncom_output        = args.syncom_output
    translated_output    = args.translated_output
    real_comments        = args.real_comments
    combined_output      = args.combined_output

    if docket_id:
        if attachments_dir is None:
            attachments_dir = os.path.join(docket_id, "comment_attachments")
        if preprocessed_output is None:
            preprocessed_output = os.path.join(
                docket_id, "shuffled_comments", "preprocessed_real.csv"
            )
        if syncom_output is None:
            syncom_output = os.path.join(docket_id, "synthetic_comments", "synthetic.txt")
        if translated_output is None:
            translated_output = os.path.join(docket_id, "shuffled_comments", "synthetic_cms.csv")
        if real_comments is None:
            real_comments = os.path.join(docket_id, "comments", f"{docket_id}.csv")
        if combined_output is None:
            combined_output = os.path.join(docket_id, "shuffled_comments", "combined.csv")

    # Validate that required args are present
    missing = []
    if translated_output is None:
        missing.append("--translated-output")
    if real_comments is None:
        missing.append("--real-comments")
    if combined_output is None:
        missing.append("--combined-output")
    if missing:
        print(
            f"Error: the following arguments are required: {', '.join(missing)}\n"
            f"       (or provide --docket-id to use convention-based defaults)",
            file=sys.stderr,
        )
        return 1

    # Ensure output directories exist
    for path in [translated_output, combined_output]:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # ── Step 0: Pre-process real comments (resolve attachment text) ─────────
    # The file passed to shuffle_comments() is either the preprocessed CSV
    # (attachment text substituted) or the original CSV (if --skip-preprocess).
    if args.skip_preprocess:
        if verbose:
            print(f"[shuffler] Skipping pre-processing — using {real_comments} directly")
        real_cms_for_shuffle = real_comments
    else:
        if not os.path.exists(real_comments):
            print(
                f"Error: real comments file not found: {real_comments}",
                file=sys.stderr,
            )
            return 1
        if attachments_dir is None or not os.path.isdir(attachments_dir):
            # No attachments dir — fall back gracefully
            if verbose:
                print(
                    f"[shuffler] Attachments directory not found "
                    f"({attachments_dir}); skipping pre-processing."
                )
            real_cms_for_shuffle = real_comments
        else:
            try:
                preprocess_real_comments(
                    real_cms_file=real_comments,
                    attachments_dir=attachments_dir,
                    output_file=preprocessed_output,
                    verbose=verbose,
                )
                real_cms_for_shuffle = preprocessed_output
            except Exception as exc:
                print(f"Error during pre-processing: {exc}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                return 1

    # ── Step 1: Translation ─────────────────────────────────────────────────
    if args.skip_translation:
        if not os.path.exists(translated_output):
            print(
                f"Error: --skip-translation was set but translated file not found: "
                f"{translated_output}",
                file=sys.stderr,
            )
            return 1
        if verbose:
            print(f"[shuffler] Skipping translation — using {translated_output}")
    else:
        if syncom_output is None:
            print(
                "Error: --syncom-output is required unless --skip-translation is set.",
                file=sys.stderr,
            )
            return 1
        if not os.path.exists(syncom_output):
            print(
                f"Error: syncom output file not found: {syncom_output}",
                file=sys.stderr,
            )
            return 1

        try:
            translate_syncom_to_cms(
                syncom_input=syncom_output,
                cms_output=translated_output,
                verbose=verbose,
            )
        except Exception as exc:
            print(f"Error during translation: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    # ── Step 2: Shuffle ─────────────────────────────────────────────────────
    if not os.path.exists(real_cms_for_shuffle):
        print(
            f"Error: real comments file not found: {real_cms_for_shuffle}",
            file=sys.stderr,
        )
        return 1

    try:
        shuffle_comments(
            synthetic_cms_file=translated_output,
            real_cms_file=real_cms_for_shuffle,
            combined_output=combined_output,
            key_output=args.key_output,
            seed=args.seed,
            verbose=verbose,
        )
    except Exception as exc:
        print(f"Error during shuffle: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="slop",
        description=(
            "Generate synthetic regulatory public comments for comment-spam "
            "detection research."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Convention-based defaults (derived from --docket-id)
-----------------------------------------------------
  --rule-text     {docket_id}/rule/rule.txt
  --output        {docket_id}/synthetic_comments/synthetic.txt
  --campaign-plan {docket_id}/campaign/campaign_plan.json  (auto-detected if present)

Simplest invocations
--------------------
  # With a campaign plan already in place:
  python cli.py --docket-id CMS-2025-0050 --volume 50

  # Direct mode (no campaign plan):
  python cli.py --docket-id CMS-2025-0050 --volume 10 \\
      --vector 2 --objective "oppose the proposed rule"
""",
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
        default=None,
        metavar="PATH_OR_TEXT",
        help=(
            "Path to a file containing the proposed rule text, OR the rule text "
            "itself as a string. "
            "Default: {docket_id}/rule/rule.txt"
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
        default=None,
        metavar="PATH",
        help=(
            "Destination file path (use .txt extension for ♔ delimited format). "
            "Default: {docket_id}/synthetic_comments/synthetic.txt"
        ),
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
            "--vector can still be used to override the plan's vector mix. "
            "Auto-detected at {docket_id}/campaign/campaign_plan.json if it exists."
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
    api.add_argument("--embed-api-base-url", metavar="URL", default=None,
                     help="Embedding API base URL (overrides SLOP_EMBED_API_BASE_URL).")
    api.add_argument("--embed-api-key", metavar="KEY", default=None,
                     help="Embedding API key (overrides SLOP_EMBED_API_KEY).")
    api.add_argument("--embed-model", metavar="MODEL", default=None,
                     help="Embedding model name (overrides SLOP_EMBED_MODEL).")
    
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
    # Normalise argv so we can inspect it before handing to a sub-parser
    if argv is None:
        argv = sys.argv[1:]

    # ── Subcommand dispatch ─────────────────────────────────────────────────
    if argv and argv[0] == "shuffle":
        return run_shuffle(argv[1:])

    # ── Generate (default) mode ─────────────────────────────────────────────
    parser = build_parser()
    args = parser.parse_args(argv)

    # Import here so the module is usable without installed deps when just
    # running --help
    from config import Config
    from syncom.pipeline import run, run_async, run_campaign, run_campaign_async

    docket_id = args.docket_id

    # ── Resolve convention-based defaults from docket-id ───────────────────
    rule_text_path = args.rule_text
    output_path    = args.output
    campaign_plan  = args.campaign_plan

    if rule_text_path is None:
        rule_text_path = os.path.join(docket_id, "rule", "rule.txt")
        if not args.quiet:
            print(f"[cli] --rule-text not set, using {rule_text_path}", file=sys.stderr)

    if output_path is None:
        output_path = os.path.join(docket_id, "synthetic_comments", "synthetic.txt")
        if not args.quiet:
            print(f"[cli] --output not set, using {output_path}", file=sys.stderr)

    # Auto-detect campaign plan if it exists and wasn't explicitly overridden
    if campaign_plan is None:
        default_plan = os.path.join(docket_id, "campaign", "campaign_plan.json")
        if os.path.exists(default_plan):
            campaign_plan = default_plan
            if not args.quiet:
                print(f"[cli] Auto-detected campaign plan: {campaign_plan}", file=sys.stderr)

    # Validate rule text path
    if not os.path.exists(rule_text_path) and len(rule_text_path.split()) == 1:
        print(f"Error: --rule-text file not found: {rule_text_path}", file=sys.stderr)
        return 1

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Validate argument combinations
    use_campaign = campaign_plan is not None
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

    rule_text = resolve_rule_text(rule_text_path)

    try:
        if use_campaign:
            # ── Campaign-plan mode ────────────────────────────────────────
            if not args.quiet:
                print(f"Using campaign plan: {campaign_plan}", file=sys.stderr)

            common_kwargs = dict(
                docket_id=docket_id,
                rule_text=rule_text,
                campaign_plan_path=campaign_plan,
                volume=args.volume,
                output_path=output_path,
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
                    docket_id=docket_id,
                    rule_text=rule_text,
                    vector=args.vector,
                    objective=args.objective,
                    volume=args.volume,
                    output_path=output_path,
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
                    docket_id=docket_id,
                    rule_text=rule_text,
                    vector=args.vector,
                    objective=args.objective,
                    volume=args.volume,
                    output_path=output_path,
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
