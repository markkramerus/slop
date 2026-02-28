#!/usr/bin/env python3
"""
planner.py — Campaign planner: decompose a natural-language scenario into a
structured campaign plan for syncom.

This is a standalone CLI application. It takes:
  - A scenario brief (text file or inline string) describing the user's
    position, rationale, and the stakeholder types who would share it
  - The proposed rule text (for grounding the argument angles in real policy)

And produces:
  - A campaign_plan.json file that can be reviewed, edited, and then passed
    to syncom's pipeline via --campaign-plan

Usage
-----
    python campaign/planner.py \\
        --rule-text HTI-5-Proposed-2025-23896.txt \\
        --scenario scenario_brief.txt \\
        --output campaign_plan.json

    python campaign/planner.py \\
        --rule-text HTI-5-Proposed-2025-23896.txt \\
        --scenario "I oppose HTI-5's removal of AI model card requirements..." \\
        --output campaign_plan.json

Environment variables (or .env file):
    SLOP_API_BASE_URL   Base URL for the chat/completion API
    SLOP_API_KEY        API key for chat/completion
    SLOP_CHAT_MODEL     Chat/completion model name
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Add the project root to the path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from campaign.campaign_models import CampaignPlan, ArgumentAngle


# ── LLM prompts ──────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
You are a regulatory comment campaign strategist for a research project on
detecting synthetic public comments. Given a user's scenario (their position
on a proposed rule, why they hold it, and who would agree with them) plus the
actual rule text, you decompose the scenario into a structured campaign plan.

The campaign plan specifies:
1. A refined objective statement
2. Distinct argument angles (each a different lens on the position)
3. Which stakeholder archetypes should be emphasized
4. What mix of comment styles (attack vectors) to use

You must output ONLY valid JSON — no prose, no markdown fences, no explanation.
"""

_PLANNER_USER_TEMPLATE = """\
=== USER'S SCENARIO ===
{scenario}

=== PROPOSED RULE TEXT (first 8000 chars for context) ===
{rule_text}

=== INSTRUCTIONS ===
Decompose this scenario into a campaign plan. Produce a JSON object with
this exact schema:

{{
  "scenario_summary": "<2-3 sentence summary of the user's position and goals>",

  "objective": "<refined, specific objective statement — the position all comments should advance>",

  "argument_angles": [
    {{
      "id": "<short_snake_case_id>",
      "angle": "<one sentence describing this specific argument angle>",
      "weight": <float 0.05-0.40, relative importance of this angle>,
      "best_archetypes": ["<archetype1>", "<archetype2>"]
    }}
  ],

  "stakeholder_emphasis": {{
    "individual_consumer": <float 0.0-1.0>,
    "advocacy_group": <float 0.0-1.0>,
    "industry": <float 0.0-1.0>,
    "academic": <float 0.0-1.0>,
    "government": <float 0.0-1.0>
  }},

  "vector_mix": {{
    "1": <float, weight for Semantic Variance — same argument, varied surface forms>,
    "2": <float, weight for Persona Mimicry — diverse stakeholders, same position>,
    "3": <float, weight for Citation Flooding — comments with many references>,
    "4": <float, weight for Dilution/Noise — brief, vague agreement>
  }},

  "notes": "<strategic rationale: why this mix of angles, stakeholders, and vectors>"
}}

RULES:
- Generate 4-8 distinct argument angles. Each should be a genuinely different
  lens on the position, not just rephrasing.
- Stakeholder emphasis should reflect who would REALISTICALLY comment on this
  rule with this position. Don't include archetypes that wouldn't plausibly
  comment unless the user's scenario suggests otherwise.
- The valid archetype names are EXACTLY: individual_consumer, advocacy_group,
  industry, academic, government. Use only these.
- Vector weights should reflect a realistic campaign strategy. Persona Mimicry
  (vector 2) is usually the primary vector. Citation Flooding (vector 3) works
  best for academic/industry archetypes. Dilution (vector 4) should be a small
  proportion unless the user specifically wants volume over substance.
- All weight values are relative — they will be normalized at runtime.
- The notes field should explain YOUR strategic reasoning.
"""


def _resolve_text(path_or_text: str) -> str:
    """If the argument looks like a file path that exists, read it; else use as-is."""
    if os.path.exists(path_or_text):
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
        for encoding in encodings:
            try:
                with open(path_or_text, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        with open(path_or_text, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    return path_or_text


def generate_campaign_plan(
    scenario: str,
    rule_text: str,
    config: Config,
    verbose: bool = True,
) -> CampaignPlan:
    """
    Use an LLM to decompose a scenario into a structured CampaignPlan.

    Parameters
    ----------
    scenario : str
        Natural-language description of the user's position, rationale,
        and the stakeholder types who would share it.
    rule_text : str
        Full text of the proposed rule (used for grounding).
    config : Config
        API configuration.
    verbose : bool
        Print progress to stderr.

    Returns
    -------
    CampaignPlan
    """
    config.validate()
    client = config.openai_client()

    if verbose:
        print("[1/2] Analyzing scenario and rule text…", file=sys.stderr)

    prompt = _PLANNER_USER_TEMPLATE.format(
        scenario=scenario,
        rule_text=rule_text[:8000],
    )

    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=2000,
    )

    raw = (response.choices[0].message.content or "{}").strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"Warning: LLM returned invalid JSON, attempting recovery: {exc}", file=sys.stderr)
        parsed = {}

    if verbose:
        print("[2/2] Building campaign plan…", file=sys.stderr)

    # Build ArgumentAngle objects
    angles = []
    for a in parsed.get("argument_angles", []):
        angles.append(ArgumentAngle(
            id=a.get("id", "unknown"),
            angle=a.get("angle", ""),
            weight=float(a.get("weight", 0.15)),
            best_archetypes=a.get("best_archetypes", []),
        ))

    # Build vector_mix with int keys
    raw_vm = parsed.get("vector_mix", {"1": 0.3, "2": 0.4, "3": 0.15, "4": 0.15})
    vector_mix = {int(k): float(v) for k, v in raw_vm.items()}

    plan = CampaignPlan(
        objective=parsed.get("objective", scenario[:200]),
        scenario_summary=parsed.get("scenario_summary", scenario[:300]),
        argument_angles=angles,
        stakeholder_emphasis=parsed.get("stakeholder_emphasis", {
            "individual_consumer": 0.2,
            "advocacy_group": 0.2,
            "industry": 0.2,
            "academic": 0.2,
            "government": 0.2,
        }),
        vector_mix=vector_mix,
        notes=parsed.get("notes", ""),
    )

    return plan


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="campaign-planner",
        description=(
            "Decompose a natural-language scenario into a structured campaign "
            "plan for syncom. The output JSON can be reviewed, edited, and then "
            "passed to syncom via --campaign-plan."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Convention-based defaults (when --docket-id is provided)
---------------------------------------------------------
  --scenario   {docket_id}/campaign/scenario_brief.txt
  --rule-text  {docket_id}/rule/rule.txt
  --output     {docket_id}/campaign/campaign_plan.json

Example (all defaults from docket directory):
  python campaign/planner.py --docket-id CMS-2025-0050
""",
    )

    p.add_argument(
        "--docket-id",
        default=None,
        metavar="ID",
        help=(
            "Docket identifier (e.g., 'CMS-2025-0050'). When provided, "
            "--scenario, --rule-text, and --output default to conventional "
            "paths inside the docket directory."
        ),
    )

    req = p.add_argument_group("path arguments (optional when --docket-id is set)")
    req.add_argument(
        "--scenario",
        default=None,
        metavar="PATH_OR_TEXT",
        help=(
            "Path to a scenario brief file, OR the scenario text itself as a "
            "string. Describe your position, why you hold it, and who would "
            "agree with you. "
            "Default: {docket_id}/campaign/scenario_brief.txt"
        ),
    )
    req.add_argument(
        "--rule-text",
        default=None,
        metavar="PATH_OR_TEXT",
        help=(
            "Path to the proposed rule text file, OR the text itself. "
            "Default: {docket_id}/rule/rule.txt"
        ),
    )
    req.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Destination path for the campaign plan JSON file. "
            "Default: {docket_id}/campaign/campaign_plan.json"
        ),
    )

    api = p.add_argument_group("API configuration")
    api.add_argument("--api-base-url", metavar="URL", default=None,
                     help="Chat API base URL (overrides SLOP_API_BASE_URL).")
    api.add_argument("--api-key", metavar="KEY", default=None,
                     help="Chat API key (overrides SLOP_API_KEY).")
    api.add_argument("--chat-model", metavar="MODEL", default=None,
                     help="Chat model name (overrides SLOP_CHAT_MODEL).")

    p.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = Config()
    if args.api_base_url:
        config.api_base_url = args.api_base_url
    if args.api_key:
        config.api_key = args.api_key
    if args.chat_model:
        config.chat_model = args.chat_model

    # Only validate chat API (planner doesn't need embeddings)
    if not config.api_key:
        print("Error: No API key found. Set SLOP_API_KEY or pass --api-key.", file=sys.stderr)
        return 1

    # ── Resolve defaults from docket-id ────────────────────────────────────
    docket_id = args.docket_id

    scenario_arg = args.scenario
    rule_text_arg = args.rule_text
    output_arg = args.output

    if docket_id:
        if scenario_arg is None:
            scenario_arg = os.path.join(docket_id, "campaign", "scenario_brief.txt")
        if rule_text_arg is None:
            rule_text_arg = os.path.join(docket_id, "rule", "rule.txt")
        if output_arg is None:
            output_arg = os.path.join(docket_id, "campaign", "campaign_plan.json")

    # Validate that required args are present
    missing = []
    if scenario_arg is None:
        missing.append("--scenario")
    if rule_text_arg is None:
        missing.append("--rule-text")
    if output_arg is None:
        missing.append("--output")
    if missing:
        print(
            f"Error: the following arguments are required: {', '.join(missing)}\n"
            f"       (or provide --docket-id to use convention-based defaults)",
            file=sys.stderr,
        )
        return 1

    # Validate that scenario and rule text files exist (when they look like paths)
    for label, path in [("--scenario", scenario_arg), ("--rule-text", rule_text_arg)]:
        if os.path.sep in path or (len(path) < 260 and not path.startswith("http")):
            if not os.path.exists(path) and len(path.split()) == 1:
                print(f"Error: {label} file not found: {path}", file=sys.stderr)
                return 1

    # Ensure output directory exists
    output_dir = os.path.dirname(output_arg)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    scenario = _resolve_text(scenario_arg)
    rule_text = _resolve_text(rule_text_arg)

    if not args.quiet:
        print(f"Scenario:  {scenario_arg}", file=sys.stderr)
        print(f"Rule text: {rule_text_arg}", file=sys.stderr)
        print(f"Output:    {output_arg}", file=sys.stderr)

    try:
        plan = generate_campaign_plan(
            scenario=scenario,
            rule_text=rule_text,
            config=config,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Save the plan
    plan.save(output_arg)

    if not args.quiet:
        print(f"\nCampaign plan saved to: {output_arg}", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(plan.summary(), file=sys.stderr)
        print(f"\nReview and edit the plan, then generate:", file=sys.stderr)
        if docket_id:
            print(f"  python cli.py --docket-id {docket_id} --volume N", file=sys.stderr)
        else:
            print(
                f"  python cli.py --campaign-plan {output_arg} "
                f"--docket-id ... --volume N",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
