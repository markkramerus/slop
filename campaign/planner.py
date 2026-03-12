#!/usr/bin/env python3
"""
planner.py — Campaign planner: decompose a natural-language scenario into a
structured campaign plan for syncom.

This is a standalone CLI application. It takes:
  - A scenario brief (text file or inline string) describing the user's
    position, rationale, and the stakeholder types who would share it
  - The proposed rule text (for grounding the argument angles in real policy)
  - The docket's stylometry index (for voice group awareness)

And produces:
  - A campaign_plan.json (v2.0) file that can be reviewed, edited, and then
    passed to syncom's pipeline via --campaign-plan

The v2.0 plan uses a Bayesian allocation framework:
    P(V, A) = P(V) × P(A|V)
    P(A|V) ∝ w(A) × f(A,V)
    f(A,V) = affinity_boost if V ∈ best_voices(A), else 1.0

Usage
-----
    python campaign/planner.py --docket-id CMS-2025-0050

    python campaign/planner.py \\
        --docket-id CMS-2025-0050 \\
        --rule-text HTI-5-Proposed-2025-23896.txt \\
        --scenario scenario_brief.txt \\
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
from pathlib import Path

# Add the project root to the path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from campaign.campaign_models import CampaignPlan, ArgumentAngle


# ── Stylometry loading ────────────────────────────────────────────────────────

def _load_stylometry_summary(docket_id: str) -> tuple[dict[str, dict], dict[str, float]]:
    """
    Load the stylometry index.json and build a summary of voice groups
    with their base population rates.

    Returns
    -------
    voice_info : dict[str, dict]
        voice_id → {archetype, sophistication, sample_size, description}
    base_population : dict[str, float]
        voice_id → proportion of total comments (excluding 'unknown' archetypes)
    """
    index_path = Path(docket_id) / "stylometry" / "index.json"
    if not index_path.exists():
        return {}, {}

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    voice_info: dict[str, dict] = {}
    total_known = 0

    for vg in index.get("voice_groups", []):
        archetype = vg["archetype"]
        # Exclude 'unknown' archetypes from campaign allocation
        if archetype == "unknown":
            continue
        voice_id = vg["voice_id"]
        sample_size = vg["sample_size"]
        total_known += sample_size

        # Build a short description for the LLM
        soph = vg.get("sophistication", "medium")
        is_org = voice_id.endswith("-org")
        if archetype == "individual_consumer":
            desc = f"Individual commenters, {soph} sophistication"
        elif archetype == "advocacy_group":
            desc = f"Advocacy/nonprofit organizations, {soph} sophistication"
        elif archetype == "industry":
            desc = f"Industry/corporate organizations, {soph} sophistication"
        elif archetype == "academic":
            desc = f"Academic/research institutions, {soph} sophistication"
        elif archetype == "government":
            desc = f"State/local government entities, {soph} sophistication"
        else:
            desc = f"{archetype}, {soph} sophistication"

        voice_info[voice_id] = {
            "archetype": archetype,
            "sophistication": soph,
            "sample_size": sample_size,
            "is_org": is_org,
            "description": desc,
        }

    # Compute base population proportions (among known voices only)
    base_population: dict[str, float] = {}
    if total_known > 0:
        for vid, info in voice_info.items():
            base_population[vid] = info["sample_size"] / total_known

    return voice_info, base_population


def _format_voice_summary_for_prompt(
    voice_info: dict[str, dict],
    base_population: dict[str, float],
    total_comments: int | None = None,
) -> str:
    """Format the voice group summary for inclusion in the LLM prompt."""
    if not voice_info:
        return "(No stylometry data available)"

    lines = []
    total_known = sum(info["sample_size"] for info in voice_info.values())
    if total_comments:
        lines.append(
            f"This docket received {total_comments} total comments. "
            f"Of the {total_known} that were classified into known archetypes:"
        )
    else:
        lines.append(f"Of {total_known} classified comments:")

    for vid in sorted(voice_info.keys()):
        info = voice_info[vid]
        pct = base_population.get(vid, 0.0)
        lines.append(
            f"  {vid:35s}  {info['sample_size']:>4d} ({pct:5.1%})  — {info['description']}"
        )

    return "\n".join(lines)


# ── LLM prompts ──────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
You are a regulatory comment campaign strategist for a research project on
detecting synthetic public comments. Given a user's scenario (their position
on a proposed rule, why they hold it, and who would agree with them), the
actual rule text, and the existing voice distribution from a real docket,
you decompose the scenario into a structured campaign plan.

The campaign plan specifies:
1. A refined objective statement
2. Distinct argument angles (each a different lens on the position)
3. A campaign voice distribution — how to shift emphasis across voice groups
4. Which voices are best suited for each argument angle

You must output ONLY valid JSON — no prose, no markdown fences, no explanation.
"""

_PLANNER_USER_TEMPLATE = """\
=== USER'S SCENARIO ===
{scenario}

=== PROPOSED RULE TEXT (first 8000 chars for context) ===
{rule_text}

=== EXISTING COMMENT POPULATION (from stylometry analysis) ===
{voice_summary}

The voice IDs above are the EXACT identifiers you must use in your output.
Each voice has a distinct writing style, sophistication level, and perspective
that was learned from analyzing real comments in this docket.

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
      "weight": <float 0.05-0.40, base rate importance of this angle>,
      "best_voices": ["<voice_id_1>", "<voice_id_2>"]
    }}
  ],

  "campaign_voices": {{
    "<voice_id>": <float, campaign weight for this voice>,
    ...
  }},

  "affinity_boost": <float, typically 3.0, the multiplier for preferred voice-argument pairings>,

  "notes": "<strategic rationale: why this mix of voices, angles, and affinities>"
}}

RULES:
- Generate 4-8 distinct argument angles. Each should be a genuinely different
  lens on the position, not just rephrasing.
- For campaign_voices, you MUST use the exact voice_id strings from the
  population data above. Only include voices with archetype != 'unknown'.
- campaign_voices weights represent the campaign's desired emphasis. These
  may differ from the base population. A campaign might amplify certain
  voices (e.g., increase advocacy_group from 18% to 30%) but the result
  should remain plausible — avoid extreme shifts (e.g., 50% academic when
  the base is 2%).
- best_voices on each argument angle should list 2-4 voice_ids that are
  most naturally suited to make that argument. These voices get an
  affinity_boost multiplier when the argument is assigned.
- affinity_boost controls how strongly voice identity channels argument
  selection. 3.0 means preferred voices are 3x more likely to get that
  argument. Use 2.0-5.0 depending on how strongly you want to channel.
- All weight values are relative — they will be normalized at runtime.
- The notes field should explain YOUR strategic reasoning, including why
  you shifted certain voices up or down from the base population.

HOW THE FRAMEWORK WORKS:
For each comment, the pipeline:
  1. Draws a voice V from campaign_voices (probability P(V))
  2. Draws an argument A with probability P(A|V) ∝ w(A) × f(A,V)
     where f(A,V) = affinity_boost if V is in that angle's best_voices, else 1.0
  3. Generates a comment in that voice making that argument

So the overall campaign argument distribution P(A) emerges naturally from
the interaction of voice weights and argument affinities — you don't need
to specify it separately.
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
    docket_id: str = "",
    verbose: bool = True,
) -> CampaignPlan:
    """
    Use an LLM to decompose a scenario into a structured CampaignPlan (v2.0).

    The planner is aware of the docket's stylometry voice groups and produces
    a plan that references actual voice_ids from the stylometry index.

    Parameters
    ----------
    scenario : str
        Natural-language description of the user's position, rationale,
        and the stakeholder types who would share it.
    rule_text : str
        Full text of the proposed rule (used for grounding).
    config : Config
        API configuration.
    docket_id : str
        Docket identifier for loading stylometry data.
    verbose : bool
        Print progress to stderr.

    Returns
    -------
    CampaignPlan
    """
    config.validate()
    client = config.openai_client()

    # ── Load stylometry data ──────────────────────────────────────────────
    voice_info: dict[str, dict] = {}
    base_population: dict[str, float] = {}
    total_comments = None

    if docket_id:
        if verbose:
            print(f"[1/3] Loading stylometry for {docket_id}…", file=sys.stderr)
        voice_info, base_population = _load_stylometry_summary(docket_id)

        index_path = Path(docket_id) / "stylometry" / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                idx = json.load(f)
            total_comments = idx.get("total_comments")

        if voice_info:
            if verbose:
                print(f"      Found {len(voice_info)} voice groups", file=sys.stderr)
                for vid, info in sorted(voice_info.items()):
                    pct = base_population.get(vid, 0)
                    print(
                        f"        {vid:35s} {info['sample_size']:>4d} ({pct:5.1%})",
                        file=sys.stderr,
                    )
        else:
            if verbose:
                print("      No stylometry data found — using generic plan", file=sys.stderr)
    else:
        if verbose:
            print("[1/3] No docket-id — skipping stylometry", file=sys.stderr)

    voice_summary = _format_voice_summary_for_prompt(
        voice_info, base_population, total_comments
    )

    # ── Call LLM ──────────────────────────────────────────────────────────
    if verbose:
        print(f"[2/3] Analyzing scenario and rule text…", file=sys.stderr)

    prompt = _PLANNER_USER_TEMPLATE.format(
        scenario=scenario,
        rule_text=rule_text[:8000],
        voice_summary=voice_summary,
    )

    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=2500,
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
        print("[3/3] Building campaign plan…", file=sys.stderr)

    # Build ArgumentAngle objects
    angles = []
    for a in parsed.get("argument_angles", []):
        angles.append(ArgumentAngle(
            id=a.get("id", "unknown"),
            angle=a.get("angle", ""),
            weight=float(a.get("weight", 0.15)),
            best_voices=a.get("best_voices", []),
        ))

    # Build campaign_voices — validate against known voice_ids
    raw_cv = parsed.get("campaign_voices", {})
    campaign_voices: dict[str, float] = {}
    for vid, w in raw_cv.items():
        campaign_voices[vid] = float(w)
    # If no valid voices, fall back to uniform over known voices
    if not campaign_voices and voice_info:
        campaign_voices = {vid: 1.0 for vid in voice_info}

    # Affinity boost
    affinity_boost = float(parsed.get("affinity_boost", 3.0))

    plan = CampaignPlan(
        objective=parsed.get("objective", scenario[:200]),
        scenario_summary=parsed.get("scenario_summary", scenario[:300]),
        argument_angles=angles,
        campaign_voices=campaign_voices,
        base_population=base_population,
        affinity_boost=affinity_boost,
        notes=parsed.get("notes", ""),
    )

    return plan


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="campaign-planner",
        description=(
            "Decompose a natural-language scenario into a structured campaign "
            "plan for syncom. The planner loads stylometry voice groups from "
            "the docket to produce a plan with actual voice_ids. The output JSON "
            "can be reviewed, edited, and then passed to syncom via --campaign-plan."
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
            "paths inside the docket directory. Also loads stylometry data "
            "for voice group awareness."
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

    # Preview option
    p.add_argument(
        "--preview-volume",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set, print the allocation matrix for N comments after "
            "generating the plan. Useful for reviewing the plan before "
            "running generation."
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
            docket_id=docket_id or "",
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

        if args.preview_volume:
            print(f"", file=sys.stderr)
            print(plan.allocation_summary(args.preview_volume), file=sys.stderr)

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
