#!/usr/bin/env python3
"""
test_rewriter.py — Interactive step-through test for the judge→rewrite loop.

Generates ONE synthetic comment, then lets you step through each judge and
rewrite iteration interactively, examining the full state at each step.

This is a diagnostic tool for verifying that the adversarial rewrite loop
is actually making comments more human-sounding.

Usage
-----
  # Generate one comment via the full pipeline, then step through rewriting:
  python -m syncom.test_rewriter --docket CMS-2025-0050

  # More rewrite iterations:
  python -m syncom.test_rewriter --docket CMS-2025-0050 --max-rewrites 5

  # Test with a pre-existing comment (skip generation):
  python -m syncom.test_rewriter --comment-file my_comment.txt

  # Non-interactive (no pauses, just print everything):
  python -m syncom.test_rewriter --docket CMS-2025-0050 --no-pause

  # Save full history to JSON:
  python -m syncom.test_rewriter --docket CMS-2025-0050 --save-json results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# ── Visual formatting helpers ─────────────────────────────────────────────────

_WIDTH = 88

def _hr(char="─"):
    return char * _WIDTH

def _box(title: str, content: str, char="│"):
    lines = [
        f"┌{'─' * (_WIDTH - 2)}┐",
        f"│ {title:^{_WIDTH - 4}} │",
        f"├{'─' * (_WIDTH - 2)}┤",
    ]
    for line in content.split("\n"):
        # Wrap long lines
        wrapped = textwrap.wrap(line, width=_WIDTH - 6) or [""]
        for w in wrapped:
            lines.append(f"│  {w:<{_WIDTH - 5}} │")
    lines.append(f"└{'─' * (_WIDTH - 2)}┘")
    return "\n".join(lines)


def _score_bar(score: int) -> str:
    """Visual score bar: [████████░░░░░░░░░░░░] 40/100"""
    filled = score // 5
    empty = 20 - filled
    bar = "█" * filled + "░" * empty
    label = "PASS" if score > 50 else "FAIL"
    return f"[{bar}] {score}/100  ({label})"


def _word_count(text: str) -> int:
    return len(text.split())


def _wait_for_user(no_pause: bool):
    """Pause and wait for user input unless --no-pause is set."""
    if no_pause:
        return "continue"
    print()
    print(f"  Press ENTER to continue, 'q' to quit, 's' to skip to summary...")
    try:
        response = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "quit"
    if response == "q":
        return "quit"
    elif response == "s":
        return "skip"
    return "continue"


# ── Comment generation (full pipeline, rewriting disabled) ────────────────────

def _choose_voice_interactive(plan, no_pause: bool = False) -> str | None:
    """
    Present the available voices from the campaign plan and let the user pick one.
    Returns the chosen voice_id, or None to sample randomly.
    """
    voice_weights = plan.normalized_voice_weights()
    voices = sorted(voice_weights.items(), key=lambda x: -x[1])

    print()
    print(f"  Available voices:")
    print(f"  {'#':>4}  {'Voice ID':<35}  {'Weight':>7}")
    print(f"  {'─'*4}  {'─'*35}  {'─'*7}")
    for i, (vid, weight) in enumerate(voices, 1):
        pct = weight * 100
        print(f"  {i:4d}  {vid:<35}  {pct:5.1f}%")
    print()
    print(f"  Enter a number (1-{len(voices)}), or press ENTER for random:")

    if no_pause:
        return None

    try:
        response = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if not response:
        return None

    try:
        idx = int(response) - 1
        if 0 <= idx < len(voices):
            return voices[idx][0]
        else:
            print(f"  Invalid choice, sampling randomly.")
            return None
    except ValueError:
        # Maybe they typed the voice_id directly
        if response in voice_weights:
            return response
        print(f"  Invalid choice, sampling randomly.")
        return None


def _generate_one_comment(
    docket_id: str,
    seed: int = 42,
    verbose: bool = True,
    voice_override: str | None = None,
    no_pause: bool = False,
):
    """
    Generate a single comment using the full pipeline with rewriting disabled.

    Returns (comment_text, persona, persona_context_dict, metadata) or raises on error.
    """
    import numpy as np
    from config import Config
    from syncom.world_model import build_world_model
    from syncom.persona import sample_persona_by_voice_id
    from syncom.argument_mapper import build_campaign_frame
    from syncom.generator import _build_and_call
    from syncom.rewriter import build_persona_context
    from stylometry.stylometry_loader import build_population_model
    from campaign.campaign_models import CampaignPlan

    config = Config()
    config.validate()
    rng = np.random.default_rng(seed)

    # ── Load campaign plan ────────────────────────────────────────────────
    plan_path = os.path.join(docket_id, "campaign", "campaign_plan.json")
    if not os.path.exists(plan_path):
        raise FileNotFoundError(f"Campaign plan not found: {plan_path}")

    plan = CampaignPlan.load(plan_path)
    objective = plan.objective

    if verbose:
        print(f"\n  Objective: {objective[:80]}...", file=sys.stderr)

    # ── Load population ───────────────────────────────────────────────────
    if verbose:
        print(f"  Loading population model...", file=sys.stderr)
    population = build_population_model(docket_id)

    # ── Load rule text ────────────────────────────────────────────────────
    rule_path = os.path.join(docket_id, "rule", "rule.txt")
    if not os.path.exists(rule_path):
        raise FileNotFoundError(f"Rule text not found: {rule_path}")
    
    encodings = ['utf-8', 'latin-1', 'cp1252']
    rule_text = None
    for enc in encodings:
        try:
            with open(rule_path, "r", encoding=enc) as f:
                rule_text = f.read()
            break
        except UnicodeDecodeError:
            continue
    if rule_text is None:
        with open(rule_path, "r", encoding="utf-8", errors="replace") as f:
            rule_text = f.read()

    # ── Build world model ─────────────────────────────────────────────────
    if verbose:
        print(f"  Building world model...", file=sys.stderr)
    world_model = build_world_model(
        rule_text=rule_text,
        population=population,
        config=config,
        docket_id=docket_id,
    )

    # ── Choose voice ──────────────────────────────────────────────────────
    voice_weights = plan.normalized_voice_weights()

    if voice_override:
        # Validate the override
        if voice_override not in voice_weights:
            available = ", ".join(sorted(voice_weights.keys()))
            raise ValueError(
                f"Voice '{voice_override}' not found in campaign plan.\n"
                f"  Available voices: {available}"
            )
        voice_id = voice_override
        if verbose:
            print(f"  Voice (selected): {voice_id}", file=sys.stderr)
    else:
        # Interactive selection or random sampling
        chosen = _choose_voice_interactive(plan, no_pause=no_pause)
        if chosen:
            voice_id = chosen
            if verbose:
                print(f"  Voice (chosen): {voice_id}", file=sys.stderr)
        else:
            voice_ids = list(voice_weights.keys())
            voice_probs = [voice_weights[v] for v in voice_ids]
            voice_idx = int(rng.choice(len(voice_ids), p=voice_probs))
            voice_id = voice_ids[voice_idx]
            if verbose:
                print(f"  Voice (random): {voice_id}", file=sys.stderr)

    # Sample argument angle using P(A|V)
    if plan.argument_angles:
        probs = plan.compute_angle_distribution(voice_id)
        angle_idx = int(rng.choice(len(plan.argument_angles), p=probs))
        angle = plan.argument_angles[angle_idx]
        angle_text = angle.angle
        angle_id = angle.id
    else:
        angle_text = ""
        angle_id = ""

    if verbose:
        print(f"  Voice: {voice_id}", file=sys.stderr)
        print(f"  Angle: {angle_id}", file=sys.stderr)

    # ── Instantiate persona ───────────────────────────────────────────────
    if verbose:
        print(f"  Instantiating persona...", file=sys.stderr)
    persona = sample_persona_by_voice_id(
        voice_id, world_model, config, rng, docket_id=docket_id,
    )

    # ── Build expression frame ────────────────────────────────────────────
    if verbose:
        print(f"  Building expression frame...", file=sys.stderr)
    frame = build_campaign_frame(
        objective, angle_text, persona, world_model, config, rng,
    )

    # ── Generate raw comment (NO rewriting) ───────────────────────────────
    if verbose:
        print(f"  Generating comment (rewriting disabled)...", file=sys.stderr)
    comment_text = _build_and_call(persona, frame, world_model, config, rng)

    # Build persona context for the rewriter
    persona_context = build_persona_context(persona)

    return comment_text, persona, persona_context, {
        "voice_id": voice_id,
        "angle_id": angle_id,
        "angle_text": angle_text,
        "objective": objective,
        "rule_title": world_model.rule_title,
    }


# ── Interactive step-through loop ─────────────────────────────────────────────

def run_step_through(
    comment_text: str,
    persona_context: dict[str, str],
    max_rewrites: int = 5,
    no_pause: bool = False,
    save_json: str | None = None,
    metadata: dict | None = None,
):
    """
    Run the judge→rewrite loop interactively, pausing at each step.
    """
    from syncom.rewriter import (
        RewriterConfig, judge_comment, rewrite_comment,
    )

    config = RewriterConfig()
    if not config.is_available():
        print("\n  ERROR: Judge/Rewriter API keys not configured.", file=sys.stderr)
        print("  Set JUDGE_API_KEY and REWRITE_COMMENT_API_KEY in .env", file=sys.stderr)
        sys.exit(1)

    # Override max_rewrites
    config.max_rewrites = max_rewrites

    current_text = comment_text
    history = []
    score_progression = []

    # ── Show initial comment ──────────────────────────────────────────────
    print()
    print(_hr("═"))
    print(f"  SYNCOM REWRITER STEP-THROUGH TEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(_hr("═"))

    if metadata:
        print()
        print(f"  Voice ID     : {metadata.get('voice_id', '?')}")
        print(f"  Angle        : {metadata.get('angle_id', '?')}")
        print(f"  Rule         : {metadata.get('rule_title', '?')}")

    print()
    print(f"  Persona:")
    print(f"    Name       : {persona_context.get('name', '?')}")
    print(f"    Occupation : {persona_context.get('occupation', '?')}")
    print(f"    Org        : {persona_context.get('org_name', '?')}")
    print(f"    State      : {persona_context.get('state', '?')}")
    print(f"    Age        : {persona_context.get('age', '?')}")
    print(f"    Archetype  : {persona_context.get('archetype', '?')}")

    print()
    print(f"  Max rewrites : {max_rewrites}")
    print(f"  Judge model  : {config.judge_model}")
    print(f"  Rewrite model: {config.rewrite_model}")
    print()
    print(_hr("═"))

    print()
    print(_box(
        "INITIAL COMMENT (raw from generator)",
        f"Word count: {_word_count(current_text)}\n\n{current_text}"
    ))

    history.append({
        "step": "initial",
        "iteration": -1,
        "text": current_text,
        "word_count": _word_count(current_text),
    })

    action = _wait_for_user(no_pause)
    if action == "quit":
        return

    skip_to_end = (action == "skip")
    rewrites_done = 0

    # ── Judge → Rewrite loop ──────────────────────────────────────────────
    for iteration in range(max_rewrites + 1):

        # ── JUDGE step ────────────────────────────────────────────────────
        print()
        print(_hr("─"))
        print(f"  ITERATION {iteration}  —  JUDGE STEP")
        print(_hr("─"))
        print()
        print(f"  Calling judge model ({config.judge_model})...")

        try:
            verdict = judge_comment(current_text, config)
        except Exception as e:
            print(f"\n  ERROR: Judge call failed: {e}")
            break

        score_progression.append(verdict.human_author_probability)

        history.append({
            "step": "judge",
            "iteration": iteration,
            "text": current_text,
            "word_count": _word_count(current_text),
            "score": verdict.human_author_probability,
            "reasons": verdict.reasons,
        })

        # Display verdict
        print()
        print(f"  Human Author Probability: {_score_bar(verdict.human_author_probability)}")
        print()
        print(f"  Judge's reasons:")
        for line in verdict.reasons.split("\n"):
            for wrapped in textwrap.wrap(line, width=_WIDTH - 6) or [""]:
                print(f"    {wrapped}")

        # Show raw API response for debugging
        # if verdict.raw_response:
        #     print()
        #     print(f"  Raw API response (first 500 chars):")
        #     raw_preview = verdict.raw_response[:500]
        #     if len(verdict.raw_response) > 500:
        #         raw_preview += "..."
        #     for line in raw_preview.split("\n"):
        #         print(f"    {line}")

        # Show the comment being judged
        # print()
        # print(_box(
        #     f"COMMENT BEING JUDGED (iteration {iteration})",
        #     f"Word count: {_word_count(current_text)}\n\n{current_text}"
        # ))

        # ── Early exit: passes as real ────────────────────────────────────
        if verdict.human_author_probability > 50:
            print()
            print(f"  ✓ PASSED — Judge considers this comment more likely human-written.")
            print(f"    Score {verdict.human_author_probability}/100 > 50 threshold")
            print(f"    Rewrites performed: {rewrites_done}")
            break

        # ── Early exit: max rewrites reached ──────────────────────────────
        if rewrites_done >= max_rewrites:
            print()
            print(f"  ✗ MAX REWRITES REACHED ({max_rewrites})")
            print(f"    Final score: {verdict.human_author_probability}/100")
            break

        if not skip_to_end:
            action = _wait_for_user(no_pause)
            if action == "quit":
                break
            if action == "skip":
                skip_to_end = True

        # ── REWRITE step ──────────────────────────────────────────────────
        print()
        print(_hr("─"))
        print(f"  ITERATION {iteration}  —  REWRITE STEP (pass {rewrites_done + 1}/{max_rewrites})")
        print(_hr("─"))
        print()
        print(f"  Calling rewrite model ({config.rewrite_model})...")
        print(f"  Addressing judge's criticisms (score was {verdict.human_author_probability}/100)...")

        try:
            rewritten = rewrite_comment(
                current_text,
                verdict.human_author_probability,
                verdict.reasons,
                persona_context,
                config,
            )
        except Exception as e:
            print(f"\n  ERROR: Rewrite call failed: {e}")
            break

        old_wc = _word_count(current_text)
        new_wc = _word_count(rewritten)
        delta_wc = new_wc - old_wc
        delta_sign = "+" if delta_wc >= 0 else ""

        history.append({
            "step": "rewrite",
            "iteration": iteration,
            "text": rewritten,
            "word_count": new_wc,
            "previous_score": verdict.human_author_probability,
            "previous_reasons": verdict.reasons,
        })

        # Display rewritten comment
        print()
        print(_box(
            f"REWRITTEN COMMENT (pass {rewrites_done + 1})",
            f"Word count: {new_wc} ({delta_sign}{delta_wc} from previous)\n\n{rewritten}"
        ))

        # Show a quick text-level comparison
        print()
        print(f"  Word count change: {old_wc} → {new_wc} ({delta_sign}{delta_wc})")

        current_text = rewritten
        rewrites_done += 1

        if not skip_to_end:
            action = _wait_for_user(no_pause)
            if action == "quit":
                break
            if action == "skip":
                skip_to_end = True

    # ── Final summary ─────────────────────────────────────────────────────
    print()
    print(_hr("═"))
    print(f"  SUMMARY")
    print(_hr("═"))
    print()

    if score_progression:
        progression_str = " → ".join(str(s) for s in score_progression)
        print(f"  Score progression : {progression_str}")
        print(f"  Rewrites performed: {rewrites_done}")
        final_score = score_progression[-1]
        passed = final_score > 50
        print(f"  Final score       : {final_score}/100  ({'PASSED' if passed else 'FAILED'})")

        if len(score_progression) > 1:
            delta = score_progression[-1] - score_progression[0]
            delta_sign = "+" if delta >= 0 else ""
            direction = "IMPROVING ↑" if delta > 0 else ("NO CHANGE" if delta == 0 else "DEGRADING ↓")
            print(f"  Score change      : {delta_sign}{delta}  ({direction})")
    else:
        print(f"  No judge evaluations were completed.")

    # ── Show all versions side by side ────────────────────────────────────
    # print()
    # print(_hr("─"))
    # print(f"  ALL COMMENT VERSIONS")
    # print(_hr("─"))

    # version_num = 0
    # for entry in history:
    #     if entry["step"] in ("initial", "rewrite"):
    #         label = "INITIAL (raw)" if entry["step"] == "initial" else f"REWRITE {version_num}"
    #         print()
    #         # Find the judge score for this version (if any)
    #         score_label = ""
    #         for h in history:
    #             if h["step"] == "judge" and h.get("text") == entry["text"]:
    #                 score_label = f" — Judge score: {h['score']}/100"
    #                 break
    #         print(_box(
    #             f"VERSION {version_num}: {label}{score_label}",
    #             f"Word count: {entry['word_count']}\n\n{entry['text']}"
    #         ))
    #         version_num += 1

    print()
    print(_hr("═"))

    # ── Save to JSON ──────────────────────────────────────────────────────
    if save_json:
        output = {
            "timestamp": datetime.now().isoformat(),
            "persona_context": persona_context,
            "metadata": metadata or {},
            "config": {
                "max_rewrites": max_rewrites,
                "judge_model": config.judge_model,
                "rewrite_model": config.rewrite_model,
            },
            "score_progression": score_progression,
            "rewrites_performed": rewrites_done,
            "passed": score_progression[-1] > 50 if score_progression else False,
            "history": history,
        }
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to: {save_json}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="test_rewriter",
        description=(
            "Interactive step-through test for the syncom judge→rewrite loop. "
            "Generates ONE comment, then lets you examine each judge and rewrite "
            "step to verify the loop improves humanness."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Full pipeline: generate + step through
              python -m syncom.test_rewriter --docket CMS-2025-0050

              # More iterations
              python -m syncom.test_rewriter --docket CMS-2025-0050 --max-rewrites 5

              # Test an existing comment
              python -m syncom.test_rewriter --comment-file my_comment.txt

              # Non-interactive (no pauses)
              python -m syncom.test_rewriter --docket CMS-2025-0050 --no-pause

              # Save results
              python -m syncom.test_rewriter --docket CMS-2025-0050 --save-json out.json
        """),
    )

    # Input source (mutually exclusive)
    source = p.add_argument_group("comment source (pick one)")
    source.add_argument(
        "--docket",
        metavar="ID",
        default=None,
        help=(
            "Docket ID to generate a comment from (e.g. CMS-2025-0050). "
            "Uses the full pipeline: campaign plan → persona → frame → comment."
        ),
    )
    source.add_argument(
        "--comment-file",
        metavar="PATH",
        default=None,
        help=(
            "Path to a text file containing a pre-existing comment to test. "
            "Skips the generation step."
        ),
    )
    source.add_argument(
        "--comment-text",
        metavar="TEXT",
        default=None,
        help="Inline comment text to test (for quick tests).",
    )

    # Persona context (used when --comment-file or --comment-text)
    persona = p.add_argument_group("persona context (for --comment-file / --comment-text)")
    persona.add_argument("--persona-name", default="Jane Smith")
    persona.add_argument("--persona-occupation", default="retired nurse")
    persona.add_argument("--persona-org", default="None")
    persona.add_argument("--persona-state", default="Ohio")
    persona.add_argument("--persona-age", default="62")
    persona.add_argument("--persona-archetype", default="individual_consumer")

    # Voice selection
    voice = p.add_argument_group("voice selection (for --docket mode)")
    voice.add_argument(
        "--voice",
        metavar="VOICE_ID",
        default=None,
        help=(
            "Voice ID to use (e.g. 'individual_consumer-low', 'industry_high_org'). "
            "If omitted, you'll be prompted to choose interactively."
        ),
    )

    # Rewriter options
    rewrite = p.add_argument_group("rewriter options")
    rewrite.add_argument(
        "--max-rewrites",
        type=int,
        default=None,
        metavar="N",
        help="Max rewrite iterations (default: 2).",
    )

    # Behavior
    p.add_argument(
        "--no-pause",
        action="store_true",
        help="Don't pause between steps (print everything at once).",
    )
    p.add_argument(
        "--save-json",
        metavar="PATH",
        default=None,
        help="Save full history to a JSON file.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for comment generation (default 42).",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    comment_text = None
    persona_context = None
    metadata = None

    # ── Determine comment source ──────────────────────────────────────────
    sources = sum([
        args.docket is not None,
        args.comment_file is not None,
        args.comment_text is not None,
    ])

    if sources == 0:
        # Default to generating from CMS-2025-0050 if it exists
        if os.path.exists("CMS-2025-0050"):
            args.docket = "CMS-2025-0050"
        else:
            print("Error: provide --docket, --comment-file, or --comment-text", file=sys.stderr)
            return 1
    elif sources > 1:
        print("Error: provide only one of --docket, --comment-file, --comment-text", file=sys.stderr)
        return 1

    # ── Source A: Generate via full pipeline ───────────────────────────────
    if args.docket:
        print(f"\n  Generating one comment from docket {args.docket}...")
        print(f"  (This involves several LLM calls: world model, persona hook, frame, comment)")
        print()

        try:
            comment_text, persona_obj, persona_context, metadata = _generate_one_comment(
                docket_id=args.docket,
                seed=args.seed,
                verbose=True,
                voice_override=args.voice,
                no_pause=args.no_pause,
            )
        except Exception as e:
            print(f"\n  ERROR during generation: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    # ── Source B: Load from file ──────────────────────────────────────────
    elif args.comment_file:
        if not os.path.exists(args.comment_file):
            print(f"Error: file not found: {args.comment_file}", file=sys.stderr)
            return 1
        with open(args.comment_file, "r", encoding="utf-8") as f:
            comment_text = f.read().strip()

    # ── Source C: Inline text ─────────────────────────────────────────────
    elif args.comment_text:
        comment_text = args.comment_text.strip()

    # ── Build persona context if not from pipeline ────────────────────────
    if persona_context is None:
        persona_context = {
            "name": args.persona_name,
            "occupation": args.persona_occupation,
            "org_name": args.persona_org,
            "state": args.persona_state,
            "age": args.persona_age,
            "archetype": args.persona_archetype,
        }

    if not comment_text:
        print("Error: no comment text to test.", file=sys.stderr)
        return 1

    # ── Determine max rewrites ────────────────────────────────────────────
    max_rewrites = args.max_rewrites if args.max_rewrites is not None else 2

    # ── Run the step-through ──────────────────────────────────────────────
    run_step_through(
        comment_text=comment_text,
        persona_context=persona_context,
        max_rewrites=max_rewrites,
        no_pause=args.no_pause,
        save_json=args.save_json,
        metadata=metadata,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
