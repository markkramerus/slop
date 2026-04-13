"""
pipeline.py — Orchestrates the full synthetic comment generation pipeline.

This module provides two modes:

1. **Direct mode** (run): Manual specification of vector, objective,
   and volume. Uses the original attack vector taxonomy (1-4).

2. **Campaign mode** (run_campaign): Driven by a v2.0
   campaign plan with Bayesian voice×argument allocation:
   - P(V) from campaign_voices
   - P(A|V) ∝ w(A) × f(A,V) where f = affinity_boost for best_voices
   - No vector taxonomy; style emerges from voice profiles

Both modes handle progress reporting (tqdm), retry logic, and QC.
"""

from __future__ import annotations
import random
import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from .argument_mapper import (
    map_argument_async, AttackVector,
    build_campaign_frame_async,
)
from config import Config
from .export import export_to_txt
from shuffler.translate_to_psv_format import translate_synthetic_to_psv
from .generator import generate_comment_async, GeneratedComment, PromptControls, build_user_prompt_template
from .org_pool import load_or_build_org_pool
from .persona import (
    sample_persona_async, sample_persona_by_voice_id_async,
)
from .phrase_check import run_phrase_check
from .phrase_fix import run_phrase_fix, build_rule_ngrams
from .quality_control import QualityController
from .world_model import build_or_load_world_model, WorldModel
from stylometry.stylometry_loader import build_population_model


# ── Run result ────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    total_attempted: int = 0
    total_accepted: int = 0
    total_qc_failed: int = 0
    total_skipped: int = 0
    output_path: str = ""
    world_model_summary: dict = field(default_factory=dict)
    comments: list[GeneratedComment] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Run complete:",
            f"  Attempted : {self.total_attempted}",
            f"  Accepted  : {self.total_accepted}",
            f"  QC failed : {self.total_qc_failed}",
            f"  Skipped   : {self.total_skipped}",
            f"  Output    : {self.output_path}",
        ]
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _distribute_volume(
    volume: int,
    weights: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, int]:
    """
    Distribute `volume` items across keys according to `weights`.
    Returns a dict mapping key → count, where counts sum to `volume`.
    """
    total_w = sum(weights.values())
    if total_w == 0:
        keys = list(weights.keys())
        base = volume // len(keys)
        remainder = volume % len(keys)
        result = {k: base for k in keys}
        for k in rng.choice(keys, size=remainder, replace=False):
            result[k] += 1
        return result

    raw = {k: (v / total_w) * volume for k, v in weights.items()}
    allocation: dict[str, int] = {}
    assigned = 0

    for k, v in raw.items():
        allocation[k] = int(v)
        assigned += allocation[k]

    remainder = volume - assigned
    if remainder > 0:
        fractional = {k: v - int(v) for k, v in raw.items()}
        keys_sorted = sorted(fractional.keys(), key=lambda k: (-fractional[k], rng.random()))
        for i in range(remainder):
            allocation[keys_sorted[i % len(keys_sorted)]] += 1

    return allocation


def _sample_argument_angle_for_voice(
    plan,
    voice_id: str,
    rng: np.random.Generator,
) -> tuple[str, str, list[str], str, list[str]]:
    """
    Sample an argument angle for a given voice using P(A|V).
    Returns (angle_id, angle_text, key_claims, rhetorical_approach, avoid).
    """
    if not plan.argument_angles:
        return ("", "", [], "", [])

    probs = plan.compute_angle_distribution(voice_id)
    idx = int(rng.choice(len(plan.argument_angles), p=probs))
    angle = plan.argument_angles[idx]
    return (
        angle.id,
        angle.angle,
        angle.key_claims,
        angle.rhetorical_approach,
        angle.avoid,
    )

# ── Campaign pipeline ──────────────────────────────────────────────────

async def _generate_one_campaign_comment_async(
    world_model: WorldModel,
    voice_id: str,
    objective: str,
    argument_angle: str,
    config: Config,
    qc: QualityController,
    rng: np.random.Generator,
    max_retries: int,
    verbose: bool,
    docket_id: str,
    key_claims: list[str] | None = None,
    rhetorical_approach: str = "",
    avoid: list[str] | None = None,
    org_name: str = "",
    scenario_brief: str = "",
    prompt_template: str = "",
    skip_judge_rewrite: bool = False,
) -> tuple[GeneratedComment | None, int, int]:
    """
    Async helper: generate one campaign-plan-aware comment with retries.
    Returns (comment, attempts, qc_failures).

    Note: org_name is pre-assigned before this coroutine is launched to avoid
    race conditions on the shared OrgPool when running concurrently.
    """
    attempts = 0
    qc_failures = 0
    last_comment: GeneratedComment | None = None  # retained if all retries fail (for --include-failed-qc)

    for attempt in range(max_retries):
        attempts += 1
        try:
            persona = await sample_persona_by_voice_id_async(
                voice_id, world_model, config, rng, docket_id=docket_id,
                org_pool=None,  # org_name already pre-assigned below
            )
            # Override the org_name with the pre-assigned value
            if org_name:
                persona.org_name = org_name
            frame = await build_campaign_frame_async(
                objective, argument_angle, persona, world_model, config, rng,
                key_claims=key_claims,
                rhetorical_approach=rhetorical_approach,
                avoid=avoid,
            )
            comment = await generate_comment_async(
                persona, frame, world_model, 0, objective, config, rng,
                scenario_brief=scenario_brief,
                prompt_template=prompt_template,
                skip_judge_rewrite=skip_judge_rewrite,
            )
            comment.argument_angle = argument_angle
            comment.voice_id = voice_id
            last_comment = comment  # save before QC so we can return it if all retries fail
            qc_result = await qc.check_async(comment)

            if qc_result.passed:
                return comment, attempts, qc_failures
            else:
                qc_failures += 1

        except Exception as exc:
            if verbose:
                print(f"      Error in async campaign generation (attempt {attempt+1}/{max_retries}): {exc}", file=sys.stderr)

    # All retries exhausted.  Return the last generated comment (qc_passed=False is already
    # set by qc.check_async) so that --include-failed-qc can still export it.
    return last_comment, attempts, qc_failures


def run_campaign_async(
    docket_id: str,
    rule_text: str,
    campaign_plan_path: str,
    volume: int,
    output_path: str,
    config: Config | None = None,
    seed: int = random.randint(0, 2**32 - 1),
    similarity_threshold: float = 0.92,
    max_retries: int = 3,
    comment_period_days: int = 60,
    include_failed_qc: bool = False,
    skip_relevance_check: bool = False,
    skip_argument_check: bool = False,
    skip_embedding_check: bool = False,
    skip_word_count_check: bool = False,
    skip_judge_rewrite: bool = False,
    skip_phrase_check: bool = False,
    skip_phrase_fix: bool = False,
    verbose: bool = True,
    max_concurrent: int = 10,
    rebuild_world_model: bool = False,
    rebuild_org_pool: bool = False,
    prompt_controls: PromptControls | None = None,
) -> RunResult:
    """
    Run the v2.0 campaign-plan-aware pipeline with async parallelization.
    """
    from campaign.campaign_models import CampaignPlan

    if config is None:
        config = Config()
    config.validate()

    rng = np.random.default_rng(seed)
    result = RunResult()

    # ── Load campaign plan ────────────────────────────────────────────────
    if verbose:
        print(f"[0/4] Loading campaign plan: {campaign_plan_path}", file=sys.stderr)
    plan = CampaignPlan.load(campaign_plan_path)
    objective = plan.objective

    if verbose:
        obj_display = f"{objective[:80]}…" if len(objective) > 80 else objective
        print(f"      Objective: {obj_display}", file=sys.stderr)
        print(f"      Angles: {len(plan.argument_angles)}", file=sys.stderr)
        print(f"      Voices: {len(plan.campaign_voices)}", file=sys.stderr)
        print(f"      Affinity boost (α): {plan.affinity_boost}", file=sys.stderr)

    # ── Stage 1: Load population from stylometry ─────────────────────────
    if verbose:
        print(f"[1/4] Loading population model from stylometry: {docket_id}", file=sys.stderr)
    population = build_population_model(docket_id)

    # ── Stage 2: Build world model ────────────────────────────────────────
    if verbose:
        print(f"[2/4] Analysing proposed rule…", file=sys.stderr)
    world_model = build_or_load_world_model(
        rule_text=rule_text,
        population=population,
        config=config,
        docket_id=docket_id,
        rebuild=rebuild_world_model,
        verbose=verbose,
    )
    result.world_model_summary = world_model.to_dict()
    if verbose:
        print(f"      Rule  : {world_model.rule_title}", file=sys.stderr)
        print(f"      Agency: {world_model.agency}", file=sys.stderr)

    # ── Stage 2b: Compute voice allocation and load/build org pool ────────
    # Voice allocation is computed here (before pool build) so we can pass
    # per-archetype counts to the pool builder for right-sized generation.
    voice_allocation = _distribute_volume(
        volume, plan.normalized_voice_weights(), rng
    )

    # Derive per-archetype counts from the voice allocation
    from syncom.persona import parse_voice_id as _parse_voice_id_for_pool_async
    archetype_counts_async: dict[str, int] = {}
    for voice_id, count in voice_allocation.items():
        archetype, _ = _parse_voice_id_for_pool_async(voice_id)
        archetype_counts_async[archetype] = archetype_counts_async.get(archetype, 0) + count

    if verbose:
        print(f"[2b/4] Loading org pool…", file=sys.stderr)
    org_pool = load_or_build_org_pool(
        world_model=world_model,
        population=population,
        config=config,
        docket_id=docket_id,
        volume_hint=volume,
        archetype_counts=archetype_counts_async,
        rebuild=rebuild_org_pool,
        verbose=verbose,
    )

    # ── Stage 3: Distribute and generate (async) ─────────────────────────
    # (voice_allocation already computed above)

    if verbose:
        print(f"[3/4] Generating {volume} comment(s) with {max_concurrent}-way parallelism:", file=sys.stderr)
        for v, count in sorted(voice_allocation.items(), key=lambda x: -x[1]):
            print(f"      {v:35s} {count:>4d} comments", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(plan.allocation_summary(volume), file=sys.stderr)
        print(f"", file=sys.stderr)

    # Skip the word-count structural check when explicitly requested OR when
    # the structural prompt block (which carries the target word count) was not
    # included in the generator prompt — the LLM was never told the target, so
    # checking against it is meaningless and will silently reject valid comments.
    _pctrl = prompt_controls or PromptControls()
    _skip_wc = skip_word_count_check or not _pctrl.use_voice_stats_block

    qc = QualityController(
        config=config,
        objective=objective,
        similarity_threshold=similarity_threshold,
        skip_relevance_check=skip_relevance_check,
        skip_argument_check=skip_argument_check,
        skip_embedding_check=skip_embedding_check,
        skip_word_count_check=_skip_wc,
    )

    # Resolve the full scenario brief to pass into each generator call.
    # Prefer the raw brief text; fall back to the LLM-generated summary for
    # older plans that were saved before scenario_brief was added to the schema.
    scenario_brief = plan.scenario_brief or plan.scenario_summary or ""

    # Build the prompt template once for this run (shared across all comments)
    prompt_template = build_user_prompt_template(prompt_controls or PromptControls())

    # Build task list: (voice_id, angle_text, key_claims, rhetorical_approach, avoid, org_name)
    # Org names are pre-assigned here (before async tasks launch) to avoid race conditions
    # on the shared OrgPool when multiple coroutines run concurrently.
    from syncom.persona import parse_voice_id as _parse_voice_id
    task_specs: list[tuple[str, str, list[str], str, list[str], str]] = []
    for voice_id, voice_count in sorted(voice_allocation.items()):
        for _ in range(voice_count):
            _, angle_text, key_claims, rhetorical_approach, avoid = (
                _sample_argument_angle_for_voice(plan, voice_id, rng)
            )
            # Pre-assign org name synchronously to avoid concurrent pool access
            archetype, _ = _parse_voice_id(voice_id)
            pre_org_name = org_pool.sample(archetype, rng=rng) if archetype != "individual" else ""
            task_specs.append((voice_id, angle_text, key_claims, rhetorical_approach, avoid, pre_org_name))

    # Run
    async def _run_all_asynch():
        semaphore = asyncio.Semaphore(max_concurrent)

        async def gen_with_semaphore_async(
            vid: str, angle: str, kc: list[str], ra: str, av: list[str], org: str
        ):
            async with semaphore:
                return await _generate_one_campaign_comment_async(
                    world_model, vid, objective, angle,
                    config, qc, rng,
                    max_retries, verbose, docket_id,
                    key_claims=kc,
                    rhetorical_approach=ra,
                    avoid=av,
                    org_name=org,
                    scenario_brief=scenario_brief,
                    prompt_template=prompt_template,
                    skip_judge_rewrite=skip_judge_rewrite,
                )

        tasks = [gen_with_semaphore_async(v, a, kc, ra, av, org) for v, a, kc, ra, av, org in task_specs]

        all_comments = []
        total_attempted = 0
        total_accepted = 0
        total_qc_failed = 0

        if _TQDM_AVAILABLE and verbose:
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating", unit="comment"):
                comment, attempts, qc_failures = await coro
                total_attempted += attempts
                total_qc_failed += qc_failures
                if comment:
                    all_comments.append(comment)
                    if comment.qc_passed:
                        total_accepted += 1
        else:
            results = await asyncio.gather(*tasks)
            for comment, attempts, qc_failures in results:
                total_attempted += attempts
                total_qc_failed += qc_failures
                if comment:
                    all_comments.append(comment)
                    if comment.qc_passed:
                        total_accepted += 1

        return all_comments, total_attempted, total_accepted, total_qc_failed

    all_comments, attempted, accepted, qc_failed = asyncio.run(_run_all_asynch())

    result.total_attempted = attempted
    result.total_accepted = accepted
    result.total_qc_failed = qc_failed
    result.total_skipped = volume - len(all_comments)  # slots where generation failed entirely
    result.comments = all_comments

    # ── Stage 4: Export ───────────────────────────────────────────────────
    if verbose:
        print(f"[4/4] Exporting → {output_path}", file=sys.stderr)

    n_written = export_to_txt(
        comments=all_comments,
        output_path=output_path,
        timing_deciles=population.timing_deciles,
        comment_period_days=comment_period_days,
        include_failed_qc=include_failed_qc,
        seed=seed,
    )
    result.output_path = output_path

    # ── Phrase repetition check ───────────────────────────────────────────
    phrase_report_path = str(Path(output_path).with_suffix("_phrase_report.md"))
    _world_model_path = os.path.join(docket_id, "world_model.json")
    _wm_path = _world_model_path if os.path.exists(_world_model_path) else None

    # Build the rule n-gram index once so it can be shared by both the phrase
    # check (to filter rule-anchored phrases from the report at collection time)
    # and phrase_fix (for triage).  This avoids rebuilding the index twice and
    # keeps both the initial report and the post-fix re-check report clean.
    _rule_ngrams = build_rule_ngrams(rule_text, _wm_path) if rule_text else None

    if skip_phrase_check:
        if verbose:
            print("[phrase-check] Skipping — disabled by skip_phrase_check.", file=sys.stderr)
    else:
        run_phrase_check(
            comments=all_comments,
            output_path=phrase_report_path,
            only_passed_qc=not include_failed_qc,
            verbose=verbose,
            rule_ngrams=_rule_ngrams,
        )

    # ── Phrase fix: classify and rewrite suspicious repeated phrases ──────
    if skip_phrase_fix:
        if verbose:
            print("[phrase-fix] Skipping — disabled by skip_phrase_fix.", file=sys.stderr)
    else:
        run_phrase_fix(
            psv_path=output_path,
            report_path=phrase_report_path,
            rule_text=rule_text,
            output_path=output_path,
            world_model_path=_wm_path,
            verbose=verbose,
        )

    # ── PSV export (always) ───────────────────────────────────────────────
    psv_path = str(Path(output_path).with_suffix(".psv"))
    if verbose:
        print(f"[export] Writing PSV → {psv_path}", file=sys.stderr)
    translate_synthetic_to_psv(output_path, psv_path)

    if verbose:
        print(result.summary(), file=sys.stderr)

    return result

# Backward-compatible alias — cli.py and __init__.py import `run_campaign`
run_campaign = run_campaign_async

# ── Async direct-mode pipeline ────────────────────────────────────────────────

async def _generate_one_comment_async(
    world_model: WorldModel,
    vector: AttackVector,
    objective: str,
    config: Config,
    qc: QualityController,
    rng: np.random.Generator,
    max_retries: int,
    verbose: bool,
    docket_id: str,
    prompt_template: str = "",
    skip_judge_rewrite: bool = False,
) -> tuple[GeneratedComment | None, int, int]:
    """Async helper: generate and QC one comment (direct mode)."""
    attempts = 0
    qc_failures = 0
    last_comment: GeneratedComment | None = None  # retained if all retries fail (for --include-failed-qc)

    for attempt in range(max_retries):
        attempts += 1
        try:
            persona = await sample_persona_async(world_model, config, rng, docket_id=docket_id)
            frame = await map_argument_async(objective, vector, persona, world_model, config, rng)
            comment = await generate_comment_async(
                persona, frame, world_model, vector, objective, config, rng,
                prompt_template=prompt_template,
                skip_judge_rewrite=skip_judge_rewrite,
            )
            last_comment = comment  # save before QC so we can return it if all retries fail
            # make QC optional
            if qc:
                qc_result = await qc.check_async(comment)

                if qc_result.passed:
                    return comment, attempts, qc_failures
                else:
                    qc_failures += 1
            else:
                qc_failures = 0

        except Exception as exc:
            if verbose:
                print(f"      Error in async generation (attempt {attempt+1}/{max_retries}): {exc}", file=sys.stderr)

    # All retries exhausted.  Return the last generated comment (qc_passed=False is already
    # set by qc.check_async) so that --include-failed-qc can still export it.
    return last_comment, attempts, qc_failures


def run(
    docket_id: str,
    rule_text: str,
    vector: AttackVector,
    objective: str,
    volume: int,
    output_path: str,
    config: Config | None = None,
    seed: int = random.randint(0, 2**32 - 1),
    similarity_threshold: float = 0.92,
    max_retries: int = 3,
    comment_period_days: int = 60,
    include_failed_qc: bool = False,
    skip_relevance_check: bool = False,
    skip_argument_check: bool = False,
    skip_embedding_check: bool = False,
    skip_word_count_check: bool = False,
    skip_judge_rewrite: bool = False,
    skip_phrase_check: bool = False,
    skip_phrase_fix: bool = False,
    verbose: bool = True,
    max_concurrent: int = 10,
    rebuild_world_model: bool = False,
    prompt_controls: PromptControls | None = None,
) -> RunResult:
    """Async direct-mode pipeline (backward compatible)."""
    if config is None:
        config = Config()
    config.validate()

    rng = np.random.default_rng(seed)
    result = RunResult()

    if verbose:
        print(f"[1/4] Loading population model from stylometry: {docket_id}", file=sys.stderr)
    population = build_population_model(docket_id)

    if verbose:
        print(f"[2/4] Analysing proposed rule…", file=sys.stderr)

    world_model = build_or_load_world_model(
        rule_text=rule_text,
        population=population,
        config=config,
        docket_id=docket_id,
        rebuild=rebuild_world_model,
        verbose=verbose,
    )
    result.world_model_summary = world_model.to_dict()
    if verbose:
        print(f"      Rule  : {world_model.rule_title}", file=sys.stderr)
        print(f"      Agency: {world_model.agency}", file=sys.stderr)

    if verbose:
        print(f"[3/4] Generating {volume} comment(s) (vector {vector}) with {max_concurrent}-way parallelism…", file=sys.stderr)

    # Build the prompt template once for this run (shared across all comments)
    prompt_template = build_user_prompt_template(prompt_controls or PromptControls())

    # Skip the word-count structural check when explicitly requested OR when
    # the structural prompt block (which carries the target word count) was not
    # included in the generator prompt — the LLM was never told the target, so
    # checking against it is meaningless and will silently reject valid comments.
    _pctrl = prompt_controls or PromptControls()
    _skip_wc = skip_word_count_check or not _pctrl.use_voice_stats_block

    qc = QualityController(
        config=config,
        objective=objective,
        similarity_threshold=similarity_threshold,
        skip_relevance_check=skip_relevance_check,
        skip_argument_check=skip_argument_check,
        skip_embedding_check=skip_embedding_check,
        skip_word_count_check=_skip_wc,
    )

    async def _run_all_asynch():
        semaphore = asyncio.Semaphore(max_concurrent)

        async def gen_with_semaphore_async():
            async with semaphore:
                return await _generate_one_comment_async(
                    world_model, vector, objective, config, qc, rng,
                    max_retries, verbose, docket_id,
                    prompt_template=prompt_template,
                    skip_judge_rewrite=skip_judge_rewrite,
                )

        tasks = [gen_with_semaphore_async() for _ in range(volume)]
        all_comments = []
        total_attempted = 0
        total_accepted = 0
        total_qc_failed = 0

        if _TQDM_AVAILABLE and verbose:
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating", unit="comment"):
                comment, attempts, qc_failures = await coro
                total_attempted += attempts
                total_qc_failed += qc_failures
                if comment:
                    all_comments.append(comment)
                    if comment.qc_passed:
                        total_accepted += 1
        else:
            results = await asyncio.gather(*tasks)
            for comment, attempts, qc_failures in results:
                total_attempted += attempts
                total_qc_failed += qc_failures
                if comment:
                    all_comments.append(comment)
                    if comment.qc_passed:
                        total_accepted += 1

        return all_comments, total_attempted, total_accepted, total_qc_failed

    all_comments, attempted, accepted, qc_failed = asyncio.run(_run_all_asynch())

    result.total_attempted = attempted
    result.total_accepted = accepted
    result.total_qc_failed = qc_failed
    result.total_skipped = volume - len(all_comments)  # slots where generation failed entirely
    result.comments = all_comments

    if verbose:
        print(f"[4/4] Exporting → {output_path}", file=sys.stderr)

    n_written = export_to_txt(
        comments=all_comments,
        output_path=output_path,
        timing_deciles=population.timing_deciles,
        comment_period_days=comment_period_days,
        include_failed_qc=include_failed_qc,
        seed=seed,
    )
    result.output_path = output_path

    # ── Phrase repetition check ───────────────────────────────────────────
    phrase_report_path = str(Path(output_path).with_suffix("_phrase_report.md"))
    _world_model_path = os.path.join(docket_id, "world_model.json")
    _wm_path = _world_model_path if os.path.exists(_world_model_path) else None

    # Build the rule n-gram index once so it can be shared by both the phrase
    # check (to filter rule-anchored phrases from the report at collection time)
    # and phrase_fix (for triage).  This avoids rebuilding the index twice and
    # keeps both the initial report and the post-fix re-check report clean.
    _rule_ngrams = build_rule_ngrams(rule_text, _wm_path) if rule_text else None

    if skip_phrase_check:
        if verbose:
            print("[phrase-check] Skipping — disabled by skip_phrase_check.", file=sys.stderr)
    else:
        run_phrase_check(
            comments=all_comments,
            output_path=phrase_report_path,
            only_passed_qc=not include_failed_qc,
            verbose=verbose,
            rule_ngrams=_rule_ngrams,
        )

    # ── Phrase fix: classify and rewrite suspicious repeated phrases ──────
    if skip_phrase_fix:
        if verbose:
            print("[phrase-fix] Skipping — disabled by skip_phrase_fix.", file=sys.stderr)
    else:
        run_phrase_fix(
            psv_path=output_path,
            report_path=phrase_report_path,
            rule_text=rule_text,
            output_path=output_path,
            world_model_path=_wm_path,
            verbose=verbose,
        )

    # ── PSV export (always) ───────────────────────────────────────────────
    psv_path = str(Path(output_path).with_suffix(".psv"))
    if verbose:
        print(f"[export] Writing PSV → {psv_path}", file=sys.stderr)
    translate_synthetic_to_psv(output_path, psv_path)

    if verbose:
        print(result.summary(), file=sys.stderr)

    return result
