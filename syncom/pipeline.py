"""
pipeline.py — Orchestrates the full synthetic comment generation pipeline.

This module provides two modes:

1. **Direct mode** (run / run_async): Manual specification of vector, objective,
   and volume. Uses the original attack vector taxonomy (1-4).

2. **Campaign mode** (run_campaign / run_campaign_async): Driven by a v2.0
   campaign plan with Bayesian voice×argument allocation:
   - P(V) from campaign_voices
   - P(A|V) ∝ w(A) × f(A,V) where f = affinity_boost for best_voices
   - No vector taxonomy; style emerges from voice profiles

Both modes handle progress reporting (tqdm), retry logic, and QC.
"""

from __future__ import annotations

import asyncio
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
    map_argument, map_argument_async, AttackVector,
    build_campaign_frame, build_campaign_frame_async,
)
from config import Config
from .export import export_to_txt
from .generator import generate_comment, generate_comment_async, GeneratedComment
from .persona import (
    sample_persona, sample_persona_async,
    sample_persona_by_voice_id, sample_persona_by_voice_id_async,
)
from .quality_control import QualityController
from .world_model import build_world_model, WorldModel
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
) -> tuple[str, str]:
    """
    Sample an argument angle for a given voice using P(A|V).
    Returns (angle_id, angle_text).
    """
    if not plan.argument_angles:
        return ("", "")

    probs = plan.compute_angle_distribution(voice_id)
    idx = int(rng.choice(len(plan.argument_angles), p=probs))
    angle = plan.argument_angles[idx]
    return (angle.id, angle.angle)


# ── Campaign-plan-aware pipeline (v2.0) ──────────────────────────────────────

def run_campaign(
    docket_id: str,
    rule_text: str,
    campaign_plan_path: str,
    volume: int,
    output_path: str,
    config: Config | None = None,
    seed: int = 42,
    similarity_threshold: float = 0.92,
    max_retries: int = 3,
    comment_period_days: int = 60,
    include_failed_qc: bool = False,
    skip_relevance_check: bool = False,
    skip_argument_check: bool = False,
    skip_embedding_check: bool = False,
    verbose: bool = True,
) -> RunResult:
    """
    Run the synthetic comment generation pipeline using a v2.0 campaign plan.

    The campaign plan specifies:
    - campaign_voices: P(V) distribution over voice groups
    - argument_angles with best_voices: used to compute P(A|V)
    - affinity_boost: the α multiplier for voice-argument channeling

    For each comment:
    1. Sample voice V from P(V)
    2. Sample argument A from P(A|V) ∝ w(A) × f(A,V)
    3. Instantiate persona for voice V
    4. Build expression frame for (A, persona)
    5. Generate comment
    6. QC check
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
    world_model = build_world_model(
        rule_text=rule_text,
        population=population,
        config=config,
        docket_id=docket_id,
    )
    result.world_model_summary = world_model.to_dict()
    if verbose:
        print(f"      Rule  : {world_model.rule_title}", file=sys.stderr)
        print(f"      Agency: {world_model.agency}", file=sys.stderr)

    # ── Stage 3: Distribute volume and generate ──────────────────────────
    voice_allocation = _distribute_volume(
        volume, plan.normalized_voice_weights(), rng
    )

    if verbose:
        print(f"[3/4] Generating {volume} comment(s) across voices:", file=sys.stderr)
        for v, count in sorted(voice_allocation.items(), key=lambda x: -x[1]):
            print(f"      {v:35s} {count:>4d} comments", file=sys.stderr)

        # Show allocation matrix
        print(f"", file=sys.stderr)
        print(plan.allocation_summary(volume), file=sys.stderr)
        print(f"", file=sys.stderr)

    qc = QualityController(
        config=config,
        objective=objective,
        similarity_threshold=similarity_threshold,
        skip_relevance_check=skip_relevance_check,
        skip_argument_check=skip_argument_check,
        skip_embedding_check=skip_embedding_check,
    )

    all_comments: list[GeneratedComment] = []
    accepted = 0
    attempted = 0
    skipped = 0

    total_to_generate = sum(voice_allocation.values())

    if _TQDM_AVAILABLE and verbose:
        pbar = tqdm(total=total_to_generate, desc="Generating", unit="comment")
    else:
        pbar = None

    for voice_id, voice_count in sorted(voice_allocation.items()):
        for _ in range(voice_count):
            success = False
            # Sample argument angle using P(A|V)
            angle_id, angle_text = _sample_argument_angle_for_voice(plan, voice_id, rng)

            for attempt in range(max_retries):
                attempted += 1
                try:
                    persona = sample_persona_by_voice_id(
                        voice_id, world_model, config, rng, docket_id=docket_id,
                    )
                    frame = build_campaign_frame(
                        objective, angle_text, persona, world_model, config, rng,
                    )
                    comment = generate_comment(
                        persona, frame, world_model, 0, objective, config,
                    )
                    comment.argument_angle = angle_text
                    comment.voice_id = voice_id
                    qc_result = qc.check(comment)
                    all_comments.append(comment)

                    if qc_result.passed:
                        accepted += 1
                        success = True
                        break
                    else:
                        result.total_qc_failed += 1

                except Exception as exc:
                    if verbose and pbar is None:
                        print(f"      Error (attempt {attempt+1}/{max_retries}): {exc}", file=sys.stderr)

            if not success:
                skipped += 1

            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    result.total_attempted = attempted
    result.total_accepted = accepted
    result.total_skipped = skipped
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

    if verbose:
        print(result.summary(), file=sys.stderr)

    return result


# ── Async campaign pipeline ──────────────────────────────────────────────────

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
) -> tuple[GeneratedComment | None, int, int]:
    """
    Async helper: generate one campaign-plan-aware comment with retries.
    Returns (comment, attempts, qc_failures).
    """
    attempts = 0
    qc_failures = 0

    for attempt in range(max_retries):
        attempts += 1
        try:
            persona = await sample_persona_by_voice_id_async(
                voice_id, world_model, config, rng, docket_id=docket_id,
            )
            frame = await build_campaign_frame_async(
                objective, argument_angle, persona, world_model, config, rng,
            )
            comment = await generate_comment_async(
                persona, frame, world_model, 0, objective, config,
            )
            comment.argument_angle = argument_angle
            comment.voice_id = voice_id
            qc_result = await qc.check_async(comment)

            if qc_result.passed:
                return comment, attempts, qc_failures
            else:
                qc_failures += 1

        except Exception as exc:
            if verbose:
                print(f"      Error in async campaign generation (attempt {attempt+1}/{max_retries}): {exc}", file=sys.stderr)

    return None, attempts, qc_failures


def run_campaign_async(
    docket_id: str,
    rule_text: str,
    campaign_plan_path: str,
    volume: int,
    output_path: str,
    config: Config | None = None,
    seed: int = 42,
    similarity_threshold: float = 0.92,
    max_retries: int = 3,
    comment_period_days: int = 60,
    include_failed_qc: bool = False,
    skip_relevance_check: bool = False,
    skip_argument_check: bool = False,
    skip_embedding_check: bool = False,
    verbose: bool = True,
    max_concurrent: int = 10,
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
    world_model = build_world_model(
        rule_text=rule_text,
        population=population,
        config=config,
        docket_id=docket_id,
    )
    result.world_model_summary = world_model.to_dict()
    if verbose:
        print(f"      Rule  : {world_model.rule_title}", file=sys.stderr)
        print(f"      Agency: {world_model.agency}", file=sys.stderr)

    # ── Stage 3: Distribute and generate (async) ─────────────────────────
    voice_allocation = _distribute_volume(
        volume, plan.normalized_voice_weights(), rng
    )

    if verbose:
        print(f"[3/4] Generating {volume} comment(s) with {max_concurrent}-way parallelism:", file=sys.stderr)
        for v, count in sorted(voice_allocation.items(), key=lambda x: -x[1]):
            print(f"      {v:35s} {count:>4d} comments", file=sys.stderr)
        print(f"", file=sys.stderr)
        print(plan.allocation_summary(volume), file=sys.stderr)
        print(f"", file=sys.stderr)

    qc = QualityController(
        config=config,
        objective=objective,
        similarity_threshold=similarity_threshold,
        skip_relevance_check=skip_relevance_check,
        skip_argument_check=skip_argument_check,
        skip_embedding_check=skip_embedding_check,
    )

    # Build task list: (voice_id, argument_angle) for each comment
    task_specs: list[tuple[str, str]] = []
    for voice_id, voice_count in sorted(voice_allocation.items()):
        for _ in range(voice_count):
            _, angle_text = _sample_argument_angle_for_voice(plan, voice_id, rng)
            task_specs.append((voice_id, angle_text))

    # Run async
    async def _run_all():
        semaphore = asyncio.Semaphore(max_concurrent)

        async def gen_with_semaphore(vid: str, angle: str):
            async with semaphore:
                return await _generate_one_campaign_comment_async(
                    world_model, vid, objective, angle,
                    config, qc, rng,
                    max_retries, verbose, docket_id,
                )

        tasks = [gen_with_semaphore(v, a) for v, a in task_specs]

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
                    total_accepted += 1
        else:
            results = await asyncio.gather(*tasks)
            for comment, attempts, qc_failures in results:
                total_attempted += attempts
                total_qc_failed += qc_failures
                if comment:
                    all_comments.append(comment)
                    total_accepted += 1

        return all_comments, total_attempted, total_accepted, total_qc_failed

    all_comments, attempted, accepted, qc_failed = asyncio.run(_run_all())

    result.total_attempted = attempted
    result.total_accepted = accepted
    result.total_qc_failed = qc_failed
    result.total_skipped = volume - accepted
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

    if verbose:
        print(result.summary(), file=sys.stderr)

    return result


# ── Direct-mode pipeline (backward compatible) ───────────────────────────────

def run(
    docket_id: str,
    rule_text: str,
    vector: AttackVector,
    objective: str,
    volume: int,
    output_path: str,
    config: Config | None = None,
    seed: int = 42,
    similarity_threshold: float = 0.92,
    max_retries: int = 3,
    comment_period_days: int = 60,
    include_failed_qc: bool = False,
    skip_relevance_check: bool = False,
    skip_argument_check: bool = False,
    skip_embedding_check: bool = False,
    verbose: bool = True,
) -> RunResult:
    """
    Run the full synthetic comment generation pipeline (direct mode).
    Uses a single attack vector for all comments.
    """
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
    world_model = build_world_model(
        rule_text=rule_text,
        population=population,
        config=config,
        docket_id=docket_id,
    )
    result.world_model_summary = world_model.to_dict()
    if verbose:
        print(f"      Rule  : {world_model.rule_title}", file=sys.stderr)
        print(f"      Agency: {world_model.agency}", file=sys.stderr)

    if verbose:
        print(f"[3/4] Generating {volume} comment(s) (vector {vector})…", file=sys.stderr)

    qc = QualityController(
        config=config,
        objective=objective,
        similarity_threshold=similarity_threshold,
        skip_relevance_check=skip_relevance_check,
        skip_argument_check=skip_argument_check,
        skip_embedding_check=skip_embedding_check,
    )

    all_comments: list[GeneratedComment] = []
    accepted = 0
    attempted = 0
    skipped = 0

    iter_obj = range(volume)
    if _TQDM_AVAILABLE and verbose:
        iter_obj = tqdm(iter_obj, desc="Generating", unit="comment")

    for _ in iter_obj:
        success = False
        for attempt in range(max_retries):
            attempted += 1
            try:
                persona = sample_persona(world_model, config, rng, docket_id=docket_id)
                frame = map_argument(objective, vector, persona, world_model, config, rng)
                comment = generate_comment(persona, frame, world_model, vector, objective, config)
                qc_result = qc.check(comment)
                all_comments.append(comment)

                if qc_result.passed:
                    accepted += 1
                    success = True
                    break
                else:
                    result.total_qc_failed += 1
            except Exception as exc:
                if verbose and not _TQDM_AVAILABLE:
                    print(f"      Error (attempt {attempt+1}/{max_retries}): {exc}", file=sys.stderr)

        if not success:
            skipped += 1

    result.total_attempted = attempted
    result.total_accepted = accepted
    result.total_skipped = skipped
    result.comments = all_comments

    if verbose:
        print(f"[4/4] Exporting CSV → {output_path}", file=sys.stderr)

    n_written = export_to_txt(
        comments=all_comments,
        output_path=output_path,
        timing_deciles=population.timing_deciles,
        comment_period_days=comment_period_days,
        include_failed_qc=include_failed_qc,
        seed=seed,
    )
    result.output_path = output_path

    if verbose:
        print(result.summary(), file=sys.stderr)

    return result


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
) -> tuple[GeneratedComment | None, int, int]:
    """Async helper: generate and QC one comment (direct mode)."""
    attempts = 0
    qc_failures = 0

    for attempt in range(max_retries):
        attempts += 1
        try:
            persona = await sample_persona_async(world_model, config, rng, docket_id=docket_id)
            frame = await map_argument_async(objective, vector, persona, world_model, config, rng)
            comment = await generate_comment_async(persona, frame, world_model, vector, objective, config)
            qc_result = await qc.check_async(comment)

            if qc_result.passed:
                return comment, attempts, qc_failures
            else:
                qc_failures += 1

        except Exception as exc:
            if verbose:
                print(f"      Error in async generation (attempt {attempt+1}/{max_retries}): {exc}", file=sys.stderr)

    return None, attempts, qc_failures


def run_async(
    docket_id: str,
    rule_text: str,
    vector: AttackVector,
    objective: str,
    volume: int,
    output_path: str,
    config: Config | None = None,
    seed: int = 42,
    similarity_threshold: float = 0.92,
    max_retries: int = 3,
    comment_period_days: int = 60,
    include_failed_qc: bool = False,
    skip_relevance_check: bool = False,
    skip_argument_check: bool = False,
    skip_embedding_check: bool = False,
    verbose: bool = True,
    max_concurrent: int = 10,
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
    world_model = build_world_model(
        rule_text=rule_text,
        population=population,
        config=config,
        docket_id=docket_id,
    )
    result.world_model_summary = world_model.to_dict()
    if verbose:
        print(f"      Rule  : {world_model.rule_title}", file=sys.stderr)
        print(f"      Agency: {world_model.agency}", file=sys.stderr)

    if verbose:
        print(f"[3/4] Generating {volume} comment(s) (vector {vector}) with {max_concurrent}-way parallelism…", file=sys.stderr)

    qc = QualityController(
        config=config,
        objective=objective,
        similarity_threshold=similarity_threshold,
        skip_relevance_check=skip_relevance_check,
        skip_argument_check=skip_argument_check,
        skip_embedding_check=skip_embedding_check,
    )

    async def _run_all():
        semaphore = asyncio.Semaphore(max_concurrent)

        async def gen_with_semaphore():
            async with semaphore:
                return await _generate_one_comment_async(
                    world_model, vector, objective, config, qc, rng,
                    max_retries, verbose, docket_id,
                )

        tasks = [gen_with_semaphore() for _ in range(volume)]
        all_comments = []
        total_attempted = 0
        total_accepted = 0
        total_qc_failed = 0

        if _TQDM_AVAILABLE and verbose:
            for coro in tqdm(asyncio.as_completed(tasks), total=volume, desc="Generating", unit="comment"):
                comment, attempts, qc_failures = await coro
                total_attempted += attempts
                total_qc_failed += qc_failures
                if comment:
                    all_comments.append(comment)
                    total_accepted += 1
        else:
            results = await asyncio.gather(*tasks)
            for comment, attempts, qc_failures in results:
                total_attempted += attempts
                total_qc_failed += qc_failures
                if comment:
                    all_comments.append(comment)
                    total_accepted += 1

        return all_comments, total_attempted, total_accepted, total_qc_failed

    all_comments, attempted, accepted, qc_failed = asyncio.run(_run_all())

    result.total_attempted = attempted
    result.total_accepted = accepted
    result.total_qc_failed = qc_failed
    result.total_skipped = volume - accepted
    result.comments = all_comments

    if verbose:
        print(f"[4/4] Exporting CSV → {output_path}", file=sys.stderr)

    n_written = export_to_txt(
        comments=all_comments,
        output_path=output_path,
        timing_deciles=population.timing_deciles,
        comment_period_days=comment_period_days,
        include_failed_qc=include_failed_qc,
        seed=seed,
    )
    result.output_path = output_path

    if verbose:
        print(result.summary(), file=sys.stderr)

    return result
