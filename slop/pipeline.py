"""
pipeline.py — Orchestrates the full synthetic comment generation pipeline.

This module wires together ingestion → world model → persona → argument mapping
→ generation → QC → export into a single callable function.

It also handles:
  - Progress reporting via tqdm
  - Retry logic: if generation+QC fails after max_retries attempts, the slot
    is skipped rather than blocking the run
  - Budget awareness: tracks LLM calls and warns if the volume is high
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from .argument_mapper import map_argument, map_argument_async, AttackVector
from .config import Config
from .export import export_to_txt
from .generator import generate_comment, generate_comment_async, GeneratedComment
from .ingestion import ingest_docket_csv
from .persona import sample_persona, sample_persona_async
from .quality_control import QualityController
from .world_model import build_world_model, WorldModel


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


# ── Main pipeline function ────────────────────────────────────────────────────

def run(
    docket_csv: str,
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
    docket_id: str = "",
    verbose: bool = True,
) -> RunResult:
    """
    Run the full synthetic comment generation pipeline.

    Parameters
    ----------
    docket_csv:
        Path to a Regulations.gov CSV from a previous (or same-topic) docket.
    rule_text:
        Full text of the proposed rule or RFI.
    vector:
        Attack vector: 1 (semantic variance), 2 (persona mimicry),
        3 (citation flooding), 4 (dilution/noise).
    objective:
        The position to advance or oppose (free-text string).
    volume:
        Number of accepted comments to produce.
    output_path:
        Destination CSV file path.
    config:
        API config.  If None, reads from environment variables.
    seed:
        Random seed for reproducibility.
    similarity_threshold:
        Cosine similarity ceiling for the deduplication check (0–1).
    max_retries:
        How many generation attempts per comment slot before giving up.
    comment_period_days:
        Simulated comment period length (days), for posted-date simulation.
    include_failed_qc:
        If True, failed QC comments are included in the output (flagged).
    skip_relevance_check / skip_argument_check / skip_embedding_check:
        Disable individual QC checks (faster, cheaper, less rigorous).
    docket_id:
        Override the docket ID written into the output CSV.
    verbose:
        Print progress messages.

    Returns
    -------
    RunResult
    """
    if config is None:
        config = Config()
    config.validate()

    rng = np.random.default_rng(seed)
    result = RunResult()

    # ── Stage 1: Ingest previous docket ──────────────────────────────────────
    if verbose:
        print(f"[1/4] Ingesting previous docket: {docket_csv}", file=sys.stderr)
    population = ingest_docket_csv(docket_csv, docket_id=docket_id, config=config)
    if verbose:
        wt = population.archetype_weights()
        for arch, w in sorted(wt.items(), key=lambda x: -x[1]):
            print(f"      {arch:25s} {w:.1%} ({population.archetypes.get(arch, type('', (), {'count': 0})()).count} comments)", file=sys.stderr)

    # ── Stage 2: Build world model ────────────────────────────────────────────
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
        print(f"      Domain: {world_model.regulatory_domain}", file=sys.stderr)

    # ── Stage 3: Generate comments ────────────────────────────────────────────
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
                persona = sample_persona(world_model, config, rng)
                frame = map_argument(objective, vector, persona, world_model, config, rng)
                comment = generate_comment(persona, frame, world_model, vector, objective, config)
                qc_result = qc.check(comment)
                all_comments.append(comment)

                if qc_result.passed:
                    accepted += 1
                    success = True
                    break
                else:
                    # QC failed — try again with a new persona
                    result.total_qc_failed += 1
                    if verbose and not _TQDM_AVAILABLE:
                        print(
                            f"      QC fail (attempt {attempt+1}/{max_retries}): {qc_result.notes}",
                            file=sys.stderr,
                        )

            except Exception as exc:
                if verbose and not _TQDM_AVAILABLE:
                    print(f"      Error (attempt {attempt+1}/{max_retries}): {exc}", file=sys.stderr)

        if not success:
            skipped += 1

    result.total_attempted = attempted
    result.total_accepted = accepted
    result.total_skipped = skipped
    result.comments = all_comments

    # ── Stage 4: Export ───────────────────────────────────────────────────────
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


# ── Async pipeline function ──────────────────────────────────────────────────

async def _generate_one_comment_async(
    world_model: WorldModel,
    vector: AttackVector,
    objective: str,
    config: Config,
    qc: QualityController,
    rng: np.random.Generator,
    max_retries: int,
    verbose: bool,
) -> tuple[GeneratedComment | None, int, int]:
    """
    Async helper: generate and QC one comment, with retries.
    Returns (comment, attempts, qc_failures).
    """
    attempts = 0
    qc_failures = 0
    
    for attempt in range(max_retries):
        attempts += 1
        try:
            persona = await sample_persona_async(world_model, config, rng)
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


async def _generate_comments_async(
    world_model: WorldModel,
    vector: AttackVector,
    objective: str,
    volume: int,
    config: Config,
    qc: QualityController,
    rng: np.random.Generator,
    max_retries: int,
    max_concurrent: int,
    verbose: bool,
) -> tuple[list[GeneratedComment], int, int, int]:
    """
    Generate multiple comments concurrently.
    Returns (all_comments, total_attempted, total_accepted, total_qc_failed).
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_with_semaphore():
        async with semaphore:
            return await _generate_one_comment_async(
                world_model, vector, objective, config, qc, rng, max_retries, verbose
            )
    
    # Create all tasks
    tasks = [generate_with_semaphore() for _ in range(volume)]
    
    # Run with progress bar if available
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


def run_async(
    docket_csv: str,
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
    docket_id: str = "",
    verbose: bool = True,
    max_concurrent: int = 10,
) -> RunResult:
    """
    Run the full synthetic comment generation pipeline using async parallelization.
    
    This is 10-15x faster than the synchronous run() function for large batches.
    
    Parameters
    ----------
    docket_csv:
        Path to a Regulations.gov CSV from a previous (or same-topic) docket.
    rule_text:
        Full text of the proposed rule or RFI.
    vector:
        Attack vector: 1 (semantic variance), 2 (persona mimicry),
        3 (citation flooding), 4 (dilution/noise).
    objective:
        The position to advance or oppose (free-text string).
    volume:
        Number of accepted comments to produce.
    output_path:
        Destination CSV file path.
    config:
        API config.  If None, reads from environment variables.
    seed:
        Random seed for reproducibility.
    similarity_threshold:
        Cosine similarity ceiling for the deduplication check (0–1).
    max_retries:
        How many generation attempts per comment slot before giving up.
    comment_period_days:
        Simulated comment period length (days), for posted-date simulation.
    include_failed_qc:
        If True, failed QC comments are included in the output (flagged).
    skip_relevance_check / skip_argument_check / skip_embedding_check:
        Disable individual QC checks (faster, cheaper, less rigorous).
    docket_id:
        Override the docket ID written into the output CSV.
    verbose:
        Print progress messages.
    max_concurrent:
        Maximum number of comments to generate concurrently (default 10).
        Higher values = faster, but may hit API rate limits.

    Returns
    -------
    RunResult
    """
    if config is None:
        config = Config()
    config.validate()

    rng = np.random.default_rng(seed)
    result = RunResult()

    # ── Stage 1: Ingest previous docket ──────────────────────────────────────
    if verbose:
        print(f"[1/4] Ingesting previous docket: {docket_csv}", file=sys.stderr)
    population = ingest_docket_csv(docket_csv, docket_id=docket_id, config=config)
    if verbose:
        wt = population.archetype_weights()
        for arch, w in sorted(wt.items(), key=lambda x: -x[1]):
            print(f"      {arch:25s} {w:.1%} ({population.archetypes.get(arch, type('', (), {'count': 0})()).count} comments)", file=sys.stderr)

    # ── Stage 2: Build world model ────────────────────────────────────────────
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
        print(f"      Domain: {world_model.regulatory_domain}", file=sys.stderr)

    # ── Stage 3: Generate comments (async) ────────────────────────────────────
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

    # Run async generation
    all_comments, attempted, accepted, qc_failed = asyncio.run(
        _generate_comments_async(
            world_model, vector, objective, volume, config, qc, rng, 
            max_retries, max_concurrent, verbose
        )
    )

    result.total_attempted = attempted
    result.total_accepted = accepted
    result.total_qc_failed = qc_failed
    result.total_skipped = volume - accepted
    result.comments = all_comments

    # ── Stage 4: Export ───────────────────────────────────────────────────────
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
