"""
quality_control.py — Post-generation quality control.

Three independent checks run per generated comment:

1. Topical relevance check (LLM)
   Is the comment actually about the proposed rule?  Comments that wander
   entirely off-topic are regenerated.

2. Argument presence check (LLM)
   Does the comment advance the objective (even obliquely)?  Comments that
   fail to do so are noise rather than attack specimens.

3. Embedding-based uniqueness check
   Compute an embedding for the comment and compare it to all previously
   accepted embeddings.  If cosine similarity to the nearest neighbour exceeds
   a configurable threshold, the comment is flagged as a near-duplicate and
   dropped from the output dataset (the attack is weaker if all comments are
   nearly identical in embedding space).

A QCResult carries the outcome of all three checks plus optional notes for
post-hoc analysis (e.g. what the nearest-neighbour similarity was).

Usage
-----
  qc = QualityController(config, objective, similarity_threshold=0.92)
  for raw_comment in raw_batch:
      result = qc.check(raw_comment)
      if result.passed:
          accepted.append(raw_comment)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Config
from .generator import GeneratedComment


# ── LLM-based checks ──────────────────────────────────────────────────────────

_RELEVANCE_SYSTEM = """\
You are a regulatory-comment quality analyst.
Given a proposed rule description and a public comment, decide whether the
comment is topically relevant to the rule.

Respond with a JSON object: {"relevant": true/false, "reason": "<one sentence>"}
Output ONLY the JSON object.
"""

_RELEVANCE_USER = """\
Rule: {rule_title}
Core change: {core_change}

Comment:
---
{comment_text}
---

Is this comment topically relevant to the rule?
"""

_ARGUMENT_SYSTEM = """\
You are a regulatory-comment quality analyst.
Given an attack objective and a public comment, decide whether the comment
advances or opposes the objective (even indirectly or obliquely).

Respond with a JSON object: {"on_message": true/false, "reason": "<one sentence>"}
Output ONLY the JSON object.
"""

_ARGUMENT_USER = """\
Objective: {objective}

Comment:
---
{comment_text}
---

Does the comment advance or oppose the objective?
"""


def _parse_json_response(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _check_relevance(comment: GeneratedComment, config: Config) -> tuple[bool, str]:
    client = config.openai_client()
    prompt = _RELEVANCE_USER.format(
        rule_title=comment.rule_title,
        core_change=comment.frame.core_arguments[0] if comment.frame.core_arguments else "",
        comment_text=comment.comment_text[:1500],
    )
    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _RELEVANCE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=150,
    )
    parsed = _parse_json_response(response.choices[0].message.content or "{}")
    relevant = bool(parsed.get("relevant", True))
    reason = str(parsed.get("reason", ""))
    return relevant, reason


async def _check_relevance_async(comment: GeneratedComment, config: Config) -> tuple[bool, str]:
    client = config.async_openai_client()
    prompt = _RELEVANCE_USER.format(
        rule_title=comment.rule_title,
        core_change=comment.frame.core_arguments[0] if comment.frame.core_arguments else "",
        comment_text=comment.comment_text[:1500],
    )
    response = await client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _RELEVANCE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=150,
    )
    parsed = _parse_json_response(response.choices[0].message.content or "{}")
    relevant = bool(parsed.get("relevant", True))
    reason = str(parsed.get("reason", ""))
    return relevant, reason


def _check_argument(comment: GeneratedComment, config: Config) -> tuple[bool, str]:
    client = config.openai_client()
    prompt = _ARGUMENT_USER.format(
        objective=comment.objective,
        comment_text=comment.comment_text[:1500],
    )
    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _ARGUMENT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=150,
    )
    parsed = _parse_json_response(response.choices[0].message.content or "{}")
    on_message = bool(parsed.get("on_message", True))
    reason = str(parsed.get("reason", ""))
    return on_message, reason


async def _check_argument_async(comment: GeneratedComment, config: Config) -> tuple[bool, str]:
    client = config.async_openai_client()
    prompt = _ARGUMENT_USER.format(
        objective=comment.objective,
        comment_text=comment.comment_text[:1500],
    )
    response = await client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _ARGUMENT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=150,
    )
    parsed = _parse_json_response(response.choices[0].message.content or "{}")
    on_message = bool(parsed.get("on_message", True))
    reason = str(parsed.get("reason", ""))
    return on_message, reason


# ── Embedding computation ──────────────────────────────────────────────────────

def _get_embedding(text: str, config: Config) -> list[float]:
    client = config.embedding_client()
    response = client.embeddings.create(
        model=config.embed_model,
        input=text[:8000],
    )
    return response.data[0].embedding


async def _get_embedding_async(text: str, config: Config) -> list[float]:
    client = config.async_embedding_client()
    response = await client.embeddings.create(
        model=config.embed_model,
        input=text[:8000],
    )
    return response.data[0].embedding


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


# ── QC result ─────────────────────────────────────────────────────────────────

@dataclass
class QCResult:
    passed: bool
    relevant: bool
    on_message: bool
    unique: bool
    nearest_similarity: float = 0.0
    notes: str = ""


# ── QualityController ──────────────────────────────────────────────────────────

class QualityController:
    """
    Stateful QC runner that maintains an embedding store of accepted comments
    for deduplication.

    Parameters
    ----------
    config:
        API configuration.
    objective:
        The attack objective string (used for argument-presence check).
    similarity_threshold:
        Maximum cosine similarity to the nearest accepted comment before a new
        comment is flagged as a near-duplicate.  Default 0.92.
    skip_relevance_check:
        Skip the LLM relevance check (faster, cheaper).
    skip_argument_check:
        Skip the LLM argument-presence check.
    skip_embedding_check:
        Skip the embedding-based deduplication.
    """

    def __init__(
        self,
        config: Config,
        objective: str,
        similarity_threshold: float = 0.92,
        skip_relevance_check: bool = False,
        skip_argument_check: bool = False,
        skip_embedding_check: bool = False,
    ) -> None:
        self.config = config
        self.objective = objective
        self.similarity_threshold = similarity_threshold
        self.skip_relevance_check = skip_relevance_check
        self.skip_argument_check = skip_argument_check
        self.skip_embedding_check = skip_embedding_check
        self._accepted_embeddings: list[list[float]] = []

    def check(self, comment: GeneratedComment) -> QCResult:
        """
        Run all QC checks on `comment`.  Mutates `comment.qc_passed` and
        `comment.qc_notes` and `comment.embedding` in-place, and also returns
        a QCResult for the caller.
        """
        notes: list[str] = []

        # 1. Topical relevance
        relevant = True
        if not self.skip_relevance_check:
            relevant, reason = _check_relevance(comment, self.config)
            if not relevant:
                notes.append(f"relevance_fail: {reason}")

        # 2. Argument presence
        on_message = True
        if not self.skip_argument_check:
            on_message, reason = _check_argument(comment, self.config)
            if not on_message:
                notes.append(f"argument_fail: {reason}")

        # 3. Embedding uniqueness
        unique = True
        nearest_sim = 0.0
        if not self.skip_embedding_check:
            emb = _get_embedding(comment.comment_text, self.config)
            comment.embedding = emb

            if self._accepted_embeddings:
                sims = [_cosine_similarity(emb, e) for e in self._accepted_embeddings]
                nearest_sim = max(sims)
                if nearest_sim >= self.similarity_threshold:
                    unique = False
                    notes.append(
                        f"near_duplicate: nearest_similarity={nearest_sim:.4f}"
                    )

        passed = relevant and on_message and unique

        if passed and not self.skip_embedding_check and comment.embedding:
            self._accepted_embeddings.append(comment.embedding)

        result = QCResult(
            passed=passed,
            relevant=relevant,
            on_message=on_message,
            unique=unique,
            nearest_similarity=nearest_sim,
            notes="; ".join(notes),
        )

        comment.qc_passed = passed
        comment.qc_notes = result.notes

        return result

    async def check_async(self, comment: GeneratedComment) -> QCResult:
        """
        Async version of check.
        Run all QC checks on `comment` using async API calls. Mutates 
        `comment.qc_passed` and `comment.qc_notes` and `comment.embedding` 
        in-place, and also returns a QCResult for the caller.
        """
        notes: list[str] = []

        # Run relevance and argument checks concurrently
        relevance_task = None
        argument_task = None
        
        if not self.skip_relevance_check:
            relevance_task = asyncio.create_task(_check_relevance_async(comment, self.config))
        
        if not self.skip_argument_check:
            argument_task = asyncio.create_task(_check_argument_async(comment, self.config))
        
        # Wait for both checks to complete
        relevant = True
        on_message = True
        
        if relevance_task:
            relevant, reason = await relevance_task
            if not relevant:
                notes.append(f"relevance_fail: {reason}")
        
        if argument_task:
            on_message, reason = await argument_task
            if not on_message:
                notes.append(f"argument_fail: {reason}")

        # 3. Embedding uniqueness
        unique = True
        nearest_sim = 0.0
        if not self.skip_embedding_check:
            emb = await _get_embedding_async(comment.comment_text, self.config)
            comment.embedding = emb

            if self._accepted_embeddings:
                sims = [_cosine_similarity(emb, e) for e in self._accepted_embeddings]
                nearest_sim = max(sims)
                if nearest_sim >= self.similarity_threshold:
                    unique = False
                    notes.append(
                        f"near_duplicate: nearest_similarity={nearest_sim:.4f}"
                    )

        passed = relevant and on_message and unique

        if passed and not self.skip_embedding_check and comment.embedding:
            self._accepted_embeddings.append(comment.embedding)

        result = QCResult(
            passed=passed,
            relevant=relevant,
            on_message=on_message,
            unique=unique,
            nearest_similarity=nearest_sim,
            notes="; ".join(notes),
        )

        comment.qc_passed = passed
        comment.qc_notes = result.notes

        return result

    @property
    def accepted_count(self) -> int:
        return len(self._accepted_embeddings)

    def embedding_matrix(self) -> np.ndarray | None:
        """Return accepted embeddings as a numpy matrix (for offline analysis)."""
        if not self._accepted_embeddings:
            return None
        return np.array(self._accepted_embeddings, dtype=np.float32)
