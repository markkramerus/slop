"""
rewriter.py — Per-comment adversarial judge→rewrite loop.

After a comment is generated, it goes through an iterative refinement loop:

  1. JUDGE evaluates the comment: returns a confidence score (0=definitely AI,
     50=toss-up, 100=definitely real) and specific reasons.
  2. If score > 50 → stop (comment passes as real).
  3. If MAX_REWRITES reached → stop (record final state regardless).
  4. REWRITER rewrites the comment, addressing the judge's specific criticisms
     while preserving: arguments, length, identity, and tone.
  5. Go to 1 with the rewritten comment.

The key insight: we do NOT attempt to craft the ultimate all-purpose humanizing
prompt.  Instead, the rewriter is given the judge's SPECIFIC criticisms of THIS
SPECIFIC comment and told to address them directly.  The rewriter prompt is
intentionally minimal — just 4 core constraints plus the judge's feedback.

Three potentially different LLM endpoints are used:
  - GENERATOR_*         → initial comment generation (existing)
  - JUDGE_*             → classifies comments as AI or real
  - REWRITE_COMMENT_*   → rewrites comments using judge feedback
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class RewriterConfig:
    """Configuration for the judge and rewriter models."""

    # Judge model — scores comments on AI-vs-Real scale
    judge_api_base_url: str = field(
        default_factory=lambda: os.getenv("JUDGE_API_BASE_URL", "https://api.openai.com/v1")
    )
    judge_api_key: str = field(
        default_factory=lambda: os.getenv("JUDGE_API_KEY", "")
    )
    judge_model: str = field(
        default_factory=lambda: os.getenv("JUDGE_CHAT_MODEL", "gpt-4o")
    )

    # Comment-rewrite model — rewrites comments using judge feedback
    rewrite_api_base_url: str = field(
        default_factory=lambda: os.getenv("REWRITE_COMMENT_API_BASE_URL", "https://api.openai.com/v1")
    )
    rewrite_api_key: str = field(
        default_factory=lambda: os.getenv("REWRITE_COMMENT_API_KEY", "")
    )
    rewrite_model: str = field(
        default_factory=lambda: os.getenv("REWRITE_COMMENT_MODEL", "gpt-4o")
    )

    # Loop parameters
    max_rewrites: int = 2

    def _make_client(self, base_url: str, api_key: str):
        import openai
        return openai.OpenAI(base_url=base_url, api_key=api_key)

    def _make_async_client(self, base_url: str, api_key: str):
        import openai
        return openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    def judge_client(self):
        return self._make_client(self.judge_api_base_url, self.judge_api_key)

    def judge_client_async(self):
        return self._make_async_client(self.judge_api_base_url, self.judge_api_key)

    def rewrite_client(self):
        return self._make_client(self.rewrite_api_base_url, self.rewrite_api_key)

    def rewrite_client_async(self):
        return self._make_async_client(self.rewrite_api_base_url, self.rewrite_api_key)

    def is_available(self) -> bool:
        """Return True if both judge and rewriter API keys are configured."""
        return bool(self.judge_api_key) and bool(self.rewrite_api_key)

    def validate(self) -> None:
        missing = []
        if not self.judge_api_key:
            missing.append("JUDGE_API_KEY")
        if not self.rewrite_api_key:
            missing.append("REWRITE_COMMENT_API_KEY")
        if missing:
            raise ValueError(f"Missing API keys for rewriter: {', '.join(missing)}")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class JudgeVerdict:
    """Result of a single judge evaluation."""
    human_author_probability: int   # 0=definitely AI, 50=toss-up, 100=definitely real
    reasons: str            # Specific, actionable reasons for the rating
    raw_response: str = ""  # Raw API response text (for debugging)


@dataclass
class RewriteResult:
    """Full result of the judge→rewrite loop for one comment."""
    final_text: str
    final_score: int
    final_reasons: str
    rewrites_performed: int
    passed: bool                 # True if final score > 50
    history: list[dict] = field(default_factory=list)
    # Each history entry: {"step": "judge"|"rewrite", "text": str,
    #                      "score": int, "reasons": str}


# ── Retry helper ──────────────────────────────────────────────────────────────

def _retry_with_backoff(
    fn: Callable,
    max_retries: int = 5,
    base_delay: float = 4.0,
    label: str = "API call",
) -> Any:
    """Call *fn()* with exponential backoff on transient errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            err_str = str(e).lower()
            is_transient = any(k in err_str for k in (
                "connection error", "rate limit", "429", "503", "502",
                "500", "timeout", "resource_exhausted", "overloaded",
            ))
            if is_transient and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                time.sleep(delay)
            else:
                raise


async def _retry_with_backoff_async(
    fn: Callable,
    max_retries: int = 5,
    base_delay: float = 4.0,
    label: str = "API call",
) -> Any:
    """Call awaitable *fn()* with exponential backoff on transient errors."""
    import asyncio
    for attempt in range(1, max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            err_str = str(e).lower()
            is_transient = any(k in err_str for k in (
                "connection error", "rate limit", "429", "503", "502",
                "500", "timeout", "resource_exhausted", "overloaded",
            ))
            if is_transient and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                await asyncio.sleep(delay)
            else:
                raise


# ── JSON parsing helper ───────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict[str, Any]:
    """Parse a JSON response, stripping markdown code fences if present.

    If standard JSON parsing fails (e.g. because the response was truncated
    mid-string by a max_tokens limit), falls back to regex extraction of
    known fields.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.rsplit("```", 1)[0]
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        pass

    # ── Regex fallback for truncated JSON ─────────────────────────────
    # The model may have returned valid JSON that was cut off by max_tokens.
    # Try to extract individual fields from the raw text.
    result: dict[str, Any] = {}

    # Extract human_author_probability (integer)
    score_match = re.search(r'"human_author_probability"\s*:\s*(\d+)', raw)
    if score_match:
        result["human_author_probability"] = int(score_match.group(1))

    # Extract reasons (string — may be truncated)
    reasons_match = re.search(r'"reasons"\s*:\s*"(.*?)(?:"\s*[,}]|$)', raw, re.DOTALL)
    if reasons_match:
        reasons_text = reasons_match.group(1)
        # Unescape JSON string escapes
        reasons_text = reasons_text.replace('\\"', '"').replace('\\n', '\n')
        if not reasons_text.rstrip().endswith(('.', '"', ')')):
            reasons_text = reasons_text.rstrip() + " [truncated]"
        result["reasons"] = reasons_text

    return result


# ── Judge prompts ─────────────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = """\
You are an expert forensic linguist specializing in detecting AI-generated text.

You will be given a single public comment submitted to a U.S. federal regulatory \
docket. Your job is to determine whether it was written by a real human or \
generated by an AI model.

You must return a JSON object with exactly two fields:

{
  "human_author_probability": <integer between 0-100>,
  "reasons": "<string>"
}

human_author_probability:
  0  = many distinctive AI features marking it definitively as AI-generated text with zero possibility of human authorship 
  20 = some distinctive AI features making it highly likely to be AI-generated
  40 = a few typical AI features, enough to conclude the text is AI-generated
  60 = mostly resembles human text with some features that hinting at AI generation
  80 = human features far outweigh anything that might hint at AI generation
  100 = absolutely human writing with no discernable features of AI-generated text

reasons:
  If you think the comment was written by AI (human_author_probability < 50), then create a numbered list including **every** AI-like feature you detected, described in enough detail to be rectified \
  (e.g., "1. perfectly balanced rhetoric in paragraph 4; 2. rule-of-three pattern in the \
  opening; 3. too smooth a narrative arc from anecdote to policy argument, 4. em dash in paragrah 6, \
  5. copula avoidance — uses 'serves as' instead of 'is'"). \
  Make sure the list is comprehensive, with specific, concrete observations. \
  \
  Conversely, if you think the comment is human-written, briefly explain what convinced you \
  (e.g., "inconsistent capitalization", "single-issue frustration without \
  comprehensive policy framing", "abrupt ending mid-thought", "genuine typos").

Output ONLY the JSON object. No preamble, no explanation outside the JSON.
"""

_JUDGE_USER_TEMPLATE = """\
Classify the following public comment as AI-generated or human-written.

=== COMMENT ===
{comment_text}
=== END COMMENT ===
"""


# ── Rewriter prompts ─────────────────────────────────────────────────────────

_REWRITER_SYSTEM_PROMPT = """\
You are rewriting a public comment that was flagged as AI-generated by a \
forensic judge. Your job is to address the judge's specific criticisms so \
the comment reads as authentically human-written.

You MUST follow these constraints:

1. PRESERVE THE ARGUMENTS: Keep the same arguments, points, and positions as the \
original. If the original argues for transparency, your rewrite must argue \
for transparency. Do not add new arguments or remove existing ones.

2. DO NOT THE COMMENT MAKE SHORTER: The rewritten comment must be approximately the same \
length as the original (within ±20%).

3. PRESERVE THE COMMENTER'S IDENTITY: Keep the same commenter identity — same name, \
organization, role, and perspective.

4. MATCH VOICE TO IDENTITY: The tone, vocabulary, and style must be consistent with \
who the commenter is. A retired nurse writes differently than a corporate \
compliance officer.

5. FOCUS EXCLUSIVELY ON ADDRESSING THE JUDGE'S FEEDBACK: The judge identified the specific \
reasons this comment looks AI-generated. Fix those specific issues. This is your primary \
task — do not apply generic "humanizing" rules, focus on what the judge actually flagged.

Output ONLY the rewritten comment text. No labels, no preamble, no \
"Here is the rewritten comment:" — just the comment itself.

SPECIAL SYMBOLS:
Use the symbol ⏎ to indicate new lines (paragraph breaks). Do not insert new lines and do not use backslash-n. \
Use the ♔ symbol as column separator.
"""

_REWRITER_USER_TEMPLATE = """\
=== COMMENTER IDENTITY ===
Name: {name}
Occupation: {occupation}
Organization: {org_name}
State: {state}
Age: {age}
Archetype: {archetype}

=== JUDGE'S CRITICISMS (address these specifically) ===
Confidence score: {score}/100 (0=definitely AI, 100=definitely real)
Reasons flagged as AI:
{reasons}

{org_writing_constraint}
=== ORIGINAL COMMENT ===
{comment_text}
=== END COMMENT ===

Rewrite this comment to address the judge's specific criticisms while \
preserving the arguments, length, identity, and tone.
"""

# Injected into the rewriter user prompt whenever the commenter is an
# organization.  Prevents the rewriter from "humanizing" an org comment
# by introducing lowercase sentence starts or missing punctuation — tactics
# that would be plausible for an individual but are never appropriate for
# a professional organization.
_ORG_REWRITER_CONSTRAINT = """\
=== ORGANIZATIONAL WRITING CONSTRAINT (non-negotiable) ===
This comment is submitted by an organization ({org_name}).
Regardless of what the judge flagged, you MUST NOT introduce:
- Lowercase sentence starts
- Missing end-of-sentence punctuation
- Phone-typing shortcuts, voice-to-text artifacts, or all-lowercase passages
Organizations always use proper capitalization and punctuation.
If the judge flagged "too polished" or "too consistent capitalization," \
address that through structural imperfections (e.g., an incomplete citation, \
slightly inconsistent spacing, a dangling clause) — NOT typographic errors.

"""


# ── Judge function ────────────────────────────────────────────────────────────

def judge_comment(
    comment_text: str,
    config: RewriterConfig,
) -> JudgeVerdict:
    """
    Call the judge model to evaluate a single comment.

    Returns a JudgeVerdict with human_author_probability (0–100) and reasons.
    """
    client = config.judge_client()
    user_prompt = _JUDGE_USER_TEMPLATE.format(
        comment_text=comment_text[:6000],
    )

    def _call():
        return client.chat.completions.create(
            model=config.judge_model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=50000,
        )

    response = _retry_with_backoff(_call, label="judge")
    raw = (response.choices[0].message.content or "").strip()
    parsed = _parse_json_response(raw)

    score = parsed.get("human_author_probability", 0)
    # Clamp to 0–100
    if isinstance(score, (int, float)):
        score = max(0, min(100, int(score)))
    else:
        score = 0

    reasons = str(parsed.get("reasons", raw))

    return JudgeVerdict(human_author_probability=score, reasons=reasons, raw_response=raw)


async def judge_comment_async(
    comment_text: str,
    config: RewriterConfig,
) -> JudgeVerdict:
    """Async version of judge_comment."""
    client = config.judge_client_async()
    user_prompt = _JUDGE_USER_TEMPLATE.format(
        comment_text=comment_text[:6000],
    )

    async def _call():
        return await client.chat.completions.create(
            model=config.judge_model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=50000,
        )

    response = await _retry_with_backoff_async(_call, label="judge")
    raw = (response.choices[0].message.content or "").strip()
    parsed = _parse_json_response(raw)

    score = parsed.get("human_author_probability", 0)
    if isinstance(score, (int, float)):
        score = max(0, min(100, int(score)))
    else:
        score = 0

    reasons = str(parsed.get("reasons", raw))

    return JudgeVerdict(human_author_probability=score, reasons=reasons, raw_response=raw)


# ── Rewriter function ────────────────────────────────────────────────────────

def rewrite_comment(
    comment_text: str,
    judge_score: int,
    judge_reasons: str,
    persona_context: dict[str, str],
    config: RewriterConfig,
) -> str:
    """
    Rewrite a single comment to address the judge's specific criticisms.

    Parameters
    ----------
    comment_text : str
        The current comment text to rewrite.
    judge_score : int
        The judge's confidence score (0-100).
    judge_reasons : str
        The judge's specific reasons for flagging as AI.
    persona_context : dict
        Identity fields: name, occupation, org_name, state, age, archetype.
    config : RewriterConfig
        API configuration.

    Returns
    -------
    str : The rewritten comment text.
    """
    client = config.rewrite_client()

    org_name = persona_context.get("org_name", "None")
    is_org = org_name not in ("", "None")
    org_writing_constraint = (
        _ORG_REWRITER_CONSTRAINT.format(org_name=org_name) if is_org else ""
    )

    user_prompt = _REWRITER_USER_TEMPLATE.format(
        name=persona_context.get("name", "Unknown"),
        occupation=persona_context.get("occupation", "Unknown"),
        org_name=org_name,
        state=persona_context.get("state", "Unknown"),
        age=persona_context.get("age", "Unknown"),
        archetype=persona_context.get("archetype", "Unknown"),
        score=judge_score,
        reasons=judge_reasons,
        org_writing_constraint=org_writing_constraint,
        comment_text=comment_text,
    )

    def _call():
        return client.chat.completions.create(
            model=config.rewrite_model,
            messages=[
                {"role": "system", "content": _REWRITER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            max_tokens=50000,
        )

    response = _retry_with_backoff(_call, label="rewrite")
    rewritten = (response.choices[0].message.content or "").strip()
    return rewritten if rewritten else comment_text


async def rewrite_comment_async(
    comment_text: str,
    judge_score: int,
    judge_reasons: str,
    persona_context: dict[str, str],
    config: RewriterConfig,
) -> str:
    """Async version of rewrite_comment."""
    client = config.rewrite_client_async()

    org_name = persona_context.get("org_name", "None")
    is_org = org_name not in ("", "None")
    org_writing_constraint = (
        _ORG_REWRITER_CONSTRAINT.format(org_name=org_name) if is_org else ""
    )

    user_prompt = _REWRITER_USER_TEMPLATE.format(
        name=persona_context.get("name", "Unknown"),
        occupation=persona_context.get("occupation", "Unknown"),
        org_name=org_name,
        state=persona_context.get("state", "Unknown"),
        age=persona_context.get("age", "Unknown"),
        archetype=persona_context.get("archetype", "Unknown"),
        score=judge_score,
        reasons=judge_reasons,
        org_writing_constraint=org_writing_constraint,
        comment_text=comment_text,
    )

    async def _call():
        return await client.chat.completions.create(
            model=config.rewrite_model,
            messages=[
                {"role": "system", "content": _REWRITER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            max_tokens=50000,
        )

    response = await _retry_with_backoff_async(_call, label="rewrite")
    rewritten = (response.choices[0].message.content or "").strip()
    return rewritten if rewritten else comment_text


# ── Main loop ─────────────────────────────────────────────────────────────────

def judge_rewrite_loop(
    comment_text: str,
    persona_context: dict[str, str],
    config: RewriterConfig,
    verbose: bool = False,
) -> RewriteResult:
    """
    Run the judge→rewrite loop on a single comment.

    Sequence (with MAX_REWRITES=2):
      Judge → Rewrite → Judge → Rewrite → Judge

    Stops early if the judge scores the comment > 50 (passes as real).

    Parameters
    ----------
    comment_text : str
        The initially generated comment.
    persona_context : dict
        Identity fields for the rewriter: name, occupation, org_name,
        state, age, archetype.
    config : RewriterConfig
        Judge and rewriter API configuration.
    verbose : bool
        Print progress to stderr.

    Returns
    -------
    RewriteResult with the final text, score, reasons, and full history.
    """
    import sys

    current_text = comment_text
    history: list[dict] = []
    rewrites_done = 0
    max_rewrites = config.max_rewrites

    for iteration in range(max_rewrites + 1):
        # ── Judge step ────────────────────────────────────────────────
        try:
            verdict = judge_comment(current_text, config)
        except Exception as e:
            if verbose:
                print(f"    [rewriter] Judge error: {e}", file=sys.stderr)
            # If judge fails, return current text with no score
            return RewriteResult(
                final_text=current_text,
                final_score=-1,
                final_reasons=f"Judge error: {e}",
                rewrites_performed=rewrites_done,
                passed=False,
                history=history,
            )

        history.append({
            "step": "judge",
            "iteration": iteration,
            "text": current_text,
            "score": verdict.human_author_probability,
            "reasons": verdict.reasons,
        })

        if verbose:
            print(
                f"    [rewriter] Judge iteration {iteration}: "
                f"score={verdict.human_author_probability}/100 "
                f"({'PASS' if verdict.human_author_probability > 50 else 'FAIL'})",
                file=sys.stderr,
            )

        # ── Early exit: passes as real ────────────────────────────────
        if verdict.human_author_probability > 50:
            return RewriteResult(
                final_text=current_text,
                final_score=verdict.human_author_probability,
                final_reasons=verdict.reasons,
                rewrites_performed=rewrites_done,
                passed=True,
                history=history,
            )

        # ── Early exit: max rewrites reached ──────────────────────────
        if rewrites_done >= max_rewrites:
            if verbose:
                print(
                    f"    [rewriter] Max rewrites ({max_rewrites}) reached. "
                    f"Final score: {verdict.human_author_probability}/100",
                    file=sys.stderr,
                )
            return RewriteResult(
                final_text=current_text,
                final_score=verdict.human_author_probability,
                final_reasons=verdict.reasons,
                rewrites_performed=rewrites_done,
                passed=False,
                history=history,
            )

        # ── Rewrite step ──────────────────────────────────────────────
        if verbose:
            print(
                f"    [rewriter] Rewriting (pass {rewrites_done + 1}/{max_rewrites})...",
                file=sys.stderr,
            )

        try:
            rewritten = rewrite_comment(
                current_text,
                verdict.human_author_probability,
                verdict.reasons,
                persona_context,
                config,
            )
        except Exception as e:
            if verbose:
                print(f"    [rewriter] Rewrite error: {e}", file=sys.stderr)
            # If rewrite fails, return current text with last judge score
            return RewriteResult(
                final_text=current_text,
                final_score=verdict.human_author_probability,
                final_reasons=verdict.reasons,
                rewrites_performed=rewrites_done,
                passed=False,
                history=history,
            )

        history.append({
            "step": "rewrite",
            "iteration": iteration,
            "text": rewritten,
            "score": verdict.human_author_probability,
            "reasons": verdict.reasons,
        })

        current_text = rewritten
        rewrites_done += 1

    # Should not reach here, but just in case
    return RewriteResult(
        final_text=current_text,
        final_score=-1,
        final_reasons="Loop ended unexpectedly",
        rewrites_performed=rewrites_done,
        passed=False,
        history=history,
    )


async def judge_rewrite_loop_async(
    comment_text: str,
    persona_context: dict[str, str],
    config: RewriterConfig,
    verbose: bool = False,
) -> RewriteResult:
    """
    Async version of judge_rewrite_loop.

    Runs the judge→rewrite loop on a single comment. Each step within this
    comment is strictly sequential (judge, then rewrite, then judge, ...),
    but multiple comments can run their loops concurrently.
    """
    import sys

    current_text = comment_text
    history: list[dict] = []
    rewrites_done = 0
    max_rewrites = config.max_rewrites

    for iteration in range(max_rewrites + 1):
        # ── Judge step ────────────────────────────────────────────────
        try:
            verdict = await judge_comment_async(current_text, config)
        except Exception as e:
            if verbose:
                print(f"    [rewriter] Judge error: {e}", file=sys.stderr)
            return RewriteResult(
                final_text=current_text,
                final_score=-1,
                final_reasons=f"Judge error: {e}",
                rewrites_performed=rewrites_done,
                passed=False,
                history=history,
            )

        history.append({
            "step": "judge",
            "iteration": iteration,
            "text": current_text,
            "score": verdict.human_author_probability,
            "reasons": verdict.reasons,
        })

        if verbose:
            print(
                f"    [rewriter] Judge iteration {iteration}: "
                f"score={verdict.human_author_probability}/100 "
                f"({'PASS' if verdict.human_author_probability > 50 else 'FAIL'})",
                file=sys.stderr,
            )

        # ── Early exit: passes as real ────────────────────────────────
        if verdict.human_author_probability > 50:
            return RewriteResult(
                final_text=current_text,
                final_score=verdict.human_author_probability,
                final_reasons=verdict.reasons,
                rewrites_performed=rewrites_done,
                passed=True,
                history=history,
            )

        # ── Early exit: max rewrites reached ──────────────────────────
        if rewrites_done >= max_rewrites:
            if verbose:
                print(
                    f"    [rewriter] Max rewrites ({max_rewrites}) reached. "
                    f"Final score: {verdict.human_author_probability}/100",
                    file=sys.stderr,
                )
            return RewriteResult(
                final_text=current_text,
                final_score=verdict.human_author_probability,
                final_reasons=verdict.reasons,
                rewrites_performed=rewrites_done,
                passed=False,
                history=history,
            )

        # ── Rewrite step ──────────────────────────────────────────────
        if verbose:
            print(
                f"    [rewriter] Rewriting (pass {rewrites_done + 1}/{max_rewrites})...",
                file=sys.stderr,
            )

        try:
            rewritten = await rewrite_comment_async(
                current_text,
                verdict.human_author_probability,
                verdict.reasons,
                persona_context,
                config,
            )
        except Exception as e:
            if verbose:
                print(f"    [rewriter] Rewrite error: {e}", file=sys.stderr)
            return RewriteResult(
                final_text=current_text,
                final_score=verdict.human_author_probability,
                final_reasons=verdict.reasons,
                rewrites_performed=rewrites_done,
                passed=False,
                history=history,
            )

        history.append({
            "step": "rewrite",
            "iteration": iteration,
            "text": rewritten,
            "score": verdict.human_author_probability,
            "reasons": verdict.reasons,
        })

        current_text = rewritten
        rewrites_done += 1

    # Should not reach here, but just in case
    return RewriteResult(
        final_text=current_text,
        final_score=-1,
        final_reasons="Loop ended unexpectedly",
        rewrites_performed=rewrites_done,
        passed=False,
        history=history,
    )


def build_persona_context(persona) -> dict[str, str]:
    """
    Extract persona identity fields into a dict for the rewriter.

    Accepts a Persona object (from syncom.persona) and returns a flat dict
    with the fields the rewriter needs to preserve identity.
    """
    return {
        "name": persona.full_name,
        "occupation": persona.occupation,
        "org_name": persona.org_name or "None",
        "state": persona.state,
        "age": str(persona.age),
        "archetype": persona.archetype,
    }
