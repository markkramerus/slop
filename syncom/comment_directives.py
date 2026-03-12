"""
comment_directives.py — Pre-sample structural directives from voice statistics.

CommentDirectives are sampled ONCE per comment BEFORE any LLM call. They
provide deterministic structural targets that neither the argument_mapper
nor the generator LLM may override:

  - target_word_count  (sampled from voice distribution)
  - use_bullets        (coin flip at voice's bullet-usage rate)
  - use_headings       (coin flip at voice's heading-usage rate)
  - target_citations   (sampled from voice's citation distribution)
  - first_person_level (from voice stats: "none", "light", "heavy")
  - max_tokens         (derived from target_word_count)

These directives flow through the pipeline as an immutable specification.
The argument_mapper LLM generates framing/arguments but cannot change
structural parameters. The generator prompt encodes these as hard rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from stylometry.stylometry_loader import parse_statistical_profile, VoiceStatistics


@dataclass(frozen=True)
class CommentDirectives:
    """Immutable structural specification for one synthetic comment."""
    target_word_count: int
    use_bullets: bool
    use_headings: bool
    target_citations: int
    first_person_level: str   # "none" | "light" | "heavy"
    max_tokens: int
    paragraph_count: int

    def structural_prompt_block(self) -> str:
        """
        Return explicit structural directives for the generator prompt.
        These are framed as hard requirements, not suggestions.
        """
        lines = ["=== STRUCTURAL REQUIREMENTS (mandatory — do not deviate) ==="]

        # Word count
        lo = max(int(self.target_word_count * 0.8), 30)
        hi = int(self.target_word_count * 1.2)
        lines.append(
            f"- Write EXACTLY {self.target_word_count} words (acceptable range: {lo}–{hi})."
        )

        # Paragraphs
        lines.append(f"- Use approximately {self.paragraph_count} paragraphs.")

        # Bullets
        if self.use_bullets:
            lines.append(
                "- MUST include bullet points or a numbered list in at least one section."
            )
        else:
            lines.append("- Do NOT use bullet points or numbered lists.")

        # Headings
        if self.use_headings:
            lines.append(
                "- MUST include section headings (e.g., bold text or ALL-CAPS headers) "
                "to organize the comment."
            )
        else:
            lines.append("- Do NOT use section headings. Write in flowing prose paragraphs.")

        # Citations
        if self.target_citations >= 3:
            lines.append(
                f"- Include approximately {self.target_citations} regulatory or literature "
                f"citations (e.g., CFR sections, Federal Register pages, published studies). "
                f"Weave them into the text naturally."
            )
        elif self.target_citations >= 1:
            lines.append(
                f"- Include {self.target_citations} citation(s) to regulations or published sources."
            )
        else:
            lines.append("- Do NOT include regulatory citations or academic references.")

        # First-person
        if self.first_person_level == "heavy":
            lines.append(
                "- Use first-person pronouns freely (I, me, my, we, our). "
                "This is a personal comment."
            )
        elif self.first_person_level == "light":
            lines.append(
                "- Use first-person sparingly — an occasional 'we' or 'our organization' "
                "is acceptable, but the tone should be largely institutional."
            )
        else:  # none
            lines.append(
                "- Avoid first-person pronouns (I, me, my). Write in third person or "
                "use the organization's name. This is a formal institutional submission."
            )

        return "\n".join(lines)


def sample_directives(
    voice_skill: str,
    rng: np.random.Generator,
    sophistication: str = "medium",
) -> CommentDirectives:
    """
    Sample CommentDirectives from a voice skill's statistical profile.

    Parameters
    ----------
    voice_skill : str
        Raw markdown content of the voice skill file.
    rng : np.random.Generator
        Seeded random number generator.
    sophistication : str
        Persona sophistication level.

    Returns
    -------
    CommentDirectives
        Immutable structural directives for one comment.
    """
    stats = parse_statistical_profile(voice_skill)
    return _sample_from_stats(stats, rng, sophistication)


def sample_directives_default(
    rng: np.random.Generator,
    sophistication: str = "medium",
) -> CommentDirectives:
    """
    Sample CommentDirectives using generic defaults (no voice skill).
    """
    stats = VoiceStatistics()  # all defaults
    return _sample_from_stats(stats, rng, sophistication)


def _sample_from_stats(
    stats: VoiceStatistics,
    rng: np.random.Generator,
    sophistication: str,
) -> CommentDirectives:
    """Core sampling logic from parsed VoiceStatistics."""

    # ── Word count: sample from Normal(median, std) ──────────────────────
    wc_std = max(1.0, (stats.word_count_high - stats.word_count_low) / 4.0)
    sampled_wc = int(rng.normal(stats.word_count_median, wc_std))
    # Clamp to ±20% beyond observed range
    lo_clamp = max(30, int(stats.word_count_low * 0.8))
    hi_clamp = int(stats.word_count_high * 1.2)
    target_wc = max(lo_clamp, min(sampled_wc, hi_clamp))

    # ── Bullets: Bernoulli at voice's usage rate ─────────────────────────
    bullet_prob = stats.uses_bullet_points_pct / 100.0
    use_bullets = bool(rng.random() < bullet_prob)

    # ── Headings: Bernoulli at voice's usage rate ────────────────────────
    heading_prob = stats.uses_headings_pct / 100.0
    use_headings = bool(rng.random() < heading_prob)

    # ── Citations: sample from Poisson(citation_frequency) ───────────────
    if stats.citation_frequency > 0.5:
        target_citations = int(rng.poisson(stats.citation_frequency))
    else:
        target_citations = 0

    # ── First-person level from percentage ───────────────────────────────
    if stats.first_person_pct >= 2.0:
        first_person_level = "heavy"
    elif stats.first_person_pct >= 0.5:
        first_person_level = "light"
    else:
        first_person_level = "none"

    # ── Paragraph count: proportional to word count ──────────────────────
    if stats.paragraphs_median > 0:
        # Scale paragraphs relative to the skill's median word count
        para_ratio = stats.paragraphs_median / max(stats.word_count_median, 1)
        para_count = max(1, int(round(target_wc * para_ratio)))
    else:
        # Fallback: ~100 words per paragraph
        para_count = max(1, target_wc // 100)

    # ── max_tokens: words × 1.5 (tokens per word), with a floor ─────────
    max_tokens = max(256, int(target_wc * 1.5))

    return CommentDirectives(
        target_word_count=target_wc,
        use_bullets=use_bullets,
        use_headings=use_headings,
        target_citations=target_citations,
        first_person_level=first_person_level,
        max_tokens=max_tokens,
        paragraph_count=para_count,
    )
