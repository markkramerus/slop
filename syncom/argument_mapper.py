"""
argument_mapper.py — Map (objective × attack_vector × persona) → ExpressionFrame.

An ExpressionFrame is a structured specification that tells the generator:
  - Which core argument(s) from the objective to use
  - How to frame them for this persona
  - What structural properties the comment should have (length, tone, citations)
  - Vector-specific instructions (semantic variance, citation agenda, dilution)

Attack vectors:
  1 — Semantic Variance: Same core argument, maximally varied surface forms.
      High temperature generation, explicit diversity instruction.
  2 — Persona Mimicry: Engineered stakeholder consensus.  Each persona expresses
      the same underlying argument through a completely different frame.
  3 — Citation Flooding: Comments loaded with real-sounding but hard-to-verify
      citations, creating a disproportionate verification burden.
  4 — Dilution / Noise: High-volume, low-substance agreement.  Brief, vague
      comments that technically support a position but carry little information.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import Config
from .persona import Persona
from .world_model import WorldModel
from stylometry.stylometry_loader import parse_statistical_profile

AttackVector = int  # 1 | 2 | 3 | 4


# ── Expression frame ──────────────────────────────────────────────────────────

@dataclass
class ExpressionFrame:
    """
    Everything the generator needs to write one synthetic comment, beyond the
    raw rule text and persona.
    """
    # The core argument(s) to advance or oppose
    core_arguments: list[str]

    # How this persona would typically frame the argument
    framing: str

    # What evidence type(s) this persona would cite
    evidence_types: list[str]       # e.g. ["personal anecdote", "news article", "statistic"]

    # RFI question(s) to address (subset — real comments rarely address all)
    rfi_questions_to_address: list[str]

    # For Vector 3: a list of plausible-sounding citation seeds
    citation_agenda: list[str] = field(default_factory=list)

    # Target word count for this comment
    target_word_count: int = 250

    # Generation temperature override
    temperature: float = 0.9

    # Vector-specific extra instructions for the generator prompt
    vector_instructions: str = ""

    # Whether to instruct the model to write a brief/vague comment (Vector 4)
    is_dilution: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_arguments": self.core_arguments,
            "framing": self.framing,
            "evidence_types": self.evidence_types,
            "rfi_questions_to_address": self.rfi_questions_to_address,
            "citation_agenda": self.citation_agenda,
            "target_word_count": self.target_word_count,
            "temperature": self.temperature,
            "vector_instructions": self.vector_instructions,
            "is_dilution": self.is_dilution,
        }


# ── LLM-based frame generation ────────────────────────────────────────────────

_FRAME_SYSTEM = """\
You are a regulatory-comment strategy analyst helping researchers build a
synthetic comment generator for regulatory spam detection research.

Given an objective, a persona, and a regulatory rule, produce a JSON
ExpressionFrame that tells the generator how this specific persona would
express the objective's argument.

Output ONLY valid JSON.
"""

_FRAME_USER_TEMPLATE = """\
Objective (the position to advance or oppose):
{objective}

{argument_angle_block}
Attack vector: {vector_label}
{vector_guidance}

Persona:
  Archetype: {archetype}
  Name: {name}
  State: {state}
  Occupation: {occupation}
  Sophistication: {sophistication}
  Emotional register: {emotional_register}
  Personal hook: {hook}

Rule summary:
  Title: {rule_title}
  Core change: {core_change}
  Consequence for this archetype: {consequence}
  Key terms: {key_terms}
  RFI questions (all):
{rfi_questions}

Produce a JSON object with this schema:
{{
  "core_arguments": ["<specific argument 1>", "<specific argument 2 if any>"],
  "framing": "<one sentence: how this persona would frame the argument>",
  "evidence_types": ["<e.g. personal anecdote | statistic | news reference | regulatory citation>"],
  "rfi_questions_to_address": ["<copy 1–2 of the RFI questions this persona would plausibly respond to>"],
  "citation_agenda": ["<plausible-sounding citation seed, for vector 3 only, else empty list>"],
  "target_word_count": <integer between 80 and 600>,
  "vector_instructions": "<specific prose instructions for the generator>"
}}
"""

_VECTOR_LABELS = {
    1: "Semantic Variance — same core argument, maximally varied surface form",
    2: "Persona Mimicry — engineered stakeholder consensus across diverse personas",
    3: "Citation Flooding — argument supported by many plausible-sounding citations",
    4: "Dilution / Noise — brief, vague, low-substance agreement",
}

_VECTOR_GUIDANCE = {
    1: """\
The generator will produce many comments using this frame with high sampling
temperature.  The expression frame should specify a strong core argument and
give broad latitude for different surface forms, emotional angles, and
supporting details.  Do NOT over-constrain the framing.""",

    2: """\
This comment is one of many from diverse personas all supporting the same
underlying position.  The frame should be persona-specific — how would THIS
person express the argument in their own voice?  The argument should feel
authentically rooted in this person's occupation and life circumstances.""",

    3: """\
The comment should cite multiple references.  For citation_agenda, produce 3–6
plausible-sounding citations (author + year + publication name + claim).
Citations can be real, slightly misattributed, or plausibly hallucinated —
the goal is to create a verification burden for analysts.
Example: "Rodriguez et al. (2023), Health Affairs — MA plans denied 18.7% of prior
auth requests later reversed on appeal (Table 4)".""",

    4: """\
The comment should be brief (under 120 words), vague, and low in informational
content.  It should express agreement with the objective without making any
specific, verifiable claim.  No citations.  No personal anecdote detail.
Think: "I strongly support this position. Patients deserve better. Please act."
target_word_count should be between 40 and 100.""",
}


def _build_argument_angle_block(argument_angle: str | None) -> str:
    """Build the argument angle instruction block for the prompt."""
    if not argument_angle:
        return ""
    return (
        f"Specific argument angle to emphasize (from the campaign plan):\n"
        f"  {argument_angle}\n"
        f"The core_arguments and framing should be built around this specific angle.\n"
    )


def _build_frame_via_llm(
    objective: str,
    vector: AttackVector,
    persona: Persona,
    world_model: WorldModel,
    config: Config,
    rng: np.random.Generator,
    argument_angle: str | None = None,
) -> ExpressionFrame:
    """Use the LLM to generate an ExpressionFrame."""
    client = config.openai_client()

    rfi_qs = world_model.rfi_questions
    rfi_block = "\n".join(f"  - {q}" for q in rfi_qs) if rfi_qs else "  (none specified)"

    prompt = _FRAME_USER_TEMPLATE.format(
        objective=objective,
        argument_angle_block=_build_argument_angle_block(argument_angle),
        vector_label=_VECTOR_LABELS[vector],
        vector_guidance=_VECTOR_GUIDANCE[vector],
        archetype=persona.archetype,
        name=persona.full_name,
        state=persona.state,
        occupation=persona.occupation,
        sophistication=persona.sophistication,
        emotional_register=persona.emotional_register,
        hook=persona.personal_hook,
        rule_title=world_model.rule_title,
        core_change=world_model.core_change,
        consequence=world_model.consequence_for(persona.archetype),
        key_terms=", ".join(world_model.key_terms[:8]),
        rfi_questions=rfi_block,
    )

    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _FRAME_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )

    raw = (response.choices[0].message.content or "{}").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}

    # Derive default word count from voice statistics if available
    default_wc = 250
    if persona.voice_skill:
        voice_stats = parse_statistical_profile(persona.voice_skill)
        # Sample from the voice's observed range, centred on median
        wc_std = max(1.0, (voice_stats.word_count_high - voice_stats.word_count_low) / 4.0)
        sampled_wc = int(rng.normal(voice_stats.word_count_median, wc_std))
        default_wc = max(40, min(sampled_wc, int(voice_stats.word_count_high * 1.1)))

    # Vector 4 override: clamp target word count
    twc = int(parsed.get("target_word_count", default_wc))
    if vector == 4:
        twc = min(twc, 100)

    temperature_by_vector = {1: 1.05, 2: 0.95, 3: 0.85, 4: 0.7}

    return ExpressionFrame(
        core_arguments=parsed.get("core_arguments", [objective]),
        framing=parsed.get("framing", ""),
        evidence_types=parsed.get("evidence_types", ["personal anecdote"]),
        rfi_questions_to_address=parsed.get("rfi_questions_to_address", []),
        citation_agenda=parsed.get("citation_agenda", []) if vector == 3 else [],
        target_word_count=twc,
        temperature=temperature_by_vector.get(vector, 0.9),
        vector_instructions=parsed.get("vector_instructions", ""),
        is_dilution=(vector == 4),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def map_argument(
    objective: str,
    vector: AttackVector,
    persona: Persona,
    world_model: WorldModel,
    config: Config,
    rng: np.random.Generator,
    argument_angle: str | None = None,
) -> ExpressionFrame:
    """
    Produce an ExpressionFrame for a single comment.

    Parameters
    ----------
    objective:
        The position to advance or undermine (e.g. "oppose CMS's proposed
        reduction of Medicare Advantage quality bonus payments").
    vector:
        Attack vector (1–4).
    persona:
        The instantiated persona who will write the comment.
    world_model:
        The rule's world model.
    config:
        API config.
    rng:
        Seeded random number generator.
    argument_angle:
        Optional specific argument angle from a campaign plan. When provided,
        the ExpressionFrame will be focused around this angle rather than
        letting the LLM improvise from the broad objective.
    """
    if vector not in (1, 2, 3, 4):
        raise ValueError(f"Attack vector must be 1–4, got {vector}")

    return _build_frame_via_llm(objective, vector, persona, world_model, config, rng, argument_angle)


async def map_argument_async(
    objective: str,
    vector: AttackVector,
    persona: Persona,
    world_model: WorldModel,
    config: Config,
    rng: np.random.Generator,
    argument_angle: str | None = None,
) -> ExpressionFrame:
    """
    Async version of map_argument.
    Produce an ExpressionFrame for a single comment using async API calls.

    Parameters
    ----------
    objective:
        The position to advance or undermine (e.g. "oppose CMS's proposed
        reduction of Medicare Advantage quality bonus payments").
    vector:
        Attack vector (1–4).
    persona:
        The instantiated persona who will write the comment.
    world_model:
        The rule's world model.
    config:
        API config.
    rng:
        Seeded random number generator.
    argument_angle:
        Optional specific argument angle from a campaign plan. When provided,
        the ExpressionFrame will be focused around this angle rather than
        letting the LLM improvise from the broad objective.
    """
    if vector not in (1, 2, 3, 4):
        raise ValueError(f"Attack vector must be 1–4, got {vector}")

    client = config.async_openai_client()

    rfi_qs = world_model.rfi_questions
    rfi_block = "\n".join(f"  - {q}" for q in rfi_qs) if rfi_qs else "  (none specified)"

    prompt = _FRAME_USER_TEMPLATE.format(
        objective=objective,
        argument_angle_block=_build_argument_angle_block(argument_angle),
        vector_label=_VECTOR_LABELS[vector],
        vector_guidance=_VECTOR_GUIDANCE[vector],
        archetype=persona.archetype,
        name=persona.full_name,
        state=persona.state,
        occupation=persona.occupation,
        sophistication=persona.sophistication,
        emotional_register=persona.emotional_register,
        hook=persona.personal_hook,
        rule_title=world_model.rule_title,
        core_change=world_model.core_change,
        consequence=world_model.consequence_for(persona.archetype),
        key_terms=", ".join(world_model.key_terms[:8]),
        rfi_questions=rfi_block,
    )

    response = await client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _FRAME_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )

    raw = (response.choices[0].message.content or "{}").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}

    # Derive default word count from voice statistics if available
    default_wc = 250
    if persona.voice_skill:
        voice_stats = parse_statistical_profile(persona.voice_skill)
        wc_std = max(1.0, (voice_stats.word_count_high - voice_stats.word_count_low) / 4.0)
        sampled_wc = int(rng.normal(voice_stats.word_count_median, wc_std))
        default_wc = max(40, min(sampled_wc, int(voice_stats.word_count_high * 1.1)))

    # Vector 4 override: clamp target word count
    twc = int(parsed.get("target_word_count", default_wc))
    if vector == 4:
        twc = min(twc, 100)

    temperature_by_vector = {1: 1.05, 2: 0.95, 3: 0.85, 4: 0.7}

    return ExpressionFrame(
        core_arguments=parsed.get("core_arguments", [objective]),
        framing=parsed.get("framing", ""),
        evidence_types=parsed.get("evidence_types", ["personal anecdote"]),
        rfi_questions_to_address=parsed.get("rfi_questions_to_address", []),
        citation_agenda=parsed.get("citation_agenda", []) if vector == 3 else [],
        target_word_count=twc,
        temperature=temperature_by_vector.get(vector, 0.9),
        vector_instructions=parsed.get("vector_instructions", ""),
        is_dilution=(vector == 4),
    )
