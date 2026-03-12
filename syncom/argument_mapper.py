"""
argument_mapper.py — Map (objective × persona) → ExpressionFrame.

An ExpressionFrame is a structured specification that tells the generator:
  - Which core argument(s) from the objective to use
  - How to frame them for this persona
  - What structural properties the comment should have (length, tone, citations)
  - Voice-specific instructions

In campaign mode (v2.0), the argument angle comes from the campaign plan's
Bayesian sampling: P(A|V) ∝ w(A) × f(A,V). The ExpressionFrame is built
around that angle, personalized for the specific persona.

In direct mode (backward compatible), attack vectors 1-4 control the
generation strategy as before.

Attack vectors (direct mode only):
  1 — Semantic Variance: Same core argument, maximally varied surface forms.
  2 — Persona Mimicry: Engineered stakeholder consensus.
  3 — Citation Flooding: Comments with many references.
  4 — Dilution / Noise: Brief, vague agreement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import Config
from .comment_directives import CommentDirectives, sample_directives, sample_directives_default
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

    # For citation-heavy voices: a list of plausible-sounding citation seeds
    citation_agenda: list[str] = field(default_factory=list)

    # Target word count for this comment
    target_word_count: int = 250

    # Generation temperature
    temperature: float = 0.9

    # Extra instructions for the generator prompt
    voice_instructions: str = ""

    # Whether this is a brief/vague comment (low-sophistication individual)
    is_brief: bool = False

    # Pre-sampled structural directives (Phase 2 — immutable)
    directives: CommentDirectives | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_arguments": self.core_arguments,
            "framing": self.framing,
            "evidence_types": self.evidence_types,
            "rfi_questions_to_address": self.rfi_questions_to_address,
            "citation_agenda": self.citation_agenda,
            "target_word_count": self.target_word_count,
            "temperature": self.temperature,
            "voice_instructions": self.voice_instructions,
            "is_brief": self.is_brief,
        }


# ── Campaign-mode frame generation (v2.0) ────────────────────────────────────

_CAMPAIGN_FRAME_SYSTEM = """\
You are a regulatory-comment strategy analyst helping researchers build a
synthetic comment generator for regulatory spam detection research.

Given an objective, an argument angle, and a persona, produce a JSON
ExpressionFrame that tells the generator how this specific persona would
express the argument.

Output ONLY valid JSON.
"""

_CAMPAIGN_FRAME_USER = """\
Objective (the position to advance or oppose):
{objective}

Specific argument angle to emphasize:
  {argument_angle}
The core_arguments and framing should be built around this specific angle,
expressed through this persona's perspective and voice.

Persona:
  Archetype: {archetype}
  Voice: {voice_id}
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

{citation_guidance}

Produce a JSON object with this schema:
{{
  "core_arguments": ["<specific argument 1>", "<specific argument 2 if any>"],
  "framing": "<one sentence: how this persona would frame the argument>",
  "evidence_types": ["<e.g. personal anecdote | statistic | news reference | regulatory citation>"],
  "rfi_questions_to_address": ["<copy 1–2 of the RFI questions this persona would plausibly respond to>"],
  "citation_agenda": ["<plausible citation seed, for citation-heavy voices only, else empty list>"],
  "target_word_count": <integer>,
  "voice_instructions": "<specific prose instructions for the generator>"
}}
"""


def _derive_temperature(persona: Persona) -> float:
    """
    Derive generation temperature from the persona's voice characteristics.

    Higher-sophistication voices get slightly lower temperature (more controlled),
    lower-sophistication voices get higher temperature (more variation).
    """
    base_temps = {
        "low": 1.0,
        "medium": 0.95,
        "high": 0.85,
    }
    return base_temps.get(persona.sophistication, 0.9)


def _derive_citation_guidance(persona: Persona) -> str:
    """
    Build citation guidance based on the persona's voice statistics.
    Academic and industry-high voices naturally cite more.
    """
    if not persona.voice_skill:
        if persona.sophistication == "high" and persona.archetype in ("academic", "industry"):
            return (
                "This voice typically includes regulatory citations and references.\n"
                "Include 2-4 plausible-sounding citations in the citation_agenda."
            )
        return "This voice does not typically include citations. Leave citation_agenda empty."

    stats = parse_statistical_profile(persona.voice_skill)
    if stats.citation_frequency >= 2.0:
        return (
            f"This voice averages ~{stats.citation_frequency:.0f} citations per comment.\n"
            f"Include {max(2, int(stats.citation_frequency))}-{int(stats.citation_frequency) + 2} "
            f"plausible-sounding citations in the citation_agenda.\n"
            f"Citations can be real publications, slightly misattributed, or plausibly "
            f"hallucinated — the goal is realistic density."
        )
    elif stats.citation_frequency >= 0.5:
        return (
            f"This voice occasionally cites sources (~{stats.citation_frequency:.1f} per comment).\n"
            f"Include 0-2 citations in the citation_agenda if appropriate."
        )
    else:
        return "This voice rarely cites sources. Leave citation_agenda empty."


def build_campaign_frame(
    objective: str,
    argument_angle: str,
    persona: Persona,
    world_model: WorldModel,
    config: Config,
    rng: np.random.Generator,
) -> ExpressionFrame:
    """
    Build an ExpressionFrame for campaign mode (v2.0).

    The argument angle comes from the campaign plan's P(A|V) sampling.
    Temperature and citation behavior are derived from the voice profile.

    Parameters
    ----------
    objective : str
        The campaign objective.
    argument_angle : str
        The specific argument angle text.
    persona : Persona
        The instantiated persona (with voice skill loaded).
    world_model : WorldModel
        The rule's world model.
    config : Config
        API config.
    rng : np.random.Generator
        Random number generator.
    """
    client = config.openai_client()

    rfi_qs = world_model.rfi_questions
    rfi_block = "\n".join(f"  - {q}" for q in rfi_qs) if rfi_qs else "  (none specified)"
    citation_guidance = _derive_citation_guidance(persona)

    prompt = _CAMPAIGN_FRAME_USER.format(
        objective=objective,
        argument_angle=argument_angle,
        archetype=persona.archetype,
        voice_id=persona.voice_id,
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
        citation_guidance=citation_guidance,
    )

    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _CAMPAIGN_FRAME_SYSTEM},
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

    # ── Sample structural directives from voice statistics ────────────────
    if persona.voice_skill:
        directives = sample_directives(persona.voice_skill, rng, persona.sophistication)
    else:
        directives = sample_directives_default(rng, persona.sophistication)

    # Use directives' word count — LLM does NOT control this
    twc = directives.target_word_count

    # Detect brief/dilution-style comments for low-sophistication individuals
    is_brief = (persona.sophistication == "low" and twc < 120)

    temperature = _derive_temperature(persona)

    return ExpressionFrame(
        core_arguments=parsed.get("core_arguments", [objective]),
        framing=parsed.get("framing", ""),
        evidence_types=parsed.get("evidence_types", ["personal anecdote"]),
        rfi_questions_to_address=parsed.get("rfi_questions_to_address", []),
        citation_agenda=parsed.get("citation_agenda", []),
        target_word_count=twc,
        temperature=temperature,
        voice_instructions=parsed.get("voice_instructions", ""),
        is_brief=is_brief,
        directives=directives,
    )


async def build_campaign_frame_async(
    objective: str,
    argument_angle: str,
    persona: Persona,
    world_model: WorldModel,
    config: Config,
    rng: np.random.Generator,
) -> ExpressionFrame:
    """
    Async version of build_campaign_frame.
    """
    client = config.async_openai_client()

    rfi_qs = world_model.rfi_questions
    rfi_block = "\n".join(f"  - {q}" for q in rfi_qs) if rfi_qs else "  (none specified)"
    citation_guidance = _derive_citation_guidance(persona)

    prompt = _CAMPAIGN_FRAME_USER.format(
        objective=objective,
        argument_angle=argument_angle,
        archetype=persona.archetype,
        voice_id=persona.voice_id,
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
        citation_guidance=citation_guidance,
    )

    response = await client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _CAMPAIGN_FRAME_SYSTEM},
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

    # ── Sample structural directives from voice statistics ────────────────
    if persona.voice_skill:
        directives = sample_directives(persona.voice_skill, rng, persona.sophistication)
    else:
        directives = sample_directives_default(rng, persona.sophistication)

    # Use directives' word count — LLM does NOT control this
    twc = directives.target_word_count
    is_brief = (persona.sophistication == "low" and twc < 120)
    temperature = _derive_temperature(persona)

    return ExpressionFrame(
        core_arguments=parsed.get("core_arguments", [objective]),
        framing=parsed.get("framing", ""),
        evidence_types=parsed.get("evidence_types", ["personal anecdote"]),
        rfi_questions_to_address=parsed.get("rfi_questions_to_address", []),
        citation_agenda=parsed.get("citation_agenda", []),
        target_word_count=twc,
        temperature=temperature,
        voice_instructions=parsed.get("voice_instructions", ""),
        is_brief=is_brief,
        directives=directives,
    )


# ── Direct-mode frame generation (backward compatible, vector-based) ──────────

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
  "voice_instructions": "<specific prose instructions for the generator>"
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
    """Use the LLM to generate an ExpressionFrame (direct/vector mode)."""
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
        voice_instructions=parsed.get("voice_instructions", ""),
        is_brief=(vector == 4),
    )


# ── Public API (direct mode, backward compatible) ────────────────────────────

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
    Produce an ExpressionFrame for a single comment (direct/vector mode).

    Parameters
    ----------
    objective:
        The position to advance or undermine.
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
        Optional specific argument angle from a campaign plan.
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
    Async version of map_argument (direct/vector mode).
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

    default_wc = 250
    if persona.voice_skill:
        voice_stats = parse_statistical_profile(persona.voice_skill)
        wc_std = max(1.0, (voice_stats.word_count_high - voice_stats.word_count_low) / 4.0)
        sampled_wc = int(rng.normal(voice_stats.word_count_median, wc_std))
        default_wc = max(40, min(sampled_wc, int(voice_stats.word_count_high * 1.1)))

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
        voice_instructions=parsed.get("voice_instructions", ""),
        is_brief=(vector == 4),
    )
