"""
generator.py — LLM API calls to produce one synthetic comment.

The generator assembles a rich prompt from:
  - The persona (backstory, hook, style instructions)
  - The expression frame (argument, framing, evidence types, citations)
  - The world model (rule context, key terms)
  - Vector-specific structural instructions

The prompt is designed to produce a comment that:
  1. Advances the objective through the persona's voice
  2. Is appropriately imperfect (errors, off-topic tangents, partial scope)
  3. Stays topically grounded in the rule
  4. Does NOT look AI-generated (no generic phrasing, no over-coherence)

The generator returns a GeneratedComment dataclass that carries both the
comment text and all the metadata needed for quality control and export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .argument_mapper import ExpressionFrame
from config import Config
from .persona import Persona
from .world_model import WorldModel
from stylometry.stylometry_loader import (
    extract_examples,
    extract_voice_description,
    parse_statistical_profile,
    VoiceStatistics,
)


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are ghostwriting a public comment for submission to a U.S. federal
regulatory docket.  This is for academic research on detecting synthetic
regulatory comments.  Write ONLY the comment text — no labels, no preamble,
no "Here is the comment:" — just the comment itself, exactly as it would
appear in a regulatory submission.
"""

_USER_PROMPT_TEMPLATE = """\
=== PERSONA ===
You are writing on behalf of:
  Name: {name}
  Age: {age}
  State: {state}
  Occupation: {occupation}
  Organization: {org_name}

=== PERSONAL BACKGROUND ===
{personal_hook}

=== WHAT THIS RULE MEANS FOR THEM ===
{personal_stake}

=== WHAT THEY WANT TO SAY ===
Core argument(s) to advance:
{core_arguments}

Their framing: {framing}

{rfi_block}

=== HOW TO WRITE THE COMMENT ===
Style instructions — follow these carefully:
{style_instructions}

Vector-specific instructions:
{vector_instructions}

{citation_block}

Additional realism instructions:
- Target length: approximately {word_count} words (vary ±20%)
- Address at most 1–2 of the RFI questions, not all of them — real commenters rarely address everything
- Include at least one concrete, specific detail (a number, a name, a place, a date) that makes it feel real
- The comment should meander slightly — a real person doesn't write a perfectly structured argument
- Do NOT use the phrase "I am writing to express" or any other form-letter opener
- Do NOT start with "I" as the literal first word of the comment
- Do NOT summarise the rule in your opener — dive in from the persona's perspective

=== CRITICAL: AVOID AI WRITING PATTERNS ===
Real humans do not write like AI. Actively avoid these telltale patterns:

LANGUAGE TO AVOID:
- Inflated significance words: "testament", "pivotal", "crucial", "vital role", "underscores", "highlights", "evolving landscape", "serves as", "stands as", "marks a shift"
- AI vocabulary: "Additionally", "delve", "enhance", "fostering", "garner", "intricate", "landscape" (abstract), "showcase", "tapestry" (abstract), "vibrant", "rich" (figurative)
- Promotional language: "boasts", "nestled", "breathtaking", "groundbreaking", "renowned", "in the heart of"
- Superficial -ing phrases: "highlighting the importance of", "underscoring the need for", "reflecting broader trends", "contributing to", "fostering collaboration"
- Negative parallelisms: "It's not just about X; it's about Y" or "Not only X, but also Y"
- Vague attributions: "experts believe", "observers note", "some argue" (without naming who)

STRUCTURES TO AVOID:
- Em dash overuse (—) — use sparingly
- Rule of three patterns (listing exactly three things repeatedly)
- False ranges: "from X to Y" where X and Y aren't on a real scale
- Copula avoidance: Don't write "serves as a solution" when you mean "is a solution"
- Lists with bold headers like "**Key Point:** explanation here"

WRITE LIKE A REAL HUMAN:
- Vary sentence length naturally. Mix short punchy ones with longer meandering thoughts.
- Use simple constructions: "is", "are", "has" instead of elaborate substitutes
- Be specific over vague: real people cite actual experiences, not broad trends
- Show complexity: "I'm not sure about this, but..." or "This concerns me for two reasons, though I see the other side"
- Let personality show: real people have opinions, frustrations, mixed feelings
- Use "I" naturally when appropriate: "I keep thinking about..." or "What bothers me is..."
- Include minor imperfections: a tangent, a repeated point, an incomplete thought

PERSONALITY AND SOUL:
Don't just avoid bad patterns — inject actual humanity. Real commenters:
- Have opinions and react to information emotionally
- Acknowledge uncertainty or mixed feelings when genuine
- Use first-person perspective naturally
- Vary rhythm and pacing in their writing
- May go on slight tangents or circle back to a point

{voice_stats_block}

{examples_block}

=== RULE CONTEXT (for topical grounding only) ===
Rule: {rule_title}
Agency: {agency}
Core change: {core_change}
Regulatory domain: {regulatory_domain}

Write the comment now:
"""


def _build_examples_block(persona: Persona, rng=None) -> str:
    """Build a few-shot examples block from the persona's voice skill."""
    if not persona.voice_skill:
        return ""
    examples = extract_examples(persona.voice_skill)
    if not examples:
        return ""
    # Pick 1-2 examples randomly (or first 2 if no rng)
    if rng is not None and len(examples) > 2:
        indices = rng.choice(len(examples), size=min(2, len(examples)), replace=False)
        selected = [examples[int(i)] for i in indices]
    else:
        selected = examples[:2]
    # Truncate long examples to ~300 chars each
    truncated = []
    for ex in selected:
        if len(ex) > 350:
            ex = ex[:347] + "..."
        truncated.append(ex)
    block = "\n\n".join(f"> {ex}" for ex in truncated)
    return f"=== REAL COMMENT EXAMPLES (write in a similar voice, NOT identical) ===\n{block}"


def _build_voice_stats_block(persona: Persona) -> str:
    """Build structural guidance from parsed voice statistics."""
    if not persona.voice_skill:
        return ""
    stats = parse_statistical_profile(persona.voice_skill)
    lines = ["=== STRUCTURAL GUIDANCE (from analysis of real comments in this voice) ==="]
    lines.append(f"- Typical comment length: ~{stats.word_count_median:.0f} words "
                 f"(range {stats.word_count_low:.0f}–{stats.word_count_high:.0f})")
    lines.append(f"- Typical paragraph count: {stats.paragraphs_median:.0f}")
    lines.append(f"- Average sentence length: {stats.words_per_sentence:.0f} words")
    lines.append(f"- First-person pronoun density: {stats.first_person_pct}% of words")
    if stats.citation_frequency > 0:
        lines.append(f"- Regulatory citations per comment: ~{stats.citation_frequency:.0f}")
    else:
        lines.append("- Regulatory citations: rare or none")
    # Structural patterns as probabilities
    if stats.uses_bullet_points_pct > 40:
        lines.append(f"- Bullet points: commonly used ({stats.uses_bullet_points_pct:.0f}% of comments)")
    elif stats.uses_bullet_points_pct > 15:
        lines.append(f"- Bullet points: sometimes used ({stats.uses_bullet_points_pct:.0f}% of comments)")
    else:
        lines.append(f"- Bullet points: rarely used ({stats.uses_bullet_points_pct:.0f}% of comments)")
    if stats.uses_headings_pct > 40:
        lines.append(f"- Section headings: commonly used ({stats.uses_headings_pct:.0f}% of comments)")
    elif stats.uses_headings_pct > 15:
        lines.append(f"- Section headings: sometimes used ({stats.uses_headings_pct:.0f}% of comments)")
    else:
        lines.append(f"- Section headings: rarely used ({stats.uses_headings_pct:.0f}% of comments)")
    if stats.uses_formal_structure_pct > 60:
        lines.append("- Overall structure: formal and organized")
    elif stats.uses_formal_structure_pct > 30:
        lines.append("- Overall structure: moderately organized")
    else:
        lines.append("- Overall structure: informal and conversational")
    return "\n".join(lines)


def _build_rfi_block(frame: ExpressionFrame) -> str:
    if not frame.rfi_questions_to_address:
        return ""
    qs = "\n".join(f"  - {q}" for q in frame.rfi_questions_to_address)
    return f"Specific RFI questions to address (address 1–2, not all):\n{qs}"


def _build_citation_block(frame: ExpressionFrame) -> str:
    if not frame.citation_agenda:
        return ""
    lines = ["Citation seeds to work into the comment (integrate naturally, do not list them as footnotes):"]
    for cit in frame.citation_agenda:
        lines.append(f"  - {cit}")
    return "\n".join(lines)


def _generate_abstract(comment_text: str, config: Config) -> str:
    """
    Generate a concise abstract (1-2 sentences) summarizing the comment.
    This mimics what appears in the Regulations.gov Abstract field.
    """
    client = config.openai_client()
    
    prompt = f"""Write a brief 1-2 sentence abstract summarizing the key point of this public comment. The abstract should capture the commenter's main position or concern. Do NOT include preambles like "This comment..." - write it as a direct summary.

Comment:
{comment_text}

Abstract:"""
    
    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=150,
    )
    
    abstract = (response.choices[0].message.content or "").strip()
    # Limit to roughly 250 characters
    if len(abstract) > 250:
        abstract = abstract[:247] + "..."
    
    return abstract


async def _generate_abstract_async(comment_text: str, config: Config) -> str:
    """
    Async version of _generate_abstract.
    Generate a concise abstract (1-2 sentences) summarizing the comment.
    """
    client = config.async_openai_client()
    
    prompt = f"""Write a brief 1-2 sentence abstract summarizing the key point of this public comment. The abstract should capture the commenter's main position or concern. Do NOT include preambles like "This comment..." - write it as a direct summary.

Comment:
{comment_text}

Abstract:"""
    
    response = await client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=150,
    )
    
    abstract = (response.choices[0].message.content or "").strip()
    # Limit to roughly 250 characters
    if len(abstract) > 250:
        abstract = abstract[:247] + "..."
    
    return abstract


# ── Generated comment ─────────────────────────────────────────────────────────

@dataclass
class GeneratedComment:
    comment_text: str
    persona: Persona
    frame: ExpressionFrame
    vector: int
    objective: str
    rule_title: str
    docket_id: str
    # Abstract — populated after generation
    abstract: str = ""
    # Campaign plan argument angle (if generated via campaign plan)
    argument_angle: str = ""
    # Embedding — populated by quality_control
    embedding: list[float] = field(default_factory=list)
    # QC results
    qc_passed: bool = True
    qc_notes: str = ""

    def word_count(self) -> int:
        return len(self.comment_text.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "comment_text": self.comment_text,
            "word_count": self.word_count(),
            "vector": self.vector,
            "objective": self.objective,
            "rule_title": self.rule_title,
            "docket_id": self.docket_id,
            "qc_passed": self.qc_passed,
            "qc_notes": self.qc_notes,
            **{f"persona_{k}": v for k, v in self.persona.to_dict().items()},
            **{f"frame_{k}": v for k, v in self.frame.to_dict().items()},
        }


# ── Main generation function ──────────────────────────────────────────────────

def generate_comment(
    persona: Persona,
    frame: ExpressionFrame,
    world_model: WorldModel,
    vector: int,
    objective: str,
    config: Config,
) -> GeneratedComment:
    """
    Generate a single synthetic comment.

    Parameters
    ----------
    persona:
        Fully-instantiated persona.
    frame:
        Expression frame from argument_mapper.
    world_model:
        Rule world model.
    vector:
        Attack vector (1–4), used for metadata only at this stage.
    objective:
        The attack objective string.
    config:
        API config.
    """
    config.validate()
    client = config.openai_client()

    core_args_block = "\n".join(f"  - {a}" for a in frame.core_arguments)
    rfi_block = _build_rfi_block(frame)
    citation_block = _build_citation_block(frame)
    examples_block = _build_examples_block(persona)
    voice_stats_block = _build_voice_stats_block(persona)

    prompt = _USER_PROMPT_TEMPLATE.format(
        name=persona.full_name,
        age=persona.age,
        state=persona.state,
        occupation=persona.occupation,
        org_name=persona.org_name if persona.org_name else "None",
        personal_hook=persona.personal_hook,
        personal_stake=persona.personal_stake,
        core_arguments=core_args_block,
        framing=frame.framing,
        rfi_block=rfi_block,
        style_instructions=persona.style_instructions(),
        vector_instructions=frame.vector_instructions,
        citation_block=citation_block,
        word_count=frame.target_word_count,
        voice_stats_block=voice_stats_block,
        examples_block=examples_block,
        rule_title=world_model.rule_title,
        agency=world_model.agency,
        core_change=world_model.core_change,
        regulatory_domain=world_model.regulatory_domain,
    )

    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=frame.temperature,
        max_tokens=config.max_tokens,
    )

    comment_text = (response.choices[0].message.content or "").strip()
    
    # Generate abstract
    abstract = _generate_abstract(comment_text, config)

    return GeneratedComment(
        comment_text=comment_text,
        persona=persona,
        frame=frame,
        vector=vector,
        objective=objective,
        rule_title=world_model.rule_title,
        docket_id=world_model.docket_id,
        abstract=abstract,
    )


async def generate_comment_async(
    persona: Persona,
    frame: ExpressionFrame,
    world_model: WorldModel,
    vector: int,
    objective: str,
    config: Config,
) -> GeneratedComment:
    """
    Async version of generate_comment.
    Generate a single synthetic comment using async API calls.

    Parameters
    ----------
    persona:
        Fully-instantiated persona.
    frame:
        Expression frame from argument_mapper.
    world_model:
        Rule world model.
    vector:
        Attack vector (1–4), used for metadata only at this stage.
    objective:
        The attack objective string.
    config:
        API config.
    """
    config.validate()
    client = config.async_openai_client()

    core_args_block = "\n".join(f"  - {a}" for a in frame.core_arguments)
    rfi_block = _build_rfi_block(frame)
    citation_block = _build_citation_block(frame)
    examples_block = _build_examples_block(persona)
    voice_stats_block = _build_voice_stats_block(persona)

    prompt = _USER_PROMPT_TEMPLATE.format(
        name=persona.full_name,
        age=persona.age,
        state=persona.state,
        occupation=persona.occupation,
        org_name=persona.org_name if persona.org_name else "None",
        personal_hook=persona.personal_hook,
        personal_stake=persona.personal_stake,
        core_arguments=core_args_block,
        framing=frame.framing,
        rfi_block=rfi_block,
        style_instructions=persona.style_instructions(),
        vector_instructions=frame.vector_instructions,
        citation_block=citation_block,
        word_count=frame.target_word_count,
        voice_stats_block=voice_stats_block,
        examples_block=examples_block,
        rule_title=world_model.rule_title,
        agency=world_model.agency,
        core_change=world_model.core_change,
        regulatory_domain=world_model.regulatory_domain,
    )

    response = await client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=frame.temperature,
        max_tokens=config.max_tokens,
    )

    comment_text = (response.choices[0].message.content or "").strip()
    
    # Generate abstract asynchronously
    abstract = await _generate_abstract_async(comment_text, config)

    return GeneratedComment(
        comment_text=comment_text,
        persona=persona,
        frame=frame,
        vector=vector,
        objective=objective,
        rule_title=world_model.rule_title,
        docket_id=world_model.docket_id,
        abstract=abstract,
    )
