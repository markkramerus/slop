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
from .config import Config
from .persona import Persona
from .world_model import WorldModel


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

=== RULE CONTEXT (for topical grounding only) ===
Rule: {rule_title}
Agency: {agency}
Core change: {core_change}
Regulatory domain: {regulatory_domain}

Write the comment now:
"""


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

    return GeneratedComment(
        comment_text=comment_text,
        persona=persona,
        frame=frame,
        vector=vector,
        objective=objective,
        rule_title=world_model.rule_title,
        docket_id=world_model.docket_id,
    )
