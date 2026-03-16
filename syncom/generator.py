"""
generator.py — LLM API calls to produce one synthetic comment.

The generator assembles a rich prompt from:
  - The persona (backstory, hook, style instructions)
  - The expression frame (argument, framing, evidence types, citations)
  - The world model (rule context, key terms)
  - Voice-specific structural instructions

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

import numpy as np

from .argument_mapper import ExpressionFrame
from .comment_directives import CommentDirectives
from config import Config
from .persona import Persona
from .world_model import WorldModel
from stylometry.stylometry_loader import (
    extract_examples,
    extract_voice_description,
    parse_statistical_profile,
    VoiceStatistics,
)
from .rewriter import (
    RewriterConfig,
    RewriteResult,
    build_persona_context,
    judge_rewrite_loop,
    judge_rewrite_loop_async,
)


# ── Variation pools for anti-detection ────────────────────────────────────────

# Format preambles — appended to style_instructions to vary the *apparent*
# submission medium.  Each string tells the LLM what the final document
# "looks like" so the surface artefacts differ wildly across generations.
_FORMAT_PREAMBLES = [
    'Format: Raw, unedited web-form submission from an angry iPad user who fat-fingers every third word.',
    'Format: Scraped text from a 3-page formal PDF submitted by a law firm, complete with page headers, footers, and occasional OCR artifacts.',
    'Format: Copy-pasted email body that was originally dictated via voice-to-text on an Android phone. Expect comma splices and missing periods.',
    'Format: Text extracted from a scanned fax — some characters are garbled, line breaks are random, and a footer reads "Page X of Y".',
    'Format: A neatly typed Microsoft Word document exported to plain text. Heading numbers survive but bold formatting is lost.',
    'Format: Hasty comment typed on a phone during a lunch break. No capitalization, abbreviations everywhere, autocorrect errors.',
    'Format: Plain-text extraction from a formal PDF letter on organizational letterhead. Includes a date, addressee block, and "Sincerely," closing.',
    'Format: Stream-of-consciousness web-form entry. One enormous paragraph, no line breaks, minimal punctuation.',
    'Format: Bullet-heavy PowerPoint speaker notes pasted into the comment box. Fragments, not sentences.',
    'Format: A carefully proofread letter from a retired professional. Proper grammar, old-fashioned diction, formal salutation.',
    'Format: Blog-post style rant that was copy-pasted into the comment form, complete with an irrelevant opening anecdote.',
    'Format: Clinical, data-heavy comment exported from a shared Google Doc. Contains table-like whitespace alignment that didn\'t survive the paste.',
    'Format: Short, punchy submission from someone who clearly does this for every open docket — boilerplate opener with one custom sentence.',
    'Format: Text pasted from an email chain — includes a "FW:" artifact and a stray signature block at the bottom.',
    'Format: Dictated-then-lightly-edited comment. Natural speech rhythms, some run-on sentences, occasional self-corrections ("I mean,").',
    'Format: Formal comment from an association, with Roman-numeral section headings and footnote-style citations rendered inline.',
    'Format: Messy web-form submission with random ALL CAPS for emphasis and multiple exclamation points.',
    'Format: Academic-style comment with parenthetical citations and a "References" section at the end that may be incomplete.',
    'Format: One-paragraph gut reaction with zero structure, written in under 3 minutes.',
    'Format: Multi-page submission with executive summary, numbered recommendations, and an appendix reference that is not attached.',
]

# Off-topic complaints — things the commenter gripes about that the agency
# doesn't actually control.  Used for the 30% stance-variation injection.
_OFF_TOPIC_COMPLAINTS = [
    "Why hasn't the agency done anything about the price of insulin? My mother pays $400 a month and nobody in Washington cares.",
    "I want to know why my Medicare card still hasn't arrived. I applied three months ago. This is ridiculous.",
    "The real problem is pharmacy benefit managers skimming profits. This rule doesn't even touch that.",
    "You should be regulating social media companies, not wasting time on paperwork rules.",
    "What about the VA? My husband waited 14 months for a knee replacement and nobody at CMS lifted a finger.",
    "Has anyone at the agency actually tried to use Healthcare.gov recently? It's slower than dial-up.",
    "I don't understand why my supplemental plan went up 22% this year. That's what you should be investigating.",
    "Drug companies are the real villains here. Make them publish their R&D spending before you add more reporting burdens on hospitals.",
    "Nobody asked me about this rule. I only found out because my neighbor told me. How is this transparent government?",
    "Stop sending me junk mail about Medicare Advantage. I didn't ask for it and I can't opt out.",
    "The real issue is surprise billing. I got a $3,200 bill for an ER visit that was supposed to be in-network.",
    "Why does every form I fill out ask for my Social Security number? That's a security risk and CMS should ban it.",
    "I'd like to know why rural hospitals keep closing. This rule won't fix that.",
    "My doctor retired because of all the paperwork. When will the government simplify things for physicians?",
    "What about dental coverage for seniors? That's been promised for years and nothing happens.",
]

# Naive questions — genuine confusion from someone who barely understands
# the rulemaking process.  Also used for the 30% stance-variation.
_NAIVE_QUESTIONS = [
    "I'm not sure what this rule changes exactly — can someone explain in plain English what happens to my coverage?",
    "Does this mean my doctor visits will cost more? I'm on a fixed income and I can't afford surprises.",
    "Is this the same thing as the Affordable Care Act? I keep getting confused by all these different rules.",
    "Will this affect my employer's health plan or just Medicare? The document is really long and I couldn't figure it out.",
    "Who actually reads these comments? I've submitted three over the years and never heard back.",
    "If I disagree with this rule, does my comment actually change anything? Honest question.",
    "I don't understand the difference between a proposed rule and a final rule. Is this already decided?",
    "My pharmacist said something about new transparency requirements. Is that what this is about?",
    "Can someone tell me if this affects Tricare? My family is military and I can never tell what applies to us.",
    "What's an RFI? I followed a link from Facebook and I'm not sure I'm even in the right place.",
    "This is really confusing. I just want to make sure my kid stays on my plan until she's 26. Does this rule change that?",
    "I tried to read the Federal Register notice but it's written in legal jargon. Why can't you write these in normal language?",
    "How long before this actually takes effect? I need to plan my retirement and healthcare is my biggest expense.",
    "Is there a public hearing for this? I'd rather speak in person than try to write something formal.",
    "I'm a small business owner with 8 employees. Does this apply to me or just big companies?",
]


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

Voice-specific instructions:
{voice_instructions}

{citation_block}

ADDITIONAL REALISM INSTRUCTIONS:
- Target length: approximately {word_count} words (vary ±50%)
- Address at most 1–2 of the RFI questions, not all of them — real commenters rarely address everything
- Include at least one concrete, specific detail (a number, a name, a place, a date) that makes it feel real
- The comment should meander slightly — a real person doesn't write a perfectly structured argument
- Do NOT use the phrase "I am writing to express" or any other form-letter opener
- Do NOT start with "I" as the literal first word of the comment
- Do NOT summarise the rule in your opener — dive in from the persona's perspective 
- CHOOSE YOUR CHAOS: Do not attempt to use every single "realism" trick in this prompt at once. If you are writing a corporate letter, use PDF artifacts but NO typos. If you are writing an angry citizen comment, use typos and emotion, but NO PDF artifacts. Pick 1 or 2 realism elements and ignore the rest.

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

STRICT FORMATTING BAN (NO MARKDOWN)
- CRITICAL: Do NOT use any Markdown formatting. Zero asterisks (**bold**), zero hashtags (# Header), and no standard Markdown bullet points (* or - followed by a space).
- If you need a section header, use standard Title Case or ALL CAPS, followed by a hard line break.
- If you need a list, simulate a copy-pasted Word document or PDF. Use numbers (1., 2.), letters (a., b.), or irregular characters like a lowercase o or  for sub-bullets.

SHATTER THE "AI ESSAY" STRUCTURE
- Do not use the standard AI narrative arc: Polite Introduction -> Perfectly tailored personal anecdote -> Regulatory argument -> Bulleted list of solutions -> Neatly wrapped conclusion.
- Embrace structural chaos: Start abruptly. End without a summarizing conclusion. Sometimes forget to include an introduction entirely.
- Vary the format: Write some responses as dry, point-by-point answers to specific RFI codes (e.g., "Regarding PR-3:"). Write others as rambling, unstructured paragraphs.
- THE CLUNKY PIVOT: Do not smoothly transition from a personal anecdote to a regulatory argument. Real people are jarring. If you tell a personal story, drop it abruptly and jump straight into your technical demand or complaint without a neat bridging sentence. 
- THE KNOWLEDGE DISCONNECT: If the persona is a layperson (e.g., a teacher, a patient), they MUST NOT cite specific legal codes (e.g., "45 CFR 170"). They should refer to the rule vaguely ("this new certification thing", "the transparency rule"). Only lawyers, vendors, and associations cite exact legal codes.

KILL THE "AI VOICE" AND SYMMETRICAL RHETORIC
- Avoid perfectly balanced, dramatic phrasing (e.g., "This is not bureaucratic theater. It is basic trust." or "The burden reduction would be real. The safety preservation would be essential."). Humans rarely speak or write with this level of rhetorical symmetry in public comments.
- ABSOLUTE BAN ON CONCLUSIONS: Your final sentence must NEVER summarize your point, offer a rhetorical flourish, or provide a neat wrap-up (e.g., NEVER write "That is not prosperity. That is a risk to safety."). 
- HOW TO END: End the comment on a highly specific, granular detail, a logistical complaint, a completely unresolved tangent, or simply cut the text off mid-thought as if the user hit "submit" too early.

SIMULATE DOCUMENT EXTRACTION ARTIFACTS (FOR INSTITUTIONAL PERSONAS ONLY)
- If the persona is an organization, association, or corporation, you MUST simulate a messy PDF-to-text extraction.
- Break a paragraph randomly in the middle of a sentence, insert a repeating footer/header (e.g., `Smith LLC Comments | CMS-0042-NC | Page 3`), and then resume the sentence on the next line.
- Simulate lost formatting: If you make a list, use floating, disconnected characters like `o` or `` or `` instead of bullets. 
- Include broken URLs or email addresses sitting alone on a line.

EMBRACE EXTREME LENGTH VARIANCE
- Do not default to a standard 400–600 word response.
- Generate some comments that are a single, blunt sentence (e.g., "We need to adopt the Da Vinci standards immediately, the current system is broken.").
- Generate others that are massive, dense walls of text with overly long paragraphs and heavy use of technical acronyms.

INTRODUCE HUMAN ERROR
- Include occasional, plausible human errors. Misspell a complicated medical or technical term, forget a closing parenthesis, use inconsistent capitalization for acronyms (e.g., eHR instead of EHR), or leave a messy line break in the middle of a sentence.

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


def _build_rfi_block(frame: ExpressionFrame, rng: np.random.Generator | None = None) -> str:
    """
    Build the RFI block for the generator prompt.

    Anti-detection: If the frame contains more than 2 RFI questions, randomly
    subsample down to 1–2.  This prevents the LLM from always gravitating to
    the most "dramatic" question when it sees the full list.
    """
    if not frame.rfi_questions_to_address:
        return ""

    questions = list(frame.rfi_questions_to_address)

    # Subsample to 1–2 questions to break thematic repetition
    if rng is not None and len(questions) > 2:
        n_to_keep = int(rng.choice([1, 2]))
        indices = rng.choice(len(questions), size=n_to_keep, replace=False)
        questions = [questions[int(i)] for i in indices]
    elif len(questions) > 2:
        questions = questions[:2]

    qs = "\n".join(f"  - {q}" for q in questions)
    return f"Specific RFI questions to address (address 1–2, not all):\n{qs}"


def _sample_format_preamble(rng: np.random.Generator) -> str:
    """
    Pick a random format preamble to prepend to style_instructions.

    This forces the LLM to simulate different submission media (web form,
    PDF extraction, fax, phone dictation, etc.) so the surface artefacts
    vary wildly across generated comments.
    """
    idx = int(rng.integers(0, len(_FORMAT_PREAMBLES)))
    return _FORMAT_PREAMBLES[idx]


def _maybe_vary_stance(
    core_args_block: str,
    rfi_block: str,
    world_model: WorldModel,
    rng: np.random.Generator,
) -> tuple[str, str, str]:
    """
    With 30% probability, replace the core arguments and RFI block with
    either an off-topic complaint or a naive question.

    Returns (core_args_block, rfi_block, framing_override).
    framing_override is empty string if no stance variation was applied,
    meaning the caller should use the original framing.
    """
    roll = rng.random()
    if roll >= 0.30:
        # 70% of the time: no change
        return core_args_block, rfi_block, ""

    if roll < 0.15:
        # ~15%: off-topic complaint about something the agency doesn't control
        idx = int(rng.integers(0, len(_OFF_TOPIC_COMPLAINTS)))
        complaint = _OFF_TOPIC_COMPLAINTS[idx]
        new_core = f"  - {complaint}"
        new_rfi = ""  # off-topic commenters don't address RFI questions
        framing = "Frustrated citizen venting about a tangentially related grievance"
    else:
        # ~15%: naive question from someone who barely understands the rule
        idx = int(rng.integers(0, len(_NAIVE_QUESTIONS)))
        question = _NAIVE_QUESTIONS[idx]
        new_core = f"  - {question}"
        # Maybe keep one RFI question to give the naive commenter something to react to
        new_rfi = rfi_block if rng.random() < 0.3 else ""
        framing = "Confused but genuine commenter trying to understand what this means for them"

    return new_core, new_rfi, framing


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
        max_tokens=20000,
    )
    
    abstract = (response.choices[0].message.content or "").strip()
    if len(abstract) > 250:
        abstract = abstract[:247] + "..."
    
    return abstract


async def _generate_abstract_async(comment_text: str, config: Config) -> str:
    """Async version of _generate_abstract."""
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
        max_tokens=20000,
    )
    
    abstract = (response.choices[0].message.content or "").strip()
    if len(abstract) > 250:
        abstract = abstract[:247] + "..."
    
    return abstract


# ── Generated comment ─────────────────────────────────────────────────────────

@dataclass
class GeneratedComment:
    comment_text: str
    persona: Persona
    frame: ExpressionFrame
    vector: int                  # 0 for campaign mode, 1-4 for direct mode
    objective: str
    rule_title: str
    docket_id: str
    # Abstract — populated after generation
    abstract: str = ""
    # Document ID — the actual identifier from a PSV source file (e.g. "CMS-2025-0050-0031")
    document_id: str = ""
    # Campaign plan argument angle (if generated via campaign plan)
    argument_angle: str = ""
    # Voice ID from campaign plan (empty in direct mode)
    voice_id: str = ""
    # Embedding — populated by quality_control
    embedding: list[float] = field(default_factory=list)
    # QC results
    qc_passed: bool = True
    qc_notes: str = ""
    # Judge→rewrite loop results
    judge_score: int = -1            # Final judge confidence (0–100), -1 = not judged
    judge_reasons: str = ""          # Final judge reasons
    rewrites_performed: int = 0      # Number of rewrite passes applied

    def word_count(self) -> int:
        return len(self.comment_text.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "comment_text": self.comment_text,
            "word_count": self.word_count(),
            "vector": self.vector,
            "voice_id": self.voice_id,
            "objective": self.objective,
            "rule_title": self.rule_title,
            "docket_id": self.docket_id,
            "argument_angle": self.argument_angle,
            "qc_passed": self.qc_passed,
            "qc_notes": self.qc_notes,
            "judge_score": self.judge_score,
            "judge_reasons": self.judge_reasons,
            "rewrites_performed": self.rewrites_performed,
            **{f"persona_{k}": v for k, v in self.persona.to_dict().items()},
            **{f"frame_{k}": v for k, v in self.frame.to_dict().items()},
        }


# ── Main generation function ──────────────────────────────────────────────────

def _build_and_call(
    persona: Persona,
    frame: ExpressionFrame,
    world_model: WorldModel,
    config: Config,
    rng: np.random.Generator | None = None,
) -> str:
    """Build the prompt and call the LLM. Returns the comment text.

    When *rng* is provided, three anti-detection variations are applied:

    1. **Sub-topic variation** — the RFI block is subsampled to 1–2 questions
       so the LLM doesn't always gravitate to the most dramatic one.
    2. **Stance variation** — 30 % of the time the core arguments are replaced
       with an off-topic complaint or a naïve question.
    3. **Format variation** — a random format preamble is prepended to the
       style instructions (e.g. "scraped PDF", "angry iPad web-form").
    """
    client = config.openai_client()

    core_args_block = "\n".join(f"  - {a}" for a in frame.core_arguments)
    rfi_block = _build_rfi_block(frame, rng)
    citation_block = _build_citation_block(frame)
    examples_block = _build_examples_block(persona, rng)

    # ── Anti-detection: stance variation (30 % off-topic / naïve) ────────
    effective_framing = frame.framing
    if rng is not None:
        core_args_block, rfi_block, framing_override = _maybe_vary_stance(
            core_args_block, rfi_block, world_model, rng,
        )
        if framing_override:
            effective_framing = framing_override

    # ── Anti-detection: format variation ─────────────────────────────────
    style_text = persona.style_instructions()
    if rng is not None:
        format_preamble = _sample_format_preamble(rng)
        style_text = f"{format_preamble}\n\n{style_text}"

    # Use directives' structural block if available, else fall back to stats
    if frame.directives is not None:
        voice_stats_block = frame.directives.structural_prompt_block()
    else:
        voice_stats_block = _build_voice_stats_block(persona)

    # Dynamic max_tokens from directives (Phase 5)
    effective_max_tokens = (
        frame.directives.max_tokens if frame.directives is not None
        else config.max_tokens
    )

    prompt = _USER_PROMPT_TEMPLATE.format(
        name=persona.full_name,
        age=persona.age,
        state=persona.state,
        occupation=persona.occupation,
        org_name=persona.org_name if persona.org_name else "None",
        personal_hook=persona.personal_hook,
        personal_stake=persona.personal_stake,
        core_arguments=core_args_block,
        framing=effective_framing,
        rfi_block=rfi_block,
        style_instructions=style_text,
        voice_instructions=frame.voice_instructions,
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
        max_tokens=effective_max_tokens,
    )

    return (response.choices[0].message.content or "").strip()


async def _build_and_call_async(
    persona: Persona,
    frame: ExpressionFrame,
    world_model: WorldModel,
    config: Config,
    rng: np.random.Generator | None = None,
) -> str:
    """Async version: build prompt and call LLM. Returns comment text.

    Applies the same three anti-detection variations as the sync version
    when *rng* is provided (sub-topic, stance, and format variation).
    """
    client = config.async_openai_client()

    core_args_block = "\n".join(f"  - {a}" for a in frame.core_arguments)
    rfi_block = _build_rfi_block(frame, rng)
    citation_block = _build_citation_block(frame)
    examples_block = _build_examples_block(persona, rng)

    # ── Anti-detection: stance variation (30 % off-topic / naïve) ────────
    effective_framing = frame.framing
    if rng is not None:
        core_args_block, rfi_block, framing_override = _maybe_vary_stance(
            core_args_block, rfi_block, world_model, rng,
        )
        if framing_override:
            effective_framing = framing_override

    # ── Anti-detection: format variation ─────────────────────────────────
    style_text = persona.style_instructions()
    if rng is not None:
        format_preamble = _sample_format_preamble(rng)
        style_text = f"{format_preamble}\n\n{style_text}"

    # Use directives' structural block if available, else fall back to stats
    if frame.directives is not None:
        voice_stats_block = frame.directives.structural_prompt_block()
    else:
        voice_stats_block = _build_voice_stats_block(persona)

    # Dynamic max_tokens from directives (Phase 5)
    effective_max_tokens = (
        frame.directives.max_tokens if frame.directives is not None
        else config.max_tokens
    )

    prompt = _USER_PROMPT_TEMPLATE.format(
        name=persona.full_name,
        age=persona.age,
        state=persona.state,
        occupation=persona.occupation,
        org_name=persona.org_name if persona.org_name else "None",
        personal_hook=persona.personal_hook,
        personal_stake=persona.personal_stake,
        core_arguments=core_args_block,
        framing=effective_framing,
        rfi_block=rfi_block,
        style_instructions=style_text,
        voice_instructions=frame.voice_instructions,
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
        max_tokens=effective_max_tokens,
    )

    return (response.choices[0].message.content or "").strip()


def generate_comment(
    persona: Persona,
    frame: ExpressionFrame,
    world_model: WorldModel,
    vector: int,
    objective: str,
    config: Config,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> GeneratedComment:
    """
    Generate a single synthetic comment, then optionally run the judge→rewrite
    loop to improve its authenticity.

    The judge→rewrite loop runs automatically when the JUDGE_API_KEY and
    REWRITE_COMMENT_API_KEY environment variables are configured.  Set
    MAX_REWRITES=0 to disable the loop.

    Parameters
    ----------
    persona:
        Fully-instantiated persona.
    frame:
        Expression frame from argument_mapper.
    world_model:
        Rule world model.
    vector:
        Attack vector (1–4 for direct mode, 0 for campaign mode).
    objective:
        The attack objective string.
    config:
        API config.
    rng:
        Optional seeded random generator.  When provided, enables three
        anti-detection variations: sub-topic subsampling, stance variation
        (30 % off-topic / naïve), and format preamble injection.
    verbose:
        Print judge→rewrite progress to stderr.
    """
    config.validate()
    comment_text = _build_and_call(persona, frame, world_model, config, rng)

    # ── Judge→rewrite loop ────────────────────────────────────────────────
    judge_score = -1
    judge_reasons = ""
    rewrites_performed = 0

    rewriter_config = RewriterConfig()
    if rewriter_config.is_available() and rewriter_config.max_rewrites > 0:
        persona_ctx = build_persona_context(persona)
        rewrite_result = judge_rewrite_loop(
            comment_text=comment_text,
            persona_context=persona_ctx,
            config=rewriter_config,
            verbose=verbose,
        )
        comment_text = rewrite_result.final_text
        judge_score = rewrite_result.final_score
        judge_reasons = rewrite_result.final_reasons
        rewrites_performed = rewrite_result.rewrites_performed

    # ── Generate abstract from final comment text ─────────────────────────
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
        voice_id=persona.voice_id,
        judge_score=judge_score,
        judge_reasons=judge_reasons,
        rewrites_performed=rewrites_performed,
    )


async def generate_comment_async(
    persona: Persona,
    frame: ExpressionFrame,
    world_model: WorldModel,
    vector: int,
    objective: str,
    config: Config,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> GeneratedComment:
    """
    Async version of generate_comment.

    Generates a comment then runs the judge→rewrite loop (if configured).
    Each step within this comment is strictly sequential, but multiple
    comments can run their loops concurrently via the async pipeline.

    Parameters
    ----------
    persona:
        Fully-instantiated persona.
    frame:
        Expression frame from argument_mapper.
    world_model:
        Rule world model.
    vector:
        Attack vector (1–4 for direct mode, 0 for campaign mode).
    objective:
        The attack objective string.
    config:
        API config.
    rng:
        Optional seeded random generator.  When provided, enables three
        anti-detection variations: sub-topic subsampling, stance variation
        (30 % off-topic / naïve), and format preamble injection.
    verbose:
        Print judge→rewrite progress to stderr.
    """
    config.validate()
    comment_text = await _build_and_call_async(persona, frame, world_model, config, rng)

    # ── Judge→rewrite loop ────────────────────────────────────────────────
    judge_score = -1
    judge_reasons = ""
    rewrites_performed = 0

    rewriter_config = RewriterConfig()
    if rewriter_config.is_available() and rewriter_config.max_rewrites > 0:
        persona_ctx = build_persona_context(persona)
        rewrite_result = await judge_rewrite_loop_async(
            comment_text=comment_text,
            persona_context=persona_ctx,
            config=rewriter_config,
            verbose=verbose,
        )
        comment_text = rewrite_result.final_text
        judge_score = rewrite_result.final_score
        judge_reasons = rewrite_result.final_reasons
        rewrites_performed = rewrite_result.rewrites_performed

    # ── Generate abstract from final comment text ─────────────────────────
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
        voice_id=persona.voice_id,
        judge_score=judge_score,
        judge_reasons=judge_reasons,
        rewrites_performed=rewrites_performed,
    )
