"""
persona.py — Sample archetypes and instantiate fully-specified personas.

A Persona is generated in two steps:
  1. Metadata — archetype, demographics, occupation, stake in the rule.
  2. Personal hook — a specific micro-narrative that anchors the comment in a
     plausible lived experience.

The hook is the single most important factor for authenticity.  Hooks are
generated via LLM given the persona metadata and the rule's consequence for
that archetype.  The comment is then written as a response to the hook, not
to an abstract policy position.

In campaign mode (v2.0), personas are sampled by voice_id directly from the
campaign plan's P(V) distribution, loading the exact voice skill for that
voice group.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import Config
from shared_models import ArchetypeProfile, PopulationModel
from .world_model import WorldModel
from .org_pool import OrgPool


# ── Static pools (used when the historical docket provides no metadata) ───────

_FIRST_NAMES = [
    "Patricia", "James", "Linda", "Robert", "Barbara", "Michael", "Susan",
    "William", "Jessica", "David", "Karen", "Richard", "Sarah", "Joseph",
    "Lisa", "Thomas", "Nancy", "Charles", "Betty", "Christopher", "Margaret",
    "Daniel", "Sandra", "Matthew", "Ashley", "Anthony", "Dorothy", "Mark",
    "Kimberly", "Donald", "Emily", "Paul", "Donna", "Steven", "Michelle",
    "Andrew", "Carol", "Kenneth", "Amanda", "Joshua", "Melissa", "George",
    "Deborah", "Kevin", "Stephanie", "Brian", "Rebecca", "Edward", "Sharon",
]

_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
]

_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Idaho", "Illinois",
    "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
    "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
    "Washington", "West Virginia", "Wisconsin", "Wyoming",
]

_OCCUPATIONS_BY_ARCHETYPE: dict[str, list[str]] = {
    "individual": [
        "retired teacher", "nurse", "retired nurse", "social worker",
        "small business owner", "office manager", "truck driver",
        "stay-at-home parent", "retail worker", "construction worker",
        "veteran", "caregiver", "elementary school teacher", "librarian",
        "firefighter", "administrative assistant", "warehouse worker",
        "home health aide", "customer service representative",
    ],
    "advocacy_organization": [
        "executive director", "policy director", "senior policy analyst",
        "director of government affairs", "community organizer",
        "campaign director", "research director",
    ],
    "organization": [
        "compliance officer", "chief medical officer", "VP of regulatory affairs",
        "director of government relations", "healthcare administrator",
        "hospital CFO", "practice manager", "health IT director",
    ],
    "academic_organization": [
        "professor of health policy", "associate professor", "PhD researcher",
        "postdoctoral fellow", "assistant professor", "health economist",
        "biostatistician", "public health researcher",
    ],
    "government_organization": [
        "Medicaid director", "state insurance commissioner",
        "county health officer", "city public health director",
        "state budget analyst",
    ],
}

_AGE_RANGES_BY_ARCHETYPE: dict[str, tuple[int, int]] = {
    "individual": (35, 78),
    "advocacy_organization": (30, 65),
    "organization": (35, 60),
    "academic_organization": (30, 65),
    "government_organization": (35, 60),
}

_SOPHISTICATION_BY_ARCHETYPE: dict[str, list[str]] = {
    "individual": ["low", "low", "medium"],
    "advocacy_organization": ["medium", "high"],
    "organization": ["high", "high", "medium"],
    "academic_organization": ["high", "high"],
    "government_organization": ["medium", "high"],
}

_EMOTIONAL_REGISTERS = ["frustrated", "concerned", "hopeful", "urgent", "resigned", "angry", "supportive"]


# ── Persona dataclass ─────────────────────────────────────────────────────────

@dataclass
class Persona:
    archetype: str
    first_name: str
    last_name: str
    state: str
    occupation: str
    age: int
    sophistication: str          # low | medium | high
    emotional_register: str
    org_name: str                # empty for individuals
    personal_stake: str          # How the rule affects them specifically
    personal_hook: str           # The specific micro-narrative (generated by LLM)
    voice_skill: str = ""        # Docket-specific voice skill (loaded from stylometry)
    voice_id: str = ""           # Voice group ID (e.g. "industry_high_org")

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @property
    def is_individual(self) -> bool:
        return self.archetype == "individual"

    def style_instructions(self) -> str:
        """
        Return style guidance for the comment generator based on this persona.
        Uses docket-specific voice skill if available, falls back to generic.
        """
        # Use docket-specific skill if available
        if self.voice_skill:
            from stylometry.stylometry_loader import extract_skill_instructions
            return extract_skill_instructions(self.voice_skill)
        
        # Fall back to generic instructions
        return self._generic_style_instructions()
    
    def _generic_style_instructions(self) -> str:
        """
        Generate generic style instructions (fallback when no skill available).
        """
        lines = []

        if self.sophistication == "low":
            lines += [
                "Write in simple, conversational English.",
                "Use short sentences. Some may be incomplete.",
                "Occasionally make common spelling or grammar mistakes "
                "(e.g. 'loose' for 'lose', missing commas, run-on sentences).",
                "Do NOT use bullet points or formal structure.",
                "Do NOT cite regulations by number.",
                "Use first-person ('I', 'my', 'we').",
                "Show emotion freely.",
            ]
        elif self.sophistication == "medium":
            lines += [
                "Write in clear but informal English.",
                "Mix some short sentences with longer ones.",
                "You may use a few bullet points if helpful.",
                "Occasionally reference a news article or general statistic.",
                "Use first-person voice.",
            ]
        else:  # high
            lines += [
                "Write in formal, professional prose.",
                "Use regulatory terminology where appropriate.",
                "Structure the comment with clear paragraphs, possibly with headings.",
                "Cite specific regulatory sections (e.g. '42 CFR § 422.560') where relevant.",
                "Present evidence and logical arguments.",
                "Third-person references to the organisation are acceptable.",
            ]

        # Emotional register colour
        register_map = {
            "frustrated": "Express frustration and disappointment clearly.",
            "concerned": "Express serious concern but remain measured in tone.",
            "hopeful": "Express a degree of optimism that the agency will listen.",
            "urgent": "Convey a sense of urgency — this is important and time-sensitive.",
            "resigned": "Sound somewhat resigned, as if this is a pattern you have seen before.",
            "angry": "Do not hide anger — this rule (or its absence) has caused real harm.",
            "supportive": "Express support for the agency's goals while raising specific concerns.",
        }
        lines.append(register_map.get(self.emotional_register, ""))

        return "\n".join(f"- {l}" for l in lines if l)

    def to_dict(self) -> dict[str, Any]:
        return {
            "archetype": self.archetype,
            "voice_id": self.voice_id,
            "full_name": self.full_name,
            "state": self.state,
            "occupation": self.occupation,
            "age": self.age,
            "sophistication": self.sophistication,
            "emotional_register": self.emotional_register,
            "org_name": self.org_name,
            "personal_stake": self.personal_stake,
            "personal_hook": self.personal_hook,
        }


# ── Voice ID parsing ─────────────────────────────────────────────────────────

def parse_voice_id(voice_id: str) -> tuple[str, str]:
    """
    Parse a voice_id into (archetype, sophistication).

    Format: {archetype}_{sophistication}  (all underscores, no -org suffix)

    Examples:
        "individual_low"               → ("individual", "low")
        "organization_medium"          → ("organization", "medium")
        "advocacy_organization_high"   → ("advocacy_organization", "high")
        "academic_organization_high"   → ("academic_organization", "high")
        "government_organization_high" → ("government_organization", "high")
    """
    # Split on last underscore to extract sophistication suffix
    parts = voice_id.rsplit("_", 1)
    if len(parts) == 2:
        archetype, sophistication = parts
        if sophistication in ("low", "medium", "high"):
            return archetype, sophistication

    # Fallback: treat the whole thing as archetype, default medium
    return voice_id, "medium"


# ── Hook / anchor generation ──────────────────────────────────────────────────
#
# Personas come in two fundamentally different types:
#
#   individual           → personal anecdote hook: a specific lived-experience
#                          micro-narrative that grounds the comment in a real-
#                          feeling human moment.
#
#   all org archetypes   → institutional anchor: a concrete data point, member
#   (advocacy_organization, report, or policy observation that grounds the org's
#    organization,        comment in institutional evidence — NOT a personal
#    government)           story.
#
# Keeping these separate prevents the LLM from defaulting to "Last spring, I…"
# for every persona regardless of type.

# Temporal and narrative clichés the LLM must NEVER use in hooks or anchors.
# Defined here inline to avoid a circular import
# (phrase_check → generator → persona → phrase_check).
# Keep in sync with WATCH_LIST in phrase_check.py.
_HOOK_FORBIDDEN_PHRASES: list[str] = [
    "last spring", "last summer", "last fall", "last winter",
    "last year", "last month", "last week", "a couple of years ago"
    "a few years ago", "a few months ago", "a couple of months ago"
    "recently i", "just recently", "not long ago",
    "i keep coming back to", "coming back to",
    "i genuinely don't understand",
]

_HOOK_SYSTEM = """\
You are a creative writer helping researchers generate synthetic public-comment
personas for regulatory-comment spam detection research.

Given a persona description and a regulatory context, generate ONE specific,
irreproducible personal anecdote — a "hook" — that connects this person's life
to the rule being commented on.

The hook must:
- Be 2–4 sentences
- Refer to a real-feeling specific event, relationship, or circumstance
- Use the persona's voice and sophistication level
- NOT be generic ("I am concerned about healthcare costs")
- NOT reference fictional famous people or impossible events
- NOT mention the rule number or agency name directly — just the real-world impact

FORBIDDEN — do NOT use any of the following phrases (any capitalization).
They are known AI-generation fingerprints that make every hook sound identical:
  "last spring", "last summer", "last fall", "last winter",
  "last year", "last month", "last week", 
  "a few years ago", "a few months ago", "a couple of years ago",
  "recently I", "just recently", "not long ago","a couple of months ago",
  "I keep coming back to", "coming back to"
Ground the hook in a specific event, relationship, or named circumstance —
NOT a vague temporal opener like "Last [season/year/month]..."

Output ONLY the hook text, no labels or prefixes.
"""

_HOOK_USER_TEMPLATE = """\
Persona:
  Name: {name}
  Age: {age}
  State: {state}
  Occupation: {occupation}
  Sophistication: {sophistication}
  Emotional register: {emotional_register}
  {org_line}

Regulatory context (how this rule affects someone like them):
{consequence}

Write the personal hook anecdote.
"""


def _generate_hook(persona: Persona, world_model: WorldModel, config: Config) -> str:
    """Call the LLM to generate a personal hook for this persona."""
    client = config.openai_client()
    org_line = f"Organization: {persona.org_name}" if persona.org_name else "No organizational affiliation"
    consequence = world_model.consequence_for(persona.archetype)
    if not consequence:
        consequence = world_model.core_change

    prompt = _HOOK_USER_TEMPLATE.format(
        name=persona.full_name,
        age=persona.age,
        state=persona.state,
        occupation=persona.occupation,
        sophistication=persona.sophistication,
        emotional_register=persona.emotional_register,
        org_line=org_line,
        consequence=consequence,
    )

    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _HOOK_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
    )
    return (response.choices[0].message.content or "").strip()


async def _generate_hook_async(persona: Persona, world_model: WorldModel, config: Config) -> str:
    """Async version: Call the LLM to generate a personal hook for this persona."""
    client = config.async_openai_client()
    org_line = f"Organization: {persona.org_name}" if persona.org_name else "No organizational affiliation"
    consequence = world_model.consequence_for(persona.archetype)
    if not consequence:
        consequence = world_model.core_change

    prompt = _HOOK_USER_TEMPLATE.format(
        name=persona.full_name,
        age=persona.age,
        state=persona.state,
        occupation=persona.occupation,
        sophistication=persona.sophistication,
        emotional_register=persona.emotional_register,
        org_line=org_line,
        consequence=consequence,
    )

    response = await client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _HOOK_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
    )
    return (response.choices[0].message.content or "").strip()


# ── Institutional anchor generation (org archetypes) ─────────────────────────
#
# Organizations don't have personal anecdotes.  The AMA doesn't say "Last
# spring, I..."  Instead, they cite member surveys, policy observations, and
# institutional data.  This prompt generates that kind of institutional anchor.

_ANCHOR_SYSTEM = """\
You are a professional writer helping researchers generate synthetic public-comment
contexts for regulatory-comment spam detection research.

Given an organizational description and a regulatory context, generate ONE specific
institutional data point or policy observation — an "anchor" — that grounds the
organization's comment in concrete evidence.

The anchor must:
- Be 2–4 sentences
- Reference specific numbers, member reports, studies, or institutional events
  (e.g., "A 2024 survey of our 3,400 member facilities found that 71% reported...",
   "Following our 18-month pilot across 12 academic medical centers...",
   "Member feedback collected in our Q3 2024 policy review consistently cited...")
- Use third-person or "our members / our analysis / our facilities" framing
- Sound like it came from an organization's institutional knowledge — NOT a personal story
- NOT begin with a temporal opener like "Last spring", "Recently", "A few months ago"
- NOT sound like one individual's personal anecdote

FORBIDDEN — do NOT use any of the following phrases (any capitalization).
They are known AI-generation fingerprints:
  "last spring", "last summer", "last fall", "last winter",
  "last year", "last month", "last week",
  "a few years ago", "a few months ago",
  "recently", "just recently", "not long ago"

Output ONLY the anchor text, no labels or prefixes.
"""

_ANCHOR_USER_TEMPLATE = """\
Organization:
  Name: {org_name}
  Type: {archetype}
  Representative role: {occupation}
  Sophistication: {sophistication}

Regulatory context (how this rule affects this organization's constituents):
{consequence}

Write the institutional context anchor.
"""


def _generate_anchor(persona: Persona, world_model: WorldModel, config: Config) -> str:
    """
    Call the LLM to generate an institutional anchor for an organizational persona.

    Used for advocacy_organization, organization, academic_organization, and government_organization archetypes.
    Returns a data-grounded institutional observation rather than a personal anecdote.
    """
    client = config.openai_client()
    consequence = world_model.consequence_for(persona.archetype) or world_model.core_change
    prompt = _ANCHOR_USER_TEMPLATE.format(
        org_name=persona.org_name or f"a {persona.archetype.replace('_', ' ')} organization",
        archetype=persona.archetype.replace("_", " "),
        occupation=persona.occupation,
        sophistication=persona.sophistication,
        consequence=consequence,
    )
    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _ANCHOR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
    )
    return (response.choices[0].message.content or "").strip()


async def _generate_anchor_async(persona: Persona, world_model: WorldModel, config: Config) -> str:
    """
    Async version of _generate_anchor.

    Used for advocacy_organization, organization, academic_organization, and government_organization archetypes.
    """
    client = config.async_openai_client()
    consequence = world_model.consequence_for(persona.archetype) or world_model.core_change
    prompt = _ANCHOR_USER_TEMPLATE.format(
        org_name=persona.org_name or f"a {persona.archetype.replace('_', ' ')} organization",
        archetype=persona.archetype.replace("_", " "),
        occupation=persona.occupation,
        sophistication=persona.sophistication,
        consequence=consequence,
    )
    response = await client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _ANCHOR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
    )
    return (response.choices[0].message.content or "").strip()


# ── Persona instantiation (shared logic) ──────────────────────────────────────

def _build_persona_metadata(
    archetype: str,
    sophistication: str,
    profile: ArchetypeProfile,
    world_model: WorldModel,
    rng: np.random.Generator,
    org_pool: OrgPool | None = None,
) -> dict:
    """
    Build persona metadata fields (name, state, occupation, age, etc.)
    from archetype, sophistication, and the population profile.

    Returns a dict of keyword arguments for Persona().

    Parameters
    ----------
    org_pool : OrgPool, optional
        Pre-generated pool of real (non-docket) org names. When provided,
        org names are sampled from this pool without replacement, ensuring
        no two synthetic personas share an org name and no org name overlaps
        with actual docket submitters. When None, falls back to the legacy
        behavior of sampling from profile.orgs (not recommended for production).
    """
    first = rng.choice(_FIRST_NAMES)
    last = rng.choice(_LAST_NAMES)

    # State — prefer states observed in the historical docket for this archetype
    state_pool = profile.states if profile.states else _STATES
    state = str(rng.choice(state_pool))

    # Occupation
    occ_pool = _OCCUPATIONS_BY_ARCHETYPE.get(archetype, ["citizen"])
    occupation = str(rng.choice(occ_pool))

    # Age
    age_min, age_max = _AGE_RANGES_BY_ARCHETYPE.get(archetype, (30, 70))
    age = int(rng.integers(age_min, age_max + 1))

    # Emotional register
    emotional_register = str(rng.choice(_EMOTIONAL_REGISTERS))

    # Org name — use the pre-generated pool (preferred) or fall back to legacy
    org_name = ""
    if archetype != "individual":
        if org_pool is not None:
            # Sample without replacement from the pre-generated pool of real,
            # non-docket org names. This prevents duplicates and avoids using
            # org names that appear in the real docket comments.
            org_name = org_pool.sample(archetype, rng=rng)
        elif profile.orgs:
            # Legacy fallback: sample from historical docket orgs.
            # WARNING: these are real docket submitters — using them risks
            # creating synthetic comments attributable to real organizations.
            org_name = str(rng.choice(profile.orgs[:50]))

    # Personal stake
    consequence = world_model.consequence_for(archetype)
    personal_stake = consequence[:200] if consequence else world_model.core_change[:200]

    return {
        "archetype": archetype,
        "first_name": first,
        "last_name": last,
        "state": state,
        "occupation": occupation,
        "age": age,
        "sophistication": sophistication,
        "emotional_register": emotional_register,
        "org_name": org_name,
        "personal_stake": personal_stake,
    }


def instantiate_persona(
    archetype: str,
    profile: ArchetypeProfile,
    world_model: WorldModel,
    config: Config,
    rng: np.random.Generator,
    docket_id: str = "",
    voice_id: str = "",
    sophistication_override: str = "",
    org_pool: OrgPool | None = None,
) -> Persona:
    """
    Instantiate a fully-specified Persona, drawing metadata from the historical
    docket profile and generating a hook via LLM.

    Parameters
    ----------
    archetype : str
        Archetype to instantiate
    profile : ArchetypeProfile
        Historical profile for this archetype
    world_model : WorldModel
        World model for rule context
    config : Config
        API configuration
    rng : np.random.Generator
        Random number generator
    docket_id : str, optional
        Docket ID for loading voice skills from stylometry
    voice_id : str, optional
        Specific voice_id to use for voice skill loading
    sophistication_override : str, optional
        Override sophistication level instead of sampling from archetype defaults
    org_pool : OrgPool, optional
        Pre-generated pool of real (non-docket) org names for unique assignment.
    """
    # Determine sophistication
    if sophistication_override:
        sophistication = sophistication_override
    else:
        soph_pool = _SOPHISTICATION_BY_ARCHETYPE.get(archetype, ["medium"])
        sophistication = str(rng.choice(soph_pool))

    # Build metadata
    meta = _build_persona_metadata(archetype, sophistication, profile, world_model, rng, org_pool=org_pool)

    persona = Persona(
        **meta,
        personal_hook="",
        voice_skill="",
        voice_id=voice_id,
    )

    # Load docket-specific voice skill
    if docket_id:
        from stylometry.stylometry_loader import load_voice_skill
        skill = load_voice_skill(docket_id, archetype, sophistication)
        if skill:
            persona.voice_skill = skill

    # Generate hook (individual) or institutional anchor (org archetypes)
    if archetype == "individual":
        persona.personal_hook = _generate_hook(persona, world_model, config)
    else:
        persona.personal_hook = _generate_anchor(persona, world_model, config)

    return persona


# ── Campaign-mode persona sampling (v2.0) ─────────────────────────────────────

def sample_persona_by_voice_id(
    voice_id: str,
    world_model: WorldModel,
    config: Config,
    rng: np.random.Generator,
    docket_id: str = "",
    org_pool: OrgPool | None = None,
) -> Persona:
    """
    Instantiate a Persona for a specific voice_id from the campaign plan.

    This is the primary persona creation function for v2.0 campaign plans.
    The voice_id determines archetype, sophistication, and voice skill directly.

    Parameters
    ----------
    voice_id : str
        Voice group ID from the campaign plan (e.g. "industry_high_org").
    world_model : WorldModel
        The world model (contains population model).
    config : Config
        API config.
    rng : np.random.Generator
        Seeded random number generator.
    docket_id : str
        Docket ID for loading voice skills.
    org_pool : OrgPool, optional
        Pre-generated pool of real (non-docket) org names for unique assignment.
    """
    archetype, sophistication = parse_voice_id(voice_id)
    population = world_model.population

    # Get profile for this archetype (or minimal fallback)
    profile = None
    if population:
        profile = population.archetypes.get(archetype)
    if profile is None:
        profile = ArchetypeProfile(archetype=archetype, count=1)

    return instantiate_persona(
        archetype=archetype,
        profile=profile,
        world_model=world_model,
        config=config,
        rng=rng,
        docket_id=docket_id,
        voice_id=voice_id,
        sophistication_override=sophistication,
        org_pool=org_pool,
    )


async def sample_persona_by_voice_id_async(
    voice_id: str,
    world_model: WorldModel,
    config: Config,
    rng: np.random.Generator,
    docket_id: str = "",
    org_pool: OrgPool | None = None,
) -> Persona:
    """
    Async version of sample_persona_by_voice_id.
    """
    archetype, sophistication = parse_voice_id(voice_id)
    population = world_model.population

    profile = None
    if population:
        profile = population.archetypes.get(archetype)
    if profile is None:
        profile = ArchetypeProfile(archetype=archetype, count=1)

    # Build metadata
    meta = _build_persona_metadata(archetype, sophistication, profile, world_model, rng, org_pool=org_pool)

    persona = Persona(
        **meta,
        personal_hook="",
        voice_skill="",
        voice_id=voice_id,
    )

    # Load docket-specific voice skill
    if docket_id:
        from stylometry.stylometry_loader import load_voice_skill
        skill = load_voice_skill(docket_id, archetype, sophistication)
        if skill:
            persona.voice_skill = skill

    # Generate hook (individual) or institutional anchor (org archetypes)
    if archetype == "individual":
        persona.personal_hook = await _generate_hook_async(persona, world_model, config)
    else:
        persona.personal_hook = await _generate_anchor_async(persona, world_model, config)

    return persona


# ── Direct-mode persona sampling (backward compatible) ────────────────────────

def sample_persona(
    world_model: WorldModel,
    config: Config,
    rng: np.random.Generator,
    archetype_override: str | None = None,
    docket_id: str = "",
    archetype_weights_override: dict[str, float] | None = None,
    org_pool: OrgPool | None = None,
) -> Persona:
    """
    Sample an archetype from the population distribution and instantiate a
    Persona for it.  Used in direct (non-campaign) mode.

    Parameters
    ----------
    world_model:
        The world model (contains the population model).
    config:
        API config.
    rng:
        Seeded random number generator.
    archetype_override:
        If set, use this archetype instead of sampling from the distribution.
    docket_id:
        Docket ID for loading voice skills from stylometry.
    archetype_weights_override:
        Optional dict of archetype → weight. When provided, persona sampling
        uses these weights directly (no blending with historical).
    org_pool : OrgPool, optional
        Pre-generated pool of real (non-docket) org names for unique assignment.
    """
    population = world_model.population
    if population is None:
        raise ValueError("world_model.population must be set before sampling personas.")

    if archetype_override and archetype_override in population.archetypes:
        archetype = archetype_override
    elif archetype_weights_override:
        # Use override weights directly (no 50/50 blend)
        keys = list(archetype_weights_override.keys())
        total = sum(archetype_weights_override.values())
        if total == 0:
            archetype = population.sample_archetype(rng)
        else:
            probs = [archetype_weights_override[k] / total for k in keys]
            idx = rng.choice(len(keys), p=probs)
            archetype = keys[idx]
    else:
        archetype = population.sample_archetype(rng)

    profile = population.archetypes.get(archetype)
    if profile is None:
        profile = ArchetypeProfile(archetype=archetype, count=1)

    return instantiate_persona(archetype, profile, world_model, config, rng, docket_id, org_pool=org_pool)


async def sample_persona_async(
    world_model: WorldModel,
    config: Config,
    rng: np.random.Generator,
    archetype_override: str | None = None,
    docket_id: str = "",
    archetype_weights_override: dict[str, float] | None = None,
    org_pool: OrgPool | None = None,
) -> Persona:
    """
    Async version of sample_persona.  Used in direct (non-campaign) mode.
    """
    population = world_model.population
    if population is None:
        raise ValueError("world_model.population must be set before sampling personas.")

    if archetype_override and archetype_override in population.archetypes:
        archetype = archetype_override
    elif archetype_weights_override:
        keys = list(archetype_weights_override.keys())
        total = sum(archetype_weights_override.values())
        if total == 0:
            archetype = population.sample_archetype(rng)
        else:
            probs = [archetype_weights_override[k] / total for k in keys]
            idx = rng.choice(len(keys), p=probs)
            archetype = keys[idx]
    else:
        archetype = population.sample_archetype(rng)

    profile = population.archetypes.get(archetype)
    if profile is None:
        profile = ArchetypeProfile(archetype=archetype, count=1)

    # Determine sophistication
    soph_pool = _SOPHISTICATION_BY_ARCHETYPE.get(archetype, ["medium"])
    sophistication = str(rng.choice(soph_pool))

    # Build metadata
    meta = _build_persona_metadata(archetype, sophistication, profile, world_model, rng, org_pool=org_pool)

    persona = Persona(
        **meta,
        personal_hook="",
        voice_skill="",
    )

    # Load docket-specific voice skill
    if docket_id:
        from stylometry.stylometry_loader import load_voice_skill
        skill = load_voice_skill(docket_id, archetype, sophistication)
        if skill:
            persona.voice_skill = skill

    # Generate hook (individual) or institutional anchor (org archetypes)
    if archetype == "individual":
        persona.personal_hook = await _generate_hook_async(persona, world_model, config)
    else:
        persona.personal_hook = await _generate_anchor_async(persona, world_model, config)

    return persona
