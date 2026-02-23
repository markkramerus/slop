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
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Config
from .ingestion import ArchetypeProfile, PopulationModel
from .world_model import WorldModel


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
    "individual_consumer": [
        "retired teacher", "nurse", "retired nurse", "social worker",
        "small business owner", "office manager", "truck driver",
        "stay-at-home parent", "retail worker", "construction worker",
        "veteran", "caregiver", "elementary school teacher", "librarian",
        "firefighter", "administrative assistant", "warehouse worker",
        "home health aide", "customer service representative",
    ],
    "advocacy_group": [
        "executive director", "policy director", "senior policy analyst",
        "director of government affairs", "community organizer",
        "campaign director", "research director",
    ],
    "industry": [
        "compliance officer", "chief medical officer", "VP of regulatory affairs",
        "director of government relations", "healthcare administrator",
        "hospital CFO", "practice manager", "health IT director",
    ],
    "academic": [
        "professor of health policy", "associate professor", "PhD researcher",
        "postdoctoral fellow", "assistant professor", "health economist",
        "biostatistician", "public health researcher",
    ],
    "government": [
        "Medicaid director", "state insurance commissioner",
        "county health officer", "city public health director",
        "state budget analyst",
    ],
}

_AGE_RANGES_BY_ARCHETYPE: dict[str, tuple[int, int]] = {
    "individual_consumer": (35, 78),
    "advocacy_group": (30, 65),
    "industry": (35, 60),
    "academic": (30, 65),
    "government": (35, 60),
}

_SOPHISTICATION_BY_ARCHETYPE: dict[str, list[str]] = {
    "individual_consumer": ["low", "low", "medium"],
    "advocacy_group": ["medium", "high"],
    "industry": ["high", "high", "medium"],
    "academic": ["high", "high"],
    "government": ["medium", "high"],
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

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @property
    def is_individual(self) -> bool:
        return self.archetype == "individual_consumer"

    def style_instructions(self) -> str:
        """
        Return style guidance for the comment generator based on this persona.
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


# ── Hook generation ───────────────────────────────────────────────────────────

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
        max_tokens=200,
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
        max_tokens=200,
    )
    return (response.choices[0].message.content or "").strip()


# ── Persona instantiation ─────────────────────────────────────────────────────

def instantiate_persona(
    archetype: str,
    profile: ArchetypeProfile,
    world_model: WorldModel,
    config: Config,
    rng: np.random.Generator,
) -> Persona:
    """
    Instantiate a fully-specified Persona for `archetype`, drawing metadata
    from the historical docket profile and generating a hook via LLM.
    """
    # Name
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

    # Sophistication — weighted toward archetype defaults
    soph_pool = _SOPHISTICATION_BY_ARCHETYPE.get(archetype, ["medium"])
    sophistication = str(rng.choice(soph_pool))

    # Emotional register
    emotional_register = str(rng.choice(_EMOTIONAL_REGISTERS))

    # Org name — pick from historical docket or leave blank
    org_name = ""
    if archetype != "individual_consumer" and profile.orgs:
        org_name = str(rng.choice(profile.orgs[:50]))

    # Personal stake (short phrase, derived from world model)
    consequence = world_model.consequence_for(archetype)
    personal_stake = consequence[:200] if consequence else world_model.core_change[:200]

    persona = Persona(
        archetype=archetype,
        first_name=first,
        last_name=last,
        state=state,
        occupation=occupation,
        age=age,
        sophistication=sophistication,
        emotional_register=emotional_register,
        org_name=org_name,
        personal_stake=personal_stake,
        personal_hook="",  # filled in below
    )

    # Generate the personal hook via LLM
    persona.personal_hook = _generate_hook(persona, world_model, config)

    return persona


def sample_persona(
    world_model: WorldModel,
    config: Config,
    rng: np.random.Generator,
    archetype_override: str | None = None,
) -> Persona:
    """
    Sample an archetype from the population distribution and instantiate a
    Persona for it.

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
    """
    population = world_model.population
    if population is None:
        raise ValueError("world_model.population must be set before sampling personas.")

    if archetype_override and archetype_override in population.archetypes:
        archetype = archetype_override
    else:
        archetype = population.sample_archetype(rng)

    profile = population.archetypes.get(archetype)
    if profile is None:
        # Fall back to a minimal profile
        profile = ArchetypeProfile(archetype=archetype, count=1)

    return instantiate_persona(archetype, profile, world_model, config, rng)


async def sample_persona_async(
    world_model: WorldModel,
    config: Config,
    rng: np.random.Generator,
    archetype_override: str | None = None,
) -> Persona:
    """
    Async version of sample_persona.
    Sample an archetype from the population distribution and instantiate a
    Persona for it using async API calls.

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
    """
    population = world_model.population
    if population is None:
        raise ValueError("world_model.population must be set before sampling personas.")

    if archetype_override and archetype_override in population.archetypes:
        archetype = archetype_override
    else:
        archetype = population.sample_archetype(rng)

    profile = population.archetypes.get(archetype)
    if profile is None:
        # Fall back to a minimal profile
        profile = ArchetypeProfile(archetype=archetype, count=1)

    # Name
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

    # Sophistication — weighted toward archetype defaults
    soph_pool = _SOPHISTICATION_BY_ARCHETYPE.get(archetype, ["medium"])
    sophistication = str(rng.choice(soph_pool))

    # Emotional register
    emotional_register = str(rng.choice(_EMOTIONAL_REGISTERS))

    # Org name — pick from historical docket or leave blank
    org_name = ""
    if archetype != "individual_consumer" and profile.orgs:
        org_name = str(rng.choice(profile.orgs[:50]))

    # Personal stake (short phrase, derived from world model)
    consequence = world_model.consequence_for(archetype)
    personal_stake = consequence[:200] if consequence else world_model.core_change[:200]

    persona = Persona(
        archetype=archetype,
        first_name=first,
        last_name=last,
        state=state,
        occupation=occupation,
        age=age,
        sophistication=sophistication,
        emotional_register=emotional_register,
        org_name=org_name,
        personal_stake=personal_stake,
        personal_hook="",  # filled in below
    )

    # Generate the personal hook via LLM asynchronously
    persona.personal_hook = await _generate_hook_async(persona, world_model, config)

    return persona
