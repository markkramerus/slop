"""
world_model.py — Build a structured world model from a population model and the
text of a new proposed rule.

The world model answers four questions that the generator needs:
  1. What is this rule about? (topic, affected parties, agency rationale)
  2. What specific questions does the RFI/NPRM ask commenters?
  3. What are the plausible real-world consequences for different stakeholder types?
  4. What argument landscape exists (from the historical docket)?

The world model is built via a single LLM call that analyses the rule text and
summarises it into a structured JSON object.  The historical docket's population
model is attached as-is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from config import Config
from shared_models import PopulationModel


_WORLD_MODEL_SYSTEM = """\
You are a regulatory policy analyst. Your job is to read a proposed rule or
Request for Information (RFI) and produce a structured JSON summary that will
be used to generate realistic public comments for research purposes.

Output ONLY valid JSON — no prose, no markdown fences, no explanation.
"""

_WORLD_MODEL_USER_TEMPLATE = """\
Analyse the following proposed rule text and produce a JSON object with this
exact schema (all fields required):

{{
  "rule_title": "<short descriptive title>",
  "docket_id": "<docket number if mentioned, else empty string>",
  "agency": "<issuing agency name>",
  "regulatory_domain": "<e.g. healthcare, environment, finance, telecommunications>",
  "core_change": "<one-sentence description of the primary regulatory change>",
  "stated_rationale": "<one or two sentences on why the agency says it is doing this>",
  "affected_parties": [
    "<type of entity affected, e.g. Medicare beneficiaries>",
    "..."
  ],
  "rfi_questions": [
    "<specific question the rule invites public comment on>",
    "..."
  ],
  "plausible_consequences": {{
    "individual_consumer": "<likely direct impact on ordinary people>",
    "advocacy_group": "<how advocacy orgs would likely frame impact>",
    "industry": "<likely industry concerns>",
    "academic": "<policy/research angles>",
    "government": "<intergovernmental or implementation concerns>"
  }},
  "key_terms": ["<important technical or policy term>", "..."],
  "controversy_level": "<low | medium | high>"
}}

Proposed rule text:
---
{rule_text}
---
"""


@dataclass
class WorldModel:
    """Structured understanding of the proposed rule and historical docket."""

    # Rule analysis
    rule_title: str = ""
    docket_id: str = ""
    agency: str = ""
    regulatory_domain: str = ""
    core_change: str = ""
    stated_rationale: str = ""
    affected_parties: list[str] = field(default_factory=list)
    rfi_questions: list[str] = field(default_factory=list)
    plausible_consequences: dict[str, str] = field(default_factory=dict)
    key_terms: list[str] = field(default_factory=list)
    controversy_level: str = "medium"

    # Historical docket population model
    population: PopulationModel | None = None

    # Raw rule text (kept for generator prompts)
    rule_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "rule_title": self.rule_title,
            "docket_id": self.docket_id,
            "agency": self.agency,
            "regulatory_domain": self.regulatory_domain,
            "core_change": self.core_change,
            "stated_rationale": self.stated_rationale,
            "affected_parties": self.affected_parties,
            "rfi_questions": self.rfi_questions,
            "plausible_consequences": self.plausible_consequences,
            "key_terms": self.key_terms,
            "controversy_level": self.controversy_level,
        }
        if self.population:
            d["population_summary"] = self.population.to_dict()
        return d

    def consequence_for(self, archetype: str) -> str:
        """Return the plausible consequence description for a given archetype."""
        return self.plausible_consequences.get(
            archetype,
            self.plausible_consequences.get("individual_consumer", "")
        )

    def random_rfi_question(self, rng) -> str | None:
        """Return a randomly selected RFI question (may be None)."""
        if not self.rfi_questions:
            return None
        idx = rng.integers(0, len(self.rfi_questions))
        return self.rfi_questions[int(idx)]


def build_world_model(
    rule_text: str,
    population: PopulationModel,
    config: Config,
    docket_id: str = "",
) -> WorldModel:
    """
    Analyse `rule_text` with an LLM and merge the result with the historical
    `population` model to produce a WorldModel.

    Parameters
    ----------
    rule_text:
        Full text of the proposed rule or RFI.
    population:
        Population model built from a previous (or same-topic) docket CSV.
    config:
        API configuration.
    docket_id:
        Optional override for the docket ID.
    """
    config.validate()
    client = config.openai_client()

    prompt = _WORLD_MODEL_USER_TEMPLATE.format(rule_text=rule_text[:12000])

    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _WORLD_MODEL_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,  # Low temperature for structured analysis
        max_tokens=1200,
    )

    raw = response.choices[0].message.content or "{}"

    # Strip potential markdown fences if the model added them anyway
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fall back to an empty world model — generation will still work but
        # with less grounding
        parsed = {}

    wm = WorldModel(
        rule_title=parsed.get("rule_title", "Unknown Rule"),
        docket_id=docket_id or parsed.get("docket_id", ""),
        agency=parsed.get("agency", ""),
        regulatory_domain=parsed.get("regulatory_domain", ""),
        core_change=parsed.get("core_change", ""),
        stated_rationale=parsed.get("stated_rationale", ""),
        affected_parties=parsed.get("affected_parties", []),
        rfi_questions=parsed.get("rfi_questions", []),
        plausible_consequences=parsed.get("plausible_consequences", {}),
        key_terms=parsed.get("key_terms", []),
        controversy_level=parsed.get("controversy_level", "medium"),
        population=population,
        rule_text=rule_text,
    )
    return wm
