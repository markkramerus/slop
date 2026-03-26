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
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
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

    # ── Persistence ───────────────────────────────────────────────────────

    def _serializable_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the LLM-derived fields + rule_text.

        The PopulationModel is *not* included — it is rebuilt cheaply from
        stylometry data each run and re-attached after loading.
        """
        return {
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
            "rule_text": self.rule_text,
        }

    def save(self, path: str | Path) -> None:
        """Save the world model (LLM-derived fields + rule text) to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._serializable_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(
        cls,
        path: str | Path,
        population: PopulationModel | None = None,
    ) -> "WorldModel":
        """Load a world model from a JSON file and optionally attach a population model.

        Parameters
        ----------
        path:
            Path to a ``world_model.json`` file previously written by :meth:`save`.
        population:
            Population model to attach.  If ``None`` the resulting WorldModel
            will have ``population=None``; the caller can set it afterwards.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            rule_title=data.get("rule_title", "Unknown Rule"),
            docket_id=data.get("docket_id", ""),
            agency=data.get("agency", ""),
            regulatory_domain=data.get("regulatory_domain", ""),
            core_change=data.get("core_change", ""),
            stated_rationale=data.get("stated_rationale", ""),
            affected_parties=data.get("affected_parties", []),
            rfi_questions=data.get("rfi_questions", []),
            plausible_consequences=data.get("plausible_consequences", {}),
            key_terms=data.get("key_terms", []),
            controversy_level=data.get("controversy_level", "medium"),
            population=population,
            rule_text=data.get("rule_text", ""),
        )


def _default_cache_path(docket_id: str) -> Path:
    """Return the conventional cache path: ``{docket_id}/world_model.json``."""
    return Path(docket_id, "world_model.json")


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


def build_or_load_world_model(
    rule_text: str,
    population: PopulationModel,
    config: Config,
    docket_id: str = "",
    rebuild: bool = False,
    verbose: bool = True,
) -> WorldModel:
    """Build a new world model via LLM or load a cached one from disk.

    On the first run for a docket the world model is built with an LLM call
    and saved to ``{docket_id}/world_model.json``.  Subsequent runs reuse the
    cached file unless *rebuild* is ``True``.

    Parameters
    ----------
    rule_text:
        Full text of the proposed rule or RFI.
    population:
        Population model built from stylometry data.
    config:
        API configuration (only used when an LLM call is needed).
    docket_id:
        Docket identifier — used to derive the cache path.
    rebuild:
        If ``True``, ignore any cached file and regenerate via LLM.
    verbose:
        Print status messages to stderr.
    """
    cache_path = _default_cache_path(docket_id) if docket_id else None

    # Try loading from cache
    if cache_path and cache_path.is_file() and not rebuild:
        if verbose:
            print(f"      Loading cached world model from {cache_path}", file=sys.stderr)
        wm = WorldModel.load(cache_path, population=population)
        # Re-attach rule_text in case the caller's copy differs only in
        # whitespace or encoding; the cached version is authoritative for
        # the LLM-derived fields.
        if not wm.rule_text:
            wm.rule_text = rule_text
        return wm

    # Build via LLM
    if verbose and cache_path and not rebuild:
        print(f"      No cached world model found — building via LLM…", file=sys.stderr)
    elif verbose and rebuild:
        print(f"      Rebuilding world model via LLM (--rebuild-world-model)…", file=sys.stderr)

    wm = build_world_model(
        rule_text=rule_text,
        population=population,
        config=config,
        docket_id=docket_id,
    )

    # Persist for next time
    if cache_path:
        wm.save(cache_path)
        if verbose:
            print(f"      World model saved to {cache_path}", file=sys.stderr)

    return wm
