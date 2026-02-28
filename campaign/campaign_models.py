"""
campaign_models.py — Data models for campaign plans.

A CampaignPlan is the intermediate artifact produced by the campaign planner
and consumed by syncom's pipeline. It can be serialized to JSON, reviewed and
edited by a human, and then loaded back for comment generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


PLAN_VERSION = "1.0"


@dataclass
class ArgumentAngle:
    """
    A specific argument angle that comments in the campaign should advance.

    Attributes
    ----------
    id : str
        Short snake_case identifier (e.g. "patient_safety", "bias_detection").
    angle : str
        One-sentence description of this argument angle.
    weight : float
        Relative weight for assigning this angle to comments (0–1, all weights
        are normalized at runtime). Higher weight = more comments use this angle.
    best_archetypes : list[str]
        Archetype names most naturally suited to make this argument.
    """
    id: str
    angle: str
    weight: float = 0.15
    best_archetypes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "angle": self.angle,
            "weight": self.weight,
            "best_archetypes": self.best_archetypes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArgumentAngle:
        return cls(
            id=d["id"],
            angle=d["angle"],
            weight=float(d.get("weight", 0.15)),
            best_archetypes=d.get("best_archetypes", []),
        )


@dataclass
class CampaignPlan:
    """
    A structured campaign plan that tells syncom how to distribute N synthetic
    comments across argument angles, stakeholder types, and attack vectors.

    Attributes
    ----------
    objective : str
        The refined objective statement (position to advance or oppose).
    scenario_summary : str
        Brief summary of the user's original scenario/brief.
    argument_angles : list[ArgumentAngle]
        Distinct argument angles to distribute across comments.
    stakeholder_emphasis : dict[str, float]
        Archetype → weight mapping that biases persona sampling.
        Keys should match syncom archetypes (individual_consumer, advocacy_group,
        industry, academic, government). Weights are normalized at runtime.
    vector_mix : dict[int, float]
        Attack vector → weight mapping (1–4). Determines what proportion of
        comments use each vector. Weights are normalized at runtime.
    notes : str
        Free-text notes from the planner (human-readable rationale for the plan).
    created : str
        ISO timestamp of plan creation.
    plan_version : str
        Schema version for forward compatibility.
    """
    objective: str
    scenario_summary: str = ""
    argument_angles: list[ArgumentAngle] = field(default_factory=list)
    stakeholder_emphasis: dict[str, float] = field(default_factory=dict)
    vector_mix: dict[int, float] = field(default_factory=dict)
    notes: str = ""
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    plan_version: str = PLAN_VERSION

    # ── Normalization helpers ─────────────────────────────────────────────

    def normalized_angle_weights(self) -> list[float]:
        """Return angle weights normalized to sum to 1.0."""
        if not self.argument_angles:
            return []
        total = sum(a.weight for a in self.argument_angles)
        if total == 0:
            return [1.0 / len(self.argument_angles)] * len(self.argument_angles)
        return [a.weight / total for a in self.argument_angles]

    def normalized_stakeholder_weights(self) -> dict[str, float]:
        """Return stakeholder emphasis weights normalized to sum to 1.0."""
        if not self.stakeholder_emphasis:
            return {}
        total = sum(self.stakeholder_emphasis.values())
        if total == 0:
            n = len(self.stakeholder_emphasis)
            return {k: 1.0 / n for k in self.stakeholder_emphasis}
        return {k: v / total for k, v in self.stakeholder_emphasis.items()}

    def normalized_vector_weights(self) -> dict[int, float]:
        """Return vector mix weights normalized to sum to 1.0."""
        if not self.vector_mix:
            return {1: 0.3, 2: 0.4, 3: 0.15, 4: 0.15}
        total = sum(self.vector_mix.values())
        if total == 0:
            n = len(self.vector_mix)
            return {k: 1.0 / n for k in self.vector_mix}
        return {k: v / total for k, v in self.vector_mix.items()}

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "created": self.created,
            "scenario_summary": self.scenario_summary,
            "objective": self.objective,
            "argument_angles": [a.to_dict() for a in self.argument_angles],
            "stakeholder_emphasis": self.stakeholder_emphasis,
            "vector_mix": {str(k): v for k, v in self.vector_mix.items()},
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str) -> None:
        """Write the campaign plan to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CampaignPlan:
        angles = [ArgumentAngle.from_dict(a) for a in d.get("argument_angles", [])]
        # vector_mix keys come in as strings from JSON; convert to int
        raw_vm = d.get("vector_mix", {})
        vector_mix = {int(k): float(v) for k, v in raw_vm.items()}
        return cls(
            objective=d.get("objective", ""),
            scenario_summary=d.get("scenario_summary", ""),
            argument_angles=angles,
            stakeholder_emphasis=d.get("stakeholder_emphasis", {}),
            vector_mix=vector_mix,
            notes=d.get("notes", ""),
            created=d.get("created", datetime.now(timezone.utc).isoformat()),
            plan_version=d.get("plan_version", PLAN_VERSION),
        )

    @classmethod
    def load(cls, path: str) -> CampaignPlan:
        """Load a campaign plan from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_json(cls, json_str: str) -> CampaignPlan:
        """Parse a campaign plan from a JSON string."""
        d = json.loads(json_str)
        return cls.from_dict(d)

    # ── Display ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of the plan."""
        lines = [
            f"Campaign Plan (v{self.plan_version})",
            f"  Created: {self.created}",
            f"  Objective: {self.objective}",
            f"",
            f"  Argument Angles ({len(self.argument_angles)}):",
        ]
        norm_weights = self.normalized_angle_weights()
        for angle, nw in zip(self.argument_angles, norm_weights):
            archetypes = ", ".join(angle.best_archetypes) if angle.best_archetypes else "any"
            lines.append(f"    [{angle.id}] {nw:.0%} — {angle.angle}")
            lines.append(f"      Best archetypes: {archetypes}")
        
        lines.append(f"")
        lines.append(f"  Stakeholder Emphasis:")
        for arch, w in sorted(self.normalized_stakeholder_weights().items(), key=lambda x: -x[1]):
            lines.append(f"    {arch:25s} {w:.0%}")
        
        lines.append(f"")
        lines.append(f"  Vector Mix:")
        vector_names = {1: "Semantic Variance", 2: "Persona Mimicry", 3: "Citation Flooding", 4: "Dilution/Noise"}
        for v, w in sorted(self.normalized_vector_weights().items()):
            lines.append(f"    {v} ({vector_names.get(v, '?'):20s}) {w:.0%}")
        
        if self.notes:
            lines.append(f"")
            lines.append(f"  Notes: {self.notes}")
        
        return "\n".join(lines)
