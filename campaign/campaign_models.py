"""
campaign_models.py — Data models for campaign plans.

A CampaignPlan is the intermediate artifact produced by the campaign planner
and consumed by syncom's pipeline. It can be serialized to JSON, reviewed and
edited by a human, and then loaded back for comment generation.

Plan Version History
--------------------
v1.0 — Original schema with stakeholder_emphasis, vector_mix, best_archetypes
v2.0 — Bayesian framework: campaign_voices (voice_ids from stylometry),
        affinity_boost, best_voices, no vector_mix.
        P(V,A) = P(V) × P(A|V) where P(A|V) ∝ w(A) × f(A,V)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


PLAN_VERSION = "2.0"


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
        Base rate weight w(A) for this angle (0–1). Used in computing
        P(A|V) ∝ w(A) × f(A,V). All weights are normalized at runtime.
    best_voices : list[str]
        Voice IDs (from stylometry index.json) most naturally suited to make
        this argument. These get the affinity_boost multiplier when computing
        P(A|V). Example: ["advocacy_group-high-org", "academic-high-org"]
    """
    id: str
    angle: str
    weight: float = 0.15
    best_voices: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "angle": self.angle,
            "weight": self.weight,
            "best_voices": self.best_voices,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArgumentAngle:
        # Support both v2.0 (best_voices) and v1.0 (best_archetypes) keys
        best_voices = d.get("best_voices", d.get("best_archetypes", []))
        return cls(
            id=d["id"],
            angle=d["angle"],
            weight=float(d.get("weight", 0.15)),
            best_voices=best_voices,
        )


@dataclass
class CampaignPlan:
    """
    A structured campaign plan that tells syncom how to distribute N synthetic
    comments across voice groups and argument angles.

    The plan implements a Bayesian allocation framework:

        P(V, A) = P(V) × P(A|V)

    where:
        P(V)   = campaign_voices distribution
        P(A|V) ∝ w(A) × f(A,V)
        f(A,V) = affinity_boost if V ∈ best_voices(A), else 1.0

    Attributes
    ----------
    objective : str
        The refined objective statement (position to advance or oppose).
    scenario_summary : str
        Brief summary of the user's original scenario/brief.
    argument_angles : list[ArgumentAngle]
        Distinct argument angles to distribute across comments.
    campaign_voices : dict[str, float]
        Voice ID → weight mapping. Voice IDs correspond to entries in the
        stylometry index.json (e.g. "advocacy_group-high-org",
        "individual_consumer-low"). Weights are normalized at runtime to
        give P(V).
    base_population : dict[str, float]
        Voice ID → proportion in the actual docket (before any campaign
        emphasis). Stored for reference / audit only; not used at runtime.
    affinity_boost : float
        Multiplier α applied when a voice is in an angle's best_voices list.
        Controls how strongly voice identity biases argument selection.
        α=1 → independent; α=3 → moderate preference; α=10 → strong channeling.
    notes : str
        Free-text notes from the planner (human-readable rationale).
    created : str
        ISO timestamp of plan creation.
    plan_version : str
        Schema version for forward compatibility.
    """
    objective: str
    scenario_summary: str = ""
    argument_angles: list[ArgumentAngle] = field(default_factory=list)
    campaign_voices: dict[str, float] = field(default_factory=dict)
    base_population: dict[str, float] = field(default_factory=dict)
    affinity_boost: float = 3.0
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

    def normalized_voice_weights(self) -> dict[str, float]:
        """Return campaign_voices weights normalized to sum to 1.0 → P(V)."""
        if not self.campaign_voices:
            return {}
        total = sum(self.campaign_voices.values())
        if total == 0:
            n = len(self.campaign_voices)
            return {k: 1.0 / n for k in self.campaign_voices}
        return {k: v / total for k, v in self.campaign_voices.items()}

    def compute_angle_distribution(self, voice_id: str) -> list[float]:
        """
        Compute P(A|V=voice_id) for all argument angles.

        P(A|V) ∝ w(A) × f(A,V)
        where f(A,V) = affinity_boost if V ∈ best_voices(A), else 1.0

        Returns a list of probabilities aligned with self.argument_angles.
        """
        if not self.argument_angles:
            return []

        α = self.affinity_boost
        raw = []
        for angle in self.argument_angles:
            w = angle.weight
            # Check if voice_id matches any best_voice (exact match or
            # archetype prefix match, e.g. "industry-high-org" matches "industry")
            affinity = 1.0
            for bv in angle.best_voices:
                if voice_id == bv or voice_id.startswith(bv + "-"):
                    affinity = α
                    break
                # Also allow archetype-level matching: if best_voice is
                # "industry" and voice_id is "industry-high-org"
                if "-" not in bv and voice_id.startswith(bv):
                    affinity = α
                    break
            raw.append(w * affinity)

        total = sum(raw)
        if total == 0:
            n = len(raw)
            return [1.0 / n] * n
        return [r / total for r in raw]

    def compute_allocation_matrix(
        self,
        volume: int,
    ) -> dict[str, dict[str, int]]:
        """
        Compute the expected allocation matrix: voice × argument → count.

        Returns dict[voice_id → dict[angle_id → expected_count]].
        Also returns the derived marginal P(A).
        """
        pv = self.normalized_voice_weights()
        matrix: dict[str, dict[str, int]] = {}

        for voice_id, p_voice in pv.items():
            voice_count = round(p_voice * volume)
            pa_given_v = self.compute_angle_distribution(voice_id)
            row: dict[str, int] = {}
            for angle, prob in zip(self.argument_angles, pa_given_v):
                row[angle.id] = round(voice_count * prob)
            matrix[voice_id] = row

        return matrix

    def marginal_argument_distribution(self) -> dict[str, float]:
        """
        Compute P(A) = Σ_V P(A|V) × P(V) for each argument angle.
        This is the derived (not specified) overall argument distribution.
        """
        pv = self.normalized_voice_weights()
        marginals: dict[str, float] = {a.id: 0.0 for a in self.argument_angles}

        for voice_id, p_voice in pv.items():
            pa_given_v = self.compute_angle_distribution(voice_id)
            for angle, prob in zip(self.argument_angles, pa_given_v):
                marginals[angle.id] += p_voice * prob

        return marginals

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "created": self.created,
            "scenario_summary": self.scenario_summary,
            "objective": self.objective,
            "argument_angles": [a.to_dict() for a in self.argument_angles],
            "campaign_voices": self.campaign_voices,
            "base_population": self.base_population,
            "affinity_boost": self.affinity_boost,
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
        version = d.get("plan_version", "1.0")

        # ── v1.0 → v2.0 migration ────────────────────────────────────────
        if version.startswith("1"):
            print(
                "[campaign] Warning: Loading v1.0 campaign plan. "
                "Migrating to v2.0 format (stakeholder_emphasis → campaign_voices, "
                "vector_mix removed). Review the migrated plan.",
                file=sys.stderr,
            )
            # Convert stakeholder_emphasis to campaign_voices
            # v1.0 used archetype names; v2.0 uses voice_ids.
            # Best-effort: use archetype as a prefix (e.g. "industry" → "industry")
            # The pipeline will match by prefix.
            se = d.get("stakeholder_emphasis", {})
            campaign_voices = {}
            for arch, w in se.items():
                campaign_voices[arch] = w
            base_population = {}
            affinity_boost = 3.0

            return cls(
                objective=d.get("objective", ""),
                scenario_summary=d.get("scenario_summary", ""),
                argument_angles=angles,
                campaign_voices=campaign_voices,
                base_population=base_population,
                affinity_boost=affinity_boost,
                notes=d.get("notes", ""),
                created=d.get("created", datetime.now(timezone.utc).isoformat()),
                plan_version="2.0",
            )

        # ── v2.0 loading ─────────────────────────────────────────────────
        return cls(
            objective=d.get("objective", ""),
            scenario_summary=d.get("scenario_summary", ""),
            argument_angles=angles,
            campaign_voices=d.get("campaign_voices", {}),
            base_population=d.get("base_population", {}),
            affinity_boost=float(d.get("affinity_boost", 3.0)),
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
            f"  Affinity boost (α): {self.affinity_boost}",
            f"",
            f"  Argument Angles ({len(self.argument_angles)}):",
        ]
        norm_weights = self.normalized_angle_weights()
        for angle, nw in zip(self.argument_angles, norm_weights):
            voices = ", ".join(angle.best_voices) if angle.best_voices else "any"
            lines.append(f"    [{angle.id}] w={nw:.0%} — {angle.angle}")
            lines.append(f"      Best voices: {voices}")

        lines.append(f"")
        lines.append(f"  Campaign Voices (P(V)):")
        for voice, w in sorted(
            self.normalized_voice_weights().items(), key=lambda x: -x[1]
        ):
            base = self.base_population.get(voice, 0.0)
            delta = ""
            if base > 0:
                ratio = w / base
                if ratio > 1.1:
                    delta = f" (↑{ratio:.1f}× vs base {base:.1%})"
                elif ratio < 0.9:
                    delta = f" (↓{ratio:.1f}× vs base {base:.1%})"
                else:
                    delta = f" (≈ base {base:.1%})"
            lines.append(f"    {voice:35s} {w:.1%}{delta}")

        # Derived marginal P(A)
        lines.append(f"")
        lines.append(f"  Derived Marginal P(A):")
        marginals = self.marginal_argument_distribution()
        for angle_id, pa in sorted(marginals.items(), key=lambda x: -x[1]):
            lines.append(f"    {angle_id:35s} {pa:.1%}")

        if self.notes:
            lines.append(f"")
            lines.append(f"  Notes: {self.notes}")

        return "\n".join(lines)

    def allocation_summary(self, volume: int) -> str:
        """
        Human-readable allocation matrix showing expected comment counts
        for each voice × argument combination.
        """
        matrix = self.compute_allocation_matrix(volume)
        angle_ids = [a.id for a in self.argument_angles]

        # Truncate angle IDs for display
        short_ids = [aid[:16] for aid in angle_ids]

        # Header
        header = f"{'Voice':35s} | " + " | ".join(f"{s:>16s}" for s in short_ids) + " | Total"
        sep = "-" * len(header)

        lines = [
            f"Expected allocation for {volume} comments:",
            sep,
            header,
            sep,
        ]

        voice_totals: dict[str, int] = {}
        angle_totals: dict[str, int] = {aid: 0 for aid in angle_ids}

        for voice_id in sorted(matrix.keys()):
            row = matrix[voice_id]
            row_total = sum(row.values())
            voice_totals[voice_id] = row_total
            cells = []
            for aid in angle_ids:
                count = row.get(aid, 0)
                angle_totals[aid] += count
                cells.append(f"{count:>16d}")
            lines.append(f"{voice_id:35s} | " + " | ".join(cells) + f" | {row_total:>5d}")

        # Totals row
        lines.append(sep)
        total_cells = [f"{angle_totals[aid]:>16d}" for aid in angle_ids]
        grand_total = sum(voice_totals.values())
        lines.append(f"{'TOTAL':35s} | " + " | ".join(total_cells) + f" | {grand_total:>5d}")

        return "\n".join(lines)
