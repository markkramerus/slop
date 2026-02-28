"""
shared_models.py — Shared data models used by both stylometry and syncom.

This module contains dataclasses that are used by both the stylometry analysis
application and the syncom comment generation application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ArchetypeProfile:
    """Profile for a specific archetype group."""
    archetype: str
    count: int
    # Distributions stored as (mean, std) tuples
    word_count: tuple[float, float] = (200.0, 100.0)
    mean_sentence_len: tuple[float, float] = (15.0, 8.0)
    first_person_ratio: tuple[float, float] = (0.05, 0.03)
    citation_count: tuple[float, float] = (0.0, 1.0)
    bullet_ratio: tuple[float, float] = (0.1, 0.1)
    # Representative comment snippets for style priming
    sample_texts: list[str] = field(default_factory=list)
    # Metadata pools
    states: list[str] = field(default_factory=list)
    orgs: list[str] = field(default_factory=list)

    def sample_word_count(self, rng: np.random.Generator) -> int:
        """Sample a realistic word count from a log-normal distribution."""
        mu, sigma = self.word_count
        # Fit log-normal to (mean, std)
        if sigma <= 0 or mu <= 0:
            return int(mu)
        v = sigma ** 2
        mu_ln = np.log(mu ** 2 / np.sqrt(v + mu ** 2))
        sigma_ln = np.sqrt(np.log(1 + v / mu ** 2))
        return max(30, int(rng.lognormal(mu_ln, sigma_ln)))


@dataclass
class PopulationModel:
    """Model of the comment population from a docket."""
    docket_id: str
    total_comments: int
    archetypes: dict[str, ArchetypeProfile]
    # Raw argument landscape summary (built by LLM in WorldModel stage)
    argument_landscape_raw: list[str] = field(default_factory=list)
    # Timing: fraction of comments in each decile of the comment period
    timing_deciles: list[float] = field(default_factory=lambda: [0.1] * 10)

    def archetype_weights(self) -> dict[str, float]:
        """Calculate the weight (proportion) of each archetype."""
        total = sum(p.count for p in self.archetypes.values())
        if total == 0:
            return {k: 1.0 / len(self.archetypes) for k in self.archetypes}
        return {k: p.count / total for k, p in self.archetypes.items()}

    def sample_archetype(self, rng: np.random.Generator) -> str:
        """Sample an archetype based on the population distribution."""
        weights = self.archetype_weights()
        keys = list(weights.keys())
        probs = [weights[k] for k in keys]
        idx = rng.choice(len(keys), p=probs)
        return keys[idx]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "docket_id": self.docket_id,
            "total_comments": self.total_comments,
            "archetypes": {
                k: {
                    "count": v.count,
                    "word_count_mean": v.word_count[0],
                    "word_count_std": v.word_count[1],
                    "mean_sentence_len_mean": v.mean_sentence_len[0],
                    "first_person_ratio_mean": v.first_person_ratio[0],
                    "citation_count_mean": v.citation_count[0],
                    "states_sample": v.states[:10],
                    "orgs_sample": v.orgs[:10],
                }
                for k, v in self.archetypes.items()
            },
        }
