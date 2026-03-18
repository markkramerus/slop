"""
org_pool.py — Pre-generate and cache a pool of real organization names for
synthetic persona assignment.

The pool contains real organizations that plausibly would comment on a given
rule, but that are NOT among the actual docket submitters (to avoid creating
synthetic comments that could be compared against real ones from the same org).

Design
------
- One LLM call per docket generates a large pool of candidate org names,
  organized by archetype.
- The pool is saved to {docket_id}/org_pool.json and reused across runs.
- During persona generation, orgs are sampled WITHOUT replacement so no two
  synthetic personas share the same organization name.
- If the pool runs low (< LOW_WATER_MARK remaining for an archetype), a
  warning is logged.
- If the pool is exhausted for an archetype, a top-up LLM call is made.

Usage
-----
    from syncom.org_pool import load_or_build_org_pool, OrgPool

    org_pool = load_or_build_org_pool(
        world_model=world_model,
        population=population,
        config=config,
        docket_id=docket_id,
        volume_hint=volume,
        rebuild=False,
        verbose=True,
    )

    # In persona generation:
    org_name = org_pool.sample("advocacy_group")
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from config import Config
from shared_models import PopulationModel


# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum orgs to request per archetype (regardless of volume)
_MIN_POOL_SIZE = 50

# Multiplier: request this many times the expected volume per archetype
_POOL_MULTIPLIER = 3

# Warn when remaining pool for an archetype drops below this fraction
_LOW_WATER_FRACTION = 0.2

# Minimum remaining before triggering a top-up
_TOP_UP_THRESHOLD = 10


# ── LLM prompt ────────────────────────────────────────────────────────────────

_POOL_SYSTEM = """\
You are a regulatory policy expert helping researchers generate realistic
synthetic public comments for comment-spam detection research.

Your task is to generate lists of REAL organizations that would plausibly
submit comments to a specific federal regulatory docket, organized by
stakeholder archetype.

Requirements:
- All organizations must be REAL, verifiable entities (can be googled)
- Organizations must be plausibly interested in the rule topic
- Do NOT include any organization from the exclusion list
- Each organization should appear only once across all archetypes
- Vary the types: national associations, regional groups, academic centers,
  companies, coalitions, etc.

Output ONLY valid JSON — no prose, no markdown fences.
"""

_POOL_USER_TEMPLATE = """\
Rule topic: {rule_title}
Agency: {agency}
Regulatory domain: {regulatory_domain}
Core change: {core_change}

Generate {count_per_archetype} real organizations per archetype that would
plausibly comment on this rule. Organize by archetype.

Archetypes to populate:
{archetypes_list}

EXCLUSION LIST — do NOT include any of these (they already submitted real comments):
{exclusion_list}

Return a JSON object with this exact schema:
{{
  "advocacy_group": ["Org Name 1", "Org Name 2", ...],
  "industry": ["Org Name 1", "Org Name 2", ...],
  "academic": ["Org Name 1", "Org Name 2", ...],
  "government": ["Org Name 1", "Org Name 2", ...],
  "individual_consumer": []
}}

Only include archetypes from the list above. Leave individual_consumer as an
empty list (individuals don't have org names). Generate exactly
{count_per_archetype} names per non-individual archetype.
"""

_TOP_UP_USER_TEMPLATE = """\
Rule topic: {rule_title}
Agency: {agency}
Regulatory domain: {regulatory_domain}

Generate {count} additional real organizations for the "{archetype}" archetype
that would plausibly comment on this rule.

EXCLUSION LIST — do NOT include any of these:
{exclusion_list}

Return a JSON array of organization name strings only:
["Org Name 1", "Org Name 2", ...]
"""


# ── OrgPool class ─────────────────────────────────────────────────────────────

@dataclass
class OrgPool:
    """
    A pool of pre-generated organization names, sampled without replacement.

    Attributes
    ----------
    docket_id : str
        The docket this pool was built for.
    pool : dict[str, list[str]]
        Remaining org names per archetype (consumed as sampled).
    used : dict[str, list[str]]
        Already-used org names per archetype (for audit / top-up exclusion).
    exclusion_set : set[str]
        Real docket submitters — never use these.
    """
    docket_id: str
    pool: dict[str, list[str]] = field(default_factory=dict)
    used: dict[str, list[str]] = field(default_factory=dict)
    exclusion_set: set[str] = field(default_factory=set)

    def sample(
        self,
        archetype: str,
        rng: np.random.Generator | None = None,
        config: Config | None = None,
        world_model=None,
        verbose: bool = False,
    ) -> str:
        """
        Sample one org name for the given archetype without replacement.

        If the pool for this archetype is empty, attempts a top-up LLM call
        if config and world_model are provided. Otherwise returns empty string.

        Parameters
        ----------
        archetype : str
            The persona archetype (e.g., "advocacy_group").
        rng : np.random.Generator, optional
            Random generator for shuffling. If None, uses system random.
        config : Config, optional
            API config for top-up LLM calls.
        world_model : WorldModel, optional
            World model for top-up LLM calls.
        verbose : bool
            Print warnings to stderr.
        """
        # Individuals don't have org names
        if archetype == "individual_consumer":
            return ""

        archetype_pool = self.pool.get(archetype, [])

        # Check low-water mark
        if archetype_pool and len(archetype_pool) <= _TOP_UP_THRESHOLD:
            if verbose:
                print(
                    f"[org_pool] WARNING: pool for '{archetype}' is low "
                    f"({len(archetype_pool)} remaining)",
                    file=sys.stderr,
                )

        # Top up if empty
        if not archetype_pool:
            if config is not None and world_model is not None:
                if verbose:
                    print(
                        f"[org_pool] Pool exhausted for '{archetype}', "
                        f"requesting top-up from LLM...",
                        file=sys.stderr,
                    )
                new_orgs = _top_up_pool(
                    archetype=archetype,
                    count=_MIN_POOL_SIZE,
                    world_model=world_model,
                    config=config,
                    exclusion_set=self._full_exclusion_set(archetype),
                )
                self.pool[archetype] = new_orgs
                archetype_pool = new_orgs
                if verbose:
                    print(
                        f"[org_pool] Top-up added {len(new_orgs)} orgs for '{archetype}'",
                        file=sys.stderr,
                    )
            else:
                if verbose:
                    print(
                        f"[org_pool] WARNING: pool exhausted for '{archetype}' "
                        f"and no config/world_model for top-up. Returning empty string.",
                        file=sys.stderr,
                    )
                return ""

        # Sample one (random pop)
        if rng is not None:
            idx = int(rng.integers(0, len(archetype_pool)))
        else:
            import random
            idx = random.randrange(len(archetype_pool))

        org_name = archetype_pool.pop(idx)

        # Track as used
        if archetype not in self.used:
            self.used[archetype] = []
        self.used[archetype].append(org_name)

        return org_name

    def _full_exclusion_set(self, archetype: str) -> set[str]:
        """Return the union of real docket orgs and already-used synthetic orgs."""
        used_flat = set()
        for orgs in self.used.values():
            used_flat.update(orgs)
        return self.exclusion_set | used_flat

    def remaining(self, archetype: str) -> int:
        """Return the number of remaining org names for an archetype."""
        return len(self.pool.get(archetype, []))

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict (pool + used + exclusion_set)."""
        return {
            "docket_id": self.docket_id,
            "pool": self.pool,
            "used": self.used,
            "exclusion_set": sorted(self.exclusion_set),
        }

    def save(self, path: str | Path) -> None:
        """Save the pool to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "OrgPool":
        """Load a pool from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            docket_id=data.get("docket_id", ""),
            pool=data.get("pool", {}),
            used=data.get("used", {}),
            exclusion_set=set(data.get("exclusion_set", [])),
        )


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _build_exclusion_list(population: PopulationModel) -> set[str]:
    """Collect all real org names from the population model."""
    exclusion: set[str] = set()
    for profile in population.archetypes.values():
        exclusion.update(profile.orgs)
    return exclusion


_POOL_VARIABLE_USER_TEMPLATE = """\
Rule topic: {rule_title}
Agency: {agency}
Regulatory domain: {regulatory_domain}
Core change: {core_change}

Generate real organizations that would plausibly comment on this rule.
Each archetype needs a DIFFERENT number of organizations as specified below.

Archetypes and counts needed:
{archetypes_with_counts}

EXCLUSION LIST — do NOT include any of these (they already submitted real comments):
{exclusion_list}

Return a JSON object where each key is an archetype name and the value is an
array of organization name strings. Generate EXACTLY the number of names
specified for each archetype.

Example schema:
{{
  "advocacy_group": ["Org Name 1", "Org Name 2", ...],
  "industry": ["Org Name 1", ...],
  "academic": ["Org Name 1", ...]
}}

Each organization should appear only once across all archetypes.
All organizations must be REAL, verifiable entities.
"""


def _call_pool_llm(
    world_model,
    archetypes: list[str],
    count_per_archetype: int,
    exclusion_set: set[str],
    config: Config,
) -> dict[str, list[str]]:
    """
    Call the LLM to generate a pool of org names (flat count per archetype).
    Used as a fallback when no per-archetype sizing is available.

    Returns a dict mapping archetype → list of org names.
    """
    per_archetype_sizes = {a: count_per_archetype for a in archetypes}
    return _call_pool_llm_variable(
        world_model=world_model,
        archetype_sizes=per_archetype_sizes,
        exclusion_set=exclusion_set,
        config=config,
    )


def _call_pool_llm_variable(
    world_model,
    archetype_sizes: dict[str, int],
    exclusion_set: set[str],
    config: Config,
) -> dict[str, list[str]]:
    """
    Call the LLM to generate a pool of org names with per-archetype sizing.

    Parameters
    ----------
    archetype_sizes : dict[str, int]
        Mapping of archetype → number of org names to generate.
    exclusion_set : set[str]
        Org names to exclude (real docket submitters + already used).
    config : Config
        API configuration.

    Returns a dict mapping archetype → list of org names.
    """
    client = config.openai_client()

    exclusion_list = "\n".join(f"- {org}" for org in sorted(exclusion_set)) or "(none)"
    archetypes_with_counts = "\n".join(
        f"- {a}: {cnt} organizations" for a, cnt in archetype_sizes.items()
    )

    prompt = _POOL_VARIABLE_USER_TEMPLATE.format(
        rule_title=world_model.rule_title,
        agency=world_model.agency,
        regulatory_domain=world_model.regulatory_domain,
        core_change=world_model.core_change[:500],
        archetypes_with_counts=archetypes_with_counts,
        exclusion_list=exclusion_list,
    )

    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _POOL_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=20000,
    )

    raw = (response.choices[0].message.content or "{}").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}

    # Ensure all archetypes are present and values are lists of strings
    result: dict[str, list[str]] = {}
    for archetype in archetype_sizes:
        orgs = parsed.get(archetype, [])
        if isinstance(orgs, list):
            result[archetype] = [str(o) for o in orgs if o]
        else:
            result[archetype] = []

    return result


def _top_up_pool(
    archetype: str,
    count: int,
    world_model,
    config: Config,
    exclusion_set: set[str],
) -> list[str]:
    """
    Call the LLM to generate additional org names for a single archetype.
    Used when the pool is exhausted.
    """
    client = config.openai_client()

    exclusion_list = "\n".join(f"- {org}" for org in sorted(exclusion_set)) or "(none)"

    prompt = _TOP_UP_USER_TEMPLATE.format(
        rule_title=world_model.rule_title,
        agency=world_model.agency,
        regulatory_domain=world_model.regulatory_domain,
        archetype=archetype,
        count=count,
        exclusion_list=exclusion_list,
    )

    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": _POOL_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=20000,
    )

    raw = (response.choices[0].message.content or "[]").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(o) for o in parsed if o]
    except json.JSONDecodeError:
        pass

    return []


# ── Public API ────────────────────────────────────────────────────────────────

def _default_pool_path(docket_id: str) -> Path:
    """Return the conventional cache path: {docket_id}/org_pool.json"""
    return Path(docket_id, "org_pool.json")


def build_org_pool(
    world_model,
    population: PopulationModel,
    config: Config,
    docket_id: str = "",
    volume_hint: int = 50,
    archetype_counts: dict[str, int] | None = None,
    verbose: bool = True,
) -> OrgPool:
    """
    Build a new org pool via LLM call.

    Parameters
    ----------
    world_model : WorldModel
        The world model (provides rule context for the LLM prompt).
    population : PopulationModel
        The population model (provides real org names to exclude).
    config : Config
        API configuration.
    docket_id : str
        Docket identifier (used for cache path).
    volume_hint : int
        Expected total volume of comments to generate. Used to size the pool
        when archetype_counts is not provided.
    archetype_counts : dict[str, int], optional
        Expected number of comments per archetype (from the campaign voice
        allocation). When provided, each archetype's pool is sized to
        ``count * _POOL_MULTIPLIER`` rather than a flat ``volume_hint``-based
        size. Archetypes with a count of 0 are skipped entirely.
    verbose : bool
        Print status messages to stderr.

    Returns
    -------
    OrgPool
        A freshly built org pool, saved to disk.
    """
    exclusion_set = _build_exclusion_list(population)

    # Determine which archetypes need org names (not individual_consumer)
    if archetype_counts is not None:
        # Only include archetypes that will actually be used (count > 0)
        org_archetypes = [
            a for a, cnt in archetype_counts.items()
            if a != "individual_consumer" and cnt > 0
        ]
        # Per-archetype pool sizes: count * multiplier (minimum 1 to avoid empty requests)
        per_archetype_sizes = {
            a: max(1, archetype_counts[a] * _POOL_MULTIPLIER)
            for a in org_archetypes
        }
    else:
        # Fallback: flat sizing based on total volume
        org_archetypes = [
            a for a in population.archetypes.keys()
            if a != "individual_consumer"
        ]
        # Always include standard archetypes even if not in this docket's population
        for standard in ["advocacy_group", "industry", "academic", "government"]:
            if standard not in org_archetypes:
                org_archetypes.append(standard)
        flat_count = max(_MIN_POOL_SIZE, volume_hint * _POOL_MULTIPLIER)
        per_archetype_sizes = {a: flat_count for a in org_archetypes}

    if not org_archetypes:
        if verbose:
            print(
                f"[org_pool] No org archetypes to populate (all counts are 0 or individual_consumer).",
                file=sys.stderr,
            )
        org_pool = OrgPool(docket_id=docket_id, pool={}, used={}, exclusion_set=exclusion_set)
        if docket_id:
            cache_path = _default_pool_path(docket_id)
            org_pool.save(cache_path)
        return org_pool

    if verbose:
        total_orgs = sum(per_archetype_sizes.values())
        print(
            f"[org_pool] Building org pool for {docket_id}: "
            f"{total_orgs} total orgs across {len(org_archetypes)} archetypes "
            f"(excluding {len(exclusion_set)} real docket orgs)",
            file=sys.stderr,
        )
        for a in org_archetypes:
            print(
                f"[org_pool]   {a}: {per_archetype_sizes[a]} orgs requested",
                file=sys.stderr,
            )

    pool_data = _call_pool_llm_variable(
        world_model=world_model,
        archetype_sizes=per_archetype_sizes,
        exclusion_set=exclusion_set,
        config=config,
    )

    # Log what we got
    if verbose:
        for archetype, orgs in pool_data.items():
            print(
                f"[org_pool]   {archetype}: {len(orgs)} orgs generated",
                file=sys.stderr,
            )

    org_pool = OrgPool(
        docket_id=docket_id,
        pool=pool_data,
        used={},
        exclusion_set=exclusion_set,
    )

    # Save to disk
    if docket_id:
        cache_path = _default_pool_path(docket_id)
        org_pool.save(cache_path)
        if verbose:
            print(f"[org_pool] Saved to {cache_path}", file=sys.stderr)

    return org_pool


def load_or_build_org_pool(
    world_model,
    population: PopulationModel,
    config: Config,
    docket_id: str = "",
    volume_hint: int = 50,
    archetype_counts: dict[str, int] | None = None,
    rebuild: bool = False,
    verbose: bool = True,
) -> OrgPool:
    """
    Load a cached org pool or build a new one via LLM.

    On the first run for a docket the pool is built with an LLM call and
    saved to ``{docket_id}/org_pool.json``. Subsequent runs reuse the cached
    file unless *rebuild* is ``True``.

    Parameters
    ----------
    world_model : WorldModel
        The world model (provides rule context).
    population : PopulationModel
        The population model (provides real org names to exclude).
    config : Config
        API configuration (only used when an LLM call is needed).
    docket_id : str
        Docket identifier — used to derive the cache path.
    volume_hint : int
        Expected total volume. Used to size the pool if building fresh and
        archetype_counts is not provided.
    archetype_counts : dict[str, int], optional
        Expected number of comments per archetype. When provided, each
        archetype's pool is sized to ``count * _POOL_MULTIPLIER``. Archetypes
        with count 0 are skipped. Passed through to build_org_pool().
    rebuild : bool
        If True, ignore any cached file and regenerate via LLM.
    verbose : bool
        Print status messages to stderr.

    Returns
    -------
    OrgPool
        A ready-to-use org pool.
    """
    cache_path = _default_pool_path(docket_id) if docket_id else None

    # Try loading from cache
    if cache_path and cache_path.is_file() and not rebuild:
        if verbose:
            print(
                f"[org_pool] Loading cached org pool from {cache_path}",
                file=sys.stderr,
            )
        pool = OrgPool.load(cache_path)

        # Check if the pool has enough orgs remaining for the archetypes we need
        if archetype_counts is not None:
            # Check per-archetype sufficiency
            needs_rebuild = False
            for archetype, needed in archetype_counts.items():
                if archetype == "individual_consumer" or needed == 0:
                    continue
                remaining = pool.remaining(archetype)
                if remaining < needed:
                    if verbose:
                        print(
                            f"[org_pool] Cached pool has only {remaining} orgs "
                            f"for '{archetype}' (need {needed}). Rebuilding...",
                            file=sys.stderr,
                        )
                    needs_rebuild = True
                    break
            if needs_rebuild:
                return build_org_pool(
                    world_model=world_model,
                    population=population,
                    config=config,
                    docket_id=docket_id,
                    volume_hint=volume_hint,
                    archetype_counts=archetype_counts,
                    verbose=verbose,
                )
        else:
            # Fallback: check total remaining
            total_remaining = sum(len(v) for v in pool.pool.values())
            if total_remaining < volume_hint:
                if verbose:
                    print(
                        f"[org_pool] Cached pool has only {total_remaining} orgs "
                        f"remaining (need ~{volume_hint}). Rebuilding...",
                        file=sys.stderr,
                    )
                return build_org_pool(
                    world_model=world_model,
                    population=population,
                    config=config,
                    docket_id=docket_id,
                    volume_hint=volume_hint,
                    verbose=verbose,
                )

        return pool

    # Build via LLM
    if verbose and cache_path and not rebuild:
        print(
            f"[org_pool] No cached org pool found — building via LLM...",
            file=sys.stderr,
        )
    elif verbose and rebuild:
        print(
            f"[org_pool] Rebuilding org pool via LLM (--rebuild-org-pool)...",
            file=sys.stderr,
        )

    return build_org_pool(
        world_model=world_model,
        population=population,
        config=config,
        docket_id=docket_id,
        volume_hint=volume_hint,
        archetype_counts=archetype_counts,
        verbose=verbose,
    )
