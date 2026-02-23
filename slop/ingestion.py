"""
ingestion.py — Load and analyze a previous docket CSV into a population model.

The population model captures:
  - Archetype distribution (individual consumer, organization, advocacy group, etc.)
  - Per-archetype linguistic fingerprints (length, sentence length, register, error rate)
  - Argument landscape (positions, evidence types, emotional frames)
  - Persona metadata pool (states, occupations, org types)
  - Submission timing distribution

Expected CSV columns (Regulations.gov export format).  The loader normalises
column names, so minor variations (extra spaces, different capitalisation) are
handled automatically.  The minimum required columns are a comment-text column
and ideally submitter metadata (name, org, state, comment date).

Typical Regulations.gov CSV headers:
  Comment ID, Document ID, Submitter Name, Organization Name,
  Submitter's Representative, Government Agency Type, Government Agency,
  Abstract (comment text), Comment, Attachment Files, Posted Date, ...
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import Config

# ── Column aliases ────────────────────────────────────────────────────────────

_COMMENT_ALIASES = [
    "comment", "abstract", "comment text", "text", "body",
]
_ORG_ALIASES = [
    "organization name", "organization", "org", "company",
]
_NAME_ALIASES = [
    "submitter name", "submitter", "name", "full name",
]
_STATE_ALIASES = [
    "state or province", "state", "province", "state/province",
]
_DATE_ALIASES = [
    "posted date", "date posted", "date", "received date",
]
_TYPE_ALIASES = [
    "submitter's representative", "category", "submitter type", "type",
]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and strip column names for fuzzy matching."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


# ── Archetype classification ──────────────────────────────────────────────────

_ARCHETYPE_KEYWORDS: dict[str, list[str]] = {
    "government": [
        "department", "agency", "bureau", "administration", "office of",
        "county", "city of", "state of", "federal", "municipal",
    ],
    "advocacy_group": [
        "association", "coalition", "alliance", "network", "federation",
        "foundation", "institute", "center for", "advocates", "council",
        "society", "union",
    ],
    "industry": [
        "inc", "llc", "corp", "ltd", "co.", "group", "solutions",
        "systems", "services", "technologies", "hospital", "health system",
        "medical center", "clinic",
    ],
    "academic": [
        "university", "college", "school of", "professor", "phd", "md ",
        "research", "lab",
    ],
}


def classify_archetype(org: str, name: str, comment: str) -> str:
    """Heuristically classify a submitter into one of five archetypes."""
    combined = f"{org} {name}".lower()
    for archetype, keywords in _ARCHETYPE_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return archetype
    # Short comments with no org → individual
    if not org.strip() and len(comment.split()) < 200:
        return "individual_consumer"
    return "individual_consumer"


# ── Linguistic fingerprinting ─────────────────────────────────────────────────

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def fingerprint(text: str) -> dict[str, float]:
    """Compute lightweight linguistic metrics for one comment."""
    words = text.split()
    sentences = _sentences(text) or [""]
    word_lengths = [len(w) for w in words] if words else [0]
    sent_lengths = [len(s.split()) for s in sentences]

    # Simple error proxies
    typo_like = sum(1 for w in words if len(w) > 2 and w == w.lower() and
                    re.search(r'[^a-z\'-]', w)) / max(len(words), 1)
    bullet_ratio = text.count("\n") / max(len(sentences), 1)
    first_person = sum(1 for w in words if w.lower() in
                       {"i", "me", "my", "we", "our", "us"}) / max(len(words), 1)
    citation_like = len(re.findall(r'\b\d{4}\b|\bcfr\b|\busc\b|\bfr\b', text.lower()))

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "mean_sentence_len": float(np.mean(sent_lengths)),
        "std_sentence_len": float(np.std(sent_lengths)),
        "mean_word_len": float(np.mean(word_lengths)),
        "first_person_ratio": first_person,
        "bullet_ratio": bullet_ratio,
        "citation_count": citation_like,
        "typo_proxy": typo_like,
    }


# ── Population model ──────────────────────────────────────────────────────────

@dataclass
class ArchetypeProfile:
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
    docket_id: str
    total_comments: int
    archetypes: dict[str, ArchetypeProfile]
    # Raw argument landscape summary (built by LLM in WorldModel stage)
    argument_landscape_raw: list[str] = field(default_factory=list)
    # Timing: fraction of comments in each decile of the comment period
    timing_deciles: list[float] = field(default_factory=lambda: [0.1] * 10)

    def archetype_weights(self) -> dict[str, float]:
        total = sum(p.count for p in self.archetypes.values())
        if total == 0:
            return {k: 1.0 / len(self.archetypes) for k in self.archetypes}
        return {k: p.count / total for k, p in self.archetypes.items()}

    def sample_archetype(self, rng: np.random.Generator) -> str:
        weights = self.archetype_weights()
        keys = list(weights.keys())
        probs = [weights[k] for k in keys]
        idx = rng.choice(len(keys), p=probs)
        return keys[idx]

    def to_dict(self) -> dict[str, Any]:
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


# ── Main ingestion function ───────────────────────────────────────────────────

def ingest_docket_csv(
    csv_path: str,
    docket_id: str = "",
    max_sample_texts: int = 5,
    config: Config | None = None,
) -> PopulationModel:
    """
    Load a Regulations.gov CSV and build a PopulationModel.

    Parameters
    ----------
    csv_path:
        Path to the docket CSV file.
    docket_id:
        The docket identifier (e.g. "CMS-2025-0050").  Inferred from filename
        if not provided.
    max_sample_texts:
        Maximum number of sample comment texts to store per archetype for
        style priming in the generator.
    config:
        API config (unused at ingestion time, reserved for future LLM-based
        argument extraction).
    """
    df = pd.read_csv(csv_path, dtype=str, low_memory=False).fillna("")
    df = _normalise_columns(df)

    if not docket_id:
        import os
        docket_id = os.path.splitext(os.path.basename(csv_path))[0]

    comment_col = _find_col(df, _COMMENT_ALIASES)
    org_col = _find_col(df, _ORG_ALIASES)
    name_col = _find_col(df, _NAME_ALIASES)
    state_col = _find_col(df, _STATE_ALIASES)
    date_col = _find_col(df, _DATE_ALIASES)

    if comment_col is None:
        raise ValueError(
            f"Could not find a comment-text column in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    archetypes_raw: dict[str, list[dict[str, Any]]] = {
        "individual_consumer": [],
        "advocacy_group": [],
        "industry": [],
        "academic": [],
        "government": [],
    }

    for _, row in df.iterrows():
        comment = str(row.get(comment_col, "")).strip()
        org = str(row.get(org_col, "")).strip() if org_col else ""
        name = str(row.get(name_col, "")).strip() if name_col else ""
        state = str(row.get(state_col, "")).strip() if state_col else ""

        if len(comment) < 10:
            continue

        archetype = classify_archetype(org, name, comment)
        fp = fingerprint(comment)
        archetypes_raw[archetype].append({
            "fingerprint": fp,
            "state": state,
            "org": org,
            "text": comment,
        })

    # Build ArchetypeProfile per archetype
    profiles: dict[str, ArchetypeProfile] = {}
    for archetype, records in archetypes_raw.items():
        if not records:
            continue
        fps = [r["fingerprint"] for r in records]
        word_counts = [fp["word_count"] for fp in fps]
        sent_lens = [fp["mean_sentence_len"] for fp in fps]
        fp_ratios = [fp["first_person_ratio"] for fp in fps]
        cit_counts = [fp["citation_count"] for fp in fps]
        bullet_ratios = [fp["bullet_ratio"] for fp in fps]

        # Pick representative sample texts (prefer mid-length)
        sorted_by_len = sorted(records, key=lambda r: r["fingerprint"]["word_count"])
        mid = len(sorted_by_len) // 2
        sample_slice = sorted_by_len[
            max(0, mid - max_sample_texts // 2):mid + max_sample_texts
        ]
        sample_texts = [r["text"][:800] for r in sample_slice]

        states = [r["state"] for r in records if r["state"]]
        orgs = [r["org"] for r in records if r["org"]]

        profiles[archetype] = ArchetypeProfile(
            archetype=archetype,
            count=len(records),
            word_count=(float(np.mean(word_counts)), float(np.std(word_counts))),
            mean_sentence_len=(float(np.mean(sent_lens)), float(np.std(sent_lens))),
            first_person_ratio=(float(np.mean(fp_ratios)), float(np.std(fp_ratios))),
            citation_count=(float(np.mean(cit_counts)), float(np.std(cit_counts))),
            bullet_ratio=(float(np.mean(bullet_ratios)), float(np.std(bullet_ratios))),
            sample_texts=sample_texts,
            states=states,
            orgs=orgs,
        )

    # Timing distribution (deciles of posted date)
    timing_deciles = [0.1] * 10
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if len(dates) > 1:
            min_d, max_d = dates.min(), dates.max()
            span = (max_d - min_d).total_seconds()
            if span > 0:
                fractions = ((dates - min_d).dt.total_seconds() / span).clip(0, 0.9999)
                decile_counts = np.histogram(fractions, bins=10)[0]
                total = decile_counts.sum()
                timing_deciles = (decile_counts / max(total, 1)).tolist()

    total_comments = sum(p.count for p in profiles.values())

    return PopulationModel(
        docket_id=docket_id,
        total_comments=total_comments,
        archetypes=profiles,
        timing_deciles=timing_deciles,
    )
