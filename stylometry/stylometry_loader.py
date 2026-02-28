"""
stylometry_loader.py — Load docket-specific voice skills for comment generation.

This module provides integration between stylometry_analyzer.py and the
syncom comment generator. It loads docket-specific writing style profiles
and builds population models without requiring access to the original CSV.

Key functions:
- build_population_model(): Create PopulationModel from stylometry index.json
- load_voice_skill(): Load voice-specific style instructions
- extract_skill_instructions(): Parse skill markdown for generation prompts
- parse_statistical_profile(): Extract statistical data from skill markdown
- extract_examples(): Extract real comment examples for few-shot prompting
- extract_voice_description(): Extract qualitative voice description sections
- extract_organizations(): Extract organization names from skill files

Usage:
    from stylometry.stylometry_loader import build_population_model, load_voice_skill
    
    # Build population from stylometry (replaces ingestion.py)
    population = build_population_model("CMS-2025-0050-0031")
    
    # Load voice skill for persona
    skill = load_voice_skill(docket_id, archetype, sophistication)
    
    # Extract specific sections
    stats = parse_statistical_profile(skill)
    examples = extract_examples(skill)
    voice_desc = extract_voice_description(skill)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from shared_models import ArchetypeProfile, PopulationModel

logger = logging.getLogger(__name__)


# ── Statistical profile dataclass ─────────────────────────────────────────────

@dataclass
class VoiceStatistics:
    """Parsed statistical profile from a voice skill markdown file."""
    # Length and Structure
    word_count_median: float = 200.0
    word_count_low: float = 50.0
    word_count_high: float = 500.0
    paragraphs_median: float = 3.0
    words_per_sentence: float = 15.0
    words_per_sentence_std: float = 8.0
    letters_per_word: float = 5.0
    # Voice Characteristics
    first_person_pct: float = 2.0
    exclamations_per_100: float = 0.0
    rhetorical_questions_per_100: float = 0.0
    citation_frequency: float = 0.0
    # Structural Patterns
    uses_bullet_points_pct: float = 20.0
    uses_headings_pct: float = 10.0
    uses_formal_structure_pct: float = 30.0
    # Emphasis and Style
    all_caps_pct: float = 1.0
    em_dash_per_100: float = 0.0
    # Quality Indicators
    ai_vocabulary_pct: float = 0.0


# ── Section extraction helpers ────────────────────────────────────────────────

def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (between --- markers) from markdown."""
    lines = text.split("\n")
    in_frontmatter = False
    filtered = []
    for line in lines:
        if line.strip() == "---":
            in_frontmatter = not in_frontmatter
            continue
        if not in_frontmatter:
            filtered.append(line)
    return "\n".join(filtered)


def _extract_section(text: str, heading: str) -> str:
    """Extract content under a specific ## heading until the next ## heading."""
    lines = text.split("\n")
    collecting = False
    result = []
    for line in lines:
        if collecting:
            if line.startswith("## ") and heading.lower() not in line.lower():
                break
            result.append(line)
        elif heading.lower() in line.lower() and line.strip().startswith("#"):
            collecting = True
    return "\n".join(result).strip()


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_statistical_profile(skill_markdown: str) -> VoiceStatistics:
    """
    Parse the Statistical Profile section from a voice skill markdown file.
    
    Extracts numerical values from lines like:
      - **Word count**: 108 words (range: 47-163)
      - **First-person usage**: 2.6% of words
      - **Uses bullet points**: 43% of comments
    
    Parameters
    ----------
    skill_markdown : str
        Full skill markdown content
        
    Returns
    -------
    VoiceStatistics
        Parsed statistical values (defaults used for missing fields)
    """
    stats = VoiceStatistics()
    text = skill_markdown
    
    # Word count: "108 words (range: 47-163)"
    m = re.search(r'\*\*Word count\*\*:\s*([\d,]+)\s*words?\s*\(range:\s*([\d,]+)\s*-\s*([\d,]+)\)', text)
    if m:
        stats.word_count_median = float(m.group(1).replace(",", ""))
        stats.word_count_low = float(m.group(2).replace(",", ""))
        stats.word_count_high = float(m.group(3).replace(",", ""))
    
    # Paragraphs: "3"
    m = re.search(r'\*\*Paragraphs\*\*:\s*(\d+)', text)
    if m:
        stats.paragraphs_median = float(m.group(1))
    
    # Words per sentence: "12.8 ± 10.3"
    m = re.search(r'\*\*Words per sentence\*\*:\s*([\d.]+)\s*±\s*([\d.]+)', text)
    if m:
        stats.words_per_sentence = float(m.group(1))
        stats.words_per_sentence_std = float(m.group(2))
    
    # Letters per word: "4.8"
    m = re.search(r'\*\*Letters per word\*\*:\s*([\d.]+)', text)
    if m:
        stats.letters_per_word = float(m.group(1))
    
    # First-person usage: "2.6%"
    m = re.search(r'\*\*First-person usage\*\*:\s*([\d.]+)%', text)
    if m:
        stats.first_person_pct = float(m.group(1))
    
    # Emotional markers: "0.0 exclamations per 100 words"
    m = re.search(r'\*\*Emotional markers\*\*:\s*([\d.]+)\s*exclamation', text)
    if m:
        stats.exclamations_per_100 = float(m.group(1))
    
    # Rhetorical questions: "0.0 per 100 words"
    m = re.search(r'\*\*Rhetorical questions\*\*:\s*([\d.]+)\s*per', text)
    if m:
        stats.rhetorical_questions_per_100 = float(m.group(1))
    
    # Citation frequency: "0.0 regulatory citations per comment"
    m = re.search(r'\*\*Citation frequency\*\*:\s*([\d.]+)', text)
    if m:
        stats.citation_frequency = float(m.group(1))
    
    # Structural patterns
    m = re.search(r'\*\*Uses bullet points\*\*:\s*(\d+)%', text)
    if m:
        stats.uses_bullet_points_pct = float(m.group(1))
    
    m = re.search(r'\*\*Uses headings\*\*:\s*(\d+)%', text)
    if m:
        stats.uses_headings_pct = float(m.group(1))
    
    m = re.search(r'\*\*Uses formal structure\*\*:\s*(\d+)%', text)
    if m:
        stats.uses_formal_structure_pct = float(m.group(1))
    
    # Emphasis
    m = re.search(r'\*\*ALL CAPS usage\*\*:\s*([\d.]+)%', text)
    if m:
        stats.all_caps_pct = float(m.group(1))
    
    m = re.search(r'\*\*Em dash frequency\*\*:\s*([\d.]+)\s*per', text)
    if m:
        stats.em_dash_per_100 = float(m.group(1))
    
    # AI vocabulary
    m = re.search(r'\*\*AI vocabulary frequency\*\*:\s*([\d.]+)%', text)
    if m:
        stats.ai_vocabulary_pct = float(m.group(1))
    
    return stats


def extract_voice_description(skill_markdown: str) -> str:
    """
    Extract only the qualitative voice description sections from a skill file.
    
    Returns the Voice Description content (tone/formality analysis, opening
    patterns, distinctive features) without statistical tables or examples.
    This is what should be injected into the generation prompt as style guidance.
    
    Parameters
    ----------
    skill_markdown : str
        Full skill markdown content
        
    Returns
    -------
    str
        Qualitative voice description text
    """
    text = _strip_frontmatter(skill_markdown)
    
    # The voice description lives between "## Voice Description" and
    # "## Statistical Profile"
    lines = text.split("\n")
    collecting = False
    result = []
    
    for line in lines:
        # Start collecting at Voice Description
        if "Voice Description" in line and line.strip().startswith("#"):
            collecting = True
            continue
        # Stop at Statistical Profile (or Example Excerpts / Typical Organizations)
        if collecting and line.strip().startswith("## "):
            section_lower = line.lower()
            if any(kw in section_lower for kw in [
                "statistical profile", "example excerpt", "typical organization"
            ]):
                break
        if collecting:
            result.append(line)
    
    return "\n".join(result).strip()


def extract_examples(skill_markdown: str) -> list[str]:
    """
    Extract real comment examples from the Example Excerpts section.
    
    Parameters
    ----------
    skill_markdown : str
        Full skill markdown content
        
    Returns
    -------
    list[str]
        List of example comment texts (cleaned)
    """
    text = _strip_frontmatter(skill_markdown)
    
    # Find the Example Excerpts section
    lines = text.split("\n")
    collecting = False
    examples = []
    current_example = []
    in_quote = False
    
    for line in lines:
        if "Example Excerpts" in line and line.strip().startswith("#"):
            collecting = True
            continue
        if not collecting:
            continue
        # Stop at next major section (## that isn't an example header)
        if line.strip().startswith("## ") and "example" not in line.lower():
            break
        
        # Detect example headers like "### Example 1" 
        if line.strip().startswith("### Example"):
            # Save previous example if any
            if current_example:
                examples.append("\n".join(current_example).strip())
                current_example = []
            in_quote = False
            continue
        
        # Collect blockquote lines (starting with >)
        if line.startswith("> "):
            current_example.append(line[2:])
            in_quote = True
        elif in_quote and line.strip() == "":
            # Empty line might end quote or be paragraph break within it
            current_example.append("")
        elif in_quote and not line.startswith("> ") and line.strip():
            # Non-quote line after quote — end of this example
            in_quote = False
    
    # Don't forget the last example
    if current_example:
        examples.append("\n".join(current_example).strip())
    
    return [e for e in examples if e]


def extract_organizations(skill_markdown: str) -> list[str]:
    """
    Extract organization names from the Typical Organizations section.
    
    Parameters
    ----------
    skill_markdown : str
        Full skill markdown content
        
    Returns
    -------
    list[str]
        List of organization names
    """
    section = _extract_section(skill_markdown, "Typical Organizations")
    if not section:
        return []
    
    orgs = []
    for line in section.split("\n"):
        line = line.strip()
        if line.startswith("- ") and not line.startswith("- **"):
            org = line[2:].strip()
            if org:
                orgs.append(org)
    return orgs


# ── Skill instructions (for generation prompt) ───────────────────────────────

def extract_skill_instructions(skill_markdown: str) -> str:
    """
    Extract actionable style instructions from a skill markdown file.
    
    Returns only the qualitative voice description — the sections about
    tone, opening patterns, and distinctive features. Statistical data and
    examples are consumed through separate channels.
    
    Parameters
    ----------
    skill_markdown : str
        Full skill markdown content
        
    Returns
    -------
    str
        Extracted voice description for use as style guidance in prompts
    """
    return extract_voice_description(skill_markdown)


# ── Voice skill loading ──────────────────────────────────────────────────────

def load_voice_skill(
    docket_id: str,
    archetype: str,
    sophistication: str,
    base_dir: str | None = None,
) -> Optional[str]:
    """
    Load the appropriate voice skill markdown for a persona.
    
    Parameters
    ----------
    docket_id : str
        Docket identifier (e.g., "CMS-2025-0050")
    archetype : str
        Persona archetype (individual_consumer, industry, etc.)
    sophistication : str
        Sophistication level (low, medium, high)
    base_dir : str, optional
        Override base directory. Default looks in {docket_id}/stylometry/
        
    Returns
    -------
    Optional[str]
        Skill markdown content if found, None otherwise
    """
    skill_dir = Path(base_dir) / docket_id if base_dir else Path(docket_id) / "stylometry"
    index_path = skill_dir / "index.json"
    
    if not index_path.exists():
        logger.debug(f"No stylometry skills found for docket {docket_id}")
        return None
    
    try:
        with open(index_path) as f:
            index = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load stylometry index for {docket_id}: {e}")
        return None
    
    # Try to find exact match: archetype + sophistication
    for skill in index["voice_groups"]:
        if (skill["archetype"] == archetype and 
            skill["sophistication"] == sophistication):
            skill_path = skill_dir / skill["filename"]
            
            if skill_path.exists():
                content = skill_path.read_text(encoding="utf-8")
                logger.info(
                    f"Loaded voice skill: {skill['filename']} "
                    f"(n={skill['sample_size']})"
                )
                return content
            else:
                logger.warning(f"Skill file not found: {skill_path}")
    
    # No exact match — try same archetype with different sophistication
    archetype_skills = [
        s for s in index["voice_groups"] 
        if s["archetype"] == archetype
    ]
    
    if archetype_skills:
        best_skill = max(archetype_skills, key=lambda s: s["sample_size"])
        skill_path = skill_dir / best_skill["filename"]
        
        if skill_path.exists():
            content = skill_path.read_text(encoding="utf-8")
            logger.info(
                f"Loaded approximate voice skill: {best_skill['filename']} "
                f"(requested {sophistication}, got {best_skill['sophistication']}, "
                f"n={best_skill['sample_size']})"
            )
            return content
    
    logger.debug(
        f"No matching voice skill for {archetype}/{sophistication} in {docket_id}"
    )
    return None


# ── Population model builder ─────────────────────────────────────────────────

def build_population_model(
    docket_id: str,
    base_dir: str | None = None,
) -> PopulationModel:
    """
    Build a PopulationModel from stylometry index.json (no CSV needed).
    
    Parses the statistical profiles from each voice skill markdown to populate
    ArchetypeProfile distributions (word count, sentence length, etc.) and
    metadata pools (states, organizations).
    
    Parameters
    ----------
    docket_id : str
        Docket identifier (e.g., "CMS-2025-0050")
    base_dir : str, optional
        Override base directory. Default looks in {docket_id}/stylometry/
        
    Returns
    -------
    PopulationModel
        Population model with archetype distributions and profiles
        
    Raises
    ------
    FileNotFoundError
        If stylometry data not found for this docket
    ValueError
        If index.json is invalid or missing required data
    """
    skill_dir = Path(base_dir) / docket_id if base_dir else Path(docket_id) / "stylometry"
    index_path = skill_dir / "index.json"
    
    if not index_path.exists():
        raise FileNotFoundError(
            f"Stylometry data not found for docket {docket_id}. "
            f"Run stylometry_analyzer.py first to generate voice profiles."
        )
    
    try:
        with open(index_path) as f:
            index = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load stylometry index for {docket_id}: {e}")
    
    profiles: dict[str, ArchetypeProfile] = {}
    
    for voice_group in index["voice_groups"]:
        archetype = voice_group["archetype"]
        sample_size = voice_group["sample_size"]
        
        skill_path = skill_dir / voice_group["filename"]
        if not skill_path.exists():
            logger.warning(f"Skill file not found: {skill_path}")
            continue
        
        skill_content = skill_path.read_text(encoding="utf-8")
        
        # Parse statistical profile
        stats = parse_statistical_profile(skill_content)
        
        # Extract organizations
        orgs = extract_organizations(skill_content)
        
        # Estimate std from range: std ≈ (high - low) / 4
        wc_std = max(1.0, (stats.word_count_high - stats.word_count_low) / 4.0)
        
        # Build or merge ArchetypeProfile
        if archetype not in profiles:
            profiles[archetype] = ArchetypeProfile(
                archetype=archetype,
                count=sample_size,
                word_count=(stats.word_count_median, wc_std),
                mean_sentence_len=(stats.words_per_sentence, stats.words_per_sentence_std),
                first_person_ratio=(stats.first_person_pct / 100.0, 0.02),
                citation_count=(stats.citation_frequency, max(1.0, stats.citation_frequency)),
                bullet_ratio=(stats.uses_bullet_points_pct / 100.0, 0.1),
                states=[],
                orgs=orgs,
            )
        else:
            # Merge: add counts and orgs for same archetype
            profiles[archetype].count += sample_size
            profiles[archetype].orgs.extend(orgs)
    
    # Deduplicate orgs
    for profile in profiles.values():
        profile.orgs = list(set(profile.orgs))
    
    total_comments = index.get(
        "total_comments",
        sum(p.count for p in profiles.values()),
    )
    
    logger.info(f"Built population model from stylometry for {docket_id}")
    logger.info(f"  Total comments: {total_comments}")
    logger.info(f"  Archetypes: {', '.join(profiles.keys())}")
    for arch, prof in profiles.items():
        logger.info(
            f"    {arch}: n={prof.count}, "
            f"word_count=({prof.word_count[0]:.0f}±{prof.word_count[1]:.0f}), "
            f"orgs={len(prof.orgs)}"
        )
    
    return PopulationModel(
        docket_id=docket_id,
        total_comments=total_comments,
        archetypes=profiles,
    )


# ── Discovery helpers ─────────────────────────────────────────────────────────

def get_available_dockets(base_dir: str | None = None) -> list[str]:
    """Get list of dockets with available stylometry skills.
    
    Scans for directories containing a stylometry/index.json file.
    If base_dir is provided, looks for {base_dir}/{docket_id}/index.json (legacy).
    Otherwise, looks for {docket_id}/stylometry/index.json in the current directory.
    """
    if base_dir:
        base_path = Path(base_dir)
        if not base_path.exists():
            return []
        return sorted(
            d.name for d in base_path.iterdir()
            if d.is_dir() and (d / "index.json").exists()
        )
    else:
        # Scan current directory for {docket_id}/stylometry/index.json
        cwd = Path(".")
        return sorted(
            d.name for d in cwd.iterdir()
            if d.is_dir() and (d / "stylometry" / "index.json").exists()
        )


def get_voice_groups_for_docket(
    docket_id: str,
    base_dir: str | None = None,
) -> dict[str, list[dict]]:
    """Get all available voice groups for a docket, organized by archetype."""
    skill_dir = Path(base_dir) / docket_id if base_dir else Path(docket_id) / "stylometry"
    index_path = skill_dir / "index.json"
    if not index_path.exists():
        return {}
    with open(index_path) as f:
        index = json.load(f)
    return index.get("archetype_mapping", {})


def load_voice_statistics(
    docket_id: str,
    archetype: str,
    sophistication: str,
    base_dir: str | None = None,
) -> Optional[VoiceStatistics]:
    """
    Load parsed statistical profile for a specific voice.
    
    Convenience function that loads the skill file and parses its statistics
    in one call.
    
    Parameters
    ----------
    docket_id : str
        Docket identifier
    archetype : str
        Persona archetype
    sophistication : str
        Sophistication level
    base_dir : str, optional
        Override base directory. Default looks in {docket_id}/stylometry/
        
    Returns
    -------
    Optional[VoiceStatistics]
        Parsed statistics if skill found, None otherwise
    """
    skill = load_voice_skill(docket_id, archetype, sophistication, base_dir)
    if skill is None:
        return None
    return parse_statistical_profile(skill)


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo_integration():
    """Demonstrate stylometry integration with parsed statistics."""
    print("=== Stylometry Integration Demo ===\n")
    
    dockets = get_available_dockets()
    print(f"Available dockets: {len(dockets)}")
    for d in dockets:
        print(f"  - {d}")
    print()
    
    if not dockets:
        print("No stylometry skills found. Run stylometry_analyzer.py first.")
        return
    
    docket_id = dockets[0]
    
    # Build population model  
    pop = build_population_model(docket_id)
    print(f"\nPopulation model for {docket_id}:")
    print(f"  Total comments: {pop.total_comments}")
    for arch, prof in pop.archetypes.items():
        print(f"  {arch}: n={prof.count}, word_count={prof.word_count}, orgs={len(prof.orgs)}")
    print()
    
    # Load a skill and parse it
    skill = load_voice_skill(docket_id, "individual_consumer", "low")
    if skill:
        stats = parse_statistical_profile(skill)
        print(f"Stats for individual_consumer-low:")
        print(f"  Word count: {stats.word_count_median} (range: {stats.word_count_low}-{stats.word_count_high})")
        print(f"  Words/sentence: {stats.words_per_sentence} ± {stats.words_per_sentence_std}")
        print(f"  First-person: {stats.first_person_pct}%")
        print(f"  Citations: {stats.citation_frequency}")
        print(f"  Bullet points: {stats.uses_bullet_points_pct}%")
        print()
        
        voice_desc = extract_voice_description(skill)
        print(f"Voice description ({len(voice_desc)} chars):")
        print(voice_desc[:300] + "...")
        print()
        
        examples = extract_examples(skill)
        print(f"Examples: {len(examples)}")
        for i, ex in enumerate(examples[:2], 1):
            print(f"  Example {i} ({len(ex)} chars): {ex[:100]}...")
        print()
        
        orgs = extract_organizations(skill)
        print(f"Organizations: {orgs}")


if __name__ == "__main__":
    demo_integration()
