"""
stylometry_analyzer.py — Analyze writing styles in docket comments and generate skill files.

This tool extracts stylometric patterns from real docket comments and generates
reusable "voice skill" markdown files that can be used to generate more realistic
synthetic comments.

Instead of clustering, we group comments by explicit CSV properties:
  - Archetype (individual_consumer, advocacy_group, industry, academic, government)
  - Sophistication level (low, medium, high) - computed from fingerprints
  - Organization presence (has org vs individual)

Each unique combination gets its own skill file with:
  - Statistical profile (word count, sentence length, paragraph structure)
  - Vocabulary patterns (terminology, first-person usage, emotional language)
  - Structural patterns (bullets, headings, citations)
  - Error patterns (typos, grammar)
  - AI-avoidance rules
  - Example excerpts

Usage:
  python stylometry_analyzer.py CMS-2025-0050-0031.csv
  python stylometry_analyzer.py CMS-2025-0050-0031.csv --output-dir stylometry/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to find config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from stylometry.stylometry_utils import (
    classify_archetype,
    find_col,
    fingerprint,
    get_attachment_text,
    normalise_columns
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants
MAX_SAMPLE_WORDS = 3000  # Maximum words to send to LLM for analysis


# ── Comment Cleaning ──────────────────────────────────────────────────────────

def clean_comment_for_example(text: str) -> str:
    """
    Clean boilerplate from comment text to get substantive content.
    
    Removes:
    - Formal headers (Re:, Ref:, Dear, address blocks)
    - Meta-references ("See attached", "Please refer to")
    - Signature blocks
    - Excessive whitespace
    """
    lines = text.split('\n')
    cleaned_lines = []
    skip_mode = False
    
    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        
        # Skip formal headers
        if any(line_stripped.startswith(prefix) for prefix in ['Re:', 'Ref:', 'Dear ', 'Attn:', 'Attention:']):
            continue
        
        # Skip address blocks (lines with "PO Box", zip codes, etc.)
        if re.search(r'\b\d{5}(?:-\d{4})?\b', line_stripped):  # zip code
            continue
        if 'po box' in line_lower or 'p.o. box' in line_lower:
            continue
            
        # Skip meta-references
        if any(phrase in line_lower for phrase in [
            'see attached', 'please see attached', 'refer to attached',
            'please refer to', 'attached file', 'attached comment'
        ]):
            continue
        
        # Skip document IDs like "CMS-0042-NC"
        if re.match(r'^[A-Z]{2,5}-\d{4}-[A-Z0-9]+$', line_stripped):
            continue
        
        # Skip signature blocks (lines that look like names/titles)
        if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+,?\s*$', line_stripped):  # "John Smith"
            skip_mode = True
            continue
        if skip_mode and (line_stripped == '' or re.match(r'^[A-Z][a-z\s,]+$', line_stripped)):
            continue
        else:
            skip_mode = False
        
        # Keep substantial lines
        if len(line_stripped) > 3:
            cleaned_lines.append(line_stripped)
    
    # Reassemble and clean whitespace
    cleaned = '\n'.join(cleaned_lines)
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Max 2 newlines
    cleaned = cleaned.strip()
    
    return cleaned


# ── LLM Qualitative Analysis ──────────────────────────────────────────────────

def analyze_voice_with_llm(comments: list[str], voice_id: str, config: Config) -> str:
    """
    Use LLM to generate qualitative analysis of writing style.
    
    Parameters
    ----------
    comments : list[str]
        Sample of comments from this voice group (already cleaned)
    voice_id : str
        Voice group identifier (e.g., "industry-high-org")
    config : Config
        API configuration
    
    Returns
    -------
    str
        LLM-generated voice description
    """
    # Prepare sample text (limit to MAX_SAMPLE_WORDS)
    combined_text = "\n\n---\n\n".join(comments)
    words = combined_text.split()
    if len(words) > MAX_SAMPLE_WORDS:
        combined_text = " ".join(words[:MAX_SAMPLE_WORDS]) + "\n\n[... truncated ...]"
    
    # Build prompt
    prompt = f"""Analyze these regulatory comment samples from the voice group "{voice_id}" and provide a detailed description of their writing style.

Focus on:
1. **Tone and Formality**: Is it personal/emotional, professional/measured, or somewhere in between?
2. **Opening Patterns**: How do these comments typically begin? What hooks or framing do they use?
3. **Argument Structure**: How do they build their case? (narrative/anecdotal, data-driven, logical reasoning, appeals to values)
4. **Language Characteristics**: Vocabulary level, sentence complexity, use of jargon or plain language
5. **Distinctive Features**: What makes this voice unique? Any recurring phrases, structural patterns, or stylistic quirks?

Provide specific examples from the text where possible.

CRITICAL: Your analysis must also include concrete guidance on how to write in this voice WITHOUT sounding like AI.
Real humans write with personality and imperfection. The voice description you write will be used to instruct
an LLM to generate synthetic comments, so explicitly flag what this voice AVOIDS and HOW it sounds human.
Include specific notes on:
- Whether this voice uses simple sentence structure vs. complex/elaborate phrasing
- Whether this voice uses hedging, uncertainty, or mixed feelings (human) vs. confident sweeping claims (AI)
- Whether this voice uses first-person naturally and directly
- What makes this voice feel grounded and specific vs. generic and abstract
- Any patterns in this voice that could accidentally trigger AI-sounding writing

SAMPLE COMMENTS:
{combined_text}

Provide your analysis in 3-4 paragraphs that would help a writer mimic this style while keeping it sounding human, not AI-generated."""

    try:
        client = config.openai_client()
        response = client.chat.completions.create(
            model=config.chat_model,
            messages=[
                {"role": "system", "content": "You are an expert in analyzing writing styles and linguistic patterns in regulatory comments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=1000
        )
        
        analysis = response.choices[0].message.content.strip()
        return analysis
        
    except Exception as e:
        logger.warning(f"LLM analysis failed for {voice_id}: {e}")
        return "AI analysis unavailable - using statistical metrics only."


# ── Voice Group Classification ────────────────────────────────────────────────

@dataclass
class CommentRecord:
    """A single comment with its metadata and fingerprint."""
    text: str
    archetype: str
    organization: str
    author_name: str
    state: str
    fingerprint: dict[str, float]
    document_id: str = ""  # Source document ID
    sophistication: str = ""  # Computed from fingerprint
    
    def compute_sophistication(self) -> str:
        """Determine sophistication level from fingerprint metrics."""
        fp = self.fingerprint
        word_count = fp.get("word_count", 0)
        citation_count = fp.get("citation_count", 0)
        mean_word_len = fp.get("mean_word_len", 0)
        bullet_ratio = fp.get("bullet_ratio", 0)
        
        # High sophistication: long, cites regulations, structured
        if word_count > 400 and (citation_count > 2 or bullet_ratio > 0.3):
            return "high"
        # Low sophistication: short, simple words, no structure
        elif word_count < 200 and mean_word_len < 5 and citation_count == 0:
            return "low"
        else:
            return "medium"


def classify_voice_group(record: CommentRecord) -> str:
    """
    Determine voice group ID based on explicit properties.
    
    Format: {archetype}-{sophistication}[-org]
    Examples: "individual_consumer-low", "industry-high-org"
    """
    has_org = bool(record.organization.strip())
    
    parts = [record.archetype, record.sophistication]
    if has_org and record.archetype != "individual_consumer":
        parts.append("org")
    
    return "-".join(parts)


# ── Enhanced Stylometric Analysis ─────────────────────────────────────────────

def analyze_punctuation(text: str) -> dict[str, float]:
    """Analyze punctuation patterns."""
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return {}
    
    return {
        "em_dash_freq": text.count("—") / max(word_count, 1) * 100,
        "ellipsis_freq": text.count("...") / max(word_count, 1) * 100,
        "exclamation_freq": text.count("!") / max(word_count, 1) * 100,
        "question_freq": text.count("?") / max(word_count, 1) * 100,
        "semicolon_freq": text.count(";") / max(word_count, 1) * 100,
        "colon_freq": text.count(":") / max(word_count, 1) * 100,
    }


def analyze_structure(text: str) -> dict[str, Any]:
    """Analyze document structure."""
    lines = text.split("\n")
    non_empty_lines = [l for l in lines if l.strip()]
    
    # Detect bullet points
    bullet_patterns = [r"^\s*[\-\*•]\s", r"^\s*\d+[\.\)]\s", r"^\s*[a-zA-Z][\.\)]\s"]
    bullet_count = sum(1 for line in lines if any(re.match(p, line) for p in bullet_patterns))
    
    # Detect headings (all caps lines, or lines ending with colon)
    heading_count = sum(1 for line in non_empty_lines 
                       if (line.isupper() and len(line.split()) < 10) or 
                          (line.strip().endswith(":") and len(line.split()) < 10))
    
    # Paragraph count (empty line separated)
    paragraph_count = len(re.split(r'\n\s*\n', text.strip())) if text.strip() else 0
    
    return {
        "paragraph_count": paragraph_count,
        "bullet_count": bullet_count,
        "heading_count": heading_count,
        "has_formal_structure": bullet_count > 0 or heading_count > 0,
    }


def detect_ai_vocabulary(text: str) -> dict[str, Any]:
    """Detect AI-vocabulary words from humanizer-skill.md."""
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    word_count = len(words)
    
    # AI vocabulary from humanizer-skill.md
    ai_vocab = [
        "additionally", "align", "crucial", "delve", "emphasizing", "enduring",
        "enhance", "fostering", "garner", "highlight", "intricate", "pivotal",
        "showcase", "tapestry", "testament", "underscore", "valuable", "vibrant",
        "landscape", "evolving", "nestled", "boasts", "renowned", "breathtaking",
    ]
    
    ai_phrase_patterns = [
        r"\bserves as\b", r"\bstands as\b", r"\bmarks a\b",
        r"\bnot just\b.*\bbut\b", r"\bnot only\b.*\bbut also\b",
        r"\bI hope this helps\b", r"\blet me know\b",
    ]
    
    ai_word_count = sum(1 for w in words if w in ai_vocab)
    ai_phrase_count = sum(1 for pattern in ai_phrase_patterns if re.search(pattern, text_lower))
    
    return {
        "ai_vocab_freq": ai_word_count / max(word_count, 1) * 100,
        "ai_phrase_count": ai_phrase_count,
        "contains_ai_markers": ai_word_count > 2 or ai_phrase_count > 0,
    }


def analyze_errors(text: str) -> dict[str, Any]:
    """Analyze error patterns (simple heuristics)."""
    words = text.split()
    
    # Common typos
    typo_patterns = [
        (r'\bloose\b', r'\blose\b'),  # loose vs lose
        (r"\byour\b.*\b(welcome|wrong|right)\b", "incorrect your/you're"),
    ]
    
    # Capitalization issues (word after period not capitalized)
    cap_errors = len(re.findall(r'\.\s+[a-z]', text))
    
    # Multiple spaces
    multi_space = len(re.findall(r'\s{2,}', text))
    
    return {
        "apparent_typos": sum(1 for pattern, _ in typo_patterns if re.search(pattern, text, re.IGNORECASE)),
        "capitalization_errors": cap_errors,
        "formatting_issues": multi_space,
    }


def analyze_emphasis(text: str) -> dict[str, Any]:
    """Analyze emphasis patterns (caps, bold, italics)."""
    words = text.split()
    word_count = len(words)
    
    if word_count == 0:
        return {"all_caps_freq": 0.0, "all_caps_words": 0}
    
    # Count words in ALL CAPS (excluding single letters and common acronyms)
    all_caps_words = sum(1 for w in words 
                        if len(w) > 1 and w.isupper() and w.isalpha() 
                        and w not in ["US", "USA", "FDA", "CMS", "CDC", "IT", "AI"])
    
    return {
        "all_caps_freq": all_caps_words / word_count * 100,
        "all_caps_words": all_caps_words,
    }


def full_stylometric_profile(text: str) -> dict[str, Any]:
    """Compute complete stylometric profile."""
    base = fingerprint(text)
    punct = analyze_punctuation(text)
    struct = analyze_structure(text)
    ai = detect_ai_vocabulary(text)
    errors = analyze_errors(text)
    emphasis = analyze_emphasis(text)
    
    return {
        **base,
        **punct,
        **struct,
        **ai,
        **errors,
        **emphasis,
    }


# ── Voice Group Aggregation ───────────────────────────────────────────────────

@dataclass
class VoiceGroup:
    """Aggregated statistics for a voice group."""
    voice_id: str
    archetype: str
    sophistication: str
    has_org: bool
    
    comments: list[CommentRecord] = field(default_factory=list)
    
    # Aggregated statistics
    stats: dict[str, Any] = field(default_factory=dict)
    
    def add_comment(self, record: CommentRecord):
        """Add a comment to this voice group."""
        self.comments.append(record)
    
    def compute_statistics(self):
        """Compute aggregated statistics from all comments."""
        if not self.comments:
            return
        
        # Collect all fingerprints
        all_profiles = [full_stylometric_profile(c.text) for c in self.comments]
        
        # Helper to safely get median
        def safe_median(values):
            filtered = [v for v in values if v is not None and not np.isnan(v)]
            return float(np.median(filtered)) if filtered else 0.0
        
        # Helper to safely get percentile
        def safe_percentile(values, pct):
            filtered = [v for v in values if v is not None and not np.isnan(v)]
            return float(np.percentile(filtered, pct)) if filtered else 0.0
        
        # Core statistics (medians + ranges)
        self.stats = {
            # Length and structure
            "median_word_count": safe_median([p.get("word_count", 0) for p in all_profiles]),
            "word_count_p10": safe_percentile([p.get("word_count", 0) for p in all_profiles], 10),
            "word_count_p90": safe_percentile([p.get("word_count", 0) for p in all_profiles], 90),
            "median_paragraph_count": safe_median([p.get("paragraph_count", 0) for p in all_profiles]),
            "median_words_per_sentence": safe_median([p.get("mean_sentence_len", 0) for p in all_profiles]),
            "std_sentence_len": safe_median([p.get("std_sentence_len", 0) for p in all_profiles]),
            "median_letters_per_word": safe_median([p.get("mean_word_len", 0) for p in all_profiles]),
            
            # Voice characteristics
            "median_first_person_ratio": safe_median([p.get("first_person_ratio", 0) for p in all_profiles]),
            "median_bullet_ratio": safe_median([p.get("bullet_ratio", 0) for p in all_profiles]),
            "median_citation_count": safe_median([p.get("citation_count", 0) for p in all_profiles]),
            
            # Punctuation and emphasis
            "median_exclamation_freq": safe_median([p.get("exclamation_freq", 0) for p in all_profiles]),
            "median_question_freq": safe_median([p.get("question_freq", 0) for p in all_profiles]),
            "median_em_dash_freq": safe_median([p.get("em_dash_freq", 0) for p in all_profiles]),
            "median_all_caps_freq": safe_median([p.get("all_caps_freq", 0) for p in all_profiles]),
            
            # Structural patterns (frequencies)
            "pct_uses_bullets": sum(1 for p in all_profiles if p.get("bullet_count", 0) > 0) / len(all_profiles) * 100,
            "pct_uses_headings": sum(1 for p in all_profiles if p.get("heading_count", 0) > 0) / len(all_profiles) * 100,
            "pct_has_citations": sum(1 for p in all_profiles if p.get("citation_count", 0) > 0) / len(all_profiles) * 100,
            "pct_has_formal_structure": sum(1 for p in all_profiles if p.get("has_formal_structure", False)) / len(all_profiles) * 100,
            
            # Quality indicators
            "median_ai_vocab_freq": safe_median([p.get("ai_vocab_freq", 0) for p in all_profiles]),
            "pct_with_ai_markers": sum(1 for p in all_profiles if p.get("contains_ai_markers", False)) / len(all_profiles) * 100,
        }
        
        # Extract metadata (states, orgs)
        self._extract_metadata()
        
        # Select cleaned examples (diverse, no boilerplate)
        import random
        random.seed(42)  # Consistent sampling
        
        # Clean all comments and filter out very short ones
        cleaned_comments = []
        for comment in self.comments:
            cleaned = clean_comment_for_example(comment.text)
            if len(cleaned.split()) >= 50:  # At least 50 words of substance
                cleaned_comments.append((comment, cleaned))
        
        # Sample for examples and LLM analysis
        if cleaned_comments:
            sample_size = min(10, len(cleaned_comments))
            sampled = random.sample(cleaned_comments, sample_size)
            
            # Store cleaned examples (3-5 for display) with document IDs
            example_count = min(5, len(sampled))
            self.stats["examples"] = [
                {"text": clean, "document_id": comment.document_id} 
                for comment, clean in sampled[:example_count]
            ]
            
            # Store all sampled cleaned text for LLM analysis
            self.stats["llm_samples"] = [clean for _, clean in sampled]
        else:
            # Fallback: use raw text if cleaning removes too much
            sample_size = min(10, len(self.comments))
            sampled = random.sample(self.comments, sample_size)
            self.stats["examples"] = [
                {"text": c.text[:800], "document_id": c.document_id} 
                for c in sampled[:5]
            ]
            self.stats["llm_samples"] = [c.text[:1000] for c in sampled]
    
    def _extract_metadata(self):
        """Extract metadata specific to this voice group (states, orgs, etc.)."""
        from collections import Counter
        
        # State distribution for this voice group
        states = [c.state for c in self.comments if c.state.strip()]
        state_counter = Counter(states)
        # Store as dict with counts for this voice group
        self.stats["state_distribution"] = dict(state_counter.most_common(20))
        
        if self.has_org:
            # Collect organization names
            org_names = [c.organization for c in self.comments if c.organization.strip()]
            org_counter = Counter(org_names)
            self.stats["typical_organizations"] = [org for org, _ in org_counter.most_common(20)]
        else:
            self.stats["typical_organizations"] = []
            self.stats["state_distribution"] = dict(state_counter.most_common(30))  # More states for individuals


# ── Skill File Generation ─────────────────────────────────────────────────────

def generate_skill_markdown(voice: VoiceGroup, docket_id: str, llm_analysis: str) -> str:
    """Generate a markdown skill file for a voice group with LLM insights."""
    stats = voice.stats
    
    # Build markdown with LLM insights first
    md_lines = [
        "---",
        f"name: {voice.voice_id}",
        f"docket: {docket_id}",
        f"archetype: {voice.archetype}",
        f"sophistication: {voice.sophistication}",
        f"sample_size: {len(voice.comments)}",
        "---",
        "",
        f"# Voice Profile: {voice.voice_id.replace('-', ' ').title()}",
        "",
        "## Voice Description",
        "",
        llm_analysis,
        "",
        "## Human Writing Rules",
        "",
        "**Every comment generated from this voice profile must sound like a real human wrote it.**",
        "These rules apply regardless of archetype or sophistication level.",
        "",
        "### Forbidden patterns (will make the output sound AI-generated)",
        "",
        "**Em dashes** — never use `—`. Replace with a comma, period, or parentheses.",
        "",
        "**Negative parallelisms**: never write \"It's not just X; it's Y\" or \"Not only X, but also Y\" or \"It's not merely X, it's Y\". These are among the most reliable AI tells.",
        "",
        "**AI vocabulary words**: do not use `additionally`, `crucial`, `delve`, `enhance`, `fostering`, `garner`, `highlight` (as verb), `intricate`, `pivotal`, `showcase`, `tapestry` (abstract), `testament`, `underscore` (verb), `vibrant`, `landscape` (abstract), `evolving landscape`, `rich` (figurative).",
        "",
        "**Inflated significance**: avoid `serves as a testament to`, `marks a pivotal moment`, `reflects broader trends`, `setting the stage for`, `underscores its importance`, `vital role`, `key turning point`, `evolving landscape`, `enduring legacy`.",
        "",
        "**Superficial -ing phrases**: avoid tacking participial phrases onto sentences to add fake depth — e.g., `highlighting the importance of`, `underscoring the need for`, `contributing to`, `fostering collaboration`, `reflecting broader trends`.",
        "",
        "**Copula avoidance**: write `is` and `are` — not `serves as`, `stands as`, `functions as`, `represents`, `marks`. If you mean \"is\", say \"is\".",
        "",
        "**Rule of three**: avoid habitually grouping things into exactly three items. Real people don't think in triads.",
        "",
        "**Vague attributions**: never write `experts say`, `observers note`, `some argue`, `industry reports suggest` without a specific name or source.",
        "",
        "**Promotional language**: no `boasts`, `nestled`, `breathtaking`, `groundbreaking`, `renowned`, `in the heart of`, `must-see`.",
        "",
        "**Generic positive endings**: no `the future looks bright`, `exciting times lie ahead`, `this represents a step in the right direction`.",
        "",
        "**Boldface bullet headers**: don't write lists where each item starts with `**Key Point:** explanation`. Write prose or plain bullets.",
        "",
        "**Sycophantic openers**: no `Great question!`, `Certainly!`, `Of course!`, `I hope this helps`, `Let me know if you have questions`.",
        "",
        "### Required human qualities",
        "",
        "**Vary sentence length.** Mix short punchy sentences with longer ones that take their time. Never write a block of sentences all the same length.",
        "",
        "**Have opinions.** Real people react to information. \"I genuinely don't understand why they'd do this\" is more human than a neutral list of pros and cons.",
        "",
        "**Show uncertainty and mixed feelings.** \"I'm not sure this is the right approach, but...\" or \"This concerns me, even though I see why they proposed it\" signals a real person thinking.",
        "",
        "**Use first person naturally.** `I keep coming back to...`, `What gets me is...`, `My experience has been...` — these read as human.",
        "",
        "**Be specific.** Real people cite actual numbers, dates, places, names. Generic claims (`many people are affected`) are AI. Specific claims (`my clinic saw 40 patients last month who couldn't afford this`) are human.",
        "",
        "**Let some mess in.** A tangent, a repeated point, a slightly incomplete thought — these are human. Perfect logical structure is a red flag.",
        "",
        "**Use simple constructions.** `is`, `are`, `has`, `does` — not elaborate substitutes.",
        "",
        "## Statistical Profile (Medians)",
        "",
        "### Length and Structure",
        f"- **Word count**: {stats.get('median_word_count', 0):.0f} words (range: {stats.get('word_count_p10', 0):.0f}-{stats.get('word_count_p90', 0):.0f})",
        f"- **Paragraphs**: {stats.get('median_paragraph_count', 0):.0f}",
        f"- **Words per sentence**: {stats.get('median_words_per_sentence', 0):.1f} ± {stats.get('std_sentence_len', 0):.1f}",
        f"- **Letters per word**: {stats.get('median_letters_per_word', 0):.1f}",
        "",
        "### Voice Characteristics",
        f"- **First-person usage**: {stats.get('median_first_person_ratio', 0)*100:.1f}% of words (I, me, my, we, our, us)",
        f"- **Emotional markers**: {stats.get('median_exclamation_freq', 0):.1f} exclamations per 100 words",
        f"- **Rhetorical questions**: {stats.get('median_question_freq', 0):.1f} per 100 words",
        f"- **Citation frequency**: {stats.get('median_citation_count', 0):.1f} regulatory citations per comment",
        "",
        "### Structural Patterns",
        f"- **Uses bullet points**: {stats.get('pct_uses_bullets', 0):.0f}% of comments",
        f"- **Uses headings**: {stats.get('pct_uses_headings', 0):.0f}% of comments",
        f"- **Uses formal structure**: {stats.get('pct_has_formal_structure', 0):.0f}% of comments",
        "",
        "### Emphasis and Style",
        f"- **ALL CAPS usage**: {stats.get('median_all_caps_freq', 0):.2f}% of words",
        f"- **Em dash frequency**: {stats.get('median_em_dash_freq', 0):.2f} per 100 words",
        "",
        "### Quality Indicators",
        f"- **AI vocabulary frequency**: {stats.get('median_ai_vocab_freq', 0):.1f}% (lower is more human-like)",
        f"- **Contains AI markers**: {stats.get('pct_with_ai_markers', 0):.0f}% of comments",
        "",
    ]
    
    # Add state distribution
    state_dist = stats.get("state_distribution", {})
    if state_dist:
        md_lines.append("## Geographic Distribution")
        md_lines.append("")
        md_lines.append("Top states represented in this voice group:")
        md_lines.append("")
        for state, count in sorted(state_dist.items(), key=lambda x: -x[1])[:10]:
            pct = count / len(voice.comments) * 100
            md_lines.append(f"- **{state}**: {count} comments ({pct:.1f}%)")
        md_lines.append("")
    
    # Add typical organizations
    if voice.has_org:
        orgs = stats.get("typical_organizations", [])
        if orgs:
            md_lines.append("## Typical Organizations")
            md_lines.append("")
            md_lines.append("Representative organizations in this voice group:")
            md_lines.append("")
            for org in orgs[:10]:
                md_lines.append(f"- {org}")
            md_lines.append("")
    
    # Add cleaned examples
    md_lines.append("## Example Excerpts")
    md_lines.append("")
    md_lines.append("Real examples from this voice group (cleaned):")
    md_lines.append("")
    
    examples = stats.get("examples", [])
    for i, example_data in enumerate(examples, 1):
        # Handle both dict format (with document_id) and legacy string format
        if isinstance(example_data, dict):
            example = example_data["text"]
            doc_id = example_data.get("document_id", "unknown")
        else:
            example = example_data  # Fallback for old format
            doc_id = "unknown"
        
        # Limit length for readability
        if len(example) > 3000:
            example = example[:2997] + "..."
        
        md_lines.append(f"### Example {i} (Source: {doc_id})")
        md_lines.append(f"> {example}")
        md_lines.append("")
    
    return "\n".join(md_lines)


# ── Main Analysis Function ────────────────────────────────────────────────────

def analyze_docket_stylometry(
    csv_path: str,
    output_dir: str | None = None,
    attachments_dir: str | None = None,
    min_group_size: int = 5,
) -> dict[str, Any]:
    """
    Analyze a docket CSV and generate voice skill files.
    
    Parameters
    ----------
    csv_path : str
        Path to docket CSV file (e.g., CMS-2025-0050/comments/CMS-2025-0050.csv)
    output_dir : str, optional
        Output directory for skill files. Defaults to {docket_id}/stylometry/
    attachments_dir : str, optional
        Directory containing downloaded attachments. Defaults to
        {docket_id}/comment_attachments/
    min_group_size : int
        Minimum number of comments required to generate a skill file
        
    Returns
    -------
    dict
        Analysis results and metadata
    """
    # Resolve attachments_dir — will be finalized after docket_id_base is known
    if attachments_dir:
        attachments_path = Path(attachments_dir)
        if not attachments_path.is_absolute():
            csv_parent = Path(csv_path).resolve().parent
            attachments_path = csv_parent / attachments_dir
            if not attachments_path.exists():
                attachments_path = Path.cwd() / attachments_dir
        attachments_dir = str(attachments_path)
    # If attachments_dir is None, we'll set it below once we know the docket_id_base
    
    # Extract docket ID from filename
    # Handle both "CMS-2025-0050.csv" and "CMS-2025-0050-0031.csv" formats
    docket_id_full = Path(csv_path).stem
    
    # Extract base docket ID for attachment lookup (e.g., "CMS-2025-0050" from "CMS-2025-0050-0031")
    # Most dockets follow pattern: AGENCY-YEAR-NUMBER or AGENCY-YEAR-NUMBER-DOCID
    parts = docket_id_full.split('-')
    if len(parts) >= 3:
        # Use first 3 parts as base docket ID for downloads directory
        docket_id_base = '-'.join(parts[:3])
    else:
        docket_id_base = docket_id_full
    
    # Set default attachments_dir now that we know docket_id_base
    if not attachments_dir:
        attachments_dir = str(Path(docket_id_base) / "comment_attachments")
    logger.info(f"Using attachments directory: {attachments_dir}")
    
    logger.info(f"Analyzing docket: {docket_id_full}")
    logger.info(f"Base docket ID for attachments: {docket_id_base}")
    logger.info(f"Loading CSV: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path, dtype=str, low_memory=False).fillna("")
    df = normalise_columns(df)

    # Columns to extract
    comment_col = find_col(df, "comment")
    org_col = find_col(df, "organization name")
    first_name_col = find_col(df, "first name")
    last_name_col = find_col(df, "last name")
    state_col = find_col(df, "state/province")
    doc_id_col = find_col(df, "document id")
    attachment_col = find_col(df, "attachment files")
    category = find_col(df, "category")
    
    if not comment_col:
        raise ValueError(f"Could not find comment column in {csv_path}")
    
    # Track attachment extraction stats
    attachment_stats = {"attempted": 0, "extracted": 0, "failed": 0}
    
    # Process comments into records
    logger.info("Processing comments...")
    records: list[CommentRecord] = []
    
    for _, row in df.iterrows():
        comment = str(row.get(comment_col, "")).strip()
        org = str(row.get(org_col, "")).strip() if org_col else ""
        first_name = str(row.get(first_name_col, "")).strip() if first_name_col else ""
        last_name = str(row.get(last_name_col, "")).strip() if last_name_col else ""
        state = str(row.get(state_col, "")).strip() if state_col else ""
        document_id = str(row.get(doc_id_col, "")).strip() if doc_id_col else ""
        name = f"{first_name} {last_name}".strip()
        
        # Extract text from attachments if available
        attachment_text = ""
        if attachments_dir and doc_id_col and attachment_col:
            attachment_urls = str(row.get(attachment_col, "")).strip()
            
            # Check if this row has attachments
            if document_id and attachment_urls:
                attachment_stats["attempted"] += 1
                # Use attachments_dir for attachment lookup
                attachment_text = get_attachment_text(document_id, docket_id_base, attachments_dir)
                if attachment_text:
                    attachment_stats["extracted"] += 1
                else:
                    attachment_stats["failed"] += 1
        
        # Merge CSV comment with attachment text. If attachment text is under 10 words, ignore the attachment
        if attachment_text and len(attachment_text.split()) > 10:
            csv_word_count = len(comment.split())
            if csv_word_count < 10:
                # CSV comment is short (probably something like "see attached"), then use attachment text
                comment = attachment_text
            else:
                # use both
                comment = f"{comment}\n\n{attachment_text}"
            
        # Limit total size (keep first 10,000 words)
        words = comment.split()
        if len(words) > 10000:
            comment = " ".join(words[:10000]) + "\n\n[... truncated ...]"
        elif len(words) < 10:  # Skip very short comments
            continue
        
        archetype = classify_archetype(org, name, category)
        fp = fingerprint(comment)
        
        record = CommentRecord(
            text=comment,
            archetype=archetype,
            organization=org,
            author_name=name,
            state=state,
            fingerprint=fp,
            document_id=document_id,
        )
        record.sophistication = record.compute_sophistication()
        records.append(record)
    
    logger.info(f"Processed {len(records)} valid comments")
    
    # Log attachment extraction stats
    if attachment_stats["attempted"] > 0:
        logger.info(
            f"Attachment extraction: {attachment_stats['extracted']}/{attachment_stats['attempted']} "
            f"successful, {attachment_stats['failed']} failed"
        )
    
    # Group by voice
    voice_groups: dict[str, VoiceGroup] = {}
    
    for record in records:
        voice_id = classify_voice_group(record)
        
        if voice_id not in voice_groups:
            has_org = "-org" in voice_id
            voice_groups[voice_id] = VoiceGroup(
                voice_id=voice_id,
                archetype=record.archetype,
                sophistication=record.sophistication,
                has_org=has_org,
            )
        
        voice_groups[voice_id].add_comment(record)
    
    # Filter out small groups
    voice_groups = {k: v for k, v in voice_groups.items() if len(v.comments) >= min_group_size}
    
    logger.info(f"Identified {len(voice_groups)} voice groups (min size: {min_group_size})")
    
    # Compute statistics for each group
    for voice_id, voice in voice_groups.items():
        logger.info(f"  - {voice_id}: {len(voice.comments)} comments")
        voice.compute_statistics()
    
    # Create output directory — default is {docket_id_base}/stylometry/
    if output_dir is None:
        output_path = Path(docket_id_base) / "stylometry"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")
    
    # Initialize LLM config
    logger.info("Initializing LLM for voice analysis...")
    config = Config()
    try:
        config.validate()
    except ValueError as e:
        logger.warning(f"LLM configuration invalid: {e}")
        logger.warning("Will generate skills without LLM analysis")
        config = None
    
    # Generate skill files with LLM analysis
    skill_files = []
    for voice_id, voice in voice_groups.items():
        filename = f"{voice_id}.md"
        filepath = output_path / filename
        
        # Get LLM analysis
        if config and "llm_samples" in voice.stats:
            logger.info(f"  Analyzing {voice_id} with LLM...")
            llm_analysis = analyze_voice_with_llm(
                voice.stats["llm_samples"],
                voice_id,
                config
            )
        else:
            llm_analysis = "LLM analysis not available - API credentials not configured."
        
        markdown = generate_skill_markdown(voice, docket_id_full, llm_analysis)
        filepath.write_text(markdown, encoding="utf-8")
        
        skill_files.append({
            "voice_id": voice_id,
            "filename": filename,
            "archetype": voice.archetype,
            "sophistication": voice.sophistication,
            "sample_size": len(voice.comments),
        })
        
        logger.info(f"  Created: {filename}")
    
    # Create index file
    index_data = {
        "docket_id": docket_id_full,
        "analyzed_at": pd.Timestamp.now().isoformat(),
        "total_comments": len(records),
        "voice_groups": skill_files,
        "archetype_mapping": {},
    }
    
    # Build archetype → skills mapping
    for archetype in set(v.archetype for v in voice_groups.values()):
        matching_skills = [s["filename"] for s in skill_files if s["archetype"] == archetype]
        index_data["archetype_mapping"][archetype] = matching_skills
    
    index_path = output_path / "index.json"
    index_path.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
    logger.info(f"  Created: index.json")
    
    return index_data


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze docket comment writing styles and generate skill files"
    )
    parser.add_argument(
        "csv_path",
        help=(
            "Docket ID (e.g., CMS-2025-0050) or path to the CSV file. "
            "When given a docket ID, looks for {docket_id}/comments/{docket_id}.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for skill files (default: {docket_id}/stylometry/)"
    )
    parser.add_argument(
        "--attachments-dir",
        default=None,
        help="Directory containing downloaded attachments (default: {docket_id}/comment_attachments/)"
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=5,
        help="Minimum comments required to create a voice group (default: 5)"
    )
    
    args = parser.parse_args()

    # ── Resolve docket ID or CSV path ──────────────────────────────────────
    # If the argument doesn't end in .csv, treat it as a docket ID and
    # derive the conventional path: {docket_id}/comments/{docket_id}.csv
    csv_input = args.csv_path
    import os
    if not csv_input.lower().endswith('.csv'):
        docket_id = csv_input.rstrip('/\\')
        resolved_csv = os.path.join(docket_id, 'comments', f'{docket_id}.csv')
        logger.info(f"Docket ID '{docket_id}' → using {resolved_csv}")
        if not os.path.exists(resolved_csv):
            logger.error(f"CSV file not found: {resolved_csv}")
            logger.error(
                f"Hint: create the directory structure {docket_id}/comments/ "
                f"and place {docket_id}.csv there, or pass the full CSV path."
            )
            return 1
        csv_path = resolved_csv
    else:
        csv_path = csv_input

    try:
        result = analyze_docket_stylometry(
            csv_path=csv_path,
            output_dir=args.output_dir,
            attachments_dir=args.attachments_dir,
            min_group_size=args.min_group_size,
        )
        
        print("\n" + "="*60)
        print("STYLOMETRY ANALYSIS COMPLETE")
        print("="*60)
        print(f"Docket: {result['docket_id']}")
        print(f"Total comments analyzed: {result['total_comments']}")
        print(f"Voice groups created: {len(result['voice_groups'])}")
        print("\nVoice groups:")
        for skill in result['voice_groups']:
            print(f"  - {skill['voice_id']}: {skill['sample_size']} comments")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
