"""
analyze_campaign.py — Compare generated synthetic comments against the campaign plan
and real-comment stylometry baselines.

Produces a comprehensive report covering:
  1. Voice distribution (planned vs actual)
  2. Argument angle coverage per voice
  3. Word count distributions per voice vs stylometry targets
  4. Structural features (bullets, headings, citations, em-dashes)
  5. AI vocabulary marker frequency
  6. First-person pronoun density
  7. Sentence length statistics

Usage:
  python analyze_campaign.py CMS-2025-0050
  python analyze_campaign.py CMS-2025-0050 --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ── Reuse stylometry fingerprinting ────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))

from stylometry.stylometry_utils import fingerprint as base_fingerprint
from stylometry.stylometry_analyzer import (
    analyze_punctuation,
    analyze_structure,
    detect_ai_vocabulary,
    analyze_emphasis,
)
from stylometry.stylometry_loader import parse_statistical_profile


# ── Helpers ────────────────────────────────────────────────────────────────────

def full_profile(text: str) -> dict:
    """Compute full stylometric profile for a single comment."""
    base = base_fingerprint(text)
    punct = analyze_punctuation(text)
    struct = analyze_structure(text)
    ai = detect_ai_vocabulary(text)
    emph = analyze_emphasis(text)
    return {**base, **punct, **struct, **ai, **emph}


def safe_median(vals):
    v = [x for x in vals if x is not None]
    return float(np.median(v)) if v else 0.0


def safe_mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else 0.0


def pct(num, denom):
    return num / denom * 100 if denom else 0.0


def bar(fraction, width=30):
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


# ── Load data ─────────────────────────────────────────────────────────────────

def load_synthetic_comments(docket_id: str) -> list[dict]:
    """Load synthetic comments from the PSV file."""
    path = Path(docket_id) / "synthetic_comments" / "synthetic.txt"
    if not path.exists():
        raise FileNotFoundError(f"Synthetic comments not found: {path}")

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("Synthetic file has no data rows")

    sep = "♔"
    headers = lines[0].strip().split(sep)
    rows = []
    for line in lines[1:]:
        vals = line.strip().split(sep)
        if len(vals) != len(headers):
            continue
        rows.append(dict(zip(headers, vals)))
    return rows


def load_campaign_plan(docket_id: str) -> dict:
    path = Path(docket_id) / "campaign" / "campaign_plan.json"
    with open(path) as f:
        return json.load(f)


def load_stylometry_skills(docket_id: str) -> dict[str, str]:
    """Load all voice skill markdown files, keyed by voice_id."""
    skill_dir = Path(docket_id) / "stylometry"
    skills = {}
    for md in skill_dir.glob("*.md"):
        skills[md.stem] = md.read_text(encoding="utf-8")
    return skills


# ── Derive voice_id from synth metadata ────────────────────────────────────────

def derive_voice_id(row: dict) -> str:
    """Reconstruct the voice_id from archetype + sophistication + org."""
    arch = row.get("synth_archetype", "unknown").strip()
    soph = row.get("synth_sophistication", "medium").strip()
    org = row.get("Organization Name", "").strip()
    parts = [arch, soph]
    if org and arch != "individual_consumer":
        parts.append("org")
    return "-".join(parts)


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze(docket_id: str, verbose: bool = False):
    plan = load_campaign_plan(docket_id)
    rows = load_synthetic_comments(docket_id)
    skills = load_stylometry_skills(docket_id)

    N = len(rows)
    print(f"\n{'='*72}")
    print(f"  CAMPAIGN ANALYSIS: {docket_id}")
    print(f"  {N} synthetic comments generated")
    print(f"{'='*72}\n")

    # ── 1. Voice distribution ──────────────────────────────────────────────
    planned_voices = plan.get("campaign_voices", {})
    actual_voice_counts = Counter()
    voice_rows: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        vid = derive_voice_id(row)
        actual_voice_counts[vid] += 1
        voice_rows[vid].append(row)

    all_voices = sorted(set(list(planned_voices.keys()) + list(actual_voice_counts.keys())))

    print("┌─────────────────────────────────┬──────────┬──────────┬──────────┐")
    print("│ Voice                           │ Planned% │ Actual%  │ Delta    │")
    print("├─────────────────────────────────┼──────────┼──────────┼──────────┤")
    for v in all_voices:
        p = planned_voices.get(v, 0) * 100
        a = actual_voice_counts.get(v, 0) / N * 100
        delta = a - p
        sign = "+" if delta > 0 else ""
        print(f"│ {v:<31} │ {p:6.1f}%  │ {a:6.1f}%  │ {sign}{delta:5.1f}pp │")
    print("└─────────────────────────────────┴──────────┴──────────┴──────────┘")

    # ── 2. Argument angle distribution ─────────────────────────────────────
    angles = plan.get("argument_angles", [])
    angle_ids = {a["id"]: a for a in angles}
    angle_counter = Counter()
    for row in rows:
        aa = row.get("synth_argument_angle", "").strip()
        # Match to angle id
        matched = False
        for aid, ainfo in angle_ids.items():
            if aid in aa.lower().replace(" ", "_") or (len(aa) > 20 and aa[:30] in ainfo["angle"]):
                angle_counter[aid] += 1
                matched = True
                break
        if not matched:
            angle_counter[aa[:50] if aa else "(unknown)"] += 1

    print(f"\n{'─'*72}")
    print("  ARGUMENT ANGLE DISTRIBUTION")
    print(f"{'─'*72}")
    print(f"{'Angle':<35} {'Plan%':>7} {'Actual%':>8} {'Count':>6}")
    for a in angles:
        aid = a["id"]
        pw = a["weight"] * 100
        ac = angle_counter.get(aid, 0)
        ap = ac / N * 100
        print(f"  {aid:<33} {pw:5.1f}%  {ap:6.1f}%   {ac:4d}")
    # Show unmatched
    for key, cnt in angle_counter.items():
        if key not in angle_ids:
            print(f"  {key:<33} {'n/a':>5}   {cnt/N*100:6.1f}%   {cnt:4d}")

    # ── 3. Word count per voice vs stylometry targets ──────────────────────
    print(f"\n{'─'*72}")
    print("  WORD COUNT: SYNTHETIC vs REAL-COMMENT STYLOMETRY BASELINES")
    print(f"{'─'*72}")
    print(f"{'Voice':<28} {'Styl Med':>9} {'Styl Range':>12} {'Synth Med':>10} {'Synth Range':>14} {'In-range%':>10}")

    for v in all_voices:
        vrows = voice_rows.get(v, [])
        if not vrows:
            continue

        # Get word counts from synthetic comments
        synth_wcs = []
        for r in vrows:
            wc = r.get("synth_word_count", "").strip()
            if wc and wc.isdigit():
                synth_wcs.append(int(wc))
            else:
                synth_wcs.append(len(r.get("Comment", "").split()))

        synth_med = safe_median(synth_wcs)
        synth_lo = min(synth_wcs) if synth_wcs else 0
        synth_hi = max(synth_wcs) if synth_wcs else 0

        # Get stylometry baseline
        skill = skills.get(v)
        if skill:
            stats = parse_statistical_profile(skill)
            styl_med = stats.word_count_median
            styl_lo = stats.word_count_low
            styl_hi = stats.word_count_high
            in_range = sum(1 for wc in synth_wcs if styl_lo * 0.8 <= wc <= styl_hi * 1.2)
            ir_pct = pct(in_range, len(synth_wcs))
        else:
            styl_med = styl_lo = styl_hi = 0
            ir_pct = 0

        print(f"  {v:<26} {styl_med:7.0f}   {styl_lo:5.0f}-{styl_hi:<5.0f}  {synth_med:8.0f}   {synth_lo:5d}-{synth_hi:<5d}    {ir_pct:5.1f}%")

    # ── 4-7. Per-voice structural/stylometric analysis ─────────────────────
    print(f"\n{'─'*72}")
    print("  STRUCTURAL FEATURES: SYNTHETIC vs REAL BASELINES")
    print(f"{'─'*72}")

    # Collect per-voice profiles
    for v in all_voices:
        vrows = voice_rows.get(v, [])
        if not vrows:
            continue

        comments = [r.get("Comment", "") for r in vrows]
        profiles = [full_profile(c) for c in comments]

        skill = skills.get(v)
        if skill:
            stats = parse_statistical_profile(skill)
        else:
            stats = None

        n = len(profiles)
        print(f"\n  ┌── {v} (n={n}) {'─'*(50-len(v))}")

        # Word count
        wcs = [p["word_count"] for p in profiles]
        med_wc = safe_median(wcs)
        styl_wc = stats.word_count_median if stats else 0
        print(f"  │ Word count median:       {med_wc:7.0f}  (real baseline: {styl_wc:.0f})")

        # Sentence length
        sls = [p["mean_sentence_len"] for p in profiles]
        med_sl = safe_median(sls)
        styl_sl = stats.words_per_sentence if stats else 0
        print(f"  │ Sentence length median:  {med_sl:7.1f}  (real baseline: {styl_sl:.1f})")

        # First-person ratio
        fps = [p["first_person_ratio"] * 100 for p in profiles]
        med_fp = safe_median(fps)
        styl_fp = stats.first_person_pct if stats else 0
        print(f"  │ First-person %:          {med_fp:7.1f}%  (real baseline: {styl_fp:.1f}%)")

        # Bullets
        has_bullets = sum(1 for p in profiles if p.get("bullet_count", 0) > 0)
        pct_bullets = pct(has_bullets, n)
        styl_bullets = stats.uses_bullet_points_pct if stats else 0
        print(f"  │ Uses bullets:            {pct_bullets:6.0f}%   (real baseline: {styl_bullets:.0f}%)")

        # Headings
        has_headings = sum(1 for p in profiles if p.get("heading_count", 0) > 0)
        pct_headings = pct(has_headings, n)
        styl_headings = stats.uses_headings_pct if stats else 0
        print(f"  │ Uses headings:           {pct_headings:6.0f}%   (real baseline: {styl_headings:.0f}%)")

        # Citations
        cits = [p["citation_count"] for p in profiles]
        med_cit = safe_median(cits)
        styl_cit = stats.citation_frequency if stats else 0
        print(f"  │ Citations median:        {med_cit:7.1f}  (real baseline: {styl_cit:.1f})")

        # Em-dashes
        ems = [p.get("em_dash_freq", 0) for p in profiles]
        med_em = safe_median(ems)
        styl_em = stats.em_dash_per_100 if stats else 0
        print(f"  │ Em-dash /100 words:      {med_em:7.2f}  (real baseline: {styl_em:.2f})")

        # ALL CAPS
        caps = [p.get("all_caps_freq", 0) for p in profiles]
        med_caps = safe_median(caps)
        styl_caps = stats.all_caps_pct if stats else 0
        print(f"  │ ALL CAPS %:              {med_caps:7.2f}%  (real baseline: {styl_caps:.2f}%)")

        # AI vocabulary
        ai_freqs = [p.get("ai_vocab_freq", 0) for p in profiles]
        med_ai = safe_median(ai_freqs)
        styl_ai = stats.ai_vocabulary_pct if stats else 0
        has_ai = sum(1 for p in profiles if p.get("contains_ai_markers", False))
        print(f"  │ AI vocab freq:           {med_ai:7.2f}%  (real baseline: {styl_ai:.2f}%)")
        print(f"  │ Has AI markers:          {pct(has_ai, n):6.0f}%")

        print(f"  └{'─'*56}")

    # ── Summary: Aggregate deviation scores ────────────────────────────────
    print(f"\n{'='*72}")
    print("  AGGREGATE DEVIATION SUMMARY")
    print(f"{'='*72}")

    total_wc_devs = []
    total_sl_devs = []
    total_fp_devs = []
    total_bullet_devs = []

    for v in all_voices:
        vrows = voice_rows.get(v, [])
        if not vrows:
            continue
        skill = skills.get(v)
        if not skill:
            continue
        stats = parse_statistical_profile(skill)

        comments = [r.get("Comment", "") for r in vrows]
        profiles = [full_profile(c) for c in comments]

        wcs = [p["word_count"] for p in profiles]
        sls = [p["mean_sentence_len"] for p in profiles]
        fps = [p["first_person_ratio"] * 100 for p in profiles]

        if stats.word_count_median > 0:
            total_wc_devs.append(abs(safe_median(wcs) - stats.word_count_median) / stats.word_count_median * 100)
        if stats.words_per_sentence > 0:
            total_sl_devs.append(abs(safe_median(sls) - stats.words_per_sentence) / stats.words_per_sentence * 100)
        total_fp_devs.append(abs(safe_median(fps) - stats.first_person_pct))

        has_bullets = pct(sum(1 for p in profiles if p.get("bullet_count", 0) > 0), len(profiles))
        total_bullet_devs.append(abs(has_bullets - stats.uses_bullet_points_pct))

    print(f"  Mean word-count deviation from baseline:  {safe_mean(total_wc_devs):5.1f}%")
    print(f"  Mean sentence-len deviation from baseline: {safe_mean(total_sl_devs):5.1f}%")
    print(f"  Mean first-person deviation (pp):          {safe_mean(total_fp_devs):5.1f}pp")
    print(f"  Mean bullet-usage deviation (pp):          {safe_mean(total_bullet_devs):5.1f}pp")

    # Voice distribution error
    voice_errors = []
    for v in all_voices:
        p = planned_voices.get(v, 0)
        a = actual_voice_counts.get(v, 0) / N
        voice_errors.append(abs(a - p) * 100)
    print(f"  Mean voice-distribution error (pp):        {safe_mean(voice_errors):5.1f}pp")

    # QC pass rate
    qc_passed = sum(1 for r in rows if r.get("synth_qc_passed", "").strip().upper() == "TRUE")
    print(f"\n  QC pass rate: {qc_passed}/{N} ({pct(qc_passed, N):.1f}%)")
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze synthetic comment campaign")
    parser.add_argument("docket_id", help="Docket ID (e.g. CMS-2025-0050)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    analyze(args.docket_id, args.verbose)
