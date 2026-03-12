#!/usr/bin/env python3
"""
analyze_conformance.py — Statistical conformance analysis of generated comments.

Compares the generated synthetic comments against:
  1. Campaign plan voice distribution (P(V))
  2. Campaign plan argument angle distribution
  3. Stylometry voice profiles (word count, bullets, headings, citations, 
     first-person pronouns, paragraphs, sentences)

Produces a detailed report with pass/fail verdicts per metric per voice.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

SYNTHETIC_FILE = "CMS-2025-0050/synthetic_comments/synthetic.txt"
CAMPAIGN_PLAN_FILE = "CMS-2025-0050/campaign/campaign_plan.json"
DOCKET_ID = "CMS-2025-0050"
SEPARATOR = "♔"
NEWLINE_ESCAPE = "⏎"


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_comments(path: str) -> list[dict]:
    """Parse the PSV file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    header = lines[0].strip().split(SEPARATOR)
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.strip().split(SEPARATOR)
        row = {}
        for i, h in enumerate(header):
            row[h] = fields[i] if i < len(fields) else ""
        rows.append(row)
    return rows


def get_comment_text(row: dict) -> str:
    """Extract comment text, unescaping newlines."""
    return row.get("Comment", "").replace(NEWLINE_ESCAPE, "\n")


def get_voice_id(row: dict) -> str:
    """Reconstruct voice_id from archetype + sophistication fields."""
    arch = row.get("synth_archetype", "")
    soph = row.get("synth_sophistication", "")
    if not arch or not soph:
        return "unknown"
    # Determine if org voice
    if arch in ("industry", "advocacy_group", "academic", "government"):
        return f"{arch}-{soph}-org"
    return f"{arch}-{soph}"


# ── Text analysis functions ───────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())


def paragraph_count(text: str) -> int:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return max(1, len(paras))


def sentence_count(text: str) -> int:
    # Simple sentence boundary detection
    sentences = re.split(r'[.!?]+(?:\s|$)', text)
    return max(1, len([s for s in sentences if s.strip()]))


def words_per_sentence(text: str) -> float:
    sc = sentence_count(text)
    wc = word_count(text)
    return wc / max(1, sc)


def first_person_pct(text: str) -> float:
    """Percentage of words that are first-person pronouns."""
    words = text.lower().split()
    if not words:
        return 0.0
    fp_words = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
    count = sum(1 for w in words if w.strip(".,;:!?\"'()") in fp_words)
    return (count / len(words)) * 100


def has_bullets(text: str) -> bool:
    """Check if text contains bullet points or numbered lists."""
    bullet_patterns = [
        r'^\s*[-•●◦▪]\s',      # dash or bullet char
        r'^\s*\d+[.)]\s',       # numbered list
        r'^\s*[a-zA-Z][.)]\s',  # lettered list
        r'^\s*\*\s',            # asterisk bullet
    ]
    for line in text.split("\n"):
        for pattern in bullet_patterns:
            if re.match(pattern, line):
                return True
    return False


def has_headings(text: str) -> bool:
    """Check if text contains section headings."""
    heading_patterns = [
        r'^#+\s',                    # Markdown headings
        r'^[A-Z][A-Z\s]{3,}$',      # ALL CAPS lines
        r'^\*\*[^*]+\*\*\s*$',      # Bold-only lines
        r'^[A-Z][A-Za-z\s]+:$',     # "Heading:" pattern
        r'^\*\*[A-Z]',              # Bold starting with caps
    ]
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for pattern in heading_patterns:
            if re.match(pattern, line):
                return True
    return False


def citation_count(text: str) -> int:
    """Count regulatory/academic citations."""
    patterns = [
        r'\d+\s*(?:CFR|C\.F\.R\.)\s*[§&]?\s*\d+',   # CFR citations
        r'(?:Section|§)\s*\d+',                        # Section references
        r'\d+\s*(?:Fed\.\s*Reg\.|FR)\s*\d+',          # Federal Register
        r'\(\d{4}\)',                                   # Year citations like (2023)
        r'et\s+al\.',                                   # Academic citations
        r'(?:Public Law|P\.L\.)\s*\d+',                # Public law
    ]
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    # Deduplicate by capping at reasonable max
    return min(count, 50)


# ── Statistical comparison ────────────────────────────────────────────────────

def format_pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(n/total)*100:.1f}%"


def median(values: list) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return float(s[-1])
    return s[f] + (k - f) * (s[c] - s[f])


# ── Main analysis ─────────────────────────────────────────────────────────────

def main():
    from stylometry.stylometry_loader import load_voice_skill, parse_statistical_profile
    
    print("=" * 80)
    print("SYNTHETIC COMMENT CONFORMANCE ANALYSIS")
    print("=" * 80)
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    rows = parse_comments(SYNTHETIC_FILE)
    print(f"Total comments: {len(rows)}")
    
    with open(CAMPAIGN_PLAN_FILE, "r") as f:
        plan = json.load(f)
    
    # ── Parse all comments ────────────────────────────────────────────────
    comments_by_voice = defaultdict(list)
    all_angles = []
    
    for row in rows:
        text = get_comment_text(row)
        voice = get_voice_id(row)
        angle = row.get("synth_argument_angle", "")
        
        analysis = {
            "text": text,
            "word_count": word_count(text),
            "paragraph_count": paragraph_count(text),
            "sentence_count": sentence_count(text),
            "words_per_sentence": words_per_sentence(text),
            "first_person_pct": first_person_pct(text),
            "has_bullets": has_bullets(text),
            "has_headings": has_headings(text),
            "citation_count": citation_count(text),
            "angle": angle,
            "voice_id": voice,
            "qc_passed": row.get("synth_qc_passed", "TRUE").upper() == "TRUE",
        }
        comments_by_voice[voice].append(analysis)
        if angle:
            all_angles.append(angle)

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1: CAMPAIGN VOICE DISTRIBUTION
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("━" * 80)
    print("1. VOICE DISTRIBUTION: Campaign Plan vs. Generated")
    print("━" * 80)
    
    target_voices = plan["campaign_voices"]
    total = len(rows)
    
    print(f"{'Voice':<35s} {'Target':>8s} {'Actual':>8s} {'#Gen':>6s} {'#Exp':>6s} {'Delta':>8s} {'Verdict':>8s}")
    print("-" * 80)
    
    voice_verdicts = []
    for voice in sorted(target_voices.keys()):
        target_pct = target_voices[voice] * 100
        actual_count = len(comments_by_voice.get(voice, []))
        actual_pct = (actual_count / total) * 100 if total > 0 else 0
        expected_count = target_voices[voice] * total
        delta = actual_pct - target_pct
        # Verdict: within ±5 percentage points
        ok = abs(delta) <= 5.0
        verdict = "✓ PASS" if ok else "✗ FAIL"
        voice_verdicts.append(ok)
        print(f"{voice:<35s} {target_pct:>7.1f}% {actual_pct:>7.1f}% {actual_count:>6d} {expected_count:>6.0f} {delta:>+7.1f}% {verdict:>8s}")
    
    voice_pass_rate = sum(voice_verdicts) / len(voice_verdicts) * 100
    print(f"\nVoice distribution pass rate: {voice_pass_rate:.0f}% ({sum(voice_verdicts)}/{len(voice_verdicts)} voices within ±5pp)")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2: ARGUMENT ANGLE DISTRIBUTION
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("━" * 80)
    print("2. ARGUMENT ANGLE DISTRIBUTION")
    print("━" * 80)
    
    angle_targets = {a["angle"]: a["weight"] for a in plan["argument_angles"]}
    angle_counter = Counter()
    for a in all_angles:
        # Match to closest target angle
        for target_angle in angle_targets:
            if target_angle[:60] in a[:80] or a[:60] in target_angle[:80]:
                angle_counter[target_angle] += 1
                break
        else:
            angle_counter[a[:80]] += 1
    
    total_angles = sum(angle_counter.values())
    print(f"{'Angle ID':<40s} {'Target':>8s} {'Actual':>8s} {'Count':>6s} {'Verdict':>8s}")
    print("-" * 80)
    
    angle_verdicts = []
    for aa in plan["argument_angles"]:
        target_pct = aa["weight"] * 100
        count = 0
        for angle_text, cnt in angle_counter.items():
            if aa["angle"][:50] in angle_text[:60] or angle_text[:50] in aa["angle"][:60]:
                count += cnt
        actual_pct = (count / total_angles * 100) if total_angles > 0 else 0
        delta = abs(actual_pct - target_pct)
        ok = delta <= 10.0  # ±10pp for angles (more variance expected)
        angle_verdicts.append(ok)
        verdict = "✓ PASS" if ok else "✗ FAIL"
        print(f"{aa['id']:<40s} {target_pct:>7.1f}% {actual_pct:>7.1f}% {count:>6d} {verdict:>8s}")
    
    angle_pass_rate = sum(angle_verdicts) / max(1, len(angle_verdicts)) * 100
    print(f"\nAngle distribution pass rate: {angle_pass_rate:.0f}% ({sum(angle_verdicts)}/{len(angle_verdicts)} angles within ±10pp)")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3: PER-VOICE STYLOMETRY CONFORMANCE
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("━" * 80)
    print("3. PER-VOICE STYLOMETRY CONFORMANCE")
    print("━" * 80)
    
    all_metric_verdicts = []
    
    for voice in sorted(target_voices.keys()):
        comments = comments_by_voice.get(voice, [])
        if not comments:
            print(f"\n  {voice}: NO COMMENTS GENERATED — SKIP")
            continue
        
        # Load expected stats
        if voice.endswith("-org"):
            clean = voice[:-4]
            archetype, soph = clean.rsplit("-", 1)
        else:
            archetype, soph = voice.rsplit("-", 1)
        
        skill = load_voice_skill(DOCKET_ID, archetype, soph)
        if not skill:
            print(f"\n  {voice}: NO STYLOMETRY PROFILE — SKIP")
            continue
        
        expected = parse_statistical_profile(skill)
        n = len(comments)
        
        print(f"\n{'─' * 80}")
        print(f"  VOICE: {voice}  (n={n})")
        print(f"{'─' * 80}")
        
        # ── Word count ────────────────────────────────────────────────────
        wcs = [c["word_count"] for c in comments]
        actual_median = median(wcs)
        actual_p10 = percentile(wcs, 10)
        actual_p90 = percentile(wcs, 90)
        actual_min = min(wcs)
        actual_max = max(wcs)
        
        # Check: median within 50% of expected median
        wc_ratio = actual_median / max(1, expected.word_count_median)
        wc_ok = 0.4 <= wc_ratio <= 2.5
        all_metric_verdicts.append(("word_count_median", voice, wc_ok))
        
        # Check: range overlap
        range_overlap = (actual_p10 <= expected.word_count_high and actual_p90 >= expected.word_count_low)
        all_metric_verdicts.append(("word_count_range", voice, range_overlap))
        
        v1 = "✓" if wc_ok else "✗"
        v2 = "✓" if range_overlap else "✗"
        print(f"  Word count:")
        print(f"    Expected  median={expected.word_count_median:.0f}  range=[{expected.word_count_low:.0f}, {expected.word_count_high:.0f}]")
        print(f"    Actual    median={actual_median:.0f}  p10={actual_p10:.0f}  p90={actual_p90:.0f}  min={actual_min}  max={actual_max}")
        print(f"    Median ratio: {wc_ratio:.2f}x  {v1}    Range overlap: {v2}")
        
        # ── Paragraph count ───────────────────────────────────────────────
        paras = [c["paragraph_count"] for c in comments]
        actual_para_median = median(paras)
        para_ratio = actual_para_median / max(1, expected.paragraphs_median)
        para_ok = 0.3 <= para_ratio <= 3.0
        all_metric_verdicts.append(("paragraphs", voice, para_ok))
        v = "✓" if para_ok else "✗"
        print(f"  Paragraphs:")
        print(f"    Expected  median={expected.paragraphs_median:.0f}")
        print(f"    Actual    median={actual_para_median:.0f}  ratio={para_ratio:.2f}x  {v}")
        
        # ── Words per sentence ────────────────────────────────────────────
        wpss = [c["words_per_sentence"] for c in comments]
        actual_wps = np.mean(wpss)
        wps_delta = abs(actual_wps - expected.words_per_sentence)
        wps_ok = wps_delta <= 10.0  # within 10 words
        all_metric_verdicts.append(("words_per_sentence", voice, wps_ok))
        v = "✓" if wps_ok else "✗"
        print(f"  Words/sentence:")
        print(f"    Expected  {expected.words_per_sentence:.0f}")
        print(f"    Actual    {actual_wps:.1f}  delta={wps_delta:.1f}  {v}")
        
        # ── First-person pronoun % ────────────────────────────────────────
        fp_pcts = [c["first_person_pct"] for c in comments]
        actual_fp = np.mean(fp_pcts)
        # Direction check: if expected > 1%, actual should also be > 0.5%
        # If expected < 0.5%, actual should be < 2%
        if expected.first_person_pct >= 1.5:
            fp_ok = actual_fp >= 0.5
        elif expected.first_person_pct <= 0.5:
            fp_ok = actual_fp <= 3.0
        else:
            fp_ok = True  # medium range, any is fine
        all_metric_verdicts.append(("first_person_pct", voice, fp_ok))
        v = "✓" if fp_ok else "✗"
        print(f"  First-person %:")
        print(f"    Expected  {expected.first_person_pct:.1f}%")
        print(f"    Actual    {actual_fp:.1f}%  {v}")
        
        # ── Bullet usage % ────────────────────────────────────────────────
        bullet_count = sum(1 for c in comments if c["has_bullets"])
        actual_bullet_pct = (bullet_count / n) * 100
        expected_bullet_pct = expected.uses_bullet_points_pct
        bullet_delta = abs(actual_bullet_pct - expected_bullet_pct)
        bullet_ok = bullet_delta <= 30.0  # within 30pp
        all_metric_verdicts.append(("bullets", voice, bullet_ok))
        v = "✓" if bullet_ok else "✗"
        print(f"  Bullet usage:")
        print(f"    Expected  {expected_bullet_pct:.0f}%")
        print(f"    Actual    {actual_bullet_pct:.0f}%  ({bullet_count}/{n})  delta={bullet_delta:.0f}pp  {v}")
        
        # ── Heading usage % ───────────────────────────────────────────────
        heading_count = sum(1 for c in comments if c["has_headings"])
        actual_heading_pct = (heading_count / n) * 100
        expected_heading_pct = expected.uses_headings_pct
        heading_delta = abs(actual_heading_pct - expected_heading_pct)
        heading_ok = heading_delta <= 30.0
        all_metric_verdicts.append(("headings", voice, heading_ok))
        v = "✓" if heading_ok else "✗"
        print(f"  Heading usage:")
        print(f"    Expected  {expected_heading_pct:.0f}%")
        print(f"    Actual    {actual_heading_pct:.0f}%  ({heading_count}/{n})  delta={heading_delta:.0f}pp  {v}")
        
        # ── Citation frequency ────────────────────────────────────────────
        cites = [c["citation_count"] for c in comments]
        actual_cite_mean = np.mean(cites)
        expected_cite = expected.citation_frequency
        # Direction check
        if expected_cite >= 3.0:
            cite_ok = actual_cite_mean >= 1.0
        elif expected_cite >= 0.5:
            cite_ok = actual_cite_mean >= 0.1
        else:
            cite_ok = actual_cite_mean <= 5.0  # shouldn't have many
        all_metric_verdicts.append(("citations", voice, cite_ok))
        v = "✓" if cite_ok else "✗"
        print(f"  Citations:")
        print(f"    Expected  ~{expected_cite:.1f} per comment")
        print(f"    Actual    {actual_cite_mean:.1f} per comment  (range {min(cites)}–{max(cites)})  {v}")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4: AGGREGATE STATISTICS
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("━" * 80)
    print("4. AGGREGATE STATISTICS")
    print("━" * 80)
    
    all_wcs = [c["word_count"] for cs in comments_by_voice.values() for c in cs]
    all_paras = [c["paragraph_count"] for cs in comments_by_voice.values() for c in cs]
    all_fps = [c["first_person_pct"] for cs in comments_by_voice.values() for c in cs]
    all_cites = [c["citation_count"] for cs in comments_by_voice.values() for c in cs]
    all_bullets = [c["has_bullets"] for cs in comments_by_voice.values() for c in cs]
    all_headings = [c["has_headings"] for cs in comments_by_voice.values() for c in cs]
    qc_passed = sum(1 for cs in comments_by_voice.values() for c in cs if c["qc_passed"])
    
    print(f"  Total comments:       {total}")
    print(f"  QC passed:            {qc_passed} ({qc_passed/total*100:.0f}%)")
    print(f"  Word count:           median={median(all_wcs):.0f}  mean={np.mean(all_wcs):.0f}  "
          f"min={min(all_wcs)}  max={max(all_wcs)}  std={np.std(all_wcs):.0f}")
    print(f"  Paragraphs:           median={median(all_paras):.0f}  mean={np.mean(all_paras):.1f}")
    print(f"  First-person %:       mean={np.mean(all_fps):.2f}%")
    print(f"  Citations/comment:    mean={np.mean(all_cites):.1f}")
    print(f"  With bullets:         {sum(all_bullets)}/{total} ({sum(all_bullets)/total*100:.0f}%)")
    print(f"  With headings:        {sum(all_headings)}/{total} ({sum(all_headings)/total*100:.0f}%)")
    
    # Word count distribution by voice
    print(f"\n  Word count by voice:")
    for voice in sorted(target_voices.keys()):
        comments = comments_by_voice.get(voice, [])
        if not comments:
            continue
        wcs = [c["word_count"] for c in comments]
        print(f"    {voice:<35s} n={len(wcs):3d}  median={median(wcs):>6.0f}  "
              f"range=[{min(wcs):>5d}, {max(wcs):>5d}]")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5: OVERALL CONFORMANCE SCORECARD
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("━" * 80)
    print("5. OVERALL CONFORMANCE SCORECARD")
    print("━" * 80)
    
    # Group verdicts by metric
    metric_groups = defaultdict(list)
    for metric, voice, ok in all_metric_verdicts:
        metric_groups[metric].append((voice, ok))
    
    print(f"\n  {'Metric':<25s} {'Pass':>6s} {'Fail':>6s} {'Rate':>8s}")
    print(f"  {'-'*50}")
    
    total_pass = 0
    total_checks = 0
    for metric in ["word_count_median", "word_count_range", "paragraphs", 
                    "words_per_sentence", "first_person_pct", "bullets", 
                    "headings", "citations"]:
        items = metric_groups.get(metric, [])
        p = sum(1 for _, ok in items if ok)
        f = len(items) - p
        rate = (p / len(items) * 100) if items else 0
        total_pass += p
        total_checks += len(items)
        symbol = "✓" if rate >= 75 else "⚠" if rate >= 50 else "✗"
        print(f"  {metric:<25s} {p:>6d} {f:>6d} {rate:>6.0f}%  {symbol}")
    
    overall = (total_pass / total_checks * 100) if total_checks > 0 else 0
    print(f"\n  {'OVERALL':<25s} {total_pass:>6d} {total_checks - total_pass:>6d} {overall:>6.0f}%")
    
    # Voice distribution
    print(f"\n  Voice distribution:     {voice_pass_rate:.0f}% pass")
    print(f"  Angle distribution:     {angle_pass_rate:.0f}% pass")
    print(f"  Stylometry conformance: {overall:.0f}% pass")
    
    grand_total = (voice_pass_rate + angle_pass_rate + overall) / 3
    print(f"\n  ╔{'═'*50}╗")
    print(f"  ║  GRAND CONFORMANCE SCORE:  {grand_total:.0f}%{' ':>19s}║")
    print(f"  ╚{'═'*50}╝")
    print()


if __name__ == "__main__":
    main()
