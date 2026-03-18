"""
Disposable bigram/trigram frequency experiment.

Scans both dockets' synthetic comment PSV files and reports the top
repeated bigrams and trigrams (no stopword filtering — raw counts).
This helps identify short narrative anchor phrases the LLM overuses.
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shuffler.psv_io import read_psv

# ── Config ────────────────────────────────────────────────────────────────────

FILES = [
    ("HHS-ONC-2026-0001", "HHS-ONC-2026-0001/synthetic_comments/synthetic.txt"),
    ("CMS-2025-0050",     "CMS-2025-0050/synthetic_comments/synthetic_original.txt"),
]

TOP_N = 40          # how many top phrases to show per docket
MIN_COMMENTS = 3    # minimum distinct comments a phrase must appear in

# ── Text helpers ──────────────────────────────────────────────────────────────

_CLEAN_RE = re.compile(r"[^a-z0-9' ]+")

def normalise(text: str) -> str:
    cleaned = _CLEAN_RE.sub(" ", text.lower())
    return " ".join(cleaned.split())

def tokenise(text: str) -> list[str]:
    return [w for w in text.split() if w]

def ngrams(tokens: list[str], n: int):
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i+n])

# ── Main ──────────────────────────────────────────────────────────────────────

for docket_id, psv_path in FILES:
    if not Path(psv_path).exists():
        print(f"\n{'='*60}")
        print(f"DOCKET: {docket_id}")
        print(f"  File not found: {psv_path}")
        continue

    rows, fieldnames = read_psv(psv_path)

    # Detect comment column
    comment_col = "Comment" if "Comment" in fieldnames else fieldnames[0]

    # Load comment texts
    texts = []
    for row in rows:
        text = row.get(comment_col, "").strip()
        if text:
            texts.append(text)

    print(f"\n{'='*60}")
    print(f"DOCKET: {docket_id}  ({len(texts)} comments from {psv_path})")
    print(f"{'='*60}")

    for n in (2, 3):
        # Count how many distinct comments each phrase appears in
        phrase_to_comments: dict[tuple, set] = defaultdict(set)
        for idx, text in enumerate(texts):
            tokens = tokenise(normalise(text))
            seen_in_this_comment = set()
            for gram in ngrams(tokens, n):
                if gram not in seen_in_this_comment:
                    phrase_to_comments[gram].add(idx)
                    seen_in_this_comment.add(gram)

        # Filter and sort
        results = [
            (gram, len(comment_set))
            for gram, comment_set in phrase_to_comments.items()
            if len(comment_set) >= MIN_COMMENTS
        ]
        results.sort(key=lambda x: -x[1])

        print(f"\n  Top {TOP_N} {n}-grams (appearing in ≥{MIN_COMMENTS} distinct comments):")
        print(f"  {'Phrase':<40} {'# Comments':>10}  {'% of batch':>10}")
        print(f"  {'-'*40} {'-'*10}  {'-'*10}")
        for gram, count in results[:TOP_N]:
            phrase = " ".join(gram)
            pct = count / len(texts) * 100
            print(f"  {phrase:<40} {count:>10}  {pct:>9.1f}%")

        if len(results) > TOP_N:
            print(f"  ... and {len(results) - TOP_N} more phrases with ≥{MIN_COMMENTS} comments")
