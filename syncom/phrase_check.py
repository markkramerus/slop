"""
phrase_check.py — Post-hoc batch analysis for repeated distinctive phrases.

Scans a batch of generated comments for n-gram phrases that appear in two or
more comments.  Repeated distinctive phrases ("As a registered nurse",
"a widely-used commercial algorithm", "I live in Duluth, MN") are a strong
signal that comments were mass-generated from similar prompts.

This module does NOT reject comments — it writes a human-readable Markdown
report so a reviewer can decide which comments need manual editing.

Usage
-----
    from syncom.phrase_check import run_phrase_check

    report_path = run_phrase_check(
        comments=all_comments,
        output_path="phrase_repetition_report.md",
    )
"""

from __future__ import annotations

import re
import time
import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from .generator import GeneratedComment


# ── English stopwords (no external dependency) ────────────────────────────────

_STOPWORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "get", "gets", "got", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
    "itself", "just", "let's", "me", "might", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd",
    "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "us", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "will", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves", "also", "been", "being",
    "did", "does", "doing", "done", "going", "gone", "got", "gotten",
    "had", "has", "have", "having", "here", "how", "its", "just", "like",
    "make", "many", "may", "much", "must", "need", "now", "one", "only",
    "really", "right", "said", "say", "says", "shall", "since", "still",
    "such", "take", "tell", "that", "the", "them", "then", "there",
    "these", "they", "thing", "think", "this", "those", "though",
    "three", "time", "two", "upon", "us", "use", "used", "using", "want",
    "way", "well", "went", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "yet",
}


# ── Text normalisation ────────────────────────────────────────────────────────

# Regex to strip everything except letters, digits, apostrophes, and spaces.
_CLEAN_RE = re.compile(r"[^a-z0-9' ]+")

# Sentence-splitting regex: split on .!? followed by whitespace or end-of-string.
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

# Pattern cache for _phrase_to_pattern
_PATTERN_CACHE: dict[str, re.Pattern] = {}


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation (keep apostrophes), collapse whitespace."""
    cleaned = _CLEAN_RE.sub(" ", text.lower())
    return " ".join(cleaned.split())


def _tokenise(text: str) -> list[str]:
    """Split normalised text into word tokens."""
    return [w for w in text.split() if w]


def _is_distinctive(ngram: tuple[str, ...]) -> bool:
    """
    An n-gram is distinctive if at least 50% of its words are content words
    (i.e. not stopwords).  This filters out generic constructions like
    "I would like to" or "in order to be".
    """
    content = sum(1 for w in ngram if w not in _STOPWORDS)
    return content >= len(ngram) * 0.5


def _extract_sentences(text: str) -> list[str]:
    """Split comment text into sentences."""
    sentences = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _phrase_to_pattern(phrase: str) -> re.Pattern:
    """
    Build a case-insensitive regex that matches a normalised phrase in original
    text, allowing any non-alphanumeric characters (punctuation, symbols,
    whitespace) between words.

    For example ``"45 cfr 170 315"`` → a pattern matching ``"45 CFR § 170.315"``.
    """
    if phrase in _PATTERN_CACHE:
        return _PATTERN_CACHE[phrase]

    words = phrase.lower().split()
    # Between words allow one or more non-alphanumeric characters
    pattern_str = r"[^a-zA-Z0-9]+".join(re.escape(w) for w in words)
    pat = re.compile(pattern_str, re.IGNORECASE)
    _PATTERN_CACHE[phrase] = pat
    return pat


def _find_sentence_containing(text: str, phrase: str) -> str:
    """
    Return the full sentence from *text* that contains *phrase*.

    The *phrase* comes from normalised n-gram extraction (punctuation stripped,
    lowercased) so a direct substring search against the original text can
    fail when the original contains hyphens, periods, section symbols, etc.

    We therefore try two strategies in order:
      1. Direct case-insensitive substring match (fast path).
      2. Regex match that tolerates intervening punctuation / symbols.

    Returns up to 300 characters of context.  If no match at all,
    returns ``"(sentence not found)"``.
    """
    sentences = _extract_sentences(text)
    phrase_lower = phrase.lower()

    # ── Strategy 1: direct substring match on each sentence ────────────
    for sentence in sentences:
        if phrase_lower in sentence.lower():
            if len(sentence) > 300:
                idx = sentence.lower().index(phrase_lower)
                start = max(0, idx - 80)
                end = min(len(sentence), idx + len(phrase) + 80)
                return ("…" if start > 0 else "") + sentence[start:end] + ("…" if end < len(sentence) else "")
            return sentence

    # ── Strategy 2: regex tolerating punctuation between words ─────────
    pat = _phrase_to_pattern(phrase)

    for sentence in sentences:
        m = pat.search(sentence)
        if m:
            if len(sentence) > 300:
                start = max(0, m.start() - 80)
                end = min(len(sentence), m.end() + 80)
                return ("…" if start > 0 else "") + sentence[start:end] + ("…" if end < len(sentence) else "")
            return sentence

    # ── Fallback: window around match in raw text ──────────────────────
    m = pat.search(text)
    if m:
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 80)
        return ("…" if start > 0 else "") + text[start:end] + ("…" if end < len(text) else "")

    return "(sentence not found)"


# ── N-gram extraction ─────────────────────────────────────────────────────────

def extract_distinctive_ngrams(
    text: str,
    min_n: int = 4,
    max_n: int = 8,
) -> set[tuple[str, ...]]:
    """
    Extract all distinctive n-grams of length min_n..max_n from *text*.

    Returns a set of tuples (each tuple is one n-gram).
    """
    tokens = _tokenise(_normalise(text))
    ngrams: set[tuple[str, ...]] = set()
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            if _is_distinctive(gram):
                ngrams.add(gram)
    return ngrams


# ── Submitter name helper ─────────────────────────────────────────────────────

def _build_submitter_name(persona) -> str:
    """
    Build the best available submitter identification string from persona
    fields.  Prefers First+Last name when meaningful, falls back to
    Organization Name, then "Anonymous".

    For real regulatory comments the First/Last fields are often empty while
    Organization Name carries the submitter identity.  For synthetic comments
    First/Last is always populated but Org Name may also be present.
    """
    first = getattr(persona, "first_name", "") or ""
    last = getattr(persona, "last_name", "") or ""
    org = getattr(persona, "org_name", "") or ""

    # Build person name, filtering out "Unknown" and "Anonymous" placeholders
    person_name = f"{first} {last}".strip()
    if person_name.lower() in ("", "unknown", "anonymous anonymous", "unknown unknown"):
        person_name = ""

    if person_name and org:
        return f"{person_name} ({org})"
    elif person_name:
        return person_name
    elif org:
        return org
    else:
        return "Anonymous"


def _build_submitter_detail(persona) -> str:
    """
    Build a secondary detail string from occupation and state, filtering
    out empty / unknown values.
    """
    occupation = getattr(persona, "occupation", "") or ""
    state = getattr(persona, "state", "") or ""

    parts = [p for p in (occupation, state) if p.strip()]
    return ", ".join(parts) if parts else ""


# ── Batch analysis ────────────────────────────────────────────────────────────

@dataclass
class PhraseMatch:
    """One occurrence of a repeated phrase in a specific comment."""
    comment_index: int
    comment_id: str
    persona_name: str
    persona_detail: str          # e.g. "RN, Ohio"
    sentence: str                # full sentence containing the phrase


@dataclass
class RepeatedPhrase:
    """A phrase found in two or more comments."""
    phrase: str
    ngram_length: int
    matches: list[PhraseMatch] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.matches)


def _extract_ngram_hashes(
    text: str,
    min_n: int = 4,
    max_n: int = 8,
) -> set[int]:
    """
    Extract hashed fingerprints of all distinctive n-grams from *text*.

    This is the lightweight first pass of the two-pass algorithm — it avoids
    materialising tuple objects for n-grams that turn out to be unique.
    """
    tokens = _tokenise(_normalise(text))
    hashes: set[int] = set()
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            if _is_distinctive(gram):
                hashes.add(hash(gram))
    return hashes


# ── Timing helper ─────────────────────────────────────────────────────────────

def _print_timing(label: str, elapsed: float, file=None) -> None:
    """Print a timing line to the given file (default stderr)."""
    import sys
    f = file or sys.stderr
    print(f"  ⏱  {label}: {elapsed:.3f}s", file=f)


def _sentence_at_token_offset(
    sentences: list[str],
    cum_token_counts: list[int],
    token_offset: int,
    phrase_str: str,
) -> str:
    """
    Given pre-computed sentence boundaries (as cumulative token counts),
    find the sentence containing the n-gram that starts at *token_offset*
    and return it (truncated to ~300 chars with context around the phrase).

    Falls back to ``"(sentence not found)"`` on boundary misses.
    """
    import bisect

    # bisect_right returns the index *after* the insertion point, so -1
    # gives us the sentence whose cumulative token range includes offset.
    s_idx = bisect.bisect_right(cum_token_counts, token_offset) - 1
    if s_idx < 0 or s_idx >= len(sentences):
        return "(sentence not found)"

    sentence = sentences[s_idx]

    if len(sentence) <= 300:
        return sentence

    # Sentence is long — find the phrase position for smart truncation
    phrase_lower = phrase_str.lower()
    sent_lower = sentence.lower()
    pos = sent_lower.find(phrase_lower)
    if pos >= 0:
        start = max(0, pos - 80)
        end = min(len(sentence), pos + len(phrase_str) + 80)
    else:
        # The normalised phrase may not substring-match the original
        # (e.g. "45 cfr 170 315" vs "45 CFR § 170.315").  Use regex.
        pat = _phrase_to_pattern(phrase_str)
        m = pat.search(sentence)
        if m:
            start = max(0, m.start() - 80)
            end = min(len(sentence), m.end() + 80)
        else:
            # Phrase spans a sentence boundary — show start of sentence
            start = 0
            end = min(len(sentence), 300)
    return ("…" if start > 0 else "") + sentence[start:end] + ("…" if end < len(sentence) else "")


def find_repeated_phrases(
    comments: Sequence[GeneratedComment],
    min_n: int = 4,
    max_n: int = 8,
    min_count: int = 2,
    only_passed_qc: bool = True,
    verbose: bool = True,
) -> list[RepeatedPhrase]:
    """
    Scan a batch of comments and return all distinctive n-gram phrases that
    appear in *min_count* or more comments, sorted by frequency (descending).

    Uses a two-pass algorithm for efficiency:

      * **Pass 1** — hash each n-gram and count how many comments contain
        each hash.  Only hashes appearing in ≥ *min_count* comments survive.
      * **Pass 2** — re-scan comments, materialise full n-gram tuples only
        for hashes that passed the threshold, **recording the token offset**
        of the first match per comment.

    The token offsets are then used to look up the containing sentence in
    O(log S) time via a pre-computed cumulative-token-count index, instead
    of the previous O(S × regex) scan per (phrase, comment) pair.

    Parameters
    ----------
    comments:
        The batch of generated comments to analyse.
    min_n, max_n:
        N-gram window sizes to consider.
    min_count:
        Minimum number of distinct comments a phrase must appear in to be
        reported (default 2).
    only_passed_qc:
        If True, skip comments that failed QC.
    verbose:
        If True, print timing breakdown to stderr.

    Returns
    -------
    list[RepeatedPhrase]
        Phrases appearing in *min_count*+ comments, sorted by match count
        descending.
    """
    import sys
    t_total_start = time.perf_counter()

    # ── Preprocessing: build target list ───────────────────────────────────
    t0 = time.perf_counter()
    target = [
        (i, c) for i, c in enumerate(comments)
        if not only_passed_qc or c.qc_passed
    ]
    t_preprocess = time.perf_counter() - t0

    if verbose:
        _print_timing(f"Preprocessing ({len(target)} comments)", t_preprocess)

    # ── Pre-compute per-comment data (tokens, sentence index) ──────────────
    #
    # For each target comment we cache:
    #   tokens          – normalised word list (reused in Pass 1 + 2)
    #   sentences       – original sentences from _extract_sentences()
    #   cum_tok_counts  – cumulative token counts per sentence, so that
    #                     given a token offset we can binary-search the
    #                     sentence in O(log S) instead of scanning.
    #
    t0 = time.perf_counter()

    # idx → (tokens, sentences, cum_tok_counts)
    _comment_cache: dict[int, tuple[list[str], list[str], list[int]]] = {}

    for idx, comment in target:
        text = comment.comment_text
        tokens = _tokenise(_normalise(text))
        sentences = _extract_sentences(text)

        # Build cumulative token counts: cum[0]=0, cum[k] = tokens in
        # sentences 0..k-1.  Token offset i falls in sentence
        # bisect_right(cum, i) - 1.
        cum: list[int] = [0]
        for sent in sentences:
            n_tok = len(_tokenise(_normalise(sent)))
            cum.append(cum[-1] + n_tok)

        _comment_cache[idx] = (tokens, sentences, cum)

    t_cache = time.perf_counter() - t0
    if verbose:
        _print_timing(f"Pre-computing sentence index ({len(_comment_cache)} comments)", t_cache)

    # ── Pass 1: lightweight hash counting ──────────────────────────────────
    t0 = time.perf_counter()
    # hash → set of comment indices
    hash_to_indices: dict[int, set[int]] = defaultdict(set)

    for idx, _comment in target:
        tokens = _comment_cache[idx][0]
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                gram = tuple(tokens[i:i + n])
                if _is_distinctive(gram):
                    hash_to_indices[hash(gram)].add(idx)

    t_hash = time.perf_counter() - t0
    if verbose:
        _print_timing(f"Pass 1 — hashing ({len(hash_to_indices)} unique hashes)", t_hash)

    # ── Filter: keep only hashes appearing in min_count+ comments ──────────
    t0 = time.perf_counter()
    candidate_hashes: set[int] = {
        h for h, indices in hash_to_indices.items()
        if len(indices) >= min_count
    }
    del hash_to_indices  # free memory
    t_filter = time.perf_counter() - t0

    if verbose:
        _print_timing(f"Filtering duplicate hashes ({len(candidate_hashes)} candidates)", t_filter)

    if not candidate_hashes:
        if verbose:
            _print_timing("Total", time.perf_counter() - t_total_start)
        return []

    # ── Pass 2: materialise candidate n-grams + record token offsets ───────
    #
    # phrase_to_positions maps each repeated n-gram to the set of comments
    # it appears in AND the token offset of its first occurrence in each
    # comment.  The offset is free — we already have `i` from the loop.
    #
    t0 = time.perf_counter()
    # gram → { comment_idx: token_start_position }
    phrase_to_positions: dict[tuple[str, ...], dict[int, int]] = defaultdict(dict)

    for idx, _comment in target:
        tokens = _comment_cache[idx][0]
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                gram = tuple(tokens[i:i + n])
                if hash(gram) in candidate_hashes and _is_distinctive(gram):
                    # Store only the first occurrence per comment
                    if idx not in phrase_to_positions[gram]:
                        phrase_to_positions[gram][idx] = i

    t_pass2 = time.perf_counter() - t0
    if verbose:
        _print_timing(f"Pass 2 — materialising + offsets ({len(phrase_to_positions)} phrases)", t_pass2)

    # ── Build results with O(log S) sentence lookups ───────────────────────
    t0 = time.perf_counter()
    repeated: list[RepeatedPhrase] = []

    for gram, positions in phrase_to_positions.items():
        if len(positions) < min_count:
            continue

        phrase_str = " ".join(gram)
        rp = RepeatedPhrase(phrase=phrase_str, ngram_length=len(gram))

        for idx in sorted(positions):
            comment = comments[idx]

            # Use the actual Document ID from the PSV file; fall back only
            # when no document_id was loaded (e.g. in-memory generation).
            comment_id = comment.document_id or f"IDX-{idx}"

            submitter_name = _build_submitter_name(comment.persona)
            submitter_detail = _build_submitter_detail(comment.persona)

            # Fast sentence lookup using pre-computed index + token offset
            _tokens, sentences, cum_tok_counts = _comment_cache[idx]
            tok_offset = positions[idx]
            sentence = _sentence_at_token_offset(
                sentences, cum_tok_counts, tok_offset, phrase_str,
            )

            rp.matches.append(PhraseMatch(
                comment_index=idx,
                comment_id=comment_id,
                persona_name=submitter_name,
                persona_detail=submitter_detail,
                sentence=sentence,
            ))

        repeated.append(rp)

    t_results = time.perf_counter() - t0
    if verbose:
        _print_timing(f"Building results + sentence lookup ({len(repeated)} repeated phrases)", t_results)

    del _comment_cache  # free memory

    # De-duplicate overlapping n-grams: if a longer phrase subsumes a shorter
    # one and they have the exact same set of comment indices, keep only the
    # longer one.
    t0 = time.perf_counter()
    repeated = _deduplicate_subsumed(repeated)
    t_dedup = time.perf_counter() - t0

    if verbose:
        _print_timing("Deduplication", t_dedup)

    # Sort by count descending, then alphabetically
    repeated.sort(key=lambda rp: (-rp.count, rp.phrase))

    t_total = time.perf_counter() - t_total_start
    if verbose:
        _print_timing("Total", t_total)

    return repeated


def _deduplicate_subsumed(phrases: list[RepeatedPhrase]) -> list[RepeatedPhrase]:
    """
    Remove shorter n-grams that are fully subsumed by a longer n-gram with
    the exact same set of matching comments.
    """
    if not phrases:
        return phrases

    # Group by the frozenset of comment indices
    by_indices: dict[frozenset[int], list[RepeatedPhrase]] = defaultdict(list)
    for rp in phrases:
        key = frozenset(m.comment_index for m in rp.matches)
        by_indices[key].append(rp)

    kept: list[RepeatedPhrase] = []
    for _key, group in by_indices.items():
        if len(group) == 1:
            kept.append(group[0])
            continue

        # Sort by n-gram length descending
        group.sort(key=lambda rp: -rp.ngram_length)

        # Keep only phrases that are not a substring of a longer kept phrase
        retained: list[RepeatedPhrase] = []
        for rp in group:
            is_subsumed = any(
                rp.phrase in longer.phrase
                for longer in retained
            )
            if not is_subsumed:
                retained.append(rp)

        kept.extend(retained)

    return kept


# ── Report generation ─────────────────────────────────────────────────────────

def _escape_md_table(s: str) -> str:
    """Escape pipe characters for Markdown table cells."""
    return s.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def generate_report(
    repeated: list[RepeatedPhrase],
    total_comments: int,
) -> str:
    """
    Generate a human-readable Markdown report from the analysis results.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = []
    lines.append("# Phrase Repetition Report")
    lines.append(f"Generated: {now}")
    lines.append(f"Comments analysed: {total_comments} | "
                 f"Repeated phrases found: {len(repeated)}")
    lines.append("")

    if not repeated:
        lines.append("✅ **No repeated distinctive phrases detected.** The batch looks clean.")
        return "\n".join(lines)

    lines.append("⚠️  Repeated phrases may indicate AI-generated templating. "
                 "Review the sentences below to decide if rewording is needed.")
    lines.append("")

    for i, rp in enumerate(repeated, 1):
        lines.append("---")
        lines.append("")
        lines.append(f'## {i}. "{rp.phrase}" (found in {rp.count} comments)')
        lines.append("")
        lines.append("| # | Comment ID | Submitter | Sentence |")
        lines.append("|---|-----------|---------|----------|")

        for j, match in enumerate(rp.matches, 1):
            if match.persona_detail:
                submitter_col = f"{_escape_md_table(match.persona_name)} ({_escape_md_table(match.persona_detail)})"
            else:
                submitter_col = _escape_md_table(match.persona_name)
            sentence_col = _escape_md_table(match.sentence)
            lines.append(
                f"| {j} | {match.comment_id} | {submitter_col} | {sentence_col} |"
            )

        lines.append("")

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def stream_report(
    repeated: list[RepeatedPhrase],
    total_comments: int,
    output_path: str,
) -> None:
    """
    Stream the Markdown report directly to a file instead of building a
    giant string in memory.  Functionally identical to :func:`generate_report`
    but O(1) memory overhead regardless of report size.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Phrase Repetition Report\n")
        f.write(f"Generated: {now}\n")
        f.write(f"Comments analysed: {total_comments} | "
                f"Repeated phrases found: {len(repeated)}\n\n")

        if not repeated:
            f.write("✅ **No repeated distinctive phrases detected.** The batch looks clean.\n")
            return

        f.write("⚠️  Repeated phrases may indicate AI-generated templating. "
                "Review the sentences below to decide if rewording is needed.\n\n")

        for i, rp in enumerate(repeated, 1):
            f.write("---\n\n")
            f.write(f'## {i}. "{rp.phrase}" (found in {rp.count} comments)\n\n')
            f.write("| # | Comment ID | Submitter | Sentence |\n")
            f.write("|---|-----------|---------|----------|\n")

            for j, match in enumerate(rp.matches, 1):
                if match.persona_detail:
                    submitter_col = f"{_escape_md_table(match.persona_name)} ({_escape_md_table(match.persona_detail)})"
                else:
                    submitter_col = _escape_md_table(match.persona_name)
                sentence_col = _escape_md_table(match.sentence)
                f.write(f"| {j} | {match.comment_id} | {submitter_col} | {sentence_col} |\n")

            f.write("\n")


def run_phrase_check(
    comments: Sequence[GeneratedComment],
    output_path: str,
    min_n: int = 4,
    max_n: int = 8,
    min_count: int = 2,
    only_passed_qc: bool = True,
    verbose: bool = True,
) -> str:
    """
    Run the batch phrase-repetition check and write a Markdown report.

    Parameters
    ----------
    comments:
        The batch of generated comments to analyse.
    output_path:
        Path for the output report (Markdown).
    min_n, max_n:
        N-gram window sizes to consider (default 4–8).
    min_count:
        Minimum number of distinct comments a phrase must appear in to be
        reported (default 2).
    only_passed_qc:
        If True, only analyse comments that passed QC.
    verbose:
        Print summary to stderr.

    Returns
    -------
    str
        The output_path where the report was written.
    """
    import sys

    t_total_start = time.perf_counter()

    target_count = sum(
        1 for c in comments
        if not only_passed_qc or c.qc_passed
    )

    repeated = find_repeated_phrases(
        comments, min_n, max_n, min_count=min_count,
        only_passed_qc=only_passed_qc,
        verbose=verbose,
    )

    # Stream directly to file — avoids building a 100K+ line string in memory
    t0 = time.perf_counter()
    stream_report(repeated, target_count, output_path)
    t_write = time.perf_counter() - t0

    if verbose:
        _print_timing("Writing report", t_write)

        t_total = time.perf_counter() - t_total_start
        _print_timing("Overall run_phrase_check", t_total)

        if repeated:
            print(
                f"      Phrase check: {len(repeated)} repeated phrase(s) "
                f"found across {target_count} comments → {output_path}",
                file=sys.stderr,
            )
        else:
            print(
                f"      Phrase check: no repeated phrases in "
                f"{target_count} comments ✓",
                file=sys.stderr,
            )

    return output_path


# ── PSV file loading ──────────────────────────────────────────────────────────

def _load_comments_from_psv(psv_path: str) -> list[GeneratedComment]:
    """
    Read a ♔-delimited PSV file and return lightweight GeneratedComment stubs
    suitable for phrase-check analysis.

    Auto-detects two formats:

    * **CMS format** (e.g. ``synthetic_cms.psv``): columns include
      ``Comment``, ``First Name``, ``Last Name``, ``State/Province``,
      ``Document ID``.
    * **Syncom export format** (e.g. ``synthetic.txt``): columns include
      ``Comment``, ``Submitter Name``, ``synth_persona_state``,
      ``synth_persona_occupation``, ``Comment ID``.
    """
    import sys
    from pathlib import Path

    # Allow running from repo root without installing shuffler as a package
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from shuffler.psv_io import read_psv
    from .persona import Persona
    from .argument_mapper import ExpressionFrame

    t0 = time.perf_counter()
    rows, fieldnames = read_psv(psv_path)
    t_read = time.perf_counter() - t0
    _print_timing(f"Reading PSV ({len(rows)} rows)", t_read)

    if not rows:
        return []

    # ── Detect format ──────────────────────────────────────────────────────
    has_cms_cols = "First Name" in fieldnames and "Last Name" in fieldnames
    has_syncom_cols = "Submitter Name" in fieldnames

    comments: list[GeneratedComment] = []

    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        # Extract comment text
        text = row.get("Comment", "")
        if not text.strip():
            continue

        # The Document ID from the PSV is the authoritative comment identifier
        document_id = row.get("Document ID", "")

        # Extract metadata depending on format
        if has_cms_cols:
            first_name = row.get("First Name", "")
            last_name = row.get("Last Name", "")
            state = row.get("State/Province", "")
            occupation = row.get("synth_persona_occupation", "")
            docket_id = row.get("Docket ID", "")
            org_name = row.get("Organization Name", "")
        elif has_syncom_cols:
            full = row.get("Submitter Name", "")
            parts = full.strip().split(None, 1)
            first_name = parts[0] if parts else ""
            last_name = parts[1] if len(parts) > 1 else ""
            state = row.get("synth_persona_state", "")
            occupation = row.get("synth_persona_occupation", "")
            docket_id = row.get("Document ID", "")
            org_name = row.get("Organization Name", "")
        else:
            # Best-effort fallback
            first_name = row.get("First Name", f"Row{i+1}")
            last_name = row.get("Last Name", "")
            state = row.get("State/Province", "")
            occupation = ""
            docket_id = ""
            org_name = ""

        persona = Persona(
            archetype="individual_consumer",
            first_name=first_name or "",
            last_name=last_name or "",
            state=state,
            occupation=occupation,
            age=0,
            sophistication="medium",
            emotional_register="concerned",
            org_name=org_name,
            personal_stake="",
            personal_hook="",
        )

        frame = ExpressionFrame(
            core_arguments=[],
            framing="",
            evidence_types=[],
            rfi_questions_to_address=[],
        )

        gc = GeneratedComment(
            comment_text=text,
            persona=persona,
            frame=frame,
            vector=0,
            objective="",
            rule_title="",
            docket_id=docket_id,
            document_id=document_id,
            qc_passed=True,
        )
        comments.append(gc)

    t_build = time.perf_counter() - t0
    _print_timing(f"Building {len(comments)} comment objects", t_build)

    return comments


def run_phrase_check_on_psv(
    psv_path: str,
    output_path: str | None = None,
    min_n: int = 4,
    max_n: int = 8,
    min_count: int = 2,
    verbose: bool = True,
) -> str:
    """
    Load comments from a ♔-delimited PSV file and run the phrase-repetition
    check, writing a Markdown report.

    Parameters
    ----------
    psv_path:
        Path to the input PSV file (syncom export or CMS format).
    output_path:
        Path for the Markdown report.  If None, derives from psv_path
        (e.g. ``synthetic_cms.phrase_report.md``).
    min_n, max_n:
        N-gram window sizes to consider (default 4–8).
    min_count:
        Minimum number of distinct comments a phrase must appear in to be
        reported (default 2).
    verbose:
        Print progress to stderr.

    Returns
    -------
    str
        The output_path where the report was written.
    """
    import sys
    from pathlib import Path

    if output_path is None:
        p = Path(psv_path)
        output_path = str(p.with_suffix(".phrase_report.md"))

    if verbose:
        print(f"[phrase-check] Loading comments from {psv_path} …", file=sys.stderr)

    t0 = time.perf_counter()
    comments = _load_comments_from_psv(psv_path)
    t_load = time.perf_counter() - t0

    if verbose:
        print(f"[phrase-check] Loaded {len(comments)} comments in {t_load:.3f}s", file=sys.stderr)

    return run_phrase_check(
        comments=comments,
        output_path=output_path,
        min_n=min_n,
        max_n=max_n,
        min_count=min_count,
        only_passed_qc=False,   # PSV rows are already filtered; check all
        verbose=verbose,
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """
    CLI for running phrase-repetition checks on PSV files.

    Usage::

        python -m syncom.phrase_check \\
            --input  CMS-2025-0050/shuffled_comments/synthetic_cms.psv \\
            --output phrase_report.md
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m syncom.phrase_check",
        description=(
            "Scan a ♔-delimited PSV file of comments for repeated distinctive "
            "phrases and write a human-readable Markdown report."
        ),
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="PATH",
        help="Input PSV file (syncom export or CMS format).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="PATH",
        help=(
            "Output Markdown report path.  "
            "Default: <input_stem>.phrase_report.md"
        ),
    )
    parser.add_argument(
        "--min-n",
        type=int, default=4,
        metavar="N",
        help="Minimum n-gram length (default 4).",
    )
    parser.add_argument(
        "--max-n",
        type=int, default=8,
        metavar="N",
        help="Maximum n-gram length (default 8).",
    )
    parser.add_argument(
        "--min-count",
        type=int, default=2,
        metavar="N",
        help=(
            "Minimum number of comments a phrase must appear in to be "
            "reported (default 2).  Raise to 3+ for a shorter, more "
            "actionable report."
        ),
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args(argv)

    import os
    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        report_path = run_phrase_check_on_psv(
            psv_path=args.input,
            output_path=args.output,
            min_n=args.min_n,
            max_n=args.max_n,
            min_count=args.min_count,
            verbose=not args.quiet,
        )
        if not args.quiet:
            print(f"[phrase-check] Report written to {report_path}", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
