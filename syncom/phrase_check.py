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


# ── Watch list: known LLM-favorite short phrases ─────────────────────────────
#
# These are short phrases (typically 2–3 words) that LLMs reach for
# repeatedly as narrative anchors.  They are too short to be caught by the
# standard n-gram analysis (min_n=4) but are strong AI-generation signals
# when they appear in a high fraction of comments.
#
# The watch list is checked separately from the n-gram analysis.  Any phrase
# here that appears in ≥ WATCH_LIST_MIN_COMMENTS distinct comments is
# included in the phrase report as a WATCH_LIST hit.
#
# Grow this list as new patterns are discovered across dockets.

WATCH_LIST: list[str] = [
    # Temporal narrative anchors — the LLM's favourite season is spring
    "last spring",
    "last summer",
    "last fall",
    "last winter",
    "last year",
    "last month",
    "last week",
    "a few years ago",
    "a few months ago",
    "a couple of years ago",
    "a couple of months ago",
    # Narrative openers
    "i keep coming back to",
    "coming back to",
    "i genuinely don't understand",
    "i genuinely don't",
    # Liability / accountability clichés
    "patient gets hurt",
    "who is responsible",
    "who exactly is responsible",
    "the doctor blames the",
]

# Minimum number of distinct comments a watch-list phrase must appear in
# to be included in the report.
WATCH_LIST_MIN_COMMENTS: int = 3
MIN_REPEATS_DEFAULT = 3


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

# Matches any digit character — used by _gram_has_digit.
_DIGIT_RE = re.compile(r"\d")

# Matches original-case words (letters, digits, apostrophes) for the
# mid-sentence capitalisation scanner.
_ORIG_WORD_RE = re.compile(r"[A-Za-z0-9']+")


# ── Public-comment boilerplate n-gram index ───────────────────────────────────
#
# Frozenset of 2–4 gram patterns (normalised, lowercase) that are universally
# boilerplate in regulatory public-comment filings.  Any n-gram whose token
# sequence contains one of these patterns as a consecutive sub-window is
# suppressed in Pass 1 of :func:`find_repeated_phrases`.
#
# These patterns cover:
#   • Courtesy / gratitude openers  ("the opportunity to provide feedback")
#   • Contact / closing phrases     ("please feel free to contact us")
#   • Submission procedural language ("submitted electronically via")
#   • Address / letterhead fragments ("independence avenue", "deputy secretary")
#   • URL / citation fragments       ("doi org")
#   • General public-comment terms   ("comment period")
#
# They do NOT match the useful LLM-signature phrases such as
# "one size fits all", "to keep pace with", or "patchwork of state laws".
#
# Extend this list as new docket-agnostic boilerplate patterns are discovered.

PUBLIC_COMMENT_BOILERPLATE_NGRAMS: frozenset[tuple[str, ...]] = frozenset({
    # ── Courtesy / gratitude ────────────────────────────────────────────────
    ("opportunity", "to"),                              # the opportunity to [provide/comment/respond]
    ("appreciate", "the", "opportunity"),               # we/I appreciate the opportunity
    ("appreciates", "the", "opportunity"),              # [org] appreciates the opportunity
    ("welcome", "the", "opportunity"),                  # we welcome the opportunity
    ("welcomes", "the", "opportunity"),
    ("look", "forward", "to"),                          # we look forward to
    ("looks", "forward", "to"),
    ("forward", "to", "working"),
    ("thank", "you", "for", "considering"),             # thank you for considering
    ("thank", "you", "for", "your"),                    # thank you for your consideration
    # ── Contact / closing phrases ───────────────────────────────────────────
    ("please", "feel", "free"),                         # please feel free to contact
    ("feel", "free", "to", "contact"),                  # feel free to contact us
    ("hesitate", "to", "contact"),                      # do not hesitate to contact
    ("not", "hesitate", "to", "contact"),               # please do not hesitate to contact
    ("additional", "information", "please", "contact"), # for additional information please contact
    ("have", "questions", "please"),
    ("forward", "to", "continued"),
    ("to", "continued", "collaboration"),
    # ── Submission / procedural phrases ────────────────────────────────────
    ("submitted", "electronically", "via"),             # submitted electronically via regulations.gov
    ("electronically", "via", "regulations"),           # electronically via regulations.gov
    ("behalf", "of"),                                   # on behalf of [org] (intro to org position)
    ("submits", "these", "comments"),                   # [org] submits these comments
    ("submitting", "these", "comments"),                # [org] is submitting these comments
    ("comments", "in", "response", "to"),               # these comments in response to [rule/RFI]
    ("these", "comments", "in", "response"),            # submitting these comments in response to
    ("is", "pleased", "to", "submit"), 
    # ── Address / letterhead fragments ─────────────────────────────────────
    ("independence", "avenue"),                         # Independence Avenue (physical address)
    ("department", "of", "health", "and"),              # Department of Health and Human Services
    ("health", "and", "human", "services"),             # Health and Human Services
    ("office", "of", "the", "deputy"),                  # Office of the Deputy Secretary
    ("deputy", "secretary"),                            # Deputy Secretary [name] (addressee)
    # ── URL / DOI citation fragments, web addresses ────────────────────────────────────────
    ("doi", "org"), 
    ("http", "doi"),
    ("http", "www"),
    # ── General public-comment procedural terms ─────────────────────────────
    ("for", "public", "comment"),                       # for public comment
    ("comment", "period"),                              # public comment period
    # ── Terms of art (domain specific) ─────────────────────────────
    ("clinical", "decision", "support"),
    ("electronic", "health", "records"),
    ("large", "language", "models"),
    ("large", "language", "model"),
})


def _gram_has_digit(gram: tuple[str, ...]) -> bool:
    """
    Return True if any token in *gram* contains a digit.

    Filters out docket numbers ("hhs onc 2026 0001"), CFR citations
    ("45 cfr part 170"), street addresses ("200 independence avenue sw"),
    ZIP codes ("washington d c 20201"), date fragments ("october 14 2024"),
    page footers ("page 1 of 3"), and year-based references ("21st century
    cures act").  None of the LLM-signature phrases of interest contain digits.
    """
    return any(_DIGIT_RE.search(tok) for tok in gram)


def _find_mid_sentence_cap_positions(sentence: str) -> set[int]:
    """
    Return the set of LOCAL normalised-token offsets within *sentence* where
    the corresponding original word was capitalised and was **not** the first
    word of the sentence.

    Both ALL-CAPS acronyms (``HHS``, ``EHR``, ``AI``, ``FHIR``) and title-case
    proper nouns (``Secretary Keane``, ``American Medical Association``,
    ``Independence Avenue``) are detected.  The first word of the sentence is
    always exempt — sentence-initial capitalisation is grammatically required
    and carries no signal.

    A single original word may produce more than one normalised token when it
    contains a hyphen or other punctuation stripped by :func:`_normalise`
    (e.g. ``"EHR-based"`` → ``["ehr", "based"]``).  In that case all
    resulting normalised tokens are marked.

    The returned set uses positions *local* to this sentence (starting at 0).
    Callers must add the sentence's global token offset to obtain positions
    in the comment-level normalised token sequence.
    """
    orig_words = _ORIG_WORD_RE.findall(sentence)
    if not orig_words:
        return set()

    flagged: set[int] = set()
    norm_pos = 0  # running offset into this sentence's normalised token sequence

    for orig_idx, orig_word in enumerate(orig_words):
        # Count how many normalised tokens this original word produces.
        normalised = _normalise(orig_word)
        parts = normalised.split() if normalised.strip() else []
        n_tok = len(parts)

        # Flag all normalised positions for mid-sentence-capitalised words.
        # orig_idx == 0 is the sentence-initial word → always exempt.
        if orig_idx > 0 and any(c.isupper() for c in orig_word):
            for j in range(n_tok):
                flagged.add(norm_pos + j)

        norm_pos += n_tok

    return flagged


def _gram_is_comment_boilerplate(
    gram: tuple[str, ...],
    min_n: int = 2,
    max_n: int = 4,
) -> bool:
    """
    Return True if any min_n..max_n consecutive sub-window of *gram* matches
    a pattern in :data:`PUBLIC_COMMENT_BOILERPLATE_NGRAMS`.

    This mirrors the structure of :func:`_gram_is_rule_anchored` and filters
    all-lowercase procedural phrases that the capitalisation and digit filters
    cannot catch:  courtesy openers ("the opportunity to provide"),
    contact closings ("please feel free to contact"), submission language
    ("submitted electronically via"), address fragments, and procedural terms.
    """
    n = len(gram)
    for sub_n in range(min_n, min(max_n, n) + 1):
        for i in range(n - sub_n + 1):
            if gram[i : i + sub_n] in PUBLIC_COMMENT_BOILERPLATE_NGRAMS:
                return True
    return False


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
    max_n: int = 4,
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
    slop_type: str = "unknown"   # "slopical" | "topical" | "unknown"

    @property
    def count(self) -> int:
        return len(self.matches)


def _gram_is_rule_anchored(
    gram: tuple[str, ...],
    rule_ngrams: "frozenset[tuple[str, ...]]",
    min_n: int = 2,
    max_n: int = 4,
) -> bool:
    """
    Return True if any min_n..max_n sub-window of *gram* is present in
    *rule_ngrams*.

    This mirrors the logic of :func:`~syncom.phrase_fix.is_rule_anchored` but
    operates directly on an already-tokenised tuple, avoiding the
    string→normalise→tokenise round-trip inside the tight Pass-1 inner loop.

    A 6-word comment phrase such as ``("national", "coordinator", "for",
    "health", "information", "technology")`` is rule-anchored because the
    4-gram ``("coordinator", "for", "health", "information")`` (or similar)
    appears in the rule vocabulary, even though the full 6-gram itself is not
    stored there.
    """
    n = len(gram)
    for sub_n in range(min_n, min(max_n, n) + 1):
        for i in range(n - sub_n + 1):
            if gram[i : i + sub_n] in rule_ngrams:
                return True
    return False


def _extract_ngram_hashes(
    text: str,
    min_n: int = 4,
    max_n: int = 4,
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


def find_watch_list_phrases(
    comments: Sequence[GeneratedComment],
    only_passed_qc: bool = True,
    min_count: int = WATCH_LIST_MIN_COMMENTS,
    rule_ngrams: "frozenset[tuple[str, ...]] | None" = None,
) -> list[RepeatedPhrase]:
    """
    Scan comments for occurrences of phrases on the :data:`WATCH_LIST`.

    Returns a list of :class:`RepeatedPhrase` objects (one per watch-list
    phrase that appears in *min_count* or more distinct comments), sorted
    by match count descending.

    This is a separate, lightweight scan that runs in addition to the
    standard n-gram analysis.  It catches short phrases (2–3 words) that
    are too short for the n-gram analysis but are known LLM-favourite
    narrative anchors.

    Parameters
    ----------
    rule_ngrams:
        Optional frozenset of n-gram tuples from the rule text (built by
        :func:`~syncom.phrase_fix.build_rule_ngrams`).  When provided, any
        watch-list phrase whose full normalised token tuple appears in
        *rule_ngrams* is silently skipped — it is rule-anchored vocabulary
        rather than a novel AI-generation signal.
    """
    target = [
        (i, c) for i, c in enumerate(comments)
        if not only_passed_qc or c.qc_passed
    ]

    # Normalised watch-list phrases for fast matching.
    # Skip any watch-list entry whose token n-gram is in the rule vocabulary.
    normalised_watch = [
        (phrase, _normalise(phrase))
        for phrase in WATCH_LIST
        if rule_ngrams is None
        or tuple(_tokenise(_normalise(phrase))) not in rule_ngrams
    ]

    # phrase → { comment_idx: sentence }
    phrase_to_hits: dict[str, dict[int, str]] = {p: {} for p, _ in normalised_watch}

    for idx, comment in target:
        text = comment.comment_text
        norm_text = _normalise(text)
        # Defense-in-depth: also check the personal_hook / institutional anchor
        # field.  If a watch-list phrase slipped through the upstream LLM guard
        # it will still appear here and be flagged in the report.
        hook_text = getattr(comment.persona, "personal_hook", "") or ""
        norm_hook = _normalise(hook_text) if hook_text else ""

        for phrase, norm_phrase in normalised_watch:
            if idx in phrase_to_hits[phrase]:
                continue
            if norm_phrase in norm_text:
                sentence = _find_sentence_containing(text, norm_phrase)
                phrase_to_hits[phrase][idx] = sentence
            elif norm_phrase in norm_hook:
                sentence = _find_sentence_containing(hook_text, norm_phrase)
                if sentence != "(sentence not found)":
                    phrase_to_hits[phrase][idx] = f"[synth_personal_hook] {sentence}"
                else:
                    phrase_to_hits[phrase][idx] = "[synth_personal_hook] (sentence not found)"

    repeated: list[RepeatedPhrase] = []
    for phrase, hits in phrase_to_hits.items():
        if len(hits) < min_count:
            continue

        rp = RepeatedPhrase(phrase=phrase, ngram_length=len(phrase.split()))
        for idx in sorted(hits):
            comment = comments[idx]
            suffix = str(idx + 1).zfill(4)
            comment_id = comment.document_id or f"SYNTH-{suffix}"

            rp.matches.append(PhraseMatch(
                comment_index=idx,
                comment_id=comment_id,
                persona_name=_build_submitter_name(comment.persona),
                persona_detail=_build_submitter_detail(comment.persona),
                sentence=hits[idx],
            ))
        repeated.append(rp)

    repeated.sort(key=lambda rp: (-rp.count, rp.phrase))
    return repeated


def find_repeated_phrases(
    comments: Sequence[GeneratedComment],
    min_n: int = 4,
    max_n: int = 4,
    min_count: int = MIN_REPEATS_DEFAULT,
    only_passed_qc: bool = True,
    verbose: bool = True,
    rule_ngrams: "frozenset[tuple[str, ...]] | None" = None,
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
        reported (default MIN_REPEATS_DEFAULT).
    only_passed_qc:
        If True, skip comments that failed QC.
    verbose:
        If True, print timing breakdown to stderr.
    rule_ngrams:
        Optional frozenset of n-gram tuples from the rule text (built by
        :func:`~syncom.phrase_fix.build_rule_ngrams`).  When provided, any
        n-gram whose tuple is an exact member of this set is skipped in
        Pass 1, so rule-anchored vocabulary never enters the candidate set
        and never appears in the final report.  This is more conservative
        than the fuzzy sub-sequence check in :func:`~syncom.phrase_fix.is_rule_anchored`
        — only exact matches are suppressed here; the downstream triage
        handles partial matches.

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

    # ── Pre-compute per-comment data (tokens, sentence index, cap map) ─────
    #
    # For each target comment we cache:
    #   tokens           – normalised word list (reused in Pass 1 + 2)
    #   sentences        – original sentences from _extract_sentences()
    #   cum_tok_counts   – cumulative token counts per sentence, so that
    #                      given a token offset we can binary-search the
    #                      sentence in O(log S) instead of scanning.
    #   mid_cap_positions – set of global token offsets whose corresponding
    #                      original word was capitalised and NOT the first
    #                      word of its sentence.  Built by
    #                      _find_mid_sentence_cap_positions() sentence-by-
    #                      sentence, then accumulated with the running global
    #                      token offset.  Used in Pass 1 to suppress any
    #                      n-gram containing an acronym or proper noun.
    #
    t0 = time.perf_counter()

    # idx → (tokens, sentences, cum_tok_counts, mid_cap_positions)
    _comment_cache: dict[int, tuple[list[str], list[str], list[int], set[int]]] = {}

    for idx, comment in target:
        text = comment.comment_text
        tokens = _tokenise(_normalise(text))
        sentences = _extract_sentences(text)

        # Build cumulative token counts AND mid-sentence-cap position set.
        # We process sentence-by-sentence so we can call
        # _find_mid_sentence_cap_positions() on each sentence and add its
        # local offsets to the global offset for that sentence.
        mid_cap_positions: set[int] = set()
        cum: list[int] = [0]
        global_tok_pos = 0
        for sent in sentences:
            n_tok = len(_tokenise(_normalise(sent)))
            # Accumulate mid-sentence-cap positions with the global offset.
            for local_off in _find_mid_sentence_cap_positions(sent):
                mid_cap_positions.add(global_tok_pos + local_off)
            global_tok_pos += n_tok
            cum.append(cum[-1] + n_tok)

        _comment_cache[idx] = (tokens, sentences, cum, mid_cap_positions)

    t_cache = time.perf_counter() - t0
    if verbose:
        _print_timing(f"Pre-computing sentence index ({len(_comment_cache)} comments)", t_cache)

    # ── Pass 1: lightweight hash counting ──────────────────────────────────
    #
    # When rule_ngrams is provided, n-grams that appear verbatim in the rule
    # vocabulary are skipped before hashing.  This prevents rule-anchored
    # regulatory phrases from ever becoming candidates, keeping the report
    # focused on novel personal phrases that are the real AI-generation signal.
    #
    t0 = time.perf_counter()
    # hash → set of comment indices
    hash_to_indices: dict[int, set[int]] = defaultdict(set)

    for idx, _comment in target:
        tokens, _sentences, _cum, mid_cap_positions = _comment_cache[idx]
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                gram = tuple(tokens[i:i + n])
                if _is_distinctive(gram):
                    # ── Multi-layer noise filters ────────────────────────
                    # Each filter is ordered cheapest → most expensive.
                    # A gram is dropped on the first filter that fires.

                    # 1. Rule-anchored: phrase overlaps the rule vocabulary
                    if rule_ngrams is not None and _gram_is_rule_anchored(gram, rule_ngrams):
                        continue

                    # 2. Digit filter: any token contains a digit.
                    #    Catches docket numbers, CFR citations, addresses,
                    #    ZIP codes, date fragments, year references, etc.
                    if _gram_has_digit(gram):
                        continue

                    # 3. Mid-sentence capitalisation filter: any token in
                    #    this gram came from a capitalised original word
                    #    that was NOT sentence-initial.  This catches both
                    #    ALL-CAPS acronyms (HHS, EHR, AI, FHIR) and title-
                    #    case proper nouns (Secretary Keane, American
                    #    Medical Association, Independence Avenue) without
                    #    needing a per-docket vocabulary.
                    if any((i + j) in mid_cap_positions for j in range(n)):
                        continue

                    # 4. Public-comment boilerplate: the gram contains a
                    #    known 2–4 token boilerplate sub-sequence.  Catches
                    #    all-lowercase procedural phrases that the digit and
                    #    capitalisation filters cannot reach ("the
                    #    opportunity to provide", "on behalf of", etc.).
                    if _gram_is_comment_boilerplate(gram):
                        continue

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
            comment_id = comment.document_id or f"SYNTH-{str(idx + 1).zfill(4)}"

            submitter_name = _build_submitter_name(comment.persona)
            submitter_detail = _build_submitter_detail(comment.persona)

            # Fast sentence lookup using pre-computed index + token offset
            _tokens, sentences, cum_tok_counts, _mid_cap = _comment_cache[idx]
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
    Remove phrases that are subsumed by a broader phrase already in the report.

    Phrase B is considered subsumed by phrase A when **both** conditions hold:

    1. **A's comment set is a superset of B's comment set** — A appears in
       every comment where B appears (and possibly more).  Identical comment
       sets (the old behaviour) are trivially a special case of this.

    2. **The phrases are textually related** — either:

       a. *Substring containment*: one phrase contains the other as a
          substring  (e.g. ``"a risk based framework"`` is a substring of
          ``"a risk based framework for"``), OR

       b. *Adjacent n-gram overlap*: the phrases share at least ``n − 1``
          consecutive tokens, i.e. they are neighbouring sliding-window
          n-grams over the same underlying text.  For example the 4-grams
          ``"a risk based framework"`` and ``"risk based framework for"``
          share 3 tokens — a suffix of one equals a prefix of the other.

    When B is subsumed by A, B is **dropped**.  A is kept because it covers
    at least the same comments (often more) and is therefore strictly more
    informative than B for detecting AI-generated templating.

    Phrases are processed broadest-first (largest comment set, then longest
    n-gram) so the most informative phrase always acts as the anchor that
    can subsume narrower, more-specific variants.
    """
    if not phrases:
        return phrases

    # Pre-compute comment-index frozensets and token lists for every phrase.
    phrase_data: list[tuple[RepeatedPhrase, frozenset[int], list[str]]] = [
        (rp, frozenset(m.comment_index for m in rp.matches), rp.phrase.split())
        for rp in phrases
    ]

    # Sort: broadest comment set first, then longest n-gram first.
    # Processing in this order means the most informative anchors are
    # evaluated before narrower variants, so a narrow variant will always
    # find an appropriate anchor to be subsumed by.
    phrase_data.sort(key=lambda x: (-len(x[1]), -x[0].ngram_length))

    kept: list[tuple[RepeatedPhrase, frozenset[int], list[str]]] = []

    for rp, idx_set, tokens in phrase_data:
        is_subsumed = False
        for kept_rp, kept_set, kept_tokens in kept:

            # Condition 1: kept phrase's comment set must be a superset.
            if not idx_set <= kept_set:
                continue

            # Condition 2a: one phrase is a substring of the other.
            if rp.phrase in kept_rp.phrase or kept_rp.phrase in rp.phrase:
                is_subsumed = True
                break

            # Condition 2b: adjacent n-gram overlap — a suffix of one
            # token list equals a prefix of the other (share n − 1 tokens).
            overlap_size = min(len(tokens), len(kept_tokens)) - 1
            if overlap_size > 0:
                if tokens[-overlap_size:] == kept_tokens[:overlap_size]:
                    is_subsumed = True
                    break
                if kept_tokens[-overlap_size:] == tokens[:overlap_size]:
                    is_subsumed = True
                    break

        if not is_subsumed:
            kept.append((rp, idx_set, tokens))

    return [rp for rp, _, _ in kept]


# ── Topicality scoring ────────────────────────────────────────────────────────
#
# Phrases are classified as "topical" or "slopical" using Log-Likelihood Ratio
# (LLR / G²) keyness scores built from the rule/RFI text.  LLR compares word
# frequencies in the rule against a general-English reference corpus (wordfreq),
# so it works on documents of any size — including a short 14 KB RFI preamble —
# and correctly identifies domain-specific vocabulary such as "interoperability",
# "algorithm", or "FHIR" regardless of how many sentences the document has.
#
# Percentile of non-zero LLR scores used as the topicality threshold.
# Phrases whose content-token average falls below this percentile are
# classified "slopical"; those at or above it are "topical".
# Range 0–100; lower values → more phrases classified as topical.
_TOPICALITY_PERCENTILE: int = 40

# Virtual reference corpus size used in the LLR 2×2 contingency table.
# wordfreq returns proportional frequencies; we scale them to this many tokens
# so we can compute expected counts without needing an actual reference corpus.
_LLR_REF_CORPUS_SIZE: int = 1_000_000


def build_keyness_scores(rule_text: str) -> "dict[str, float]":
    """
    Build a word-importance map from the rule/RFI text using Log-Likelihood
    Ratio (G²) keyness scoring (Dunning 1993).

    Each word's frequency in *rule_text* is compared against its frequency in
    general English (supplied by the ``wordfreq`` package).  Only words that
    are **over-represented** in the rule relative to general English receive a
    positive keyness score; these are the domain-specific terms that distinguish
    the rule's subject matter from everyday language.

    Returns a dict mapping vocabulary term → G² keyness score.  Higher scores
    indicate stronger domain-centrality.  Returns an empty dict if ``wordfreq``
    is unavailable or the text is too short (fewer than 50 tokens).

    Unlike TF-IDF, this approach:
    * Works on documents of **any** size — no sentence-splitting or ``min_df``
      constraint across internal sub-documents is needed.
    * Uses a stable, externally-grounded baseline (general English), so the
      results do not shift as the document grows or shrinks.
    * Explicitly distinguishes *over-represented* (topical) words from words
      that are merely frequent within the document but common in everyday
      language (e.g. "provide", "ensure", "important").
    """
    if not rule_text or not rule_text.strip():
        return {}

    try:
        from wordfreq import word_frequency
    except ImportError:
        return {}

    import math
    from collections import Counter

    # Tokenise: lowercase letters only, ≥ 3 characters, excluding stopwords.
    # Matches the token_pattern used in the former TfidfVectorizer call.
    tokens = [
        w for w in re.findall(r"[a-z]{3,}", rule_text.lower())
        if w not in _STOPWORDS
    ]

    if len(tokens) < 50:
        return {}

    N1 = len(tokens)                # total tokens in the rule text
    N2 = _LLR_REF_CORPUS_SIZE       # virtual reference corpus size

    counts: Counter = Counter(tokens)
    scores: dict[str, float] = {}

    def _safe_llr_term(O: float, E: float) -> float:
        return O * math.log(O / E) if O > 0 and E > 0 else 0.0

    for word, count in counts.items():
        if count < 2:               # must appear at least twice in the rule
            continue

        O1: float = float(count)
        ref_freq: float = word_frequency(word, "en")

        # Words absent from the reference corpus are treated as extremely rare
        # rather than zero, to avoid log(0) and to be conservative.
        if ref_freq == 0.0:
            ref_freq = 1e-9

        O2: float = ref_freq * N2   # expected reference count

        # Only retain words over-represented in the rule.
        # Words that are equally or more common in general English are not
        # domain-specific signals and should not influence topicality scoring.
        if O1 / N1 <= ref_freq:
            continue

        # Compute G² from the 2×2 contingency table:
        #
        #               rule     reference
        #   word        O1       O2
        #   other       N1-O1    N2-O2
        #
        E1 = N1 * (O1 + O2) / (N1 + N2)
        E2 = N2 * (O1 + O2) / (N1 + N2)

        g2 = 2.0 * (
            _safe_llr_term(O1,      E1) +
            _safe_llr_term(O2,      E2) +
            _safe_llr_term(N1 - O1, N1 - E1) +
            _safe_llr_term(N2 - O2, N2 - E2)
        )

        scores[word] = g2

    return scores


def classify_all_phrases(
    repeated: "list[RepeatedPhrase]",
    tfidf_scores: "dict[str, float]",
) -> None:
    """
    Classify each phrase in *repeated* as ``"topical"`` or ``"slopical"``
    by comparing its content-token average keyness score against a threshold
    derived from the :data:`_TOPICALITY_PERCENTILE` of the rule vocabulary.

    The *tfidf_scores* parameter accepts the dict returned by
    :func:`build_keyness_scores` (LLR / G² scores) — the parameter name is
    kept for backward-compatibility with callers that pass the dict through
    :func:`run_phrase_check`.

    A phrase whose content tokens average above the threshold uses vocabulary
    that is over-represented vs. general English in the rule's subject matter
    → **topical**.  A phrase that uses generic words not prominent in the rule
    → **slopical** (a domain-agnostic AI-generation signal).

    Mutates :attr:`RepeatedPhrase.slop_type` in-place.  Phrases with no
    scoreable content tokens (all stopwords or very short words) are marked
    "slopical".
    """
    if not tfidf_scores or not repeated:
        return

    # Pre-compute the threshold from the distribution of non-zero scores.
    nonzero = sorted(v for v in tfidf_scores.values() if v > 0)
    if not nonzero:
        return
    p_idx = max(0, min(len(nonzero) - 1,
                       int(len(nonzero) * _TOPICALITY_PERCENTILE / 100)))
    threshold = nonzero[p_idx]

    for rp in repeated:
        tokens = _tokenise(_normalise(rp.phrase))
        # Content tokens: not a stopword, ≥ 3 characters (matches vectorizer)
        content = [t for t in tokens if t not in _STOPWORDS and len(t) >= 3]
        if not content:
            rp.slop_type = "slopical"
            continue

        # Average ONLY over content tokens that have a positive keyness score.
        #
        # Tokens absent from the scores dict (either not in the rule at all, or
        # present but filtered out by the count threshold) get 0.0 from .get().
        # Including those zeros in the average unfairly penalises phrases that
        # contain a mix of domain-specific words (high score) and generic
        # connectors or rare words that happen to be missing from the small
        # rule.txt vocabulary.
        #
        # Example: "reduce administrative burden" — 'reduce' scores 24.7 but
        # 'administrative' and 'burden' each appear only once and are filtered
        # from the scores dict.  Averaging in their 0.0s drags the phrase below
        # the threshold even though it is clearly topical.
        #
        # By averaging only the scored tokens we test: "among the domain words
        # in this phrase that DO appear in the rule vocabulary, are they above
        # the topicality threshold?"  If no content token is in the vocabulary
        # at all (every lookup returns 0.0), we fall through to "slopical".
        scored = [tfidf_scores[t] for t in content if t in tfidf_scores and tfidf_scores[t] > 0]
        if not scored:
            rp.slop_type = "slopical"
            continue
        score = sum(scored) / len(scored)
        rp.slop_type = "topical" if score >= threshold else "slopical"


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
        lines.append("| # | Document ID | Submitter | Sentence |")
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

def _write_phrase_entry(f, rp: "RepeatedPhrase", counter: int, tag: str) -> None:
    """Write a single phrase entry (header + table) to an open file handle."""
    tag_str = f" [{tag}]" if tag != "UNKNOWN" else ""
    f.write(f'## {counter}. "{rp.phrase}" (found in {rp.count} comments){tag_str}\n\n')
    f.write("| # | Document ID | Submitter | Sentence |\n")
    f.write("|---|-----------|---------|----------|\n")
    for j, match in enumerate(rp.matches, 1):
        if match.persona_detail:
            submitter_col = (
                f"{_escape_md_table(match.persona_name)} "
                f"({_escape_md_table(match.persona_detail)})"
            )
        else:
            submitter_col = _escape_md_table(match.persona_name)
        sentence_col = _escape_md_table(match.sentence)
        f.write(f"| {j} | {match.comment_id} | {submitter_col} | {sentence_col} |\n")
    f.write("\n")


def stream_report(
    repeated: list[RepeatedPhrase],
    total_comments: int,
    output_path: str,
) -> None:
    """
    Stream the Markdown report directly to a file.

    When phrases have been classified (``slop_type`` set to ``"slopical"`` or
    ``"topical"`` by :func:`classify_all_phrases`), the report is split into
    two clearly-labelled sections.  Each phrase header carries a machine-
    readable tag (``[SLOPICAL]`` or ``[TOPICAL]``) that :mod:`phrase_network`
    uses to filter the visualisation.

    When no classification has been run (all ``slop_type`` values remain
    ``"unknown"``), a single unsectioned list is produced for backward
    compatibility.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Determine whether topicality classification was performed
    types_present = {rp.slop_type for rp in repeated}
    classified = bool(types_present - {"unknown"})

    if classified:
        slopical = [rp for rp in repeated if rp.slop_type != "topical"]
        topical = [rp for rp in repeated if rp.slop_type == "topical"]
    else:
        slopical = repeated
        topical = []

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Phrase Repetition Report\n")
        f.write(f"Generated: {now}\n")
        summary = (
            f"Comments analysed: {total_comments} | "
            f"Repeated phrases found: {len(repeated)}"
        )
        if classified:
            summary += (
                f" ({len(slopical)} slopical · {len(topical)} topical)"
            )
        f.write(summary + "\n\n")

        if not repeated:
            f.write(
                "✅ **No repeated distinctive phrases detected.** "
                "The batch looks clean.\n"
            )
            return

        if classified:
            f.write(
                "⚠️  Phrases are classified as **slopical** (domain-agnostic "
                "AI-generation signals) or **topical** (subject-specific "
                "vocabulary that may reflect shared professional knowledge).\n\n"
            )

            # ── Slopical section ────────────────────────────────────────────
            f.write("---\n\n")
            f.write(f"# 🤖 Slopical Phrases ({len(slopical)} phrases)\n\n")
            f.write(
                "These phrases are domain-agnostic and signal AI-generated "
                "templating regardless of subject matter.\n\n"
            )
            counter = 1
            for rp in slopical:
                _write_phrase_entry(f, rp, counter, "SLOPICAL")
                counter += 1

            # ── Topical section ─────────────────────────────────────────────
            f.write("---\n\n")
            f.write(f"# 📋 Topical Domain Phrases ({len(topical)} phrases)\n\n")
            f.write(
                "These phrases use vocabulary specific to the RFI subject "
                "matter.  Repetition may reflect shared professional knowledge "
                "rather than AI templating.\n\n"
            )
            for rp in topical:
                _write_phrase_entry(f, rp, counter, "TOPICAL")
                counter += 1

        else:
            # ── Legacy single-section (no rule text / classification) ────────
            f.write(
                "⚠️  Repeated phrases may indicate AI-generated templating. "
                "Review the sentences below to decide if rewording is needed.\n\n"
            )
            for i, rp in enumerate(repeated, 1):
                f.write("---\n\n")
                _write_phrase_entry(f, rp, i, "UNKNOWN")


def run_phrase_check(
    comments: Sequence[GeneratedComment],
    output_path: str,
    min_n: int = 4,  # min ngram length
    max_n: int = 4,  # max ngram length
    min_count: int = MIN_REPEATS_DEFAULT,
    only_passed_qc: bool = True,
    verbose: bool = True,
    rule_ngrams: "frozenset[tuple[str, ...]] | None" = None,
    tfidf_scores: "dict[str, float] | None" = None,
) -> str:
    """
    Run the batch phrase-repetition check and write a Markdown report.

    Combines two detection passes:
      1. Standard n-gram analysis (min_n–max_n, default 4–8 words).
      2. Watch-list scan for known short LLM-favourite phrases.

    Watch-list hits are merged into the results and sorted by frequency.
    If *tfidf_scores* is provided (built from the rule text via
    :func:`build_tfidf_scores`), each phrase is classified as ``"topical"``
    or ``"slopical"`` before the report is written.

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
        reported (default MIN_REPEATS_DEFAULT).
    only_passed_qc:
        If True, only analyse comments that passed QC.
    verbose:
        Print summary to stderr.
    rule_ngrams:
        Optional frozenset of n-gram tuples from the rule text.  When
        provided, rule-anchored phrases are suppressed from the report.
    tfidf_scores:
        Optional word-importance map from :func:`build_tfidf_scores`.  When
        provided, each surviving phrase is classified as ``"topical"`` or
        ``"slopical"`` and the report is split into two labelled sections.

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
        rule_ngrams=rule_ngrams,
    )

    # ── Watch-list scan ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    watch_hits = find_watch_list_phrases(
        comments,
        only_passed_qc=only_passed_qc,
        min_count=min_count,
        rule_ngrams=rule_ngrams,
    )
    t_watch = time.perf_counter() - t0
    if verbose:
        _print_timing(f"Watch-list scan ({len(watch_hits)} hits)", t_watch)

    # Merge watch-list hits: add any phrase not already in the n-gram results
    existing_phrases = {rp.phrase for rp in repeated}
    for wh in watch_hits:
        if wh.phrase not in existing_phrases:
            repeated.append(wh)
            existing_phrases.add(wh.phrase)

    # Re-sort after merge
    repeated.sort(key=lambda rp: (-rp.count, rp.phrase))

    # ── Topicality classification ──────────────────────────────────────────
    if tfidf_scores:
        t0 = time.perf_counter()
        classify_all_phrases(repeated, tfidf_scores)
        t_classify = time.perf_counter() - t0
        slopical_n = sum(1 for rp in repeated if rp.slop_type == "slopical")
        topical_n = sum(1 for rp in repeated if rp.slop_type == "topical")
        if verbose:
            _print_timing(
                f"Topicality classification "
                f"({slopical_n} slopical · {topical_n} topical)",
                t_classify,
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

    * **Regulations.gov format** (e.g. ``synthetic.psv``): columns include
      ``Comment``, ``First Name``, ``Last Name``, ``State/Province``,
      ``Document ID``.
    * **Synthetic comment export format** (e.g. ``synthetic.txt``): columns include
      ``Comment``, ``Submitter Name``, ``synth_persona_state``,
      ``synth_persona_occupation``, ``Document ID``.
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
    has_psv_cols = "First Name" in fieldnames and "Last Name" in fieldnames
    has_syncom_cols = "Submitter Name" in fieldnames

    comments: list[GeneratedComment] = []

    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        # Extract comment text
        text = row.get("Comment", "")
        if not text.strip():
            continue

        # The Document ID is the per-row unique identifier of the comment (e.g. "SYNTH-0001").
        document_id = row.get("Document ID", "")
        docket_id = row.get("Docket ID", "")

        # Extract metadata depending on format
        if has_psv_cols:
            first_name = row.get("First Name", "")
            last_name = row.get("Last Name", "")
            state = row.get("State/Province", "")
            occupation = row.get("synth_persona_occupation", "")
            org_name = row.get("Organization Name", "")
        elif has_syncom_cols:
            full = row.get("Submitter Name", "")
            parts = full.strip().split(None, 1)
            first_name = parts[0] if parts else ""
            last_name = parts[1] if len(parts) > 1 else ""
            state = row.get("synth_persona_state", "")
            occupation = row.get("synth_persona_occupation", "")
            org_name = row.get("Organization Name", "")
        else:
            # Best-effort fallback
            first_name = row.get("First Name", f"Row{i+1}")
            last_name = row.get("Last Name", "")
            state = row.get("State/Province", "")
            occupation = ""
            org_name = ""

        # Load synth_personal_hook so the watch-list scan can detect
        # watch-list phrases that slipped through in the hook field.
        personal_hook = row.get("synth_personal_hook", "") or ""
        archetype = row.get("synth_persona_archetype", "individual_consumer") or "individual_consumer"

        persona = Persona(
            archetype=archetype,
            first_name=first_name or "",
            last_name=last_name or "",
            state=state,
            occupation=occupation,
            age=0,
            sophistication="medium",
            emotional_register="concerned",
            org_name=org_name,
            personal_stake="",
            personal_hook=personal_hook,
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
    max_n: int = 4,
    min_count: int = MIN_REPEATS_DEFAULT,
    verbose: bool = True,
    rule_text: str | None = None,
    rule_ngrams: "frozenset[tuple[str, ...]] | None" = None,
) -> str:
    """
    Load comments from a ♔-delimited PSV file and run the phrase-repetition
    check, writing a Markdown report.

    Parameters
    ----------
    psv_path:
        Path to the input PSV file (synthetic or real).
    output_path:
        Path for the Markdown report.  If None, derives from psv_path
        (e.g. ``synthetic_cm_phrase_report.md``).
    min_n, max_n:
        N-gram window sizes to consider (default 4–8).
    min_count:
        Minimum number of distinct comments a phrase must appear in to be
        reported (default MIN_REPEATS_DEFAULT).
    verbose:
        Print progress to stderr.
    rule_text:
        Optional full text of the proposed rule/RFI.  When provided, two
        things happen automatically:

        1. A rule n-gram index is built and used to suppress rule-anchored
           phrases from the report (only novel phrases are shown).
        2. A TF-IDF vocabulary model is built from the rule text and used to
           classify each surviving phrase as ``"topical"`` (domain-specific)
           or ``"slopical"`` (domain-agnostic AI-generation signal).  The
           report is split into two labelled sections accordingly.
    rule_ngrams:
        Optional pre-built frozenset of rule n-grams (from
        :func:`~syncom.phrase_fix.build_rule_ngrams`).  Takes precedence
        over *rule_text* for n-gram suppression when supplied.  TF-IDF
        classification still requires *rule_text*.

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

    # ── Build rule n-gram index if not already provided ────────────────────
    if rule_ngrams is None and rule_text:
        from .phrase_fix import build_rule_ngrams
        t0 = time.perf_counter()
        rule_ngrams = build_rule_ngrams(rule_text)
        t_rn = time.perf_counter() - t0
        if verbose:
            _print_timing(f"Building rule n-gram index ({len(rule_ngrams):,} n-grams)", t_rn)

    # ── Build LLR keyness scores from the rule text ────────────────────────
    tfidf_scores: dict[str, float] = {}
    if rule_text:
        t0 = time.perf_counter()
        tfidf_scores = build_keyness_scores(rule_text)
        t_tfidf = time.perf_counter() - t0
        if verbose:
            _print_timing(
                f"Building keyness vocabulary ({len(tfidf_scores):,} terms)", t_tfidf
            )

    return run_phrase_check(
        comments=comments,
        output_path=output_path,
        min_n=min_n,
        max_n=max_n,
        min_count=min_count,
        only_passed_qc=False,   # PSV rows are already filtered; check all
        verbose=verbose,
        rule_ngrams=rule_ngrams,
        tfidf_scores=tfidf_scores or None,
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """
    CLI for running phrase-repetition checks on PSV files.

    Usage::

        python -m syncom.phrase_check \\
            --input  CMS-2025-0050/shuffled_comments/synthetic.psv \\
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
        help="Input PSV file (synthetic or real).",
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
        type=int, default= MIN_REPEATS_DEFAULT,
        metavar="N",
        help=(
            "Minimum number of comments a phrase must appear in to be "
            "reported (default 3).  Raise to 3+ for a shorter, more "
            "actionable report."
        ),
    )
    parser.add_argument(
        "--rule-text",
        default=None,
        metavar="PATH",
        help=(
            "Path to the proposed rule/RFI text file.  When provided, "
            "rule-anchored phrases are filtered from the report and each "
            "surviving phrase is classified as slopical or topical using "
            "TF-IDF scoring against the rule vocabulary."
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

    # Load rule text if provided
    rule_text: str | None = None
    if args.rule_text:
        if not os.path.exists(args.rule_text):
            print(f"Error: --rule-text file not found: {args.rule_text}", file=sys.stderr)
            return 1
        encodings = ["utf-8", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                with open(args.rule_text, "r", encoding=enc) as f:
                    rule_text = f.read()
                break
            except UnicodeDecodeError:
                continue
        if rule_text is None:
            with open(args.rule_text, "r", encoding="utf-8", errors="replace") as f:
                rule_text = f.read()

    try:
        report_path = run_phrase_check_on_psv(
            psv_path=args.input,
            output_path=args.output,
            min_n=args.min_n,
            max_n=args.max_n,
            min_count=args.min_count,
            verbose=not args.quiet,
            rule_text=rule_text,
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
