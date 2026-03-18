"""
phrase_fix.py — Targeted rewriting of comments that share suspicious repeated phrases.

After running the phrase-repetition check (phrase_check.py), this module:

  1. Parses the existing phrase report to identify which phrases are repeated
     and which Document IDs are affected.
  2. Classifies each repeated phrase as EXPECTED or SUSPICIOUS by checking
     whether any 2–4-gram sub-sequence of the phrase appears in the rule text
     or world model (key_terms, rfi_questions).  Phrases anchored to the rule
     vocabulary are expected; free-floating personal narrative phrases are not.
  3. For each SUSPICIOUS phrase, rewrites every affected comment using the
     existing rewriter infrastructure, passing the phrase collision as the
     specific "judge criticism" to address.
  4. Patches the updated comment text back into the source PSV file.
  5. Re-runs the phrase check to verify the fix worked.

Usage
-----
    python cli.py phrase-fix --docket-id HHS-ONC-2026-0001

    python -m syncom.phrase_fix \\
        --input   HHS-ONC-2026-0001/synthetic_comments/synthetic.txt \\
        --report  HHS-ONC-2026-0001/synthetic_comments/synthetic_phrase_report.md \\
        --rule-text HHS-ONC-2026-0001/rule/rule.txt \\
        --world-model HHS-ONC-2026-0001/world_model.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .phrase_check import (
    RepeatedPhrase,
    PhraseMatch,
    WATCH_LIST,
    _normalise,
    _tokenise,
    run_phrase_check_on_psv,
)
from .rewriter import RewriterConfig, rewrite_comment


# ── Versioned output path helper ──────────────────────────────────────────────

def _next_versioned_path(psv_path: str) -> str:
    """
    Given an input PSV path, return the next available versioned output path.

    Examples
    --------
    ``synthetic.txt``    → ``synthetic_r1.txt``  (if not exists)
    ``synthetic_r1.txt`` → ``synthetic_r2.txt``  (if _r1 exists)
    ``synthetic_r2.txt`` → ``synthetic_r3.txt``  (if _r2 exists)

    The suffix is always the original file extension (e.g. ``.txt``).
    """
    p = Path(psv_path)
    stem = p.stem
    suffix = p.suffix
    parent = p.parent

    # Strip any existing _rN suffix from the stem
    base_stem = re.sub(r"_r\d+$", "", stem)

    # Find the next available revision number
    rev = 1
    while True:
        candidate = parent / f"{base_stem}_r{rev}{suffix}"
        if not candidate.exists():
            return str(candidate)
        rev += 1


# ── Rule n-gram index ─────────────────────────────────────────────────────────

def build_rule_ngrams(
    rule_text: str,
    world_model_path: str | None = None,
    min_n: int = 2,
    max_n: int = 4,
) -> frozenset[tuple[str, ...]]:
    """
    Build a set of all n-grams (length min_n..max_n) extracted from the rule
    text and, optionally, the world model's key_terms and rfi_questions.

    Stopwords are NOT filtered here — we want phrases like "of the" and
    "health and human" to match expected boilerplate.

    Parameters
    ----------
    rule_text:
        The full text of the proposed rule.
    world_model_path:
        Optional path to a world_model.json file.  When provided, key_terms
        and rfi_questions are also indexed.
    min_n, max_n:
        N-gram window sizes (default 2–4).

    Returns
    -------
    frozenset of tuples — each tuple is one n-gram of normalised tokens.
    """
    # Collect all text sources to index
    sources: list[str] = [rule_text]

    if world_model_path and Path(world_model_path).exists():
        try:
            with open(world_model_path, "r", encoding="utf-8") as f:
                wm = json.load(f)
            # Add key_terms (list of strings)
            for term in wm.get("key_terms", []):
                sources.append(str(term))
            # Add rfi_questions (list of strings)
            for q in wm.get("rfi_questions", []):
                sources.append(str(q))
            # Add rule_title, agency, core_change for good measure
            for field_name in ("rule_title", "agency", "core_change"):
                val = wm.get(field_name, "")
                if val:
                    sources.append(str(val))
        except (json.JSONDecodeError, OSError):
            pass  # world model unavailable — proceed with rule text only

    ngrams: set[tuple[str, ...]] = set()
    for source in sources:
        tokens = _tokenise(_normalise(source))
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.add(tuple(tokens[i : i + n]))

    return frozenset(ngrams)


# ── Phrase triage ─────────────────────────────────────────────────────────────

def is_rule_anchored(
    phrase: str,
    rule_ngrams: frozenset[tuple[str, ...]],
    min_n: int = 2,
    max_n: int = 4,
) -> bool:
    """
    Return True if any 2–4-gram sub-sequence of *phrase* appears in
    *rule_ngrams*.

    A phrase is considered rule-anchored (and therefore an EXPECTED repetition)
    when at least one of its short n-gram windows matches the rule vocabulary.
    This is order-preserving — "private sector innovation" matches because the
    trigram ("private", "sector", "innovation") appears in the RFI questions,
    while "last spring my cardiologist" does not match because none of its
    bigrams or trigrams appear in the rule text.

    Parameters
    ----------
    phrase:
        The repeated phrase string (as it appears in the phrase report).
    rule_ngrams:
        The frozenset returned by :func:`build_rule_ngrams`.
    min_n, max_n:
        N-gram window sizes to check (default 2–4).

    Returns
    -------
    bool
    """
    tokens = _tokenise(_normalise(phrase))
    for n in range(min_n, min(max_n, len(tokens)) + 1):
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            if gram in rule_ngrams:
                return True
    return False


# Regex that matches "page N of M" patterns (e.g. "page 1 of 1", "page 2 of 3").
# These are document-formatting artefacts that are expected to repeat across
# comments and should never be flagged as suspicious.
_PAGE_N_OF_M_RE = re.compile(
    r"^page\s+\d+\s+of\s+\d+$",
    re.IGNORECASE,
)


def _is_formatting_artefact(phrase: str) -> bool:
    """
    Return True if *phrase* is a known document-formatting artefact that is
    expected to repeat across comments and should never be rewritten.

    Currently catches:
      - "page N of M" patterns (e.g. "page 1 of 1", "page 2 of 3")
    """
    return bool(_PAGE_N_OF_M_RE.match(phrase.strip()))


def triage_repeated_phrases(
    repeated: list[RepeatedPhrase],
    rule_ngrams: frozenset[tuple[str, ...]],
) -> tuple[list[RepeatedPhrase], list[RepeatedPhrase]]:
    """
    Classify each repeated phrase as EXPECTED or SUSPICIOUS.

    A phrase is EXPECTED (and therefore skipped) if:
      - It is a known document-formatting artefact (e.g. "page 1 of 1"), OR
      - Any 2–4-gram sub-sequence of the phrase appears in the rule vocabulary.

    Parameters
    ----------
    repeated:
        List of RepeatedPhrase objects from the phrase report.
    rule_ngrams:
        The frozenset returned by :func:`build_rule_ngrams`.

    Returns
    -------
    (expected, suspicious) — two lists of RepeatedPhrase objects.
    """
    expected: list[RepeatedPhrase] = []
    suspicious: list[RepeatedPhrase] = []

    for rp in repeated:
        if _is_formatting_artefact(rp.phrase) or is_rule_anchored(rp.phrase, rule_ngrams):
            expected.append(rp)
        else:
            suspicious.append(rp)

    return expected, suspicious


# ── Phrase report parser ──────────────────────────────────────────────────────

def parse_phrase_report(report_path: str) -> list[RepeatedPhrase]:
    """
    Parse an existing phrase report Markdown file and reconstruct a list of
    RepeatedPhrase objects.

    This avoids re-running the phrase check just to get the data — the report
    is already on disk.  The parser handles the Markdown table format produced
    by :func:`syncom.phrase_check.stream_report`.

    Parameters
    ----------
    report_path:
        Path to the .phrase_report.md file.

    Returns
    -------
    list[RepeatedPhrase]
    """
    text = Path(report_path).read_text(encoding="utf-8")
    repeated: list[RepeatedPhrase] = []

    # Split on section headers: ## N. "phrase" (found in K comments)
    # Pattern: ## <number>. "<phrase>" (found in <count> comments)
    section_re = re.compile(
        r'^## \d+\.\s+"([^"]+)"\s+\(found in (\d+) comments?\)',
        re.MULTILINE,
    )
    # Table row pattern: | N | Document ID | Submitter | Sentence |
    row_re = re.compile(
        r'^\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|',
        re.MULTILINE,
    )

    sections = section_re.split(text)
    # sections[0] = preamble, then groups of (phrase, count, body) repeat
    # After split: [preamble, phrase1, count1, body1, phrase2, count2, body2, ...]
    i = 1
    while i + 2 < len(sections):
        phrase = sections[i].strip()
        # count = int(sections[i + 1])  # not needed — we count matches
        body = sections[i + 2]

        matches: list[PhraseMatch] = []
        for j, m in enumerate(row_re.finditer(body)):
            comment_id = m.group(1).strip()
            submitter_raw = m.group(2).strip()
            sentence = m.group(3).strip()

            # Parse submitter: "Name (detail)" or just "Name"
            detail_match = re.match(r'^(.+?)\s+\((.+)\)$', submitter_raw)
            if detail_match:
                persona_name = detail_match.group(1).strip()
                persona_detail = detail_match.group(2).strip()
            else:
                persona_name = submitter_raw
                persona_detail = ""

            matches.append(PhraseMatch(
                comment_index=j,
                comment_id=comment_id,
                persona_name=persona_name,
                persona_detail=persona_detail,
                sentence=sentence,
            ))

        if matches:
            # Infer ngram_length from word count of phrase
            ngram_length = len(phrase.split())
            repeated.append(RepeatedPhrase(
                phrase=phrase,
                ngram_length=ngram_length,
                matches=matches,
            ))

        i += 3

    return repeated


# ── Comment rewriter ──────────────────────────────────────────────────────────

# Phrases the rewriter must NEVER introduce — the full watch list plus
# common seasonal variants.  Appended to every rewrite criticism prompt.
_DO_NOT_INTRODUCE: list[str] = WATCH_LIST + [
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
    "a couple of months months ago",
    "recently i",
    "just recently",
    "not long ago",
]
# Deduplicate while preserving order
_seen: set[str] = set()
_DO_NOT_INTRODUCE = [
    p for p in _DO_NOT_INTRODUCE
    if not (p in _seen or _seen.add(p))  # type: ignore[func-returns-value]
]
del _seen


def _do_not_introduce_clause() -> str:
    """Return a formatted 'do not introduce' clause for the rewrite prompt."""
    quoted = ", ".join(f'"{p}"' for p in _DO_NOT_INTRODUCE[:12])  # top 12 for brevity
    return (
        f"\n\nIMPORTANT — do NOT introduce any of the following phrases "
        f"(or close variants) into the rewritten text: {quoted}. "
        f"These are known LLM-favourite clichés that would make the comment "
        f"look even more AI-generated."
    )


def _build_phrase_criticism(
    phrase_list: list[tuple[str, str, int]],
) -> str:
    """
    Build the "judge criticism" text that tells the rewriter exactly what to fix.

    *phrase_list* is a list of (phrase, sentence, total_occurrences) tuples —
    all the suspicious phrases that need to be fixed in this one comment.
    The criticism is phrased as a numbered list of specific, actionable
    instructions so the LLM can fix all of them in a single rewrite pass.

    A "do not introduce" clause is appended to every criticism to prevent the
    rewriter from substituting one LLM cliché for another.
    """
    do_not = _do_not_introduce_clause()

    if len(phrase_list) == 1:
        phrase, sentence, total_occurrences = phrase_list[0]
        return (
            f'PHRASE COLLISION DETECTED: The phrase "{phrase}" appears verbatim in '
            f"{total_occurrences} different comments in this batch, which is a strong "
            f"signal of AI-generated templating.\n\n"
            f'The offending sentence is: "{sentence}"\n\n'
            f"Rewrite ONLY that sentence (and any immediately adjacent sentences that "
            f"share the same narrative beat) so that it expresses the same idea using "
            f"completely different wording, a different temporal anchor if applicable, "
            f'and no trace of the phrase "{phrase}" or any close variant. '
            f"The rest of the comment should remain unchanged."
            f"{do_not}"
        )

    # Multiple phrases — build a numbered list
    lines = [
        f"PHRASE COLLISIONS DETECTED: The following {len(phrase_list)} phrases each "
        f"appear verbatim in multiple comments in this batch, which is a strong signal "
        f"of AI-generated templating. Fix ALL of them in a single rewrite pass.\n"
    ]
    for i, (phrase, sentence, total_occurrences) in enumerate(phrase_list, 1):
        lines.append(
            f'{i}. Phrase "{phrase}" (found in {total_occurrences} comments)\n'
            f'   Offending sentence: "{sentence}"\n'
            f'   Fix: rewrite that sentence so it expresses the same idea with '
            f'completely different wording and no trace of "{phrase}" or any close variant.'
        )
    lines.append(
        "\nThe rest of the comment should remain unchanged. "
        "Fix all of the above in one pass."
        f"{do_not}"
    )
    return "\n".join(lines)


def rewrite_comment_for_phrases(
    comment_text: str,
    phrase_list: list[tuple[str, str, int]],
    persona_context: dict[str, str],
    config: RewriterConfig,
) -> str:
    """
    Rewrite a comment to eliminate ALL suspicious repeated phrases in one LLM call.

    Parameters
    ----------
    comment_text:
        The full comment text to rewrite.
    phrase_list:
        List of (phrase, sentence, total_occurrences) tuples — all the phrases
        that need to be fixed in this comment.
    persona_context:
        Identity fields for the rewriter: name, occupation, org_name,
        state, age, archetype.
    config:
        Rewriter API configuration.

    Returns
    -------
    str — The rewritten comment text.
    """
    criticism = _build_phrase_criticism(phrase_list)
    return rewrite_comment(
        comment_text=comment_text,
        judge_score=0,          # treat phrase collision as definitive AI signal
        judge_reasons=criticism,
        persona_context=persona_context,
        config=config,
    )


# Keep the single-phrase variant as a convenience wrapper
def rewrite_comment_for_phrase(
    comment_text: str,
    phrase: str,
    sentence: str,
    total_occurrences: int,
    persona_context: dict[str, str],
    config: RewriterConfig,
) -> str:
    """Single-phrase convenience wrapper around :func:`rewrite_comment_for_phrases`."""
    return rewrite_comment_for_phrases(
        comment_text=comment_text,
        phrase_list=[(phrase, sentence, total_occurrences)],
        persona_context=persona_context,
        config=config,
    )


# ── PSV patcher ───────────────────────────────────────────────────────────────

def patch_psv_comments(
    psv_path: str,
    patches: dict[str, str],   # comment_id → new_comment_text
    output_path: str,
) -> int:
    """
    Read a PSV file, replace the Comment field for each patched comment ID,
    and write the result to output_path.

    Parameters
    ----------
    psv_path:
        Path to the source PSV file.
    patches:
        Dict mapping Document ID → new comment text.
    output_path:
        Destination path (may equal psv_path for in-place update).

    Returns
    -------
    int — Number of rows that were patched.
    """
    from pathlib import Path as _Path
    import sys as _sys

    # Allow running from repo root without installing shuffler as a package
    _REPO_ROOT = _Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_REPO_ROOT))

    from shuffler.psv_io import read_psv, write_psv, NEWLINE_ENC

    rows, fieldnames = read_psv(psv_path)

    if "Document ID" not in fieldnames or "Comment" not in fieldnames:
        raise ValueError(
            f"PSV file does not have expected 'Document ID' and 'Comment' columns. "
            f"Found: {fieldnames[:10]}"
        )

    patched_count = 0
    for row in rows:
        comment_id = row.get("Document ID", "").strip()
        if comment_id in patches:
            new_text = patches[comment_id]
            # Encode newlines as ⏎ for PSV storage (same as export.py)
            new_text_encoded = (
                new_text
                .replace("\r\n", NEWLINE_ENC)
                .replace("\r", NEWLINE_ENC)
                .replace("\n", NEWLINE_ENC)
            )
            row["Comment"] = new_text_encoded
            patched_count += 1

    write_psv(output_path, fieldnames, rows)
    return patched_count


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PhraseFixResult:
    """Summary of a phrase-fix run."""
    phrases_found: int = 0          # total repeated phrases in report
    phrases_expected: int = 0       # classified as rule-anchored (skipped)
    phrases_suspicious: int = 0     # classified as needing rewrite
    comments_rewritten: int = 0     # distinct comments that were patched
    output_path: str = ""           # path to updated PSV
    report_path: str = ""           # path to regenerated phrase report
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            "Phrase-fix complete:",
            f"  Phrases in report : {self.phrases_found}",
            f"  Expected (skipped): {self.phrases_expected}",
            f"  Suspicious (fixed): {self.phrases_suspicious}",
            f"  Comments rewritten: {self.comments_rewritten}",
            f"  Output PSV        : {self.output_path}",
            f"  Updated report    : {self.report_path}",
            f"  Elapsed           : {self.elapsed_seconds:.1f}s",
        ]
        return "\n".join(lines)


# ── PSV comment loader (for persona context) ──────────────────────────────────

def _load_persona_contexts_from_psv(psv_path: str) -> dict[str, dict[str, str]]:
    """
    Load persona context dicts keyed by Document ID from a PSV file.

    Returns a dict mapping comment_id → persona_context dict suitable for
    passing to :func:`rewrite_comment_for_phrase`.
    """
    from pathlib import Path as _Path
    import sys as _sys

    _REPO_ROOT = _Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_REPO_ROOT))

    from shuffler.psv_io import read_psv

    rows, fieldnames = read_psv(psv_path)
    contexts: dict[str, dict[str, str]] = {}

    for row in rows:
        comment_id = row.get("Document ID", "").strip()
        if not comment_id:
            continue

        # Build persona context from available PSV columns
        submitter_name = row.get("Submitter Name", "")
        org_name = row.get("Organization Name", "")
        occupation = row.get("synth_persona_occupation", "")
        state = row.get("synth_persona_state", "")
        age = row.get("synth_persona_age", "")
        archetype = row.get("synth_archetype", "individual_consumer")

        contexts[comment_id] = {
            "name": submitter_name,
            "occupation": occupation,
            "org_name": org_name or "None",
            "state": state,
            "age": age,
            "archetype": archetype,
        }

    return contexts


def _load_comment_texts_from_psv(psv_path: str) -> dict[str, str]:
    """
    Load comment texts keyed by Document ID from a PSV file.
    """
    from pathlib import Path as _Path
    import sys as _sys

    _REPO_ROOT = _Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_REPO_ROOT))

    from shuffler.psv_io import read_psv

    rows, _ = read_psv(psv_path)
    texts: dict[str, str] = {}

    for row in rows:
        comment_id = row.get("Document ID", "").strip()
        if comment_id:
            texts[comment_id] = row.get("Comment", "")

    return texts


# ── Top-level orchestrator ────────────────────────────────────────────────────

def run_phrase_fix(
    psv_path: str,
    report_path: str,
    rule_text: str,
    output_path: str | None = None,
    world_model_path: str | None = None,
    min_n: int = 2,
    max_n: int = 4,
    verbose: bool = True,
) -> PhraseFixResult:
    """
    Run the full phrase-fix pipeline:

    1. Parse the phrase report to get repeated phrases.
    2. Build rule n-grams from rule text + world model.
    3. Triage phrases into EXPECTED vs SUSPICIOUS.
    4. Rewrite all comments affected by SUSPICIOUS phrases.
    5. Patch the PSV file with updated comment texts.
    6. Re-run the phrase check to regenerate the report.

    Parameters
    ----------
    psv_path:
        Path to the source PSV file (syncom export format).
    report_path:
        Path to the existing phrase report Markdown file.
    rule_text:
        The full text of the proposed rule (used to build the n-gram index).
    output_path:
        Destination PSV path.  Defaults to psv_path (in-place update).
    world_model_path:
        Optional path to world_model.json for additional vocabulary.
    min_n, max_n:
        N-gram window sizes for the rule-anchoring check (default 2–4).
    verbose:
        Print progress to stderr.

    Returns
    -------
    PhraseFixResult
    """
    t_start = time.perf_counter()
    result = PhraseFixResult()

    if output_path is None:
        output_path = psv_path

    # ── Step 1: Parse phrase report ───────────────────────────────────────
    if verbose:
        print(f"[phrase-fix] Parsing phrase report: {report_path}", file=sys.stderr)

    repeated = parse_phrase_report(report_path)
    result.phrases_found = len(repeated)

    if verbose:
        print(f"[phrase-fix] Found {len(repeated)} repeated phrases in report.", file=sys.stderr)

    if not repeated:
        if verbose:
            print("[phrase-fix] Nothing to fix.", file=sys.stderr)
        result.output_path = output_path
        result.report_path = report_path
        result.elapsed_seconds = time.perf_counter() - t_start
        return result

    # ── Step 2: Build rule n-gram index ───────────────────────────────────
    if verbose:
        print(f"[phrase-fix] Building rule n-gram index (n={min_n}–{max_n})…", file=sys.stderr)

    rule_ngrams = build_rule_ngrams(rule_text, world_model_path, min_n, max_n)

    if verbose:
        print(f"[phrase-fix] Rule n-gram index: {len(rule_ngrams):,} n-grams.", file=sys.stderr)

    # ── Step 3: Triage phrases ────────────────────────────────────────────
    expected, suspicious = triage_repeated_phrases(repeated, rule_ngrams)
    result.phrases_expected = len(expected)
    result.phrases_suspicious = len(suspicious)

    if verbose:
        print(
            f"[phrase-fix] Triage: {len(expected)} expected (rule-anchored), "
            f"{len(suspicious)} suspicious.",
            file=sys.stderr,
        )
        if expected:
            print("[phrase-fix] Expected (skipped):", file=sys.stderr)
            for rp in expected:
                print(f'  - "{rp.phrase}" ({rp.count} comments)', file=sys.stderr)
        if suspicious:
            print("[phrase-fix] Suspicious (will rewrite):", file=sys.stderr)
            for rp in suspicious:
                print(f'  - "{rp.phrase}" ({rp.count} comments)', file=sys.stderr)

    if not suspicious:
        if verbose:
            print("[phrase-fix] No suspicious phrases found — nothing to rewrite.", file=sys.stderr)
        result.output_path = output_path
        result.report_path = report_path
        result.elapsed_seconds = time.perf_counter() - t_start
        return result

    # ── Step 4: Load comment texts and persona contexts ───────────────────
    if verbose:
        print(f"[phrase-fix] Loading comments from: {psv_path}", file=sys.stderr)

    comment_texts = _load_comment_texts_from_psv(psv_path)
    persona_contexts = _load_persona_contexts_from_psv(psv_path)

    # ── Step 5: Rewrite affected comments ─────────────────────────────────
    rewriter_config = RewriterConfig()
    if not rewriter_config.is_available():
        print(
            "[phrase-fix] WARNING: Rewriter API keys not configured "
            "(REWRITE_COMMENT_API_KEY / JUDGE_API_KEY). "
            "Skipping rewrites — only the triage report will be shown.",
            file=sys.stderr,
        )
        result.output_path = output_path
        result.report_path = report_path
        result.elapsed_seconds = time.perf_counter() - t_start
        return result

    # Collect all (comment_id, phrase, sentence, total_occurrences) to rewrite.
    # A comment may be affected by multiple suspicious phrases — we process
    # them sequentially, each rewrite building on the previous.
    # Map: comment_id → list of (phrase, sentence, total_occurrences)
    rewrites_needed: dict[str, list[tuple[str, str, int]]] = {}
    for rp in suspicious:
        for match in rp.matches:
            cid = match.comment_id
            if cid not in rewrites_needed:
                rewrites_needed[cid] = []
            rewrites_needed[cid].append((rp.phrase, match.sentence, rp.count))

    patches: dict[str, str] = {}
    total_to_rewrite = len(rewrites_needed)

    if verbose:
        print(
            f"[phrase-fix] Rewriting {total_to_rewrite} comment(s) "
            f"across {len(suspicious)} suspicious phrase(s)…",
            file=sys.stderr,
        )

    for idx, (comment_id, phrase_list) in enumerate(rewrites_needed.items(), 1):
        current_text = patches.get(comment_id, comment_texts.get(comment_id, ""))
        if not current_text:
            if verbose:
                print(
                    f"[phrase-fix]   [{idx}/{total_to_rewrite}] "
                    f"WARNING: comment {comment_id} not found in PSV — skipping.",
                    file=sys.stderr,
                )
            continue

        persona_ctx = persona_contexts.get(comment_id, {
            "name": "Unknown",
            "occupation": "Unknown",
            "org_name": "None",
            "state": "Unknown",
            "age": "Unknown",
            "archetype": "individual_consumer",
        })

        # One LLM call fixes ALL suspicious phrases in this comment at once
        if verbose:
            phrases_str = ", ".join(f'"{p}"' for p, _, _ in phrase_list)
            print(
                f'[phrase-fix]   [{idx}/{total_to_rewrite}] '
                f'Rewriting {comment_id} for {len(phrase_list)} phrase(s): {phrases_str}…',
                file=sys.stderr,
            )
        try:
            current_text = rewrite_comment_for_phrases(
                comment_text=current_text,
                phrase_list=phrase_list,
                persona_context=persona_ctx,
                config=rewriter_config,
            )
        except Exception as exc:
            print(
                f"[phrase-fix]   WARNING: rewrite failed for {comment_id}: {exc}",
                file=sys.stderr,
            )

        patches[comment_id] = current_text

    result.comments_rewritten = len(patches)

    # ── Step 6: Patch PSV ─────────────────────────────────────────────────
    if patches:
        if verbose:
            print(
                f"[phrase-fix] Patching {len(patches)} comment(s) → {output_path}",
                file=sys.stderr,
            )
        patched = patch_psv_comments(psv_path, patches, output_path)
        if verbose:
            print(f"[phrase-fix] Patched {patched} row(s) in PSV.", file=sys.stderr)
    else:
        if verbose:
            print("[phrase-fix] No patches to apply.", file=sys.stderr)

    # ── Step 7: Re-run phrase check ───────────────────────────────────────
    if verbose:
        print("[phrase-fix] Re-running phrase check on updated PSV…", file=sys.stderr)

    # Derive the report path from the output PSV stem, replacing dots with
    # underscores so we get e.g. synthetic_r1_phrase_report.md
    output_stem = Path(output_path).stem.replace(".", "_")
    output_dir = Path(output_path).parent
    new_report_path = str(output_dir / f"{output_stem}_phrase_report.md")
    run_phrase_check_on_psv(
        psv_path=output_path,
        output_path=new_report_path,
        verbose=verbose,
    )

    result.output_path = output_path
    result.report_path = new_report_path
    result.elapsed_seconds = time.perf_counter() - t_start

    if verbose:
        print(result.summary(), file=sys.stderr)

    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """
    CLI for fixing repeated phrases in a synthetic comment PSV file.

    Usage::

        python -m syncom.phrase_fix \\
            --input   HHS-ONC-2026-0001/synthetic_comments/synthetic.txt \\
            --report  HHS-ONC-2026-0001/synthetic_comments/synthetic_phrase_report.md \\
            --rule-text HHS-ONC-2026-0001/rule/rule.txt \\
            --world-model HHS-ONC-2026-0001/world_model.json

        # Or using docket-id convention:
        python cli.py phrase-fix --docket-id HHS-ONC-2026-0001
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="python -m syncom.phrase_fix",
        description=(
            "Identify and rewrite synthetic comments that share suspicious "
            "repeated phrases, using the existing phrase report and rewriter "
            "infrastructure."
        ),
    )
    parser.add_argument(
        "--docket-id",
        default=None,
        metavar="ID",
        help=(
            "Docket identifier (e.g., 'HHS-ONC-2026-0001'). When provided, "
            "all file paths default to conventional locations inside the "
            "docket directory; any explicit path argument overrides the default."
        ),
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        metavar="PATH",
        help=(
            "Input PSV file (syncom export format). "
            "Default: {docket_id}/synthetic_comments/synthetic.txt"
        ),
    )
    parser.add_argument(
        "--report", "-r",
        default=None,
        metavar="PATH",
        help=(
            "Existing phrase report Markdown file. "
            "Default: {docket_id}/synthetic_comments/synthetic_phrase_report.md"
        ),
    )
    parser.add_argument(
        "--rule-text",
        default=None,
        metavar="PATH_OR_TEXT",
        help=(
            "Path to the rule text file, or the rule text itself. "
            "Default: {docket_id}/rule/rule.txt"
        ),
    )
    parser.add_argument(
        "--world-model",
        default=None,
        metavar="PATH",
        help=(
            "Path to world_model.json for additional vocabulary. "
            "Default: {docket_id}/world_model.json"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="PATH",
        help=(
            "Output PSV path.  If not provided, auto-generates a versioned "
            "filename: synthetic_r1.txt, synthetic_r2.txt, etc.  "
            "The phrase report is written alongside as "
            "synthetic_r1_phrase_report.md, etc."
        ),
    )
    parser.add_argument(
        "--min-n",
        type=int, default=2,
        metavar="N",
        help="Minimum n-gram length for rule-anchoring check (default 2).",
    )
    parser.add_argument(
        "--max-n",
        type=int, default=4,
        metavar="N",
        help="Maximum n-gram length for rule-anchoring check (default 4).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args(argv)
    verbose = not args.quiet

    # ── Resolve convention-based defaults from docket-id ─────────────────
    docket_id = args.docket_id
    psv_path = args.input
    report_path = args.report
    rule_text_path = args.rule_text
    world_model_path = args.world_model
    output_path = args.output

    if docket_id:
        base = docket_id
        if psv_path is None:
            psv_path = os.path.join(base, "synthetic_comments", "synthetic.txt")
        if report_path is None:
            report_path = os.path.join(base, "synthetic_comments", "synthetic_phrase_report.md")
        if rule_text_path is None:
            rule_text_path = os.path.join(base, "rule", "rule.txt")
        if world_model_path is None:
            candidate = os.path.join(base, "world_model.json")
            if os.path.exists(candidate):
                world_model_path = candidate

    # ── Auto-generate versioned output path if not specified ──────────────
    # Input:  synthetic.txt  → synthetic_r1.txt, synthetic_r2.txt, …
    # Input:  synthetic_r1.txt → synthetic_r2.txt, …
    if output_path is None and psv_path is not None:
        output_path = _next_versioned_path(psv_path)
        if verbose:
            print(f"[phrase-fix] Output path: {output_path}", file=sys.stderr)

    # Validate required arguments
    missing = []
    if psv_path is None:
        missing.append("--input")
    if report_path is None:
        missing.append("--report")
    if rule_text_path is None:
        missing.append("--rule-text")
    if missing:
        print(
            f"Error: the following arguments are required: {', '.join(missing)}\n"
            f"       (or provide --docket-id to use convention-based defaults)",
            file=sys.stderr,
        )
        return 1

    # Validate file existence
    for label, path in [
        ("--input", psv_path),
        ("--report", report_path),
    ]:
        if not os.path.exists(path):
            print(f"Error: {label} file not found: {path}", file=sys.stderr)
            return 1

    # Load rule text
    if os.path.exists(rule_text_path):
        encodings = ["utf-8", "latin-1", "cp1252"]
        rule_text = None
        for enc in encodings:
            try:
                with open(rule_text_path, "r", encoding=enc) as f:
                    rule_text = f.read()
                break
            except UnicodeDecodeError:
                continue
        if rule_text is None:
            with open(rule_text_path, "r", encoding="utf-8", errors="replace") as f:
                rule_text = f.read()
    else:
        # Treat as literal text
        rule_text = rule_text_path

    try:
        result = run_phrase_fix(
            psv_path=psv_path,
            report_path=report_path,
            rule_text=rule_text,
            output_path=output_path,
            world_model_path=world_model_path,
            min_n=args.min_n,
            max_n=args.max_n,
            verbose=verbose,
        )
        if verbose:
            print(f"[phrase-fix] Done. Report: {result.report_path}", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
