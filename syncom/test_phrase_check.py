"""
test_phrase_check.py — Unit tests for batch phrase-repetition detection.

Run:  python -m pytest syncom/test_phrase_check.py -v
"""

from __future__ import annotations

import os
import tempfile

import pytest

from syncom.phrase_check import (
    _is_distinctive,
    _normalise,
    _find_sentence_containing,
    _phrase_to_pattern,
    extract_distinctive_ngrams,
    find_repeated_phrases,
    generate_report,
    run_phrase_check,
    _deduplicate_subsumed,
    RepeatedPhrase,
    PhraseMatch,
)
from syncom.generator import GeneratedComment
from syncom.argument_mapper import ExpressionFrame
from syncom.persona import Persona


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_persona(name: str = "Jane Doe", occupation: str = "nurse", state: str = "Ohio") -> Persona:
    return Persona(
        archetype="individual",
        first_name=name.split()[0],
        last_name=name.split()[-1] if " " in name else "Doe",
        state=state,
        occupation=occupation,
        age=45,
        sophistication="medium",
        emotional_register="concerned",
        org_name="",
        personal_stake="This rule affects my healthcare.",
        personal_hook="My mother waited 3 months for approval.",
    )


def _make_frame() -> ExpressionFrame:
    return ExpressionFrame(
        core_arguments=["Oppose the rule"],
        framing="Personal experience framing",
        evidence_types=["personal anecdote"],
        rfi_questions_to_address=[],
    )


def _make_comment(text: str, index: int = 0, persona: Persona | None = None, qc_passed: bool = True) -> GeneratedComment:
    p = persona or _make_persona()
    return GeneratedComment(
        comment_text=text,
        persona=p,
        frame=_make_frame(),
        vector=0,
        objective="Oppose the proposed rule",
        rule_title="Test Rule",
        docket_id="TEST-2025-0001",
        qc_passed=qc_passed,
    )


# ── Unit tests: normalisation and filtering ───────────────────────────────────

class TestNormalise:
    def test_lowercases(self):
        assert _normalise("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert _normalise("Hello, World!") == "hello world"

    def test_keeps_apostrophes(self):
        assert _normalise("I can't believe it's") == "i can't believe it's"

    def test_strips_special_chars(self):
        assert _normalise("45 CFR § 422.560") == "45 cfr 422 560"


class TestIsDistinctive:
    def test_all_stopwords_rejected(self):
        # "i would like to" — all stopwords
        assert not _is_distinctive(("i", "would", "like", "to"))

    def test_mostly_content_words_accepted(self):
        # "registered nurse working rural" — all content
        assert _is_distinctive(("registered", "nurse", "working", "rural"))

    def test_mixed_at_threshold(self):
        # "as a registered nurse" — 2 stopwords, 2 content = 50% → pass
        assert _is_distinctive(("as", "a", "registered", "nurse"))

    def test_just_below_threshold(self):
        # 3 stopwords, 1 content in 4-gram = 25% → fail
        assert not _is_distinctive(("in", "the", "of", "algorithm"))


# ── Unit tests: n-gram extraction ─────────────────────────────────────────────

class TestExtractDistinctiveNgrams:
    def test_basic_extraction(self):
        text = "As a registered nurse working in a rural clinic I see the impact every day"
        ngrams = extract_distinctive_ngrams(text, min_n=4, max_n=4)
        # 4-grams are contiguous windows; "a registered nurse working" is one
        phrases = {" ".join(g) for g in ngrams}
        assert "a registered nurse working" in phrases

    def test_min_max_range(self):
        text = "The widely used commercial algorithm has significant problems for patients"
        ngrams_4 = extract_distinctive_ngrams(text, min_n=4, max_n=4)
        ngrams_6 = extract_distinctive_ngrams(text, min_n=4, max_n=6)
        # 6-gram window should produce strictly more n-grams
        assert len(ngrams_6) >= len(ngrams_4)

    def test_short_text_returns_empty(self):
        text = "Hello world"
        ngrams = extract_distinctive_ngrams(text, min_n=4, max_n=4)
        assert len(ngrams) == 0

    def test_all_stopwords_text(self):
        text = "I would like to say that it is very much the same as before"
        ngrams = extract_distinctive_ngrams(text, min_n=4, max_n=4)
        # Most 4-grams should be filtered as non-distinctive
        phrases = {" ".join(g) for g in ngrams}
        assert "i would like to" not in phrases


# ── Unit tests: sentence extraction ───────────────────────────────────────────

class TestFindSentenceContaining:
    def test_finds_correct_sentence(self):
        text = "I love dogs. As a registered nurse I see problems daily. The end."
        result = _find_sentence_containing(text, "registered nurse")
        assert "registered nurse" in result.lower()
        assert "I see problems" in result

    def test_case_insensitive(self):
        text = "My REGISTERED NURSE colleagues agree. This is bad."
        result = _find_sentence_containing(text, "registered nurse")
        assert "REGISTERED NURSE" in result

    def test_fallback_when_no_sentence_boundary(self):
        text = "As a registered nurse I see problems daily and it never gets better"
        result = _find_sentence_containing(text, "registered nurse")
        assert "registered nurse" in result.lower()

    def test_not_found_returns_placeholder(self):
        text = "This comment is about something else entirely."
        result = _find_sentence_containing(text, "nonexistent phrase xyz")
        assert result == "(sentence not found)"

    def test_finds_phrase_with_stripped_section_symbol(self):
        """The normalised phrase '45 cfr 170 315' must match '45 CFR § 170.315'."""
        text = "We must comply with 45 CFR § 170.315 to maintain certification. This is critical."
        result = _find_sentence_containing(text, "45 cfr 170 315")
        assert "45 CFR § 170.315" in result
        assert result != "(sentence not found)"

    def test_finds_phrase_with_hyphenated_words(self):
        """The normalised phrase 'year old latina woman' must match 'year-old Latina woman'."""
        text = "As a 65-year-old Latina woman living in Colorado, I depend on these services."
        result = _find_sentence_containing(text, "year old latina woman")
        assert "year-old Latina woman" in result
        assert result != "(sentence not found)"

    def test_finds_phrase_with_period_in_number(self):
        """Normalised '422 560' must match '422.560'."""
        text = "The changes to section 422.560 would affect prior authorization. We oppose this."
        result = _find_sentence_containing(text, "changes to section 422 560")
        assert "422.560" in result
        assert result != "(sentence not found)"

    def test_finds_phrase_with_slash(self):
        """Normalised 'medicare medicaid' must match 'Medicare/Medicaid'."""
        text = "Many Medicare/Medicaid dual-eligible patients face barriers. We need reform."
        result = _find_sentence_containing(text, "medicare medicaid dual eligible patients")
        assert "Medicare/Medicaid dual-eligible patients" in result
        assert result != "(sentence not found)"

    def test_fallback_with_punctuation_no_sentence_boundary(self):
        """Regex fallback works when there are no sentence boundaries."""
        text = "The rule references 45 CFR § 170.315 and other certification standards that affect us"
        result = _find_sentence_containing(text, "45 cfr 170 315")
        assert "45 CFR § 170.315" in result
        assert result != "(sentence not found)"


# ── Integration tests: find_repeated_phrases ──────────────────────────────────

class TestFindRepeatedPhrases:
    def test_detects_shared_phrase(self):
        c1 = _make_comment(
            "As a registered nurse working in Ohio, I see the damage this rule causes.",
            persona=_make_persona("Jane Doe", "nurse", "Ohio"),
        )
        c2 = _make_comment(
            "As a registered nurse working at a hospital, I oppose this rule.",
            persona=_make_persona("Maria Lopez", "nurse", "Texas"),
        )
        c3 = _make_comment(
            "I'm a small business owner and I think this rule is fine.",
            persona=_make_persona("Bob Smith", "business owner", "Maine"),
        )

        repeated = find_repeated_phrases([c1, c2, c3], min_n=4, max_n=4)
        phrases = [rp.phrase for rp in repeated]
        # "registered nurse working" should appear in some 4-gram match
        assert any("registered nurse working" in p for p in phrases)

    def test_no_repeats_returns_empty(self):
        c1 = _make_comment("The certification requirements create an undue burden on small hospitals.")
        c2 = _make_comment("I disagree with the proposed timeline for implementation of the new standards.")
        c3 = _make_comment("Rural communities will suffer disproportionately from these funding changes.")

        repeated = find_repeated_phrases([c1, c2, c3], min_n=4, max_n=4)
        assert len(repeated) == 0

    def test_respects_only_passed_qc(self):
        c1 = _make_comment(
            "As a registered nurse working here, I see problems.",
            qc_passed=True,
        )
        c2 = _make_comment(
            "As a registered nurse working there, I oppose this.",
            qc_passed=False,
        )

        # With only_passed_qc=True, c2 is excluded — no repeats
        repeated = find_repeated_phrases([c1, c2], min_n=4, max_n=4, only_passed_qc=True)
        assert len(repeated) == 0

        # With only_passed_qc=False, both are included — repeats found
        repeated = find_repeated_phrases([c1, c2], min_n=4, max_n=4, only_passed_qc=False)
        assert len(repeated) > 0

    def test_match_contains_correct_metadata(self):
        persona_a = _make_persona("Jane Doe", "nurse", "Ohio")
        persona_b = _make_persona("Maria Lopez", "nurse", "Texas")
        c1 = _make_comment(
            "As a registered nurse working at a rural clinic I see the damage.",
            persona=persona_a,
        )
        c2 = _make_comment(
            "As a registered nurse working in the ER I see real harm.",
            persona=persona_b,
        )

        repeated = find_repeated_phrases([c1, c2], min_n=4, max_n=4)
        assert len(repeated) > 0

        # Check that matches have correct persona info
        first = repeated[0]
        names = {m.persona_name for m in first.matches}
        assert "Jane Doe" in names
        assert "Maria Lopez" in names

    def test_longer_ngrams_subsume_shorter(self):
        """If 'registered nurse working rural' and 'registered nurse working'
        both repeat in the same comment set, only the longer one should survive."""
        c1 = _make_comment(
            "As a registered nurse working rural areas, I see the problem clearly.",
            persona=_make_persona("Jane Doe", "nurse", "Ohio"),
        )
        c2 = _make_comment(
            "Being a registered nurse working rural clinics, I oppose this.",
            persona=_make_persona("Maria Lopez", "nurse", "Texas"),
        )

        repeated = find_repeated_phrases([c1, c2], min_n=4, max_n=6)
        # The 5-gram "registered nurse working rural" should subsume
        # the 4-gram "registered nurse working" (if same comment set)
        phrases = [rp.phrase for rp in repeated]
        # Should have the longer phrase
        has_longer = any("registered nurse working rural" in p for p in phrases)
        # The shorter should be deduplicated if same indices
        if has_longer:
            # Check that we don't ALSO have just "a registered nurse working" 
            # with the exact same match set
            for rp in repeated:
                if rp.phrase == "a registered nurse working":
                    # This should have been subsumed
                    longer_indices = None
                    for rp2 in repeated:
                        if "registered nurse working rural" in rp2.phrase:
                            longer_indices = {m.comment_index for m in rp2.matches}
                    shorter_indices = {m.comment_index for m in rp.matches}
                    # If same indices, the shorter should have been removed
                    # (but it's OK if they differ)
                    if shorter_indices == longer_indices:
                        pytest.fail("Shorter n-gram should have been subsumed by longer one")


# ── Tests: deduplication helper ───────────────────────────────────────────────

class TestDeduplicateSubsumed:
    def test_removes_substring_with_same_indices(self):
        short = RepeatedPhrase(
            phrase="registered nurse working",
            ngram_length=3,
            matches=[
                PhraseMatch(0, "ID-1", "Jane", "nurse, OH", "sentence 1"),
                PhraseMatch(1, "ID-2", "Maria", "nurse, TX", "sentence 2"),
            ],
        )
        long = RepeatedPhrase(
            phrase="a registered nurse working here",
            ngram_length=5,
            matches=[
                PhraseMatch(0, "ID-1", "Jane", "nurse, OH", "sentence 1"),
                PhraseMatch(1, "ID-2", "Maria", "nurse, TX", "sentence 2"),
            ],
        )

        result = _deduplicate_subsumed([short, long])
        assert len(result) == 1
        assert result[0].phrase == "a registered nurse working here"

    def test_keeps_both_with_different_indices(self):
        short = RepeatedPhrase(
            phrase="registered nurse",
            ngram_length=2,
            matches=[
                PhraseMatch(0, "ID-1", "Jane", "nurse, OH", "sentence 1"),
                PhraseMatch(1, "ID-2", "Maria", "nurse, TX", "sentence 2"),
            ],
        )
        long = RepeatedPhrase(
            phrase="registered nurse working rural",
            ngram_length=4,
            matches=[
                PhraseMatch(0, "ID-1", "Jane", "nurse, OH", "sentence 1"),
                PhraseMatch(2, "ID-3", "Alice", "nurse, CA", "sentence 3"),
            ],
        )

        result = _deduplicate_subsumed([short, long])
        assert len(result) == 2


# ── Tests: report generation ──────────────────────────────────────────────────

class TestGenerateReport:
    def test_clean_report(self):
        report = generate_report([], total_comments=50)
        assert "No repeated distinctive phrases detected" in report
        assert "Comments analysed: 50" in report

    def test_report_with_findings(self):
        rp = RepeatedPhrase(
            phrase="registered nurse working rural",
            ngram_length=4,
            matches=[
                PhraseMatch(0, "TEST-SYNTH-0001", "Jane Doe", "nurse, Ohio", "As a registered nurse working rural areas, I see this."),
                PhraseMatch(1, "TEST-SYNTH-0002", "Maria Lopez", "nurse, Texas", "Being a registered nurse working rural clinics is tough."),
            ],
        )
        report = generate_report([rp], total_comments=10)
        assert "registered nurse working rural" in report
        assert "found in 2 comments" in report
        assert "Jane Doe" in report
        assert "Maria Lopez" in report
        assert "TEST-SYNTH-0001" in report
        assert "TEST-SYNTH-0002" in report

    def test_report_escapes_pipes(self):
        rp = RepeatedPhrase(
            phrase="test phrase here now",
            ngram_length=4,
            matches=[
                PhraseMatch(0, "ID-1", "Name|Pipe", "job, ST", "Sentence with | pipe."),
            ],
        )
        report = generate_report([rp], total_comments=1)
        # Pipes should be escaped
        assert "Name\\|Pipe" in report


# ── Tests: run_phrase_check (end-to-end with file output) ─────────────────────

class TestRunPhraseCheck:
    def test_writes_report_file(self):
        c1 = _make_comment("As a registered nurse working at a rural clinic I see the damage.")
        c2 = _make_comment("As a registered nurse working in the ER I see real harm.")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = f.name

        try:
            result_path = run_phrase_check([c1, c2], path, verbose=False)
            assert result_path == path
            assert os.path.exists(path)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Phrase Repetition Report" in content
            assert "registered nurse working" in content.lower()
        finally:
            os.unlink(path)

    def test_clean_batch_writes_clean_report(self):
        c1 = _make_comment("The certification requirements create an undue burden.")
        c2 = _make_comment("Rural communities will suffer from funding changes.")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = f.name

        try:
            run_phrase_check([c1, c2], path, verbose=False)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "No repeated distinctive phrases detected" in content
        finally:
            os.unlink(path)

    def test_empty_comments_list(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = f.name

        try:
            run_phrase_check([], path, verbose=False)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Comments analysed: 0" in content
        finally:
            os.unlink(path)


# ── Tests: realistic give-away phrases from user's examples ───────────────────

class TestRealWorldExamples:
    """Test with the exact give-away phrases mentioned in the feature request."""

    def test_detects_as_a_registered_nurse(self):
        c1 = _make_comment("As a registered nurse with 20 years of experience, I oppose this rule.")
        c2 = _make_comment("As a registered nurse in a busy ER, I know the real impact.")
        repeated = find_repeated_phrases([c1, c2], min_n=4, max_n=6)
        phrases = [rp.phrase for rp in repeated]
        assert any("registered nurse" in p for p in phrases)

    def test_detects_widely_used_commercial_algorithm(self):
        c1 = _make_comment("We rely on a widely used commercial algorithm to determine eligibility.")
        c2 = _make_comment("The widely used commercial algorithm has known biases that need addressing.")
        repeated = find_repeated_phrases([c1, c2], min_n=4, max_n=6)
        phrases = [rp.phrase for rp in repeated]
        assert any("widely used commercial algorithm" in p for p in phrases)

    def test_detects_i_live_in_specific_city(self):
        c1 = _make_comment("I live in Duluth MN and the nearest hospital is 30 miles away.")
        c2 = _make_comment("I live in Duluth MN where healthcare access is already limited.")
        repeated = find_repeated_phrases([c1, c2], min_n=4, max_n=6)
        phrases = [rp.phrase for rp in repeated]
        assert any("live in duluth mn" in p for p in phrases)
