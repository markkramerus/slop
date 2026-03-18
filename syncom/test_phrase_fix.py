"""
test_phrase_fix.py — Unit tests for syncom/phrase_fix.py.

Run:  python -m pytest syncom/test_phrase_fix.py -v

These tests cover the pure, side-effect-free functions only.  The rewriter
and PSV-patching functions require live API keys and are not tested here.
"""

from __future__ import annotations

import json
import os
import tempfile
import textwrap

import pytest

from syncom.phrase_fix import (
    build_rule_ngrams,
    is_rule_anchored,
    triage_repeated_phrases,
    parse_phrase_report,
    _build_phrase_criticism,
    _next_versioned_path,
    PhraseFixResult,
)
from syncom.phrase_check import (
    RepeatedPhrase,
    PhraseMatch,
    WATCH_LIST,
    find_watch_list_phrases,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Minimal rule text that covers the HHS AI RFI vocabulary
_RULE_TEXT = """
HHS Health Sector AI Request for Information: Accelerating AI Adoption in Clinical Care.

What are the biggest barriers to private sector innovation in AI for health care
and its adoption and use in clinical care?

What regulatory, payment policy, or programmatic design changes should HHS
prioritize to incentivize the effective use of AI in clinical care?

For non-medical devices, what novel legal and implementation issues exist
relating to liability, indemnification, privacy, and security?

How can HHS best support private sector activities such as accreditation,
certification, industry-driven testing, and credentialing?

Where have AI tools deployed in clinical care met or exceeded performance
and cost expectations and where have they fallen short?

What challenges do patients and caregivers wish to see addressed by AI in
clinical care, and what concerns do they have about its adoption?

Fee-for-service reimbursement. Value-based care. Interoperability.
Algorithmic accountability. Health data privacy. Robustness testing.
Public-private partnerships. OneHHS approach.
"""

# A world model JSON with key_terms and rfi_questions
_WORLD_MODEL = {
    "rule_title": "HHS Health Sector AI RFI",
    "key_terms": [
        "Artificial intelligence (AI) in clinical care",
        "Fee-for-service reimbursement",
        "Value-based care",
        "Non-medical device AI",
        "AI evaluation methods",
        "Interoperability",
        "Algorithmic accountability",
        "Health data privacy",
        "Accreditation and certification",
        "Robustness testing",
        "Public-private partnerships",
    ],
    "rfi_questions": [
        "What are the biggest barriers to private sector innovation in AI for health care?",
        "What regulatory, payment policy, or programmatic design changes should HHS prioritize?",
        "What challenges do patients and caregivers wish to see addressed by AI in clinical care?",
    ],
}


def _make_world_model_file(tmp_path: str) -> str:
    """Write a world model JSON to a temp file and return its path."""
    path = os.path.join(tmp_path, "world_model.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_WORLD_MODEL, f)
    return path


def _make_repeated_phrase(phrase: str, comment_ids: list[str]) -> RepeatedPhrase:
    """Build a minimal RepeatedPhrase for testing."""
    matches = [
        PhraseMatch(
            comment_index=i,
            comment_id=cid,
            persona_name=f"Person {i}",
            persona_detail="occupation, State",
            sentence=f"Sentence containing {phrase}.",
        )
        for i, cid in enumerate(comment_ids)
    ]
    return RepeatedPhrase(
        phrase=phrase,
        ngram_length=len(phrase.split()),
        matches=matches,
    )


# ── Tests: build_rule_ngrams ──────────────────────────────────────────────────

class TestBuildRuleNgrams:
    def test_returns_frozenset(self):
        ngrams = build_rule_ngrams(_RULE_TEXT)
        assert isinstance(ngrams, frozenset)

    def test_contains_bigrams(self):
        ngrams = build_rule_ngrams(_RULE_TEXT)
        # "clinical care" appears in the rule text
        assert ("clinical", "care") in ngrams

    def test_contains_trigrams(self):
        ngrams = build_rule_ngrams(_RULE_TEXT)
        # "private sector innovation" appears in the rule text
        assert ("private", "sector", "innovation") in ngrams

    def test_contains_4grams(self):
        ngrams = build_rule_ngrams(_RULE_TEXT)
        # "ai in clinical care" appears in the rule text
        assert ("ai", "in", "clinical", "care") in ngrams

    def test_respects_min_max_n(self):
        ngrams_2_4 = build_rule_ngrams(_RULE_TEXT, min_n=2, max_n=4)
        ngrams_3_3 = build_rule_ngrams(_RULE_TEXT, min_n=3, max_n=3)
        # 3-only set should be a subset of 2-4 set
        assert ngrams_3_3.issubset(ngrams_2_4)
        # 2-4 set should be strictly larger (has bigrams and 4-grams too)
        assert len(ngrams_2_4) > len(ngrams_3_3)

    def test_includes_world_model_terms(self, tmp_path):
        wm_path = _make_world_model_file(str(tmp_path))
        ngrams = build_rule_ngrams("minimal rule text", world_model_path=wm_path)
        # "fee for service reimbursement" is a key term
        assert ("fee", "for", "service") in ngrams

    def test_world_model_not_found_does_not_raise(self):
        # Should silently fall back to rule text only
        ngrams = build_rule_ngrams(_RULE_TEXT, world_model_path="/nonexistent/path.json")
        assert len(ngrams) > 0

    def test_empty_rule_text_returns_empty(self):
        ngrams = build_rule_ngrams("")
        assert len(ngrams) == 0


# ── Tests: is_rule_anchored ───────────────────────────────────────────────────

class TestIsRuleAnchored:
    """
    Verify the core triage logic: rule-vocabulary phrases → True,
    free-floating personal narrative phrases → False.
    """

    def setup_method(self):
        self.rule_ngrams = build_rule_ngrams(_RULE_TEXT)

    # ── Expected (rule-anchored) phrases ──────────────────────────────────

    def test_barriers_to_private_sector_innovation(self):
        # "private sector innovation" is in the RFI questions
        assert is_rule_anchored("barriers to private sector innovation", self.rule_ngrams)

    def test_accelerating_ai_adoption_in_clinical_care(self):
        # "ai adoption in clinical care" is in the rule title
        assert is_rule_anchored("accelerating ai adoption in clinical care", self.rule_ngrams)

    def test_hhs_health_sector_ai(self):
        # "hhs health sector ai" appears in the rule text title
        assert is_rule_anchored("hhs health sector ai request for information", self.rule_ngrams)

    def test_fee_for_service_reimbursement(self):
        # "fee for service reimbursement" is a key term
        rule_ngrams_with_wm = build_rule_ngrams(_RULE_TEXT)
        # Even without world model, "fee for service" appears in rule text
        assert is_rule_anchored("fee for service reimbursement", self.rule_ngrams)

    def test_ai_in_clinical_care(self):
        assert is_rule_anchored("ai in clinical care", self.rule_ngrams)

    def test_patients_and_caregivers(self):
        # "patients and caregivers" appears in the RFI questions
        assert is_rule_anchored("patients and caregivers wish to see", self.rule_ngrams)

    def test_public_private_partnerships(self):
        assert is_rule_anchored("public private partnerships", self.rule_ngrams)

    # ── Suspicious (not rule-anchored) phrases ────────────────────────────

    def test_last_spring_my_cardiologist(self):
        # The canonical example from the user's bug report
        assert not is_rule_anchored("last spring my cardiologist", self.rule_ngrams)

    def test_a_straight_answer_about(self):
        assert not is_rule_anchored("a straight answer about", self.rule_ngrams)

    def test_patient_gets_hurt_who_exactly_is_responsible(self):
        assert not is_rule_anchored("patient gets hurt who exactly is responsible", self.rule_ngrams)

    def test_the_doctor_blames_the(self):
        assert not is_rule_anchored("the doctor blames the", self.rule_ngrams)

    def test_i_am_77_years_old(self):
        assert not is_rule_anchored("i am 77 years old", self.rule_ngrams)

    def test_last_spring_i_sat_in(self):
        assert not is_rule_anchored("last spring i sat in", self.rule_ngrams)

    def test_my_mother_nearly_died(self):
        assert not is_rule_anchored("my mother nearly died", self.rule_ngrams)

    def test_retired_teacher_michigan(self):
        assert not is_rule_anchored("retired teacher michigan", self.rule_ngrams)

    def test_single_word_phrase_not_anchored(self):
        # Single-word phrases can't form a bigram — should return False
        assert not is_rule_anchored("cardiologist", self.rule_ngrams)

    def test_empty_phrase_not_anchored(self):
        assert not is_rule_anchored("", self.rule_ngrams)


# ── Tests: triage_repeated_phrases ───────────────────────────────────────────

class TestTriageRepeatedPhrases:
    def setup_method(self):
        self.rule_ngrams = build_rule_ngrams(_RULE_TEXT)

    def test_splits_into_expected_and_suspicious(self):
        phrases = [
            _make_repeated_phrase("barriers to private sector innovation", ["ID-1", "ID-2", "ID-3"]),
            _make_repeated_phrase("last spring my cardiologist", ["ID-4", "ID-5", "ID-6"]),
            _make_repeated_phrase("ai in clinical care", ["ID-7", "ID-8", "ID-9"]),
            _make_repeated_phrase("my mother nearly died", ["ID-10", "ID-11", "ID-12"]),
        ]
        expected, suspicious = triage_repeated_phrases(phrases, self.rule_ngrams)

        expected_phrases = {rp.phrase for rp in expected}
        suspicious_phrases = {rp.phrase for rp in suspicious}

        assert "barriers to private sector innovation" in expected_phrases
        assert "ai in clinical care" in expected_phrases
        assert "last spring my cardiologist" in suspicious_phrases
        assert "my mother nearly died" in suspicious_phrases

    def test_empty_input_returns_empty_lists(self):
        expected, suspicious = triage_repeated_phrases([], self.rule_ngrams)
        assert expected == []
        assert suspicious == []

    def test_all_expected(self):
        phrases = [
            _make_repeated_phrase("barriers to private sector innovation", ["ID-1", "ID-2"]),
            _make_repeated_phrase("ai in clinical care", ["ID-3", "ID-4"]),
        ]
        expected, suspicious = triage_repeated_phrases(phrases, self.rule_ngrams)
        assert len(expected) == 2
        assert len(suspicious) == 0

    def test_all_suspicious(self):
        phrases = [
            _make_repeated_phrase("last spring my cardiologist", ["ID-1", "ID-2"]),
            _make_repeated_phrase("my mother nearly died", ["ID-3", "ID-4"]),
        ]
        expected, suspicious = triage_repeated_phrases(phrases, self.rule_ngrams)
        assert len(expected) == 0
        assert len(suspicious) == 2

    def test_preserves_match_data(self):
        rp = _make_repeated_phrase("last spring my cardiologist", ["ID-1", "ID-2", "ID-3"])
        _, suspicious = triage_repeated_phrases([rp], self.rule_ngrams)
        assert len(suspicious) == 1
        assert suspicious[0].count == 3
        assert suspicious[0].matches[0].comment_id == "ID-1"


# ── Tests: parse_phrase_report ────────────────────────────────────────────────

class TestParsePhraseReport:
    """
    Test the Markdown phrase report parser against hand-crafted report snippets.
    """

    def _write_report(self, content: str) -> str:
        """Write content to a temp file and return its path."""
        fd, path = tempfile.mkstemp(suffix=".md")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_parses_single_phrase(self):
        report = textwrap.dedent("""\
            # Phrase Repetition Report
            Generated: 2026-03-17 10:00
            Comments analysed: 50 | Repeated phrases found: 1

            ---

            ## 1. "last spring my cardiologist" (found in 3 comments)

            | # | Document ID | Submitter | Sentence |
            |---|-----------|---------|----------|
            | 1 | SYNTH-0017 | Susan Scott (retired teacher, Michigan) | Last spring my cardiologist mentioned something. |
            | 2 | SYNTH-0026 | Michael Jackson (retired teacher, Wyoming) | Last spring my cardiologist in Casper admitted something. |
            | 3 | SYNTH-0041 | Anthony Walker (truck driver, Nebraska) | Last spring my cardiologist was typing my symptoms. |
        """)
        path = self._write_report(report)
        try:
            result = parse_phrase_report(path)
            assert len(result) == 1
            rp = result[0]
            assert rp.phrase == "last spring my cardiologist"
            assert rp.count == 3
            assert rp.matches[0].comment_id == "SYNTH-0017"
            assert rp.matches[1].comment_id == "SYNTH-0026"
            assert rp.matches[2].comment_id == "SYNTH-0041"
        finally:
            os.unlink(path)

    def test_parses_multiple_phrases(self):
        report = textwrap.dedent("""\
            # Phrase Repetition Report
            Generated: 2026-03-17 10:00
            Comments analysed: 50 | Repeated phrases found: 2

            ---

            ## 1. "last spring my cardiologist" (found in 3 comments)

            | # | Document ID | Submitter | Sentence |
            |---|-----------|---------|----------|
            | 1 | SYNTH-0017 | Susan Scott (retired teacher, Michigan) | Last spring my cardiologist mentioned something. |
            | 2 | SYNTH-0026 | Michael Jackson (retired teacher, Wyoming) | Last spring my cardiologist in Casper. |
            | 3 | SYNTH-0041 | Anthony Walker (truck driver, Nebraska) | Last spring my cardiologist was typing. |

            ---

            ## 2. "barriers to private sector innovation" (found in 2 comments)

            | # | Document ID | Submitter | Sentence |
            |---|-----------|---------|----------|
            | 1 | SYNTH-0005 | Jane Doe (researcher, Ohio) | The barriers to private sector innovation are significant. |
            | 2 | SYNTH-0012 | Bob Smith (policy analyst, DC) | We must address barriers to private sector innovation. |
        """)
        path = self._write_report(report)
        try:
            result = parse_phrase_report(path)
            assert len(result) == 2
            phrases = [rp.phrase for rp in result]
            assert "last spring my cardiologist" in phrases
            assert "barriers to private sector innovation" in phrases
        finally:
            os.unlink(path)

    def test_parses_submitter_name_with_detail(self):
        report = textwrap.dedent("""\
            # Phrase Repetition Report
            Generated: 2026-03-17 10:00
            Comments analysed: 10 | Repeated phrases found: 1

            ---

            ## 1. "test phrase here now" (found in 2 comments)

            | # | Document ID | Submitter | Sentence |
            |---|-----------|---------|----------|
            | 1 | ID-1 | Jane Doe (nurse, Ohio) | Test phrase here now in context. |
            | 2 | ID-2 | Bob Smith | Test phrase here now again. |
        """)
        path = self._write_report(report)
        try:
            result = parse_phrase_report(path)
            assert len(result) == 1
            rp = result[0]
            # First match has detail
            assert rp.matches[0].persona_name == "Jane Doe"
            assert rp.matches[0].persona_detail == "nurse, Ohio"
            # Second match has no detail
            assert rp.matches[1].persona_name == "Bob Smith"
            assert rp.matches[1].persona_detail == ""
        finally:
            os.unlink(path)

    def test_empty_report_returns_empty_list(self):
        report = textwrap.dedent("""\
            # Phrase Repetition Report
            Generated: 2026-03-17 10:00
            Comments analysed: 50 | Repeated phrases found: 0

            ✅ **No repeated distinctive phrases detected.** The batch looks clean.
        """)
        path = self._write_report(report)
        try:
            result = parse_phrase_report(path)
            assert result == []
        finally:
            os.unlink(path)

    def test_parses_sentence_text(self):
        report = textwrap.dedent("""\
            # Phrase Repetition Report
            Generated: 2026-03-17 10:00
            Comments analysed: 10 | Repeated phrases found: 1

            ---

            ## 1. "last spring my cardiologist" (found in 2 comments)

            | # | Document ID | Submitter | Sentence |
            |---|-----------|---------|----------|
            | 1 | SYNTH-0017 | Susan Scott (retired teacher, Michigan) | Last spring my cardiologist mentioned that a computer program had caught something. |
            | 2 | SYNTH-0026 | Michael Jackson (retired teacher, Wyoming) | Last spring my cardiologist in Casper admitted almost in passing that the software was wrong. |
        """)
        path = self._write_report(report)
        try:
            result = parse_phrase_report(path)
            assert len(result) == 1
            rp = result[0]
            assert "cardiologist mentioned" in rp.matches[0].sentence
            assert "cardiologist in Casper" in rp.matches[1].sentence
        finally:
            os.unlink(path)


# ── Tests: _build_phrase_criticism ────────────────────────────────────────────

class TestBuildPhraseCriticism:
    """
    _build_phrase_criticism now takes a list of (phrase, sentence, count) tuples.
    The single-item case should still contain the phrase, sentence, and count.
    """

    def test_contains_phrase(self):
        criticism = _build_phrase_criticism([
            ("last spring my cardiologist", "Last spring my cardiologist mentioned something.", 3),
        ])
        assert "last spring my cardiologist" in criticism

    def test_contains_sentence(self):
        sentence = "Last spring my cardiologist mentioned something."
        criticism = _build_phrase_criticism([
            ("last spring my cardiologist", sentence, 3),
        ])
        assert sentence in criticism

    def test_contains_occurrence_count(self):
        criticism = _build_phrase_criticism([
            ("last spring my cardiologist", "Last spring my cardiologist mentioned something.", 3),
        ])
        assert "3" in criticism

    def test_is_non_empty_string(self):
        criticism = _build_phrase_criticism([("test phrase", "Test phrase sentence.", 2)])
        assert isinstance(criticism, str)
        assert len(criticism) > 50

    def test_multi_phrase_contains_all_phrases(self):
        criticism = _build_phrase_criticism([
            ("last spring my cardiologist", "Last spring my cardiologist mentioned something.", 3),
            ("i am 77 years old", "I am 77 years old and live in Omaha.", 2),
        ])
        assert "last spring my cardiologist" in criticism
        assert "i am 77 years old" in criticism
        assert "3" in criticism
        assert "2" in criticism


# ── Tests: PhraseFixResult ────────────────────────────────────────────────────

class TestPhraseFixResult:
    def test_summary_contains_all_fields(self):
        result = PhraseFixResult(
            phrases_found=10,
            phrases_expected=7,
            phrases_suspicious=3,
            comments_rewritten=5,
            output_path="/path/to/output.txt",
            report_path="/path/to/report.md",
            elapsed_seconds=12.5,
        )
        summary = result.summary()
        assert "10" in summary
        assert "7" in summary
        assert "3" in summary
        assert "5" in summary
        assert "/path/to/output.txt" in summary
        assert "/path/to/report.md" in summary
        assert "12.5" in summary

    def test_default_values(self):
        result = PhraseFixResult()
        assert result.phrases_found == 0
        assert result.phrases_expected == 0
        assert result.phrases_suspicious == 0
        assert result.comments_rewritten == 0
        assert result.output_path == ""
        assert result.report_path == ""
        assert result.elapsed_seconds == 0.0


# ── Tests: WATCH_LIST and find_watch_list_phrases ────────────────────────────

class TestWatchList:
    """Tests for the WATCH_LIST constant and find_watch_list_phrases function."""

    def test_watch_list_is_non_empty(self):
        assert len(WATCH_LIST) > 0

    def test_last_spring_in_watch_list(self):
        assert "last spring" in WATCH_LIST

    def test_watch_list_phrases_are_lowercase(self):
        for phrase in WATCH_LIST:
            assert phrase == phrase.lower(), f"Watch list phrase not lowercase: {phrase!r}"

    def test_watch_list_no_duplicates(self):
        assert len(WATCH_LIST) == len(set(WATCH_LIST))


class TestFindWatchListPhrases:
    """Tests for find_watch_list_phrases using in-memory GeneratedComment stubs."""

    def _make_comments(self, texts: list[str]):
        """Build minimal GeneratedComment stubs from a list of text strings."""
        from syncom.generator import GeneratedComment
        from syncom.persona import Persona
        from syncom.argument_mapper import ExpressionFrame

        persona = Persona(
            archetype="individual_consumer",
            first_name="Test",
            last_name="User",
            state="Ohio",
            occupation="nurse",
            age=45,
            sophistication="medium",
            emotional_register="concerned",
            org_name="",
            personal_stake="",
            personal_hook="",
        )
        frame = ExpressionFrame(
            core_arguments=[],
            framing="",
            evidence_types=[],
            rfi_questions_to_address=[],
        )
        comments = []
        for i, text in enumerate(texts):
            gc = GeneratedComment(
                comment_text=text,
                persona=persona,
                frame=frame,
                vector=0,
                objective="",
                rule_title="",
                docket_id="TEST-001",
                document_id=f"TEST-001-SYNTH-{i+1:04d}",
                qc_passed=True,
            )
            comments.append(gc)
        return comments

    def test_detects_last_spring(self):
        texts = [
            "Last spring, I visited my doctor and was surprised.",
            "Last spring my cardiologist mentioned something unusual.",
            "Last spring I got a letter from my insurance company.",
        ]
        comments = self._make_comments(texts)
        hits = find_watch_list_phrases(comments, only_passed_qc=False, min_count=2)
        phrases = [rp.phrase for rp in hits]
        assert "last spring" in phrases

    def test_does_not_flag_rare_phrase(self):
        # "last spring" appears only once — should not be flagged with min_count=2
        texts = [
            "Last spring I visited my doctor.",
            "I have concerns about AI in healthcare.",
            "The proposed rule raises important questions.",
        ]
        comments = self._make_comments(texts)
        hits = find_watch_list_phrases(comments, only_passed_qc=False, min_count=2)
        phrases = [rp.phrase for rp in hits]
        assert "last spring" not in phrases

    def test_returns_correct_match_count(self):
        texts = [
            "Last spring, I watched a colleague struggle with an AI diagnosis.",
            "Last spring my cardiologist mentioned a computer program.",
            "Last spring I sat across from my doctor.",
            "I have no concerns about AI.",
        ]
        comments = self._make_comments(texts)
        hits = find_watch_list_phrases(comments, only_passed_qc=False, min_count=2)
        last_spring_hits = [rp for rp in hits if rp.phrase == "last spring"]
        assert len(last_spring_hits) == 1
        assert last_spring_hits[0].count == 3

    def test_empty_comments_returns_empty(self):
        hits = find_watch_list_phrases([], only_passed_qc=False, min_count=2)
        assert hits == []

    def test_case_insensitive_matching(self):
        texts = [
            "LAST SPRING I visited my doctor.",
            "Last Spring my cardiologist mentioned something.",
            "last spring I got a letter.",
        ]
        comments = self._make_comments(texts)
        hits = find_watch_list_phrases(comments, only_passed_qc=False, min_count=2)
        phrases = [rp.phrase for rp in hits]
        assert "last spring" in phrases


# ── Tests: _next_versioned_path ───────────────────────────────────────────────

class TestNextVersionedPath:
    def test_base_file_gets_r1(self, tmp_path):
        # synthetic.txt does not exist → synthetic_r1.txt
        src = str(tmp_path / "synthetic.txt")
        result = _next_versioned_path(src)
        assert result == str(tmp_path / "synthetic_r1.txt")

    def test_r1_exists_gets_r2(self, tmp_path):
        # synthetic_r1.txt exists → synthetic_r2.txt
        (tmp_path / "synthetic_r1.txt").write_text("x")
        src = str(tmp_path / "synthetic.txt")
        result = _next_versioned_path(src)
        assert result == str(tmp_path / "synthetic_r2.txt")

    def test_r1_and_r2_exist_gets_r3(self, tmp_path):
        (tmp_path / "synthetic_r1.txt").write_text("x")
        (tmp_path / "synthetic_r2.txt").write_text("x")
        src = str(tmp_path / "synthetic.txt")
        result = _next_versioned_path(src)
        assert result == str(tmp_path / "synthetic_r3.txt")

    def test_input_already_versioned_strips_suffix(self, tmp_path):
        # synthetic_r1.txt → synthetic_r2.txt (strips _r1, finds next)
        (tmp_path / "synthetic_r1.txt").write_text("x")
        src = str(tmp_path / "synthetic_r1.txt")
        result = _next_versioned_path(src)
        assert result == str(tmp_path / "synthetic_r2.txt")

    def test_preserves_extension(self, tmp_path):
        src = str(tmp_path / "comments.psv")
        result = _next_versioned_path(src)
        assert result.endswith("_r1.psv")


# ── Integration: triage against real HHS-ONC world model ─────────────────────

class TestTriageWithRealWorldModel:
    """
    End-to-end triage test using the actual world_model.json from the
    HHS-ONC-2026-0001 docket (if present).  Skipped if the file is not found.
    """

    WORLD_MODEL_PATH = "HHS-ONC-2026-0001/world_model.json"

    @pytest.fixture(autouse=True)
    def skip_if_no_world_model(self):
        if not os.path.exists(self.WORLD_MODEL_PATH):
            pytest.skip(f"World model not found: {self.WORLD_MODEL_PATH}")

    def _load_rule_text(self) -> str:
        with open(self.WORLD_MODEL_PATH, "r", encoding="utf-8") as f:
            wm = json.load(f)
        return wm.get("rule_text", "")

    def test_last_spring_my_cardiologist_is_suspicious(self):
        rule_text = self._load_rule_text()
        rule_ngrams = build_rule_ngrams(rule_text, self.WORLD_MODEL_PATH)
        assert not is_rule_anchored("last spring my cardiologist", rule_ngrams)

    def test_barriers_to_private_sector_innovation_is_expected(self):
        rule_text = self._load_rule_text()
        rule_ngrams = build_rule_ngrams(rule_text, self.WORLD_MODEL_PATH)
        assert is_rule_anchored("barriers to private sector innovation", rule_ngrams)

    def test_accelerating_ai_adoption_is_expected(self):
        rule_text = self._load_rule_text()
        rule_ngrams = build_rule_ngrams(rule_text, self.WORLD_MODEL_PATH)
        assert is_rule_anchored("accelerating ai adoption in clinical care", rule_ngrams)

    def test_department_of_health_and_human_services_is_expected(self):
        rule_text = self._load_rule_text()
        rule_ngrams = build_rule_ngrams(rule_text, self.WORLD_MODEL_PATH)
        assert is_rule_anchored("department of health and human services", rule_ngrams)

    def test_my_mother_nearly_died_is_suspicious(self):
        rule_text = self._load_rule_text()
        rule_ngrams = build_rule_ngrams(rule_text, self.WORLD_MODEL_PATH)
        assert not is_rule_anchored("my mother nearly died", rule_ngrams)

    def test_i_am_77_years_old_is_suspicious(self):
        rule_text = self._load_rule_text()
        rule_ngrams = build_rule_ngrams(rule_text, self.WORLD_MODEL_PATH)
        assert not is_rule_anchored("i am 77 years old", rule_ngrams)

    def test_the_doctor_blames_the_is_suspicious(self):
        rule_text = self._load_rule_text()
        rule_ngrams = build_rule_ngrams(rule_text, self.WORLD_MODEL_PATH)
        assert not is_rule_anchored("the doctor blames the", rule_ngrams)
