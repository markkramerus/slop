"""
export.py — Emit accepted synthetic comments as a CSV that mirrors the
Regulations.gov docket export format.

The output CSV has two layers of columns:
  1. Standard Regulations.gov columns (Comment ID, Document ID, Submitter Name,
     Organization Name, Government Agency Type, Government Agency, Abstract,
     Comment, Attachment Files, Posted Date, …) — so synthetic rows can be
     interleaved with real rows for detector evaluation.
  2. Research-only columns (prefixed with "synth_") that carry the ground-truth
     metadata (archetype, vector, persona details, QC results, frame summary).
     These can be stripped when building blind evaluation datasets.
"""

from __future__ import annotations

import csv
import datetime
import io
import random
import string
from typing import Sequence

import numpy as np

from .generator import GeneratedComment


# ── Column definitions ────────────────────────────────────────────────────────

_REGS_GOV_COLUMNS = [
    "Comment ID",
    "Document ID",
    "Submitter Name",
    "Organization Name",
    "Submitter's Representative",
    "Government Agency Type",
    "Government Agency",
    "Abstract",
    "Comment",
    "Attachment Files",
    "Posted Date",
    "Received Date",
    "Comment Start Date",
    "Comment End Date",
    "Page Count",
    "Federal Register Number",
    "Exhibit Type",
    "Exhibit Location",
]

_RESEARCH_COLUMNS = [
    "synth_is_synthetic",
    "synth_vector",
    "synth_objective",
    "synth_archetype",
    "synth_sophistication",
    "synth_emotional_register",
    "synth_persona_state",
    "synth_persona_occupation",
    "synth_persona_age",
    "synth_personal_hook",
    "synth_core_arguments",
    "synth_framing",
    "synth_citation_agenda",
    "synth_qc_passed",
    "synth_qc_notes",
    "synth_word_count",
]

ALL_COLUMNS = _REGS_GOV_COLUMNS + _RESEARCH_COLUMNS


# ── ID generation ─────────────────────────────────────────────────────────────

def _make_comment_id(docket_id: str, index: int) -> str:
    """Generate a plausible-looking comment ID."""
    suffix = str(index + 1).zfill(4)
    return f"{docket_id}-SYNTH-{suffix}"


# ── Timing ────────────────────────────────────────────────────────────────────

def _posted_date(
    timing_deciles: list[float],
    comment_period_days: int,
    start_date: datetime.date,
    rng: np.random.Generator,
) -> str:
    """
    Sample a plausible posted date from the timing distribution.
    Real comments cluster near the deadline — timing_deciles encodes this.
    """
    if not timing_deciles or sum(timing_deciles) == 0:
        decile_probs = [0.1] * 10
    else:
        total = sum(timing_deciles)
        decile_probs = [d / total for d in timing_deciles]

    decile = int(rng.choice(10, p=decile_probs))
    # Day within the decile
    day_offset = int(
        decile * (comment_period_days / 10)
        + rng.uniform(0, comment_period_days / 10)
    )
    day_offset = min(day_offset, comment_period_days - 1)
    date = start_date + datetime.timedelta(days=day_offset)
    return date.strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Row builder ───────────────────────────────────────────────────────────────

def _build_row(
    comment: GeneratedComment,
    index: int,
    timing_deciles: list[float],
    comment_period_days: int,
    comment_start_date: datetime.date,
    rng: np.random.Generator,
) -> dict[str, str]:
    docket_id = comment.docket_id or "UNKNOWN"
    posted = _posted_date(timing_deciles, comment_period_days, comment_start_date, rng)

    # Regulations.gov columns
    row: dict[str, str] = {
        "Comment ID": _make_comment_id(docket_id, index),
        "Document ID": docket_id,
        "Submitter Name": comment.persona.full_name,
        "Organization Name": comment.persona.org_name,
        "Submitter's Representative": "",
        "Government Agency Type": "Government Agency" if comment.persona.archetype == "government" else "",
        "Government Agency": "",
        "Abstract": comment.comment_text[:500],
        "Comment": comment.comment_text,
        "Attachment Files": "",
        "Posted Date": posted,
        "Received Date": posted,
        "Comment Start Date": comment_start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Comment End Date": (
            comment_start_date + datetime.timedelta(days=comment_period_days)
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Page Count": str(max(1, comment.word_count() // 300)),
        "Federal Register Number": "",
        "Exhibit Type": "",
        "Exhibit Location": "",
    }

    # Research-only columns
    row.update({
        "synth_is_synthetic": "TRUE",
        "synth_vector": str(comment.vector),
        "synth_objective": comment.objective,
        "synth_archetype": comment.persona.archetype,
        "synth_sophistication": comment.persona.sophistication,
        "synth_emotional_register": comment.persona.emotional_register,
        "synth_persona_state": comment.persona.state,
        "synth_persona_occupation": comment.persona.occupation,
        "synth_persona_age": str(comment.persona.age),
        "synth_personal_hook": comment.persona.personal_hook,
        "synth_core_arguments": " | ".join(comment.frame.core_arguments),
        "synth_framing": comment.frame.framing,
        "synth_citation_agenda": " | ".join(comment.frame.citation_agenda),
        "synth_qc_passed": str(comment.qc_passed),
        "synth_qc_notes": comment.qc_notes,
        "synth_word_count": str(comment.word_count()),
    })

    return row


# ── Public API ────────────────────────────────────────────────────────────────

def export_to_csv(
    comments: Sequence[GeneratedComment],
    output_path: str,
    timing_deciles: list[float] | None = None,
    comment_period_days: int = 60,
    comment_start_date: datetime.date | None = None,
    include_failed_qc: bool = False,
    seed: int = 42,
) -> int:
    """
    Write accepted synthetic comments to a CSV file.

    Parameters
    ----------
    comments:
        List of GeneratedComment objects (may include QC failures).
    output_path:
        Destination CSV file path.
    timing_deciles:
        10-element list of relative comment-frequency per decile of the comment
        period.  If None, uniform distribution is used.
    comment_period_days:
        Length of the comment period in days (for posted-date simulation).
    comment_start_date:
        Start date of the comment period.  Defaults to 90 days ago.
    include_failed_qc:
        If True, write all comments including QC failures (with the
        synth_qc_passed column set to FALSE).
    seed:
        Random seed for date simulation.

    Returns
    -------
    int
        Number of rows written.
    """
    rng = np.random.default_rng(seed)

    if timing_deciles is None:
        timing_deciles = [0.1] * 10

    if comment_start_date is None:
        comment_start_date = datetime.date.today() - datetime.timedelta(days=90)

    target = [c for c in comments if include_failed_qc or c.qc_passed]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        for i, comment in enumerate(target):
            row = _build_row(
                comment, i, timing_deciles, comment_period_days,
                comment_start_date, rng
            )
            writer.writerow(row)

    return len(target)


def to_csv_string(
    comments: Sequence[GeneratedComment],
    timing_deciles: list[float] | None = None,
    comment_period_days: int = 60,
    comment_start_date: datetime.date | None = None,
    include_failed_qc: bool = False,
    seed: int = 42,
) -> str:
    """
    Same as export_to_csv but returns the CSV content as a string.
    Useful for testing or programmatic downstream processing.
    """
    rng = np.random.default_rng(seed)

    if timing_deciles is None:
        timing_deciles = [0.1] * 10

    if comment_start_date is None:
        comment_start_date = datetime.date.today() - datetime.timedelta(days=90)

    target = [c for c in comments if include_failed_qc or c.qc_passed]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=ALL_COLUMNS)
    writer.writeheader()
    for i, comment in enumerate(target):
        row = _build_row(
            comment, i, timing_deciles, comment_period_days,
            comment_start_date, rng
        )
        writer.writerow(row)

    return buf.getvalue()
