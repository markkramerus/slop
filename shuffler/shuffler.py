"""
shuffler/shuffler.py — Core shuffler logic for the slop pipeline.

This module:
  1. Pre-processes the real CMS comments file, substituting attachment text
     for the comment body when the attachment contains the bulk of the comment,
     then writes the result as a ♔-delimited PSV file.
  2. Translates syncom output (♔-delimited) to a ♔-PSV using the CMS schema.
  3. Randomly interleaves the translated synthetic comments with the
     pre-processed real comments file.
  4. Produces a combined ♔-PSV (with attachment URLs cleared) and a key CSV
     that identifies every row as "real" or "synthetic".

PSV format (♔-Separated Values)
--------------------------------
Field separator        : ♔  (U+2654 WHITE CHESS KING)
Record separator       : \\n (one record per line)
Newline within a field : ⏎  (U+23CE RETURN SYMBOL) — restored on read
♔ within a field value : < >

See shuffler/psv_io.py for the read/write utilities.
"""

import csv
import os
import random
import sys
from pathlib import Path

from shuffler import psv_io

# Some attachment texts are very large; raise the csv field-size limit to match.
# sys.maxsize overflows the C long on Windows 64-bit, so cap at 2**31 - 1.
csv.field_size_limit(2**31 - 1)


# ── Step 0: Pre-process real comments (resolve attachments) ───────────────────

def preprocess_real_comments(
    real_cms_file: str,
    attachments_dir: str,
    output_file: str,
    verbose: bool = True,
) -> dict:
    """
    Pre-process the real CMS comments CSV by substituting attachment text when
    the attachment contains the bulk of the comment, then write the result as a
    ♔-delimited PSV file so that downstream steps never touch the fragile
    comma-CSV format.

    For each row in real_cms_file:
      - Look up the comment's Document ID in attachments_dir/<Document ID>/
      - Read all *.txt files found there and concatenate their text.
      - Compare len(attachment_text) vs len(comment_text).
      - Use whichever is longer as the "Comment" column value.

    The "Attachment Files" column is deliberately left intact here; it will be
    cleared by shuffle_comments() in the final merged output.

    Args:
        real_cms_file:   Path to the original real CMS comments CSV.
        attachments_dir: Path to the directory containing per-comment attachment
                         subdirectories (e.g. "CMS-2025-0050/comment_attachments").
        output_file:     Path where the pre-processed PSV will be written.
        verbose:         Print progress messages.

    Returns:
        dict with keys: total_rows, rows_with_attachments, rows_substituted,
                        output_file.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Always read the source from comma-CSV (official docket export format)
    rows, fieldnames = _load_csv(real_cms_file)

    attachments_root = Path(attachments_dir)
    rows_with_attachments = 0
    rows_substituted = 0

    if verbose:
        print(f"[shuffler] Pre-processing real comments")
        print(f"           Input       : {real_cms_file}")
        print(f"           Attachments : {attachments_dir}")
        print(f"           Output      : {output_file}")

    for row in rows:
        doc_id = row.get("Document ID", "").strip()
        if not doc_id:
            continue

        attachment_subdir = attachments_root / doc_id
        if not attachment_subdir.is_dir():
            continue

        # Collect all .txt files in the attachment directory
        txt_files = sorted(attachment_subdir.glob("*.txt"))
        if not txt_files:
            continue

        rows_with_attachments += 1

        # Concatenate text from all attachment .txt files
        parts = []
        for txt_path in txt_files:
            try:
                parts.append(
                    txt_path.read_text(encoding="utf-8", errors="replace").strip()
                )
            except OSError:
                pass
        attachment_text = "\n\n".join(p for p in parts if p)

        # Compare lengths; use whichever is longer
        comment_text = row.get("Comment", "") or ""
        if len(attachment_text) > len(comment_text):
            row["Comment"] = attachment_text
            rows_substituted += 1
            if verbose:
                print(
                    f"[shuffler]   {doc_id}: attachment text used "
                    f"({len(attachment_text):,} chars vs {len(comment_text):,} in body)"
                )

    # Write pre-processed output as ♔-PSV
    # psv_io.write_psv encodes newlines as ⏎ so the file is safe to parse
    # line-by-line regardless of comment length.
    psv_io.write_psv(output_file, fieldnames, rows)

    if verbose:
        print(
            f"[shuffler] Pre-processing complete — "
            f"{rows_substituted:,} of {rows_with_attachments:,} attachment rows substituted "
            f"(total rows: {len(rows):,})\n"
        )

    return {
        "total_rows":             len(rows),
        "rows_with_attachments":  rows_with_attachments,
        "rows_substituted":       rows_substituted,
        "output_file":            output_file,
    }


# ── Step 1: Translation ────────────────────────────────────────────────────────

def translate_syncom_to_cms(
    syncom_input: str,
    cms_output: str,
    verbose: bool = True,
) -> int:
    """
    Translate a ♔-delimited syncom output file to a ♔-PSV using the CMS schema.

    Args:
        syncom_input: Path to the ♔-delimited syncom output (.txt).
        cms_output:   Path where the translated ♔-PSV will be written (.psv).
        verbose:      Print progress messages.

    Returns:
        Number of records translated.
    """
    Path(cms_output).parent.mkdir(parents=True, exist_ok=True)

    from shuffler.translate_to_cms_format import translate_synthetic_to_cms

    if verbose:
        print(f"[shuffler] Translating syncom output → CMS PSV format")
        print(f"           Input  : {syncom_input}")
        print(f"           Output : {cms_output}")

    count = translate_synthetic_to_cms(syncom_input, cms_output)

    if verbose:
        print(f"[shuffler] Translation complete — {count} synthetic records written.\n")

    return count


# ── Step 2: Shuffle ────────────────────────────────────────────────────────────

def shuffle_comments(
    synthetic_cms_file: str,
    real_cms_file: str,
    combined_output: str,
    key_output: str | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Randomly interleave synthetic comments into a real comments file and write
    the result as a ♔-delimited PSV.

    Args:
        synthetic_cms_file: Path to translated synthetic comments in CMS ♔-PSV
                            format (output of translate_syncom_to_cms).
        real_cms_file:      Path to the real comments file — either the ♔-PSV
                            produced by preprocess_real_comments (.psv) or the
                            original CMS comma-CSV (.csv) when pre-processing
                            is skipped.
        combined_output:    Path where the combined shuffled ♔-PSV will be written.
        key_output:         Path where the key CSV will be written.  If None, a
                            path is derived from combined_output by appending "_key".
        seed:               Random seed for reproducibility.
        verbose:            Print progress messages.

    Returns:
        dict with keys: real_count, synthetic_count, total_count,
                        combined_output, key_output.
    """
    # Derive default key path (key stays as CSV — it has only four short columns)
    if key_output is None:
        p = Path(combined_output)
        key_output = str(p.with_name(p.stem + "_key.csv"))

    Path(combined_output).parent.mkdir(parents=True, exist_ok=True)
    Path(key_output).parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[shuffler] Loading real comments      : {real_cms_file}")
        print(f"[shuffler] Loading synthetic comments : {synthetic_cms_file}")

    # ── Load real comments ──────────────────────────────────────────────────
    # Auto-detect format by extension for backward compatibility
    real_rows, fieldnames = _load_by_extension(real_cms_file)
    if verbose:
        print(f"[shuffler] Real comments loaded       : {len(real_rows):,} rows")

    # ── Load synthetic comments ─────────────────────────────────────────────
    synth_rows, _synth_fieldnames = _load_by_extension(synthetic_cms_file)
    if verbose:
        print(f"[shuffler] Synthetic comments loaded  : {len(synth_rows):,} rows")

    # Use real fieldnames as the canonical schema
    combined_fieldnames = fieldnames

    # Tag each row with its type (stored temporarily; not written to PSV)
    tagged_real  = [{"_type": "real",      **r} for r in real_rows]
    tagged_synth = [{"_type": "synthetic", **r} for r in synth_rows]

    # ── Shuffle ─────────────────────────────────────────────────────────────
    rng = random.Random(seed)
    combined = tagged_real + tagged_synth
    rng.shuffle(combined)

    if verbose:
        print(f"[shuffler] Shuffling with seed={seed}  → {len(combined):,} total rows")

    # ── Assign anonymous UIDs ────────────────────────────────────────────────
    # Replace every Document ID with a neutral "UID-XXXX" identifier so that
    # synthetic IDs (e.g. "CMS-2025-0050-SYNTH-0077") are indistinguishable
    # from real IDs in the combined output.
    _DOC_ID_COL     = "Document ID"
    _ATTACHMENT_COL = "Attachment Files"

    for idx, row in enumerate(combined, start=1):
        row["_original_doc_id"] = row.get(_DOC_ID_COL, "")
        row[_DOC_ID_COL]        = f"UID-{idx:04d}"

    # ── Write combined ♔-PSV ─────────────────────────────────────────────────
    # The "Attachment Files" column is cleared so that no row reveals whether
    # it originated from a real submission (only real comments have attachments).
    _INTERNAL_KEYS = {"_type", "_original_doc_id"}

    clean_rows = []
    for row in combined:
        clean = {k: v for k, v in row.items() if k not in _INTERNAL_KEYS}
        if _ATTACHMENT_COL in clean:
            clean[_ATTACHMENT_COL] = ""
        clean_rows.append(clean)

    psv_io.write_psv(combined_output, combined_fieldnames, clean_rows)

    if verbose:
        print(f"[shuffler] Combined file written      : {combined_output}")

    # ── Write key CSV ────────────────────────────────────────────────────────
    # Maps each UID back to its original Document ID so results can be
    # de-anonymised after analysis.  Kept as comma-CSV: it's small and has
    # no long free-text fields.
    key_headers = ["row_number", "uid", "original_document_id", "type"]
    with open(key_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=key_headers)
        writer.writeheader()
        for idx, row in enumerate(combined, start=1):
            writer.writerow({
                "row_number":           idx,
                "uid":                  f"UID-{idx:04d}",
                "original_document_id": row.get("_original_doc_id", ""),
                "type":                 row["_type"],
            })

    if verbose:
        print(f"[shuffler] Key file written           : {key_output}")

    real_count  = sum(1 for r in combined if r["_type"] == "real")
    synth_count = sum(1 for r in combined if r["_type"] == "synthetic")

    if verbose:
        print()
        print("=" * 60)
        print("Shuffle complete!")
        print(f"  Real comments      : {real_count:,}")
        print(f"  Synthetic comments : {synth_count:,}")
        print(f"  Total rows         : {len(combined):,}")
        print(f"  Combined output    : {combined_output}")
        print(f"  Key output         : {key_output}")
        print("=" * 60)

    return {
        "real_count":      real_count,
        "synthetic_count": synth_count,
        "total_count":     len(combined),
        "combined_output": combined_output,
        "key_output":      key_output,
    }


# ── Full pipeline convenience wrapper ─────────────────────────────────────────

def run_pipeline(
    real_cms_file: str,
    attachments_dir: str,
    syncom_input: str,
    output_dir: str,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run the full shuffler pipeline in one call:

      Step 0  Pre-process the real comments CSV, substituting attachment text
              where it is longer than the inline comment body.
              → <output_dir>/preprocessed_real.psv

      Step 1  Translate the syncom ♔-delimited output to CMS ♔-PSV format.
              → <output_dir>/synthetic_cms.psv

      Step 2  Shuffle the pre-processed real comments together with the
              translated synthetic comments.  Attachment URLs are cleared in
              the combined output so no row is distinguishable as real.
              → <output_dir>/combined.psv
              → <output_dir>/combined_key.csv   (comma-CSV; small, no free text)

    Args:
        real_cms_file:   Original real CMS comments CSV
                         (e.g. "CMS-2025-0050/comments/CMS-2025-0050.csv").
        attachments_dir: Directory of per-comment attachment subdirectories
                         (e.g. "CMS-2025-0050/comment_attachments").
        syncom_input:    ♔-delimited syncom output file.
        output_dir:      Directory for all pipeline outputs
                         (e.g. "CMS-2025-0050/shuffled_comments").
        seed:            Random seed for reproducibility.
        verbose:         Print progress messages.

    Returns:
        dict with keys from each step merged together.
    """
    out = Path(output_dir)

    preprocessed_file  = str(out / "preprocessed_real.psv")
    synthetic_cms_file = str(out / "synthetic_cms.psv")
    combined_output    = str(out / "combined.psv")
    key_output         = str(out / "combined_key.csv")

    # Step 0 – resolve attachments into the real PSV
    pre_result = preprocess_real_comments(
        real_cms_file=real_cms_file,
        attachments_dir=attachments_dir,
        output_file=preprocessed_file,
        verbose=verbose,
    )

    # Step 1 – translate syncom output to CMS PSV
    synth_count = translate_syncom_to_cms(
        syncom_input=syncom_input,
        cms_output=synthetic_cms_file,
        verbose=verbose,
    )

    # Step 2 – shuffle and merge
    shuffle_result = shuffle_comments(
        synthetic_cms_file=synthetic_cms_file,
        real_cms_file=preprocessed_file,
        combined_output=combined_output,
        key_output=key_output,
        seed=seed,
        verbose=verbose,
    )

    return {**pre_result, "synthetic_translated": synth_count, **shuffle_result}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> tuple[list[dict], list[str]]:
    """
    Load a comma-delimited CSV file and return (rows, fieldnames).
    Used only for the original CMS docket export (source of real comments).
    """
    rows: list[dict] = []
    fieldnames: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))

    return rows, fieldnames


def _load_psv(path: str) -> tuple[list[dict], list[str]]:
    """
    Load a ♔-delimited PSV file and return (rows, fieldnames).
    Newlines encoded as ⏎ are restored to \\n in all field values.
    """
    return psv_io.read_psv(path)


def _load_by_extension(path: str) -> tuple[list[dict], list[str]]:
    """
    Auto-detect file format by extension and load accordingly.
      .psv  → ♔-delimited PSV (psv_io.read_psv)
      .csv  → comma-delimited CSV (_load_csv)
    """
    if Path(path).suffix.lower() == ".psv":
        return _load_psv(path)
    return _load_csv(path)
