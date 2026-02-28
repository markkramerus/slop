"""
shuffler/shuffler.py — Core shuffler logic for the slop pipeline.

This module:
  1. Translates syncom output (♔-delimited) to CMS CSV format.
  2. Randomly interleaves the translated synthetic comments with a real
     CMS comments file.
  3. Produces a combined CMS CSV and a key CSV that identifies every row
     as "real" or "synthetic".
"""

import csv
import os
import random
from pathlib import Path


# ── Step 1: Translation ────────────────────────────────────────────────────────

def translate_syncom_to_cms(syncom_input: str, cms_output: str, verbose: bool = True) -> int:
    """
    Translate a ♔-delimited syncom output file to CMS CSV format.

    Args:
        syncom_input: Path to the ♔-delimited syncom output (.txt).
        cms_output:   Path where the translated CMS CSV will be written.
        verbose:      Print progress messages.

    Returns:
        Number of records translated.
    """
    # Ensure output directory exists
    Path(cms_output).parent.mkdir(parents=True, exist_ok=True)

    # Import the existing translator (same package, pure stdlib)
    from shuffler.translate_to_cms_format import translate_synthetic_to_cms

    if verbose:
        print(f"[shuffler] Translating syncom output → CMS format")
        print(f"           Input  : {syncom_input}")
        print(f"           Output : {cms_output}")

    translate_synthetic_to_cms(syncom_input, cms_output)

    # Count the rows that were written (minus header)
    count = 0
    with open(cms_output, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for _ in reader:
            count += 1

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
    Randomly interleave synthetic comments into a real CMS comments file.

    Args:
        synthetic_cms_file: Path to translated synthetic comments in CMS CSV format.
        real_cms_file:      Path to the real CMS comments file (comma-delimited CSV).
        combined_output:    Path where the combined shuffled CSV will be written.
        key_output:         Path where the key CSV will be written.  If None, a
                            path is derived from combined_output by appending "_key".
        seed:               Random seed for reproducibility.
        verbose:            Print progress messages.

    Returns:
        dict with keys: real_count, synthetic_count, total_count,
                        combined_output, key_output.
    """
    # Derive default key path
    if key_output is None:
        p = Path(combined_output)
        key_output = str(p.with_name(p.stem + "_key.csv"))

    # Ensure output directories exist
    Path(combined_output).parent.mkdir(parents=True, exist_ok=True)
    Path(key_output).parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[shuffler] Loading real comments      : {real_cms_file}")
        print(f"[shuffler] Loading synthetic comments : {synthetic_cms_file}")

    # ── Load real comments ──────────────────────────────────────────────────
    real_rows, fieldnames = _load_csv(real_cms_file)
    if verbose:
        print(f"[shuffler] Real comments loaded       : {len(real_rows):,} rows")

    # ── Load synthetic comments ─────────────────────────────────────────────
    synth_rows, synth_fieldnames = _load_csv(synthetic_cms_file)
    if verbose:
        print(f"[shuffler] Synthetic comments loaded  : {len(synth_rows):,} rows")

    # Use real fieldnames as the canonical schema (they may differ slightly)
    # Synthetic rows have the same CMS headers from translate_to_cms_format.py
    combined_fieldnames = fieldnames

    # Tag each row with its type (stored temporarily; not written to CSV)
    tagged_real  = [{"_type": "real",      **r} for r in real_rows]
    tagged_synth = [{"_type": "synthetic", **r} for r in synth_rows]

    # ── Shuffle ─────────────────────────────────────────────────────────────
    rng = random.Random(seed)
    combined = tagged_real + tagged_synth
    rng.shuffle(combined)

    if verbose:
        print(f"[shuffler] Shuffling with seed={seed}  → {len(combined):,} total rows")

    # ── Write combined CSV ───────────────────────────────────────────────────
    with open(combined_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=combined_fieldnames,
            extrasaction="ignore",   # drops the _type tag
        )
        writer.writeheader()
        for row in combined:
            # Remove internal tag before writing
            clean = {k: v for k, v in row.items() if k != "_type"}
            writer.writerow(clean)

    if verbose:
        print(f"[shuffler] Combined file written      : {combined_output}")

    # ── Write key CSV ────────────────────────────────────────────────────────
    key_headers = ["row_number", "document_id", "type"]
    with open(key_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=key_headers)
        writer.writeheader()
        for idx, row in enumerate(combined, start=1):
            writer.writerow({
                "row_number":  idx,
                "document_id": row.get("Document ID", ""),
                "type":        row["_type"],
            })

    if verbose:
        print(f"[shuffler] Key file written           : {key_output}")

    real_count   = sum(1 for r in combined if r["_type"] == "real")
    synth_count  = sum(1 for r in combined if r["_type"] == "synthetic")

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
        "real_count":       real_count,
        "synthetic_count":  synth_count,
        "total_count":      len(combined),
        "combined_output":  combined_output,
        "key_output":       key_output,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> tuple[list[dict], list[str]]:
    """
    Load a CSV file and return (rows, fieldnames).
    Rows is a list of dicts; fieldnames is the ordered list of headers.
    """
    rows = []
    fieldnames: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))

    return rows, fieldnames
