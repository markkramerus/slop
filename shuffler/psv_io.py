"""
shuffler/psv_io.py — Read and write ♔-delimited PSV (♔-Separated Values) files.

Format conventions
------------------
Field separator         : ♔  (U+2654 WHITE CHESS KING)
Record separator        : newline  (\\n, one record per line)
Newline within a field  : ⏎  (U+23CE RETURN SYMBOL) — replaces \\n / \\r\\n
♔ within a field value  : < >  — replaces ♔

Both characters (♔ and ⏎) are so rare in natural text that the format needs no
quoting whatsoever.  Field values may be arbitrarily long.  The consuming
application restores ⏎ to \\n on read.

Typical usage
-------------
    from shuffler.psv_io import write_psv, read_psv

    # Writing
    write_psv("output.psv", fieldnames, rows)

    # Reading
    rows, fieldnames = read_psv("output.psv")
"""

from __future__ import annotations

from pathlib import Path

# ── Encoding constants ─────────────────────────────────────────────────────────

FIELD_SEP   = "♔"   # U+2654 WHITE CHESS KING   — field delimiter
NEWLINE_ENC = "⏎"   # U+23CE RETURN SYMBOL       — encodes \n / \r\n within fields
KING_ENC    = "< >" # replaces ♔ within field values


# ── Internal helpers ───────────────────────────────────────────────────────────

def _encode_field(value: str) -> str:
    """Encode a single field value for safe storage in a PSV record."""
    # Replace the field separator first (before any other substitution)
    value = value.replace(FIELD_SEP, KING_ENC)
    # Encode all newline variants as the return-symbol placeholder
    value = value.replace("\r\n", NEWLINE_ENC)
    value = value.replace("\r",   NEWLINE_ENC)
    value = value.replace("\n",   NEWLINE_ENC)
    return value


def _decode_field(value: str) -> str:
    """Restore a field value from PSV encoding back to plain text."""
    # Restore encoded newlines to real newlines
    return value.replace(NEWLINE_ENC, "\n")
    # Note: KING_ENC (< >) is deliberately NOT restored — the original ♔
    # character cannot appear in real or synthetic comment text in practice,
    # so < > in a field value always originated as a literal < > in the source.


# ── Public API ─────────────────────────────────────────────────────────────────

def write_psv(
    path: str | Path,
    fieldnames: list[str],
    rows: list[dict],
) -> None:
    """
    Write a list of dicts to a ♔-delimited PSV file.

    Parameters
    ----------
    path:
        Destination file path.  Parent directories are created if needed.
    fieldnames:
        Ordered list of column names.  Defines both the header row and the
        order in which fields are written for every data row.
    rows:
        List of dicts mapping column name → field value.  Missing columns are
        written as empty strings; extra keys are silently ignored.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        # Header line
        f.write(FIELD_SEP.join(_encode_field(h) for h in fieldnames) + "\n")
        # Data rows — one line per record
        for row in rows:
            values = [
                _encode_field(str(row.get(col, "") or ""))
                for col in fieldnames
            ]
            f.write(FIELD_SEP.join(values) + "\n")


def read_psv(path: str | Path) -> tuple[list[dict], list[str]]:
    """
    Read a ♔-delimited PSV file produced by :func:`write_psv` or
    ``syncom/export.py``'s ``export_to_txt``.

    Returns
    -------
    (rows, fieldnames)
        rows       : list of dicts; ⏎ in values is restored to \\n.
        fieldnames : ordered list of column names from the header row.
    """
    rows: list[dict] = []
    fieldnames: list[str] = []

    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()          # handles \r\n, \r, \n uniformly

    if not lines:
        return rows, fieldnames

    # First non-empty line is the header
    fieldnames = [_decode_field(h) for h in lines[0].split(FIELD_SEP)]
    n_cols = len(fieldnames)

    for line in lines[1:]:
        if not line:
            continue                   # skip blank trailing lines
        parts = line.split(FIELD_SEP)
        # Pad to match column count (handles truncated trailing empties)
        while len(parts) < n_cols:
            parts.append("")
        row = {
            fieldnames[i]: _decode_field(parts[i])
            for i in range(n_cols)
        }
        rows.append(row)

    return rows, fieldnames
