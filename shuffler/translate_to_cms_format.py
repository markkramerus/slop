"""
shuffler/translate_to_cms_format.py — Map synthetic comment columns to the
canonical CMS export schema and write the result as a ♔-delimited PSV file.

Input  : ♔-delimited synthetic comment file produced by syncom/export.py
Output : ♔-delimited PSV file (same schema as the real CMS docket CSV export)

The output columns exactly match the CMS docket CSV export so that synthetic
rows can be interleaved with real rows without schema mismatch.
Newlines encoded as ⏎ (U+23CE) by the syncom exporter pass through unchanged
into the PSV output — psv_io.read_psv() restores them on the consuming end.
"""

import sys
from pathlib import Path

# Allow running directly as a script from the repo root
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shuffler.psv_io import read_psv, write_psv


# ── CMS column schema ──────────────────────────────────────────────────────────

CMS_HEADERS = [
    "Document ID", "Agency ID", "Docket ID", "Tracking Number",
    "Document Type", "Posted Date", "Is Withdrawn?", "Federal Register Number",
    "FR Citation", "Title", "Comment Start Date", "Comment Due Date",
    "Allow Late Comments", "Comment on Document ID", "Effective Date",
    "Implementation Date", "Postmark Date", "Received Date", "Author Date",
    "Related RIN(s)", "Authors", "CFR", "Abstract", "Legacy ID", "Media",
    "Document Subtype", "Exhibit Location", "Exhibit Type", "Additional Field 1",
    "Additional Field 2", "Topics", "Duplicate Comments", "OMB/PRA Approval Number",
    "Page Count", "Page Length", "Paper Width", "Special Instructions",
    "Source Citation", "Start End Page", "Subject", "First Name", "Last Name",
    "City", "State/Province", "Zip/Postal Code", "Country", "Organization Name",
    "Submitter Representative", "Representative's Address",
    "Representative's City, State & Zip", "Government Agency",
    "Government Agency Type", "Comment", "Category", "Restrict Reason Type",
    "Restrict Reason", "Reason Withdrawn", "Content Files", "Attachment Files",
    "Display Properties (Name, Label, Tooltip)",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_name(full_name: str) -> tuple[str, str]:
    """Split a full name string into (first_name, last_name)."""
    if not full_name or not full_name.strip():
        return "", ""
    parts = full_name.strip().split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def _docket_base(doc_id: str) -> str:
    """Strip the -SYNTH-NNNN suffix from a synthetic document ID."""
    return doc_id.split("-SYNTH")[0] if "-SYNTH" in doc_id else doc_id


# ── Main translator ────────────────────────────────────────────────────────────

def translate_synthetic_to_cms(input_file: str, output_file: str) -> int:
    """
    Translate a ♔-delimited syncom output file to a ♔-delimited PSV file
    whose columns match the CMS docket export schema.

    Parameters
    ----------
    input_file:
        Path to the syncom ♔-delimited output (e.g. synthetic.txt).
    output_file:
        Destination path for the translated PSV (e.g. synthetic_cms.psv).

    Returns
    -------
    int
        Number of records written.
    """
    # ── Read syncom ♔-PSV ──────────────────────────────────────────────────
    rows, _src_fieldnames = read_psv(input_file)

    # ── Map columns to CMS schema ──────────────────────────────────────────
    cms_rows: list[dict] = []
    for row in rows:
        comment_id = row.get("Comment ID", "")
        doc_id     = row.get("Document ID", "")
        base_id    = _docket_base(doc_id)
        first_name, last_name = _parse_name(row.get("Submitter Name", ""))

        cms_row = {
            "Document ID":           comment_id,          # Comment ID becomes the Document ID
            "Agency ID":             "CMS",
            "Docket ID":             base_id,
            "Tracking Number":       "",
            "Document Type":         "Public Submission",
            "Posted Date":           row.get("Posted Date", ""),
            "Is Withdrawn?":         "false",
            "Federal Register Number": row.get("Federal Register Number", ""),
            "FR Citation":           "",
            "Title":                 f"Comment on {base_id}" if base_id else "",
            "Comment Start Date":    row.get("Comment Start Date", ""),
            "Comment Due Date":      row.get("Comment End Date", ""),
            "Allow Late Comments":   "false",
            "Comment on Document ID": base_id,
            "Effective Date":        "",
            "Implementation Date":   "",
            "Postmark Date":         "",
            "Received Date":         row.get("Received Date", ""),
            "Author Date":           "",
            "Related RIN(s)":        "",
            "Authors":               "",
            "CFR":                   "",
            "Abstract":              row.get("Abstract", ""),
            "Legacy ID":             "",
            "Media":                 "",
            "Document Subtype":      "Public Comment",
            "Exhibit Location":      row.get("Exhibit Location", ""),
            "Exhibit Type":          row.get("Exhibit Type", ""),
            "Additional Field 1":    "",
            "Additional Field 2":    "",
            "Topics":                "",
            "Duplicate Comments":    "",
            "OMB/PRA Approval Number": "",
            "Page Count":            row.get("Page Count", ""),
            "Page Length":           "",
            "Paper Width":           "",
            "Special Instructions":  "",
            "Source Citation":       "",
            "Start End Page":        "",
            "Subject":               "",
            "First Name":            first_name,
            "Last Name":             last_name,
            "City":                  "",
            "State/Province":        row.get("synth_persona_state", ""),
            "Zip/Postal Code":       "",
            "Country":               "United States" if row.get("synth_persona_state") else "",
            "Organization Name":     row.get("Organization Name", ""),
            "Submitter Representative": row.get("Submitter's Representative", ""),
            "Representative's Address": "",
            "Representative's City, State & Zip": "",
            "Government Agency":     row.get("Government Agency", ""),
            "Government Agency Type": row.get("Government Agency Type", ""),
            # Comment text: newlines are already encoded as ⏎ by syncom/export.py;
            # psv_io.write_psv will handle them correctly — no manual replacement needed.
            "Comment":               row.get("Comment", ""),
            "Category":              "",
            "Restrict Reason Type":  "",
            "Restrict Reason":       "",
            "Reason Withdrawn":      "",
            "Content Files":         "",
            "Attachment Files":      row.get("Attachment Files", ""),
            "Display Properties (Name, Label, Tooltip)":
                "pageCount, Page Count, Number of pages In the content file",
        }
        cms_rows.append(cms_row)

    # ── Write ♔-PSV ────────────────────────────────────────────────────────
    write_psv(output_file, CMS_HEADERS, cms_rows)

    return len(cms_rows)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Synthetic Comments → CMS PSV Translator")
        print("=" * 50)
        print()
        print("Usage: python shuffler/translate_to_cms_format.py <input_file> [output_file]")
        print()
        print("Example:")
        print("  python shuffler/translate_to_cms_format.py \\")
        print("    CMS-2025-0050/synthetic_comments/synthetic.txt \\")
        print("    CMS-2025-0050/synthetic_comments/synthetic_cms.psv")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not output_file:
        p = Path(input_file)
        output_file = str(p.with_suffix("").with_name(p.stem + "_cms.psv"))

    print("Synthetic Comments → CMS PSV Translator")
    print("=" * 50)
    print(f"Input  : {input_file}")
    print(f"Output : {output_file}")
    print()

    if not Path(input_file).exists():
        print(f"Error: input file '{input_file}' not found.")
        sys.exit(1)

    n = translate_synthetic_to_cms(input_file, output_file)

    print(f"Translation complete — {n} records written to {output_file}")


if __name__ == "__main__":
    main()
