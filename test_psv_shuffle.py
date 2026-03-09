"""
Quick integration test for the PSV-based shuffle pipeline.
Runs against the real CMS-2025-0050 docket data.

Usage:  python test_psv_shuffle.py
"""
import pathlib
from shuffler.shuffler import preprocess_real_comments
from shuffler.psv_io import read_psv

OUT = pathlib.Path("CMS-2025-0050/shuffled_comments")

# ── Step 0: Pre-process real CSV → PSV ───────────────────────────────────────
print("=" * 60)
print("STEP 0 — Pre-process real comments")
print("=" * 60)
result = preprocess_real_comments(
    real_cms_file="CMS-2025-0050/comments/CMS-2025-0050.csv",
    attachments_dir="CMS-2025-0050/comment_attachments",
    output_file=str(OUT / "preprocessed_real.psv"),
    verbose=True,
)
print()

# ── Verify PSV ───────────────────────────────────────────────────────────────
psv_path = OUT / "preprocessed_real.psv"
rows, fieldnames = read_psv(psv_path)

print("PSV verification")
print(f"  Columns : {len(fieldnames)}")
print(f"  Rows    : {len(rows)}")

raw        = psv_path.read_text(encoding="utf-8")
line_count = len(raw.splitlines())
print(f"  Lines in file : {line_count}  (should be {len(rows)+1} = 1 header + {len(rows)} data)")

# Confirm row count matches line count
assert line_count == len(rows) + 1, \
    f"Line count mismatch: {line_count} lines but {len(rows)} rows — raw newlines leaked!"
print("  ✅ No raw newlines leaked into field values")

# Find a comment that came from an attachment (long text)
long_comments = sorted(rows, key=lambda r: len(r.get("Comment", "")), reverse=True)
if long_comments:
    sample = long_comments[0]
    comment_len = len(sample.get("Comment", ""))
    has_newlines = "\n" in sample.get("Comment", "")
    print(f"  Longest comment : {comment_len:,} chars (Document ID: {sample.get('Document ID','?')})")
    print(f"  Newlines decoded: {'yes' if has_newlines else 'no (short comment, no paragraphs)'}")
    if comment_len > 32_000:
        print(f"  ✅ Exceeds 32K Excel cell limit — PSV handles it fine")

print()
print("=" * 60)
print("All checks passed — preprocessed_real.psv looks correct.")
print("=" * 60)
print()
print("To test the full pipeline (requires synthetic.txt):")
print()
print("  python -c \"")
print("  from shuffler.shuffler import run_pipeline")
print("  run_pipeline(")
print("    real_cms_file='CMS-2025-0050/comments/CMS-2025-0050.csv',")
print("    attachments_dir='CMS-2025-0050/comment_attachments',")
print("    syncom_input='CMS-2025-0050/synthetic_comments/synthetic.txt',")
print("    output_dir='CMS-2025-0050/shuffled_comments',")
print("  )\"")
print()
print("Outputs written to CMS-2025-0050/shuffled_comments/:")
print("  preprocessed_real.psv  — real comments with attachment text inlined")
print("  synthetic_cms.psv      — synthetic comments in CMS column schema")
print("  combined.psv           — shuffled mix, UIDs replacing Document IDs")
print("  combined_key.csv       — UID → original ID + real/synthetic label")
