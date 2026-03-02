"""Quick spot-check of the preprocessed_real.csv output."""
import csv
import sys
csv.field_size_limit(2**31 - 1)

targets = {"CMS-2025-0050-0030", "CMS-2025-0050-0002", "CMS-2025-0050-0006"}

with open("CMS-2025-0050/shuffled_comments/preprocessed_real.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        doc_id = row["Document ID"]
        if doc_id in targets:
            comment = row["Comment"]
            attach  = row.get("Attachment Files", "")
            print(f"--- {doc_id} ---")
            print(f"  Attachment Files : {(attach[:80] + '...') if len(attach) > 80 else (attach or '(empty)')}")
            print(f"  Comment (first 120 chars): {comment[:120]}")
            print()
