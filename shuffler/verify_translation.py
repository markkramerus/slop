"""Quick verification script for the translated CMS format.

Usage:
    python shuffler/verify_translation.py <csv_file>
    python shuffler/verify_translation.py CMS-2025-0050/synthetic_comments/comments_cms.csv
"""

import csv
import sys

# Determine file to verify
if len(sys.argv) > 1:
    csv_file = sys.argv[1]
else:
    print("Usage: python shuffler/verify_translation.py <csv_file>")
    print("Example: python shuffler/verify_translation.py CMS-2025-0050/synthetic_comments/comments_cms.csv")
    sys.exit(1)

# Read the translated file
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    row = next(reader)
    
    print("Sample translated record:")
    print("=" * 60)
    print(f"Document ID: {row['Document ID']}")
    print(f"Agency ID: {row['Agency ID']}")
    print(f"Docket ID: {row['Docket ID']}")
    print(f"First Name: {row['First Name']}")
    print(f"Last Name: {row['Last Name']}")
    print(f"State: {row['State/Province']}")
    print(f"Organization: {row['Organization Name']}")
    print(f"Document Type: {row['Document Type']}")
    print(f"Posted Date: {row['Posted Date']}")
    print(f"\nComment (first 200 chars):")
    print(row['Comment'][:200] + "...")
    print("=" * 60)
