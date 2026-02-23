"""Quick verification script for the translated CMS format."""

import csv

# Read the translated file
with open('synthetic_comments_translated_to_cms.csv', 'r', encoding='utf-8') as f:
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
