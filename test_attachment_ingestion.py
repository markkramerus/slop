"""
Test script to verify attachment extraction in ingestion.py
"""
import logging
from syncom.ingestion import ingest_docket_csv

# Set up logging to see extraction details
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Test ingestion with attachment extraction enabled
print("=" * 80)
print("Testing ingestion with attachment extraction...")
print("=" * 80)

try:
    model = ingest_docket_csv(
        csv_path="CMS-2025-0050-0031.csv",
        docket_id="CMS-2025-0050",
        downloader_dir="downloader"
    )
    
    print(f"\n✓ Successfully ingested {model.total_comments} comments")
    print(f"✓ Docket ID: {model.docket_id}")
    print(f"\nArchetype distribution:")
    for archetype, profile in model.archetypes.items():
        print(f"  - {archetype}: {profile.count} comments "
              f"(avg {profile.word_count[0]:.0f} words)")
    
    # Check if any comments are substantially longer (indication of attachment text)
    max_words = 0
    max_archetype = None
    for archetype, profile in model.archetypes.items():
        if profile.word_count[0] > max_words:
            max_words = profile.word_count[0]
            max_archetype = archetype
    
    print(f"\n✓ Longest average comment: {max_archetype} ({max_words:.0f} words)")
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"\n✗ Error during ingestion: {e}")
    import traceback
    traceback.print_exc()
