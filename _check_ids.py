import sys
sys.path.insert(0, 'c:/Users/mkramer/Github/slop-github')
from shuffler.psv_io import read_psv

rows, fieldnames = read_psv('output-four.txt')

phrase = "i genuinely don"
print(f"Total rows: {len(rows)}")
print(f"\nRows containing '{phrase}':")
for i, r in enumerate(rows):
    comment = r.get('Comment', '').replace('\u23ce', '\n')
    if phrase.lower() in comment.lower():
        idx = comment.lower().index(phrase.lower())
        snippet = comment[max(0, idx-30):idx+80]
        print(f"  row[{i}] doc_id={r['Document ID']!r}: ...{snippet!r}...")

# Also check the _load_comments_from_psv behavior
print("\nSimulating _load_comments_from_psv (skipping empty Comment rows):")
comments_list = []
for i, r in enumerate(rows):
    text = r.get('Comment', '')
    if not text.strip():
        print(f"  Skipped row[{i}] doc_id={r.get('Document ID','')!r} (empty comment)")
        continue
    comments_list.append((i, r.get('Document ID', '')))

print(f"Total comments loaded: {len(comments_list)}")
print("Checking comments_list idx vs doc_id around idx 81-85:")
for enum_idx, (row_idx, doc_id) in enumerate(comments_list[79:87], start=79):
    print(f"  comments[{enum_idx}] from row[{row_idx}]: doc_id={doc_id!r}")
