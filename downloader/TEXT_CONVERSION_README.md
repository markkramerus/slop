
# Text Conversion for Comment Attachments

This document explains how to use the text conversion functionality to convert
downloaded comment attachments to text files for downstream analysis (e.g.,
stylometry).

## Overview

Text conversion is driven by **AI classification**: only PDFs that the AI
classifier labels as `comment` are converted to text. This avoids wasting time
on presentations, cover letters, marketing materials, and other non-comment
attachments.

### Three-Step Workflow

```
1. Download all attachments        → download_attachments.py
2. AI-classify each PDF            → classify_attachments_ai.py  → attachment_classification.csv
3. Convert comment PDFs to text    → text_converter.py  (reads attachment_classification.csv)
```

## Step 1: Download Attachments

```bash
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv
```

## Step 2: AI Classification

Classify every PDF in the attachment tree as `comment` or `not_comment`:

```bash
python downloader/classify_attachments_ai.py CMS-2025-0050/comment_attachments \
  --output CMS-2025-0050/attachment_classification.csv
```

This produces a CSV with one row per PDF containing the AI label, confidence,
document type, and rationale. The classifier renders the first pages of each PDF
to images and sends them to an OpenAI-compatible endpoint. It does **not** need
`.txt` files — classification is completely independent of text conversion.

**Environment variables required** (set in `.env` or your shell):

- `SLOP_CLASSIFER_API_BASE_URL`
- `SLOP_CLASSIFER_API_KEY`
- `SLOP_CLASSIFER_MODEL`

### Resume / re-run

Re-running skips PDFs already in the CSV. To force reclassification:

```bash
python downloader/classify_attachments_ai.py CMS-2025-0050/comment_attachments \
  --output CMS-2025-0050/attachment_classification.csv --force
```

## Step 3: Convert Comment PDFs to Text

After classification, convert **only** the files labeled `comment`:

```bash
# Basic usage — reads CMS-2025-0050/attachment_classification.csv by default
python downloader/text_converter.py CMS-2025-0050

# Specify a custom classification CSV
python downloader/text_converter.py CMS-2025-0050 \
  --classification-csv CMS-2025-0050/attachment_classification.csv

# Force reconversion (even if .txt files exist)
python downloader/text_converter.py CMS-2025-0050 --force
```

The converter will:
1. Read `attachment_classification.csv` to get the list of comment PDFs
2. For each comment PDF, check if a `.txt` file already exists
3. Extract text (via pdfplumber, with OCR fallback) and write `.txt`
4. Skip files that already have `.txt` versions (unless `--force`)

### Integrated Download + Convert

You can also convert as part of the download step:

```bash
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --convert-text
```

> **Note:** This requires `attachment_classification.csv` to already exist. Run the
> AI classifier (Step 2) first.

## Benefits of This Approach

- **Fewer conversions**: Only comment PDFs are converted; presentations, cover
  letters, and marketing materials are skipped entirely.
- **Better classification**: AI vision-based classification is more reliable than
  heuristic signals (PDF metadata, word density, landscape detection, image
  coverage).
- **Auditability**: The classification CSV records every decision with confidence
  scores and rationale for human review.
- **Speed**: Text files are read instantly; PDF extraction can be slow.
- **Consistency**: Once converted, the same text is reused across analyses.

## File Structure

After classification and conversion:

```
CMS-2025-0050/
  attachment_classification.csv          ← AI classification results
  comment_attachments/
    CMS-2025-0050-0004/
      attachment_1.pdf                ← classified as "comment"
      attachment_1.txt                ← converted to text
    CMS-2025-0050-0007/
      attachment_1.pdf                ← classified as "comment"
      attachment_1.txt
      attachment_2.docx               ← classified as "not_comment"
                                         (no .txt created)
    CMS-2025-0050-0021/
      attachment_1.pdf                ← classified as "not_comment"
                                         (no .txt created)
    ...
```

## Conversion Summary Output

After running, you'll see a summary like:

```
============================================================
CONVERSION SUMMARY
============================================================
Docket: CMS-2025-0050
Comment files in CSV:       142
  Newly converted:           71
  Skipped (already existed):  65
  Failed:                      6
```

## Text Extraction Details

The converter uses **pdfplumber** (built on pdfminer.six) for PDF text extraction.
pdfplumber uses layout analysis to group characters into words based on geometric
proximity, which produces much better results than simpler extractors for PDFs with
character-level glyph positioning.

If pdfplumber output looks like encoding garbage (custom font CIDs or low
alpha-character ratio), the converter automatically falls back to **OCR** via
`pdf2image` + `pytesseract`.

## Workflow Examples

### Example 1: Full Pipeline

```bash
# 1. Download attachments
python downloader/download_attachments.py CMS-2025-0050

# 2. Classify
python downloader/classify_attachments_ai.py CMS-2025-0050/comment_attachments \
  --output CMS-2025-0050/attachment_classification.csv

# 3. Convert comment PDFs to text
python downloader/text_converter.py CMS-2025-0050

# 4. Run stylometry analysis
python stylometry/stylometry_analyzer.py CMS-2025-0050/comments/CMS-2025-0050.csv
```

### Example 2: Re-classify and Reconvert

If the AI model or prompt changes and you want fresh results:

```bash
# Reclassify all PDFs
python downloader/classify_attachments_ai.py CMS-2025-0050/comment_attachments \
  --output CMS-2025-0050/attachment_classification.csv --force

# Reconvert text
python downloader/text_converter.py CMS-2025-0050 --force
```

## Command Reference

### classify_attachments_ai.py

```
python downloader/classify_attachments_ai.py ATTACHMENTS_ROOT --output CSV_PATH [OPTIONS]

Arguments:
  ATTACHMENTS_ROOT      Root directory of PDF attachments
  --output CSV_PATH     Output classification CSV

Options:
  --force               Reclassify even if already in CSV
  --limit N             Only classify N attachments (debug)
  --request-format FMT  content_parts (default) | attachments_field | multipart_form
  --max-mb MB           Skip PDFs larger than MB (default 20)
  --sleep SECONDS       Rate-limiting delay between requests
  --max-pages N         Pages to render as images (default 2)
  --dpi DPI             Image rendering DPI (default 150)
  --quiet               Suppress progress output
```

### text_converter.py

```
python downloader/text_converter.py DOCKET_ID [OPTIONS]

Arguments:
  DOCKET_ID                       Docket identifier (e.g., CMS-2025-0050)

Options:
  --attachments-dir DIR           Directory containing attachments
                                  (default: {docket_id}/comment_attachments/)
  --classification-csv CSV_PATH   Path to attachment_classification.csv
                                  (default: {docket_id}/attachment_classification.csv)
  --force                         Reconvert even if .txt files exist
```

## Troubleshooting

### "Classification CSV not found"

Run the AI classifier first (Step 2). `text_converter.py` requires the
`attachment_classification.csv` file to know which PDFs to convert.

### Text extraction fails for some files

Check the conversion output for errors and inspect the `.txt` files. The OCR
fallback handles most encoding issues, but some PDFs may need manual conversion.

### Analyzer still running slowly

Verify that `.txt` files exist for comment attachments:

```bash
dir CMS-2025-0050\comment_attachments\CMS-2025-0050-0004\
```

## Best Practices

1. **Always classify before converting**: The classification CSV is the single
   source of truth for which files get converted.

2. **Keep source files**: Don't delete PDFs after conversion — allows
   re-extraction if needed and preserves original formatting.

3. **Use `--force` sparingly**: Only reconvert when extraction logic changes or
   after reclassification.

4. **Review the classification CSV**: Spot-check the AI's rationale column to
   ensure the classifier is performing well on your docket.
