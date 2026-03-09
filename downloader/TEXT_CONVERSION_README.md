# Text Conversion for Attachment Processing

This document explains how to use the text conversion functionality to pre-convert downloaded attachments to text files, which speeds up downstream analysis (e.g., stylometry) and allows for easier inspection of extracted text.

## Overview

The downloader module supports two approaches for converting attachments to text:

1. **Integrated conversion** (recommended): Use the `--convert-text` flag when downloading
2. **Standalone conversion**: Run `text_converter.py` directly after downloading

## Presentation Filtering (Default Behavior)

Many public comment dockets receive PDF submissions that are actually PowerPoint or Keynote presentations exported to PDF. These convert very poorly—the text comes out as short, fragmented bullet points with scrambled reading order, making them useless for text analysis.

**By default, `text_converter.py` skips these files.** A PDF is classified as a presentation and skipped if either of the following is true:

| Signal | Threshold | Rationale |
|---|---|---|
| **Metadata** | Creator/Producer contains "PowerPoint", "Impress", "Keynote", "Google Slides", or "Prezi" | Most reliable; presentation software stamps these fields reliably |
| **Low word density** | Average < 100 words/page (across first 10 pages) | Slides typically have 10–75 words/slide; narrative pages have 200–500 |

Signals are checked in order; the first match causes the file to be skipped. The reason is logged at WARNING level so you can see exactly what was filtered and why.

> **Note:** A "short text block" ratio heuristic was considered but dropped. pypdf extracts text line-by-line from the PDF's internal structure, so even dense narrative prose produces many short fragments, making that signal unreliable.

To convert presentation-style PDFs anyway, add `--include-presentations`.

## Benefits of Pre-Converting to Text

- **Speed**: Text files are read instantly; PDF/DOCX extraction can be slow
- **Consistency**: Once converted, the same text is used across multiple analyses
- **Quality**: Presentation-style PDFs are filtered out, keeping only readable narrative text
- **Debugging**: Easy to inspect what text was extracted from attachments
- **Reprocessing**: Run stylometry analysis multiple times without re-extracting

## Usage

### Option 1: Download + Convert in One Step (Recommended)

```bash
# Download attachments and convert to text in one command
# (presentation-style PDFs are skipped automatically)
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --convert-text

# Force reconversion (even if .txt files exist)
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --force-convert

# Include presentation-style PDFs (not recommended for text analysis)
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --convert-text --include-presentations
```

### Option 2: Standalone Text Conversion

After downloading attachments, convert them separately:

```bash
# Basic usage — looks in CMS-2025-0050/comment_attachments/
# (presentation-style PDFs are skipped automatically)
python downloader/text_converter.py CMS-2025-0050

# Specify custom attachments directory
python downloader/text_converter.py CMS-2025-0050 --attachments-dir path/to/attachments

# Force reconversion (even if .txt files exist)
python downloader/text_converter.py CMS-2025-0050 --force

# Include presentation-style PDFs (not recommended for text analysis)
python downloader/text_converter.py CMS-2025-0050 --include-presentations
```

Both options will:
1. Find all PDF and DOCX files in `CMS-2025-0050/comment_attachments/*/`
2. Skip PDFs that look like presentation slides (unless `--include-presentations`)
3. Create corresponding `.txt` files (e.g., `attachment_1.pdf` → `attachment_1.txt`)
4. Skip files that already have `.txt` versions (unless force is used)

### Step 2: Run Stylometry Analysis

Run stylometry analysis as normal. The analyzer will automatically use `.txt` files when available:

```bash
python stylometry/stylometry_analyzer.py CMS-2025-0050/comments/CMS-2025-0050.csv
```

The analyzer will:
- Check for `.txt` files first
- Use pre-converted text when available (fast)
- Fall back to extracting from source files if no `.txt` exists (slower)

## File Structure

After conversion, your attachments directory will look like:

```
CMS-2025-0050/
  comment_attachments/
    CMS-2025-0050-0004/
      attachment_1.pdf
      attachment_1.txt          ← Created by text converter
    CMS-2025-0050-0007/
      attachment_1.pdf
      attachment_1.txt
      attachment_2.docx
      attachment_2.txt
    CMS-2025-0050-0021/
      attachment_1.pdf          ← Presentation-style PDF: no .txt created
    ...
```

## Conversion Summary Output

After running, you'll see a summary like:

```
==================================================
TEXT CONVERSION SUMMARY
==================================================
Documents processed:      87
Total source files:        94
Newly converted:           71
Presentations skipped:      9   ← filtered presentation-style PDFs
Skipped (existed):          8
Failed:                     6
==================================================
```

Filtered files are also logged individually at WARNING level:

```
WARNING:  SKIPPED (presentation) CMS-2025-0050-0021/attachment_1.pdf
          — presentation_metadata (matched 'powerpoint' in creator/producer)
          | creator: 'Microsoft PowerPoint'

WARNING:  SKIPPED (presentation) CMS-2025-0050-0047/attachment_1.pdf
          — low_density (42 words/page avg, threshold 100)
```

## Workflow Examples

### Example 1: First-Time Analysis (Recommended Workflow)

```bash
# 1. Download attachments and convert to text in one step
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --convert-text

# 2. Run stylometry analysis
python stylometry/stylometry_analyzer.py CMS-2025-0050/comments/CMS-2025-0050.csv
```

### Example 2: Reprocessing Existing Downloads

If you already have downloads but want to ensure all text is pre-converted:

```bash
# Convert any new attachments that don't have .txt files yet
python downloader/text_converter.py CMS-2025-0050

# Run analysis
python stylometry/stylometry_analyzer.py CMS-2025-0050/comments/CMS-2025-0050.csv
```

### Example 3: Updating Text Extraction

If text extraction logic improves and you want to reconvert everything:

```bash
# Force reconversion of all files
python downloader/text_converter.py CMS-2025-0050 --force

# Run analysis with updated text
python stylometry/stylometry_analyzer.py CMS-2025-0050/comments/CMS-2025-0050.csv
```

## Implementation Details

### Presentation Classifier (`text_converter.classify_pdf_readability`)

The classifier runs two signals in priority order, returning early as soon as one fires:

1. **Metadata check** (fastest): reads PDF `/Creator` and `/Producer` fields and matches
   against a list of known presentation-software keywords. No page content is read.

2. **Word density** (fast): samples the first `SAMPLE_PAGE_COUNT` (10) pages and
   computes the average word count. A threshold of `MIN_WORDS_PER_PAGE` (100) is used.

Both thresholds are defined as named constants at the top of `text_converter.py` for easy
tuning.

### Text Conversion Logic (`downloader/text_converter.py`)

The converter:
1. Finds all `.pdf`, `.docx`, and `.doc` files in the docket directory
2. For each `.pdf`, runs the readability classifier (unless `--include-presentations`)
3. Skips classified presentations; converts everything else
4. Creates a `.txt` file with the same basename
5. Skips conversion if `.txt` already exists (unless `--force` is specified)

### Downloader Integration (`download_attachments.py --convert-text`)

When the `--convert-text` flag is passed:
1. Downloads all attachments as normal
2. Determines the docket ID from the CSV filename
3. Calls `convert_docket_to_text()` with `skip_presentations=True` (default)
4. Prints a combined summary of both download and conversion statistics

### Stylometry Analyzer Integration (`stylometry_utils.get_attachment_text`)

The stylometry analyzer's `get_attachment_text` function:
1. Lists all files in a document's attachment directory
2. For each file, checks if a corresponding `.txt` file exists
3. If `.txt` exists: reads it directly (fast path)
4. If `.txt` doesn't exist: extracts from source file (slow path)
5. Avoids processing duplicates (e.g., both `attachment_1.pdf` and `attachment_1.txt`)

Since presentation PDFs are never converted to `.txt`, the analyzer will fall back to
on-the-fly extraction for them. If you want to avoid that, pass `--include-presentations`
during conversion to create `.txt` stubs (though the quality will be poor).

### Backward Compatibility

The stylometry analyzer remains fully backward compatible:
- Works without pre-converted text files (extracts on-the-fly)
- Works with a mix of converted and non-converted files
- No changes to command-line interface or existing workflows

## Tuning the Classifier

If you find the classifier is filtering too aggressively or missing some presentations,
edit the constants at the top of `downloader/text_converter.py`:

```python
PRESENTATION_CREATOR_KEYWORDS = [
    "powerpoint", "impress", "keynote", "google slides", "prezi",
]
MIN_WORDS_PER_PAGE = 100   # lower = less aggressive filtering
SAMPLE_PAGE_COUNT  = 10    # pages to sample for the word-density signal
```

To audit which files are being filtered and why, check the WARNING-level log output
during conversion, or redirect the output to a file:

```bash
python downloader/text_converter.py CMS-2025-0050 2>&1 | findstr SKIPPED
```

## Troubleshooting

### Issue: Too many files being filtered

The `low_density` signal can catch scanned documents or short PDFs that extracted
sparsely. Try lowering `MIN_WORDS_PER_PAGE` (e.g., to 50), or use
`--include-presentations` and inspect the resulting `.txt` files manually.

### Issue: Presentation PDFs are not being filtered

Some presentation exports don't embed creator metadata and happen to have dense text
(e.g., a presenter who puts full sentences on every slide). In this case, lower
`MIN_WORDS_PER_PAGE` toward 75 to catch more borderline cases.

### Issue: Text extraction fails for some files

**Solution**: Check the conversion output for errors:

```bash
python downloader/text_converter.py CMS-2025-0050
# Look for "Failed" count in summary
```

Inspect specific `.txt` files to verify content. You may need to manually convert
problematic files.

### Issue: Analyzer still running slowly

**Solution**: Verify that `.txt` files exist:

```bash
# Check if txt files exist
dir CMS-2025-0050\comment_attachments\CMS-2025-0050-0004\

# The analyzer logs when it uses pre-converted text
# Look for: "Using pre-converted text from ..."
```

## Command Reference

### download_attachments.py (with text conversion)

```
python downloader/download_attachments.py CSV_FILE [OPTIONS]

Text Conversion Options:
  --convert-text           Convert PDF/DOCX to .txt after downloading
                           (presentation-style PDFs are skipped by default)
  --force-convert          Force reconversion even if .txt exists (implies --convert-text)
  --include-presentations  Also convert presentation-style PDFs
```

### text_converter.py (standalone)

```
python downloader/text_converter.py DOCKET_ID [OPTIONS]

Arguments:
  DOCKET_ID                   Docket identifier (e.g., CMS-2025-0050)

Options:
  --attachments-dir DIR       Directory containing attachments
                              (default: {docket_id}/comment_attachments/)
  --force                     Reconvert even if .txt files exist
  --include-presentations     Also convert presentation-style PDFs
```

**Output Statistics**:
- `Documents processed`: Number of document directories found
- `Total source files`: Total PDF/DOCX files found
- `Newly converted`: Files converted in this run
- `Presentations skipped`: PDFs filtered by the readability classifier
- `Skipped`: Files with existing .txt (only shown without --force)
- `Failed`: Conversion failures

## Best Practices

1. **Use --convert-text when downloading**: Single command for download + convert
   ```bash
   python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --convert-text
   ```

2. **Check how many presentations were filtered**: The summary shows `Presentations skipped`.
   A handful is normal; if it's a large fraction of your corpus, check whether the thresholds
   need tuning.

3. **Keep source files**: Don't delete PDF/DOCX after conversion
   - Allows re-extraction if needed
   - Preserves original formatting for reference

4. **Use --force sparingly**: Only reconvert when extraction logic changes
   - Saves time by reusing existing conversions
