# Text Conversion for Attachment Processing

This document explains how to use the text conversion functionality to pre-convert downloaded attachments to text files, which speeds up downstream analysis (e.g., stylometry) and allows for easier inspection of extracted text.

## Overview

The downloader module supports two approaches for converting attachments to text:

1. **Integrated conversion** (recommended): Use the `--convert-text` flag when downloading
2. **Standalone conversion**: Run `text_converter.py` directly after downloading

## Benefits of Pre-Converting to Text

- **Speed**: Text files are read instantly; PDF/DOCX extraction can be slow
- **Consistency**: Once converted, the same text is used across multiple analyses
- **Debugging**: Easy to inspect what text was extracted from attachments
- **Reprocessing**: Run stylometry analysis multiple times without re-extracting

## Usage

### Option 1: Download + Convert in One Step (Recommended)

```bash
# Download attachments and convert to text in one command
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --convert-text

# Force reconversion (even if .txt files exist)
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --force-convert
```

### Option 2: Standalone Text Conversion

After downloading attachments, convert them separately:

```bash
# Basic usage — looks in CMS-2025-0050/comment_attachments/
python downloader/text_converter.py CMS-2025-0050

# Specify custom attachments directory
python downloader/text_converter.py CMS-2025-0050 --attachments-dir path/to/attachments

# Force reconversion (even if .txt files exist)
python downloader/text_converter.py CMS-2025-0050 --force
```

Both options will:
1. Find all PDF and DOCX files in `CMS-2025-0050/comment_attachments/*/`
2. Create corresponding `.txt` files (e.g., `attachment_1.pdf` → `attachment_1.txt`)
3. Skip files that already have `.txt` versions (unless force is used)

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
    ...
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

### Text Conversion Logic (`downloader/text_converter.py`)

The converter:
1. Finds all `.pdf`, `.docx`, and `.doc` files in the docket directory
2. For each file, creates a `.txt` file with the same basename
3. Uses `pypdf` and `python-docx` for text extraction
4. Skips conversion if `.txt` already exists (unless `--force` is specified)

### Downloader Integration (`download_attachments.py --convert-text`)

When the `--convert-text` flag is passed:
1. Downloads all attachments as normal
2. Determines the docket ID from the CSV filename
3. Calls `convert_docket_to_text()` to convert all downloaded PDF/DOCX files
4. Prints a combined summary of both download and conversion statistics

### Stylometry Analyzer Integration (`stylometry_utils.get_attachment_text`)

The stylometry analyzer's `get_attachment_text` function:
1. Lists all files in a document's attachment directory
2. For each file, checks if a corresponding `.txt` file exists
3. If `.txt` exists: reads it directly (fast path)
4. If `.txt` doesn't exist: extracts from source file (slow path)
5. Avoids processing duplicates (e.g., both `attachment_1.pdf` and `attachment_1.txt`)

### Backward Compatibility

The stylometry analyzer remains fully backward compatible:
- Works without pre-converted text files (extracts on-the-fly)
- Works with a mix of converted and non-converted files
- No changes to command-line interface or existing workflows

## Troubleshooting

### Issue: Text extraction fails for some files

**Solution**: Check the conversion output for errors:

```bash
python downloader/text_converter.py CMS-2025-0050
# Look for "Failed" count in summary
```

Inspect specific `.txt` files to verify content. You may need to manually convert problematic files.

### Issue: Want to update extraction for specific files

**Solution**: Delete the `.txt` files you want to reconvert, then run without `--force`:

```bash
# Delete specific txt file
del CMS-2025-0050\comment_attachments\CMS-2025-0050-0004\attachment_1.txt

# Reconvert only files without .txt
python downloader/text_converter.py CMS-2025-0050
```

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

```bash
python downloader/download_attachments.py CSV_FILE [OPTIONS]

Text Conversion Options:
  --convert-text           Convert PDF/DOCX to .txt after downloading
  --force-convert          Force reconversion even if .txt exists (implies --convert-text)
```

### text_converter.py (standalone)

```bash
python downloader/text_converter.py DOCKET_ID [OPTIONS]

Arguments:
  DOCKET_ID           Docket identifier (e.g., CMS-2025-0050)

Options:
  --attachments-dir DIR  Directory containing attachments (default: {docket_id}/comment_attachments/)
  --force                Reconvert even if .txt files exist
```

**Output Statistics**:
- `Documents processed`: Number of document directories found
- `Total source files`: Total PDF/DOCX files found
- `Newly converted`: Files converted in this run
- `Skipped`: Files with existing .txt (only shown without --force)
- `Failed`: Conversion failures

## Best Practices

1. **Use --convert-text when downloading**: Single command for download + convert
   ```bash
   python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --convert-text
   ```

2. **Check conversion statistics**: Ensure most files convert successfully
   - Failed conversions may indicate corrupted files or unsupported formats

3. **Keep source files**: Don't delete PDF/DOCX after conversion
   - Allows re-extraction if needed
   - Preserves original formatting for reference

4. **Use --force sparingly**: Only reconvert when extraction logic changes
   - Saves time by reusing existing conversions
