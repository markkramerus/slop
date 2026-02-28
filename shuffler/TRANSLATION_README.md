# Synthetic Comments Translation Tool

## Overview

This tool translates synthetic comments from the output format (♔-delimited) back to the original CMS CSV format (comma-delimited). This allows synthetic comments to be reformatted for submission or analysis alongside real CMS comments.

## Files

- **`translate_to_cms_format.py`** - Main translation script
- **`verify_translation.py`** - Quick verification script to check output

## Usage

### Basic Usage

```bash
python shuffler/translate_to_cms_format.py <input_file> [output_file]
```

Example:
```bash
python shuffler/translate_to_cms_format.py CMS-2025-0050/synthetic_comments/comments.txt CMS-2025-0050/synthetic_comments/comments_cms.csv
```

If the output file is omitted, a default name is derived from the input file (e.g., `comments_cms.csv`).

### Verify Translation

After translation, verify the output:

```bash
python shuffler/verify_translation.py CMS-2025-0050/synthetic_comments/comments_cms.csv
```

## Input Format

The script expects synthetic comments in the format produced by the slop pipeline:
- ♔ (crown symbol) as the delimiter
- Headers matching the synthetic comments schema
- UTF-8 encoding

## Output Format

The script produces a standard CMS CSV file with:
- Comma delimiter
- All required CMS column headers in the correct order
- Properly mapped fields from synthetic to CMS format
- UTF-8 encoding

## Field Mapping

Key mappings from synthetic format to CMS format:

| Synthetic Field | CMS Field | Notes |
|----------------|-----------|-------|
| Comment ID | Document ID | Used as primary identifier |
| Document ID | Docket ID | Strips -SYNTH suffix |
| Submitter Name | First Name, Last Name | Parsed into components |
| synth_persona_state | State/Province | - |
| Organization Name | Organization Name | - |
| Comment | Comment | Full comment text |
| Posted Date | Posted Date | - |
| Received Date | Received Date | - |

## Example Output

```
Sample translated record:
============================================================
Document ID: RIN 0955-AA09-SYNTH-0001
Agency ID: CMS
Docket ID: RIN 0955-AA09
First Name: Donna
Last Name: Perez
State: Minnesota
Organization: CareAtlas Inc.
Document Type: Public Submission
Posted Date: 2026-01-20T00:00:00Z
============================================================
```

## Technical Details

### Dependencies
- Python 3.6+
- Standard library only (csv, sys, datetime, pathlib)

### Error Handling
- Validates input file exists before processing
- Provides clear error messages
- Displays progress every 100 records
- Reports final statistics

### Encoding
- Uses UTF-8 encoding for both input and output
- Handles special characters in comments properly

## Limitations

- Some CMS fields don't have direct equivalents in synthetic format and are left empty
- Name parsing assumes "FirstName LastName" format
- All synthetic comments default to "CMS" as Agency ID

## Future Enhancements

Potential improvements:
- Support for bulk processing of multiple files
- Option to include/exclude synthetic metadata columns
- Validation against CMS schema
- Support for additional output formats
