# Attachment Downloader Utility

A Python utility to download attachment files from regulations.gov CSV exports and organize them in a docket-centric directory hierarchy.

## Overview

This utility reads CSV files exported from regulations.gov, extracts URLs from the "Attachment Files" column, and downloads them into a structured directory under the docket's master folder.

## File Organization

Files are organized under the docket directory:

```
{DOCKET_ID}/
└── comment_attachments/
    └── {DOCUMENT_ID}/
        └── {FILENAME}
```

**Example:**
```
CMS-2025-0050/
└── comment_attachments/
    ├── CMS-2025-0050-0957/
    │   └── attachment_1.pdf
    ├── CMS-2025-0050-0958/
    │   └── attachment_1.pdf
    └── CMS-2025-0050-0961/
        ├── attachment_1.pdf
        ├── attachment_2.pdf
        └── attachment_2.png
```

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements_downloader.txt
```

Or manually:

```bash
pip install requests tqdm
```

## Usage

### Basic Usage

Download all attachments from a CSV file:

```bash
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv
```

This downloads to `CMS-2025-0050/comment_attachments/` by default.

### Custom Output Directory

Specify a custom download directory:

```bash
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv -o /path/to/downloads
```

### Convert Attachments to Text

Download and automatically convert PDF/DOCX files to `.txt` for faster downstream processing:

```bash
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --convert-text
```

Force reconversion of existing `.txt` files:

```bash
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --force-convert
```

You can also run text conversion standalone (after downloading):

```bash
python downloader/text_converter.py CMS-2025-0050
python downloader/text_converter.py CMS-2025-0050 --force
```

See **TEXT_CONVERSION_README.md** for full details.

### Force Re-download

By default, the utility skips files that already exist (resume mode). To re-download all files:

```bash
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --no-resume
```

### Verbose Logging

Enable detailed logging for debugging:

```bash
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --verbose
```

### Help

View all available options:

```bash
python downloader/download_attachments.py --help
```

## AI Attachment Classification (comment vs not-comment)

Downloader includes an **AI-based classifier** that walks a docket's
`comment_attachments/` tree and labels each **PDF attachment** as:

- `comment` — the PDF itself is the substantive response / regulatory comment
- `not_comment` — the PDF is supporting material (academic paper), a slide deck,
  brochure/product sheet, or a cover/transmittal letter that primarily says
  "see attached"

The classifier output drives the text conversion step: **only PDFs classified as
`comment` are converted to text**, minimizing wasted conversions.

### Recommended Workflow

```
1. Download attachments          → download_attachments.py
2. AI-classify each PDF          → classify_attachments_ai.py  → attachment_classification.csv
3. Convert comment PDFs to text  → text_converter.py  (reads attachment_classification.csv)
```

### Configuration

Set these environment variables (in `.env` or your shell):

- `SLOP_CLASSIFER_API_BASE_URL`
- `SLOP_CLASSIFER_API_KEY`
- `SLOP_CLASSIFER_MODEL`

### Run classification

```bash
python downloader/classify_attachments_ai.py CMS-2025-0050/comment_attachments \
  --output CMS-2025-0050/attachment_classification.csv \
  --limit 25
```

Notes:
- Output is a **per-attachment CSV** (one row per PDF) with AI-only columns.
- The classifier is independent of text conversion — it sends rendered page
  images to the AI endpoint and does not need `.txt` files.

### Resume / re-run

Re-running will skip PDFs already present in the output CSV. To force
reclassification:

```bash
python downloader/classify_attachments_ai.py CMS-2025-0050/comment_attachments \
  --output CMS-2025-0050/attachment_classification.csv \
  --force
```

### Convert only comment PDFs to text

After classification, convert only the files the AI labeled `comment`:

```bash
python downloader/text_converter.py CMS-2025-0050
# reads CMS-2025-0050/attachment_classification.csv by default
```

### If your endpoint expects a different PDF upload format

The default request format sends the PDF by **rendering the first pages to images**
and including them as OpenAI-style `messages[].content` `image_url` parts. This is
generally the most compatible approach for OpenAI-compatible endpoints.

Some gateways may require a different shape; try:

```bash
python downloader/classify_attachments_ai.py CMS-2025-0050/comment_attachments \
  --output CMS-2025-0050/attachment_classification.csv \
  --request-format attachments_field
```

or (less standard, for multipart-upload gateways):

```bash
python downloader/classify_attachments_ai.py CMS-2025-0050/comment_attachments \
  --output CMS-2025-0050/attachment_classification.csv \
  --request-format multipart_form
```

## Features

- **Automatic directory creation**: Creates nested directories based on regulation and document IDs
- **Resume capability**: Skips already downloaded files by default (use `--no-resume` to override)
- **Progress tracking**: Shows progress bars for individual files and overall progress
- **Text conversion**: Optionally convert PDF/DOCX attachments to `.txt` files after downloading (`--convert-text`)
- **Error handling**: Continues downloading even if some files fail
- **Logging**: Writes detailed logs to `download_attachments.log`
- **Summary report**: Displays statistics at the end (total, downloaded, skipped, failed)
- **Multiple file types**: Handles PDF, DOCX, PNG, and other file formats

## Finding Downloaded Files

Given an original URL, you can easily find the corresponding local file:

**Original URL:**
```
https://downloads.regulations.gov/CMS-2025-0050-0961/attachment_2.pdf
```

**Local Path (default):**
```
CMS-2025-0050/comment_attachments/CMS-2025-0050-0961/attachment_2.pdf
```

The path is deterministic and can be reconstructed from the URL using the `get_local_path()` function:

```python
from download_attachments import get_local_path

url = "https://downloads.regulations.gov/CMS-2025-0050-0961/attachment_2.pdf"
local_path = get_local_path(url)
print(local_path)  # CMS-2025-0050/comment_attachments/CMS-2025-0050-0961/attachment_2.pdf
```

## Using as a Module

You can also import and use the utility functions in your own Python code:

```python
from download_attachments import parse_url, get_local_path, download_file
from pathlib import Path

# Parse a URL
regulation_id, document_id, filename = parse_url(
    "https://downloads.regulations.gov/CMS-2025-0050-0961/attachment_2.pdf"
)
print(f"Regulation: {regulation_id}")  # CMS-2025-0050
print(f"Document: {document_id}")      # CMS-2025-0050-0961
print(f"Filename: {filename}")         # attachment_2.pdf

# Get local path for a URL
local_path = get_local_path(url, base_dir="my_downloads")

# Download a single file
url = "https://downloads.regulations.gov/CMS-2025-0050-0961/attachment_1.pdf"
local_path = Path("downloads/CMS-2025-0050/CMS-2025-0050-0961/attachment_1.pdf")
success = download_file(url, local_path)
```

## CSV Format Requirements

The utility expects a CSV file with the following characteristics:

- Must have a header row
- Must contain a column named "Attachment Files"
- URLs in the "Attachment Files" column can be:
  - Single URL: `https://downloads.regulations.gov/CMS-2025-0050-0961/attachment_1.pdf`
  - Multiple URLs (comma-separated): `https://downloads.regulations.gov/.../attachment_1.pdf, https://downloads.regulations.gov/.../attachment_2.png`

## Logging

All operations are logged to `download_attachments.log` in the current directory. The log includes:

- URLs being downloaded
- Files being skipped (already exist)
- Download errors with details
- Summary statistics

## Error Handling

The utility is designed to be resilient:

- **Network errors**: Logs the error and continues with remaining downloads
- **Missing files**: HTTP 404 errors are logged but don't stop execution
- **Invalid URLs**: Logs the error and skips to the next URL
- **Disk space**: Creates directories as needed; fails gracefully if disk is full

## Exit Codes

- `0`: Success (all files downloaded or skipped)
- `1`: Partial failure (some files failed to download)

## Troubleshooting

### "Column 'Attachment Files' not found"
- Check that your CSV has the correct column name
- The utility will list available columns in the error message

### "CSV file not found"
- Verify the path to your CSV file is correct
- Use absolute paths if relative paths aren't working

### Files failing to download
- Check your internet connection
- Check the log file for specific error messages
- Use `--verbose` flag for more detailed logging
- Verify the URLs are accessible in a browser

### Permission errors
- Ensure you have write permissions in the output directory
- Try specifying a different output directory with `--output`

## Performance Notes

- Download speed depends on your internet connection and the regulations.gov server
- The utility downloads files sequentially (not in parallel) to be respectful of the server
- Large files show progress bars with transfer speed
- Resume mode significantly speeds up re-runs by skipping existing files

## License

This utility is provided as-is for working with public CMS regulations data.
