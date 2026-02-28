# Quick Start Guide - CMS Attachment Downloader

## Installation

1. Install dependencies:
```bash
pip install -r requirements_downloader.txt
```

## Basic Usage

### Download all attachments from the CSV file:

```bash
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv
```

This will:
- Extract URLs from the CSV file
- Download all attachments to `CMS-2025-0050/comment_attachments/`
- Organize files in a structured hierarchy:
  ```
  CMS-2025-0050/
  └── comment_attachments/
      ├── CMS-2025-0050-0004/
      │   └── attachment_1.pdf
      ├── CMS-2025-0050-0007/
      │   ├── attachment_1.pdf
      │   ├── attachment_2.pdf
      │   └── attachment_2.docx
      └── ...
  ```
- Skip files that already exist (resume mode enabled by default)
- Show progress bars for each download
- Create a log file: `download_attachments.log`

### Example output:
```
2026-02-25 12:00:00,000 - INFO - Extracted 980 URLs from CMS-2025-0050/comments/CMS-2025-0050.csv
2026-02-25 12:00:00,000 - INFO - Starting download of 980 files to CMS-2025-0050/comment_attachments/
Overall progress: 100%|████████████████████| 980/980 [05:30<00:00, 2.96file/s]

==================================================
DOWNLOAD SUMMARY
==================================================
Total files:      980
Downloaded:       975
Skipped:          0
Failed:           5
==================================================
```

## Finding Downloaded Files

To find a downloaded file given its original URL:

**Original URL:**
```
https://downloads.regulations.gov/CMS-2025-0050-0007/attachment_2.pdf
```

**Local path:**
```
CMS-2025-0050/comment_attachments/CMS-2025-0050-0007/attachment_2.pdf
```

The path is deterministic - you can reconstruct it from the URL by:
1. Extracting the regulation ID: `CMS-2025-0050` (everything before the last dash-number)
2. Extracting the document ID: `CMS-2025-0050-0007` (the folder in the URL)
3. Extracting the filename: `attachment_2.pdf`
4. Combining: `{regulation_id}/comment_attachments/{document_id}/{filename}`

## Common Commands

### Download and convert to text (recommended):
```bash
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --convert-text
```

### Convert attachments to text standalone:
```bash
python downloader/text_converter.py CMS-2025-0050
python downloader/text_converter.py CMS-2025-0050 --force  # Reconvert existing .txt files
```

### Custom output directory:
```bash
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv -o /path/to/downloads
```

### Force re-download all files:
```bash
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --no-resume
```

### Verbose logging for debugging:
```bash
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --verbose
```

## Using as a Python Module

```python
from download_attachments import get_local_path, parse_url

# Find where a URL will be/is downloaded
url = "https://downloads.regulations.gov/CMS-2025-0050-0007/attachment_2.pdf"
local_path = get_local_path(url)
print(local_path)
# Output: CMS-2025-0050\comment_attachments\CMS-2025-0050-0007\attachment_2.pdf

# Parse a URL to get its components
regulation_id, document_id, filename = parse_url(url)
print(f"Regulation: {regulation_id}")  # CMS-2025-0050
print(f"Document: {document_id}")      # CMS-2025-0050-0007
print(f"Filename: {filename}")         # attachment_2.pdf
```

## Troubleshooting

### Problem: Some files fail to download
- Check the log file: `download_attachments.log`
- Verify your internet connection
- Use `--verbose` flag for more details
- The URLs might be temporarily unavailable - try again later

### Problem: "Column 'Attachment Files' not found"
- Verify your CSV file has the correct column name
- The utility expects a CSV with headers

### Problem: Running out of disk space
- The utility will create directories as needed
- Make sure you have enough disk space for ~980 files
- Use `--output` to specify a different location

## Next Steps

For more detailed information, see:
- **README_DOWNLOADER.md** - Complete documentation
- **TEXT_CONVERSION_README.md** - Text conversion details
- **download_attachments.log** - Detailed execution logs
- Run `python downloader/download_attachments.py --help` for all options
