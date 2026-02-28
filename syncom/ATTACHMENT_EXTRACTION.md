# Attachment Text Extraction

## Overview

The ingestion system now automatically extracts and incorporates text from downloaded attachment files (PDFs and DOCX) when building the population model. This ensures that comments submitted as attachments are properly analyzed, not just the brief text in the CSV.

## How It Works

### 1. Attachment Discovery
When processing a CSV row, the system:
- Checks for a `Document ID` column (e.g., "CMS-2025-0050-0004")
- Checks for an `Attachment Files` column (URLs to attachments)
- Looks for downloaded files in: `downloads/{docket_id}/{document_id}/`

### 2. Text Extraction
The system supports:
- **PDF files** (.pdf) - extracted using `pypdf`
- **DOCX files** (.docx, .doc) - extracted using `python-docx`

Multiple attachments per comment are automatically combined.

### 3. Text Merging Strategy
- If CSV comment is **< 50 words** and attachments exist → Use attachment text primarily
- If CSV comment is **≥ 50 words** → Concatenate CSV + attachment text
- Long texts are **truncated at 10,000 words** to prevent memory issues

### 4. Error Handling
- Missing attachment directories are logged at DEBUG level (not an error)
- Failed PDF/DOCX extractions are logged as warnings
- Extraction statistics are reported at INFO level

## Usage

### Basic Usage (Automatic)
```python
from syncom.ingestion import ingest_docket_csv

# Attachment extraction enabled by default
model = ingest_docket_csv(
    csv_path="CMS-2025-0050-0031.csv",
    docket_id="CMS-2025-0050"
)
```

### Custom Downloads Directory
```python
model = ingest_docket_csv(
    csv_path="my_docket.csv",
    downloads_dir="path/to/downloads"
)
```

### Disable Attachment Extraction
```python
# Set downloads_dir to empty string
model = ingest_docket_csv(
    csv_path="my_docket.csv",
    downloads_dir=""
)
```

## Requirements

The attachment extraction feature requires additional dependencies:
```
pypdf>=3.17.0
python-docx>=1.1.0
```

These are included in the main `requirements.txt`. If not installed, the system gracefully falls back to CSV-only ingestion with warnings.

## Expected Directory Structure

```
downloads/
└── {docket_id}/           # e.g., CMS-2025-0050
    ├── {document_id}/     # e.g., CMS-2025-0050-0004
    │   ├── attachment1.pdf
    │   ├── attachment2.docx
    │   └── ...
    ├── {document_id}/
    │   └── ...
    └── ...
```

This structure is automatically created by the `/downloader` sub-application.

## Logging

To see detailed extraction progress, configure logging:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

You'll see messages like:
```
INFO: Extracted text from 3 attachment(s) for CMS-2025-0050-0007
INFO: Attachment extraction: 713/723 successful, 10 failed
```

## Impact on Population Model

Comments with attachments typically result in:
- **Higher average word counts** per archetype
- **More substantive linguistic fingerprints** (sentence length, citation counts, etc.)
- **Better representation** of formal organizational submissions
- **More realistic training data** for synthetic comment generation

## Troubleshooting

### PDF Parsing Warnings
You may see warnings like:
```
WARNING: Ignoring wrong pointing object 11 0 (offset 0)
```

These are normal for slightly malformed PDFs. The library handles them gracefully and continues extraction.

### Missing Attachments
If `Attachment Files` column exists but attachments weren't downloaded:
```
DEBUG: Attachment directory not found: downloads/CMS-2025-0050/CMS-2025-0050-XXXX
```

Solution: Run the `/downloader` application to download attachments first.

### Import Errors
If pypdf or python-docx aren't installed:
```
WARNING: pypdf not available, skipping PDF extraction: ...
```

Solution: `pip install pypdf python-docx`

## Performance

- Extraction speed: ~1-5 seconds per PDF (depends on size)
- Memory usage: Moderate (texts truncated at 10,000 words)
- Typical extraction rate: 95-99% success across most dockets
