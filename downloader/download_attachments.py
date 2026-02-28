#!/usr/bin/env python3
"""
Utility to download attachment files from regulations.gov CSV exports.

This script reads a CSV file (e.g., CMS-2025-0050.csv), extracts URLs from
the "Attachment Files" column, and downloads them into a docket-centric
directory hierarchy.

File organization (default):
    {DOCKET_ID}/
    └── comment_attachments/
        └── {DOCUMENT_ID}/
            └── {FILENAME}

Example:
    CMS-2025-0050/comment_attachments/CMS-2025-0050-0961/attachment_1.pdf

Usage:
    python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv
    python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv -o custom/path
    python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --convert-text
    python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --force-convert
"""

import argparse
import csv
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_attachments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_url(url: str) -> Tuple[str, str, str]:
    """
    Parse a regulations.gov download URL into its components.
    
    Args:
        url: The full download URL
        
    Returns:
        Tuple of (regulation_id, document_id, filename)
        
    Example:
        >>> parse_url('https://downloads.regulations.gov/CMS-2025-0050-0961/attachment_1.pdf')
        ('CMS-2025-0050', 'CMS-2025-0050-0961', 'attachment_1.pdf')
    """
    url = url.strip()
    parsed = urlparse(url)
    path_parts = parsed.path.strip('/').split('/')
    
    if len(path_parts) < 2:
        raise ValueError(f"Invalid URL format: {url}")
    
    document_id = path_parts[0]
    filename = path_parts[1]
    
    # Extract regulation ID (everything before the last dash and digits)
    # Example: CMS-2025-0050-0961 → CMS-2025-0050
    match = re.match(r'(.*)-\d+$', document_id)
    if match:
        regulation_id = match.group(1)
    else:
        # Fallback: use the document_id as regulation_id
        regulation_id = document_id
    
    return regulation_id, document_id, filename


def get_local_path(url: str, base_dir: str | None = None) -> Path:
    """
    Convert a download URL to a local file path.
    
    Uses the new docket-centric layout: {regulation_id}/comment_attachments/{document_id}/{filename}
    
    Args:
        url: The full download URL
        base_dir: Override base directory. If provided, uses {base_dir}/{document_id}/{filename}.
                  If not provided, uses {regulation_id}/comment_attachments/{document_id}/{filename}.
        
    Returns:
        Path object representing the local file path
        
    Example:
        >>> get_local_path('https://downloads.regulations.gov/CMS-2025-0050-0961/attachment_1.pdf')
        Path('CMS-2025-0050/comment_attachments/CMS-2025-0050-0961/attachment_1.pdf')
    """
    regulation_id, document_id, filename = parse_url(url)
    if base_dir:
        return Path(base_dir) / document_id / filename
    return Path(regulation_id) / "comment_attachments" / document_id / filename


def download_file(url: str, local_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from a URL to a local path with progress bar.
    
    Args:
        url: The URL to download from
        local_path: The local file path to save to
        chunk_size: Size of chunks to download (default: 8192 bytes)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directories if they don't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set headers to mimic a browser request and avoid 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        # Download with streaming to handle large files
        response = requests.get(url, stream=True, timeout=30, headers=headers)
        response.raise_for_status()
        
        # Get total file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(local_path, 'wb') as f, \
             tqdm(total=total_size, unit='B', unit_scale=True, 
                  desc=local_path.name, leave=False) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Downloaded: {url} -> {local_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {url}: {e}")
        return False


def extract_urls_from_csv(csv_path: str, column_name: str = 'Attachment Files') -> list:
    """
    Extract all URLs from the specified column in a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        column_name: Name of the column containing URLs
        
    Returns:
        List of URLs (each URL is stripped and cleaned)
    """
    urls = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if column_name not in reader.fieldnames:
                logger.error(f"Column '{column_name}' not found in CSV")
                logger.info(f"Available columns: {reader.fieldnames}")
                return []
            
            for row in reader:
                attachment_files = row.get(column_name, '').strip()
                
                if attachment_files:
                    # Split by comma for multiple URLs
                    file_urls = [url.strip() for url in attachment_files.split(',')]
                    urls.extend([url for url in file_urls if url])
        
        logger.info(f"Extracted {len(urls)} URLs from {csv_path}")
        return urls
        
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return []


def process_csv(csv_path: str, base_dir: str | None = None, resume: bool = True) -> dict:
    """
    Process a CSV file and download all attachments.
    
    Args:
        csv_path: Path to the CSV file
        base_dir: Override base directory for downloads. If not provided,
                  uses {regulation_id}/comment_attachments/ layout.
        resume: Skip files that already exist (default: True)
        
    Returns:
        Dictionary with download statistics
    """
    stats = {
        'total': 0,
        'downloaded': 0,
        'skipped': 0,
        'failed': 0
    }
    
    # Extract URLs from CSV
    urls = extract_urls_from_csv(csv_path)
    stats['total'] = len(urls)
    
    if not urls:
        logger.warning("No URLs found to download")
        return stats
    
    # Download each file
    dest_desc = base_dir if base_dir else "{docket_id}/comment_attachments/"
    logger.info(f"Starting download of {len(urls)} files to {dest_desc}")
    
    for url in tqdm(urls, desc="Overall progress", unit="file"):
        try:
            local_path = get_local_path(url, base_dir)
            
            # Skip if file already exists and resume is enabled
            if resume and local_path.exists():
                logger.debug(f"Skipping existing file: {local_path}")
                stats['skipped'] += 1
                continue
            
            # Download the file
            if download_file(url, local_path):
                stats['downloaded'] += 1
            else:
                stats['failed'] += 1
                
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            stats['failed'] += 1
    
    return stats


def print_summary(stats: dict, conversion_stats: dict | None = None) -> None:
    """Print a summary of download statistics."""
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"Total files:      {stats['total']}")
    print(f"Downloaded:       {stats['downloaded']}")
    print(f"Skipped:          {stats['skipped']}")
    print(f"Failed:           {stats['failed']}")
    print("=" * 50)

    if conversion_stats:
        print()
        print("=" * 50)
        print("TEXT CONVERSION SUMMARY")
        print("=" * 50)
        print(f"Documents processed:  {conversion_stats['document_count']}")
        print(f"Total source files:   {conversion_stats['total_files']}")
        print(f"Newly converted:      {conversion_stats['converted']}")
        print(f"Skipped (existed):    {conversion_stats['skipped']}")
        print(f"Failed:               {conversion_stats['failed']}")
        print("=" * 50)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Download attachment files from CMS regulations CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'csv_file',
        help=(
            'Docket ID (e.g., CMS-2025-0050) or path to the CSV file. '
            'When given a docket ID, looks for {docket_id}/comments/{docket_id}.csv.'
        ),
    )
    
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output directory for downloads (default: {docket_id}/comment_attachments/)'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Re-download files even if they exist'
    )
    
    parser.add_argument(
        '--convert-text',
        action='store_true',
        help='Convert downloaded PDF/DOCX attachments to .txt files after downloading'
    )
    
    parser.add_argument(
        '--force-convert',
        action='store_true',
        help='Force reconversion of .txt files even if they already exist (implies --convert-text)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # ── Resolve docket ID or CSV path ──────────────────────────────────────
    # If the argument doesn't end in .csv, treat it as a docket ID and
    # derive the conventional path: {docket_id}/comments/{docket_id}.csv
    csv_input = args.csv_file
    if not csv_input.lower().endswith('.csv'):
        docket_id = csv_input.rstrip('/\\')
        csv_path = os.path.join(docket_id, 'comments', f'{docket_id}.csv')
        logger.info(f"Docket ID '{docket_id}' → using {csv_path}")
    else:
        csv_path = csv_input
        # Derive docket_id from CSV filename for use in text conversion
        csv_stem = Path(csv_path).stem
        parts = csv_stem.split('-')
        docket_id = '-'.join(parts[:3]) if len(parts) >= 3 else csv_stem

    # Verify CSV file exists
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        if not csv_input.lower().endswith('.csv'):
            logger.error(
                f"Hint: create the directory structure {docket_id}/comments/ "
                f"and place {docket_id}.csv there, or pass the full CSV path."
            )
        sys.exit(1)
    
    # Process the CSV and download files
    resume = not args.no_resume
    stats = process_csv(csv_path, args.output, resume=resume)
    
    # Text conversion step (--force-convert implies --convert-text)
    conversion_stats = None
    do_convert = args.convert_text or args.force_convert
    
    if do_convert:
        from text_converter import convert_docket_to_text
        
        # docket_id was already resolved above
        
        logger.info(f"\nConverting attachments to text for docket: {docket_id}")
        try:
            conversion_stats = convert_docket_to_text(
                docket_id=docket_id,
                attachments_dir=args.output,
                force=args.force_convert,
            )
        except FileNotFoundError as e:
            logger.error(f"Text conversion failed: {e}")
        except Exception as e:
            logger.error(f"Text conversion failed: {e}", exc_info=True)
    
    # Print summary
    print_summary(stats, conversion_stats)
    
    # Exit with error code if there were failures
    if stats['failed'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
