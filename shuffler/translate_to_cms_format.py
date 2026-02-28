"""
Translate synthetic comments format back to CMS CSV format.

This script converts the synthetic comments output (with ♔ delimiter) 
back to the original CMS CSV format (comma-delimited).
"""

import csv
import sys
from datetime import datetime
from pathlib import Path


def parse_name(full_name):
    """Parse a full name into first and last name."""
    if not full_name or full_name.strip() == '':
        return '', ''
    
    parts = full_name.strip().split()
    if len(parts) == 0:
        return '', ''
    elif len(parts) == 1:
        return parts[0], ''
    else:
        # First name is first part, last name is everything else
        return parts[0], ' '.join(parts[1:])


def translate_synthetic_to_cms(input_file, output_file):
    """
    Translate synthetic comments format back to CMS CSV format.
    
    Args:
        input_file: Path to synthetic comments file (♔-delimited)
        output_file: Path to output CMS CSV file
    """
    
    # Define the CMS CSV column headers in the correct order
    cms_headers = [
        'Document ID', 'Agency ID', 'Docket ID', 'Tracking Number', 
        'Document Type', 'Posted Date', 'Is Withdrawn?', 'Federal Register Number',
        'FR Citation', 'Title', 'Comment Start Date', 'Comment Due Date',
        'Allow Late Comments', 'Comment on Document ID', 'Effective Date',
        'Implementation Date', 'Postmark Date', 'Received Date', 'Author Date',
        'Related RIN(s)', 'Authors', 'CFR', 'Abstract', 'Legacy ID', 'Media',
        'Document Subtype', 'Exhibit Location', 'Exhibit Type', 'Additional Field 1',
        'Additional Field 2', 'Topics', 'Duplicate Comments', 'OMB/PRA Approval Number',
        'Page Count', 'Page Length', 'Paper Width', 'Special Instructions',
        'Source Citation', 'Start End Page', 'Subject', 'First Name', 'Last Name',
        'City', 'State/Province', 'Zip/Postal Code', 'Country', 'Organization Name',
        'Submitter Representative', 'Representative\'s Address',
        'Representative\'s City, State & Zip', 'Government Agency',
        'Government Agency Type', 'Comment', 'Category', 'Restrict Reason Type',
        'Restrict Reason', 'Reason Withdrawn', 'Content Files', 'Attachment Files',
        'Display Properties (Name, Label, Tooltip)'
    ]
    
    records_processed = 0
    records_written = 0
    
    try:
        # Read the synthetic comments file
        with open(input_file, 'r', encoding='utf-8') as infile:
            # Use csv.DictReader with custom delimiter
            reader = csv.DictReader(infile, delimiter='♔')
            
            # Open output CSV file
            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=cms_headers)
                writer.writeheader()
                
                for row in reader:
                    records_processed += 1
                    
                    # Parse the submitter name into first and last
                    first_name, last_name = parse_name(row.get('Submitter Name', ''))
                    
                    # Extract document ID (without the -SYNTH suffix if present)
                    comment_id = row.get('Comment ID', '')
                    doc_id = row.get('Document ID', '')
                    
                    # Map synthetic format to CMS format
                    cms_row = {
                        'Document ID': comment_id,  # Use Comment ID as Document ID
                        'Agency ID': 'CMS',  # Default to CMS
                        'Docket ID': doc_id.split('-SYNTH')[0] if '-SYNTH' in doc_id else doc_id,
                        'Tracking Number': '',
                        'Document Type': 'Public Submission',
                        'Posted Date': row.get('Posted Date', ''),
                        'Is Withdrawn?': 'false',
                        'Federal Register Number': row.get('Federal Register Number', ''),
                        'FR Citation': '',
                        'Title': f"Comment on {doc_id.split('-SYNTH')[0]}" if doc_id else '',
                        'Comment Start Date': row.get('Comment Start Date', ''),
                        'Comment Due Date': row.get('Comment End Date', ''),
                        'Allow Late Comments': 'false',
                        'Comment on Document ID': doc_id.split('-SYNTH')[0] if '-SYNTH' in doc_id else doc_id,
                        'Effective Date': '',
                        'Implementation Date': '',
                        'Postmark Date': '',
                        'Received Date': row.get('Received Date', ''),
                        'Author Date': '',
                        'Related RIN(s)': '',
                        'Authors': '',
                        'CFR': '',
                        'Abstract': row.get('Abstract', ''),
                        'Legacy ID': '',
                        'Media': '',
                        'Document Subtype': 'Public Comment',
                        'Exhibit Location': row.get('Exhibit Location', ''),
                        'Exhibit Type': row.get('Exhibit Type', ''),
                        'Additional Field 1': '',
                        'Additional Field 2': '',
                        'Topics': '',
                        'Duplicate Comments': '',
                        'OMB/PRA Approval Number': '',
                        'Page Count': row.get('Page Count', ''),
                        'Page Length': '',
                        'Paper Width': '',
                        'Special Instructions': '',
                        'Source Citation': '',
                        'Start End Page': '',
                        'Subject': '',
                        'First Name': first_name,
                        'Last Name': last_name,
                        'City': '',
                        'State/Province': row.get('synth_persona_state', ''),
                        'Zip/Postal Code': '',
                        'Country': 'United States' if row.get('synth_persona_state', '') else '',
                        'Organization Name': row.get('Organization Name', ''),
                        'Submitter Representative': row.get('Submitter\'s Representative', ''),
                        'Representative\'s Address': '',
                        'Representative\'s City, State & Zip': '',
                        'Government Agency': row.get('Government Agency', ''),
                        'Government Agency Type': row.get('Government Agency Type', ''),
                        'Comment': row.get('Comment', ''),
                        'Category': '',
                        'Restrict Reason Type': '',
                        'Restrict Reason': '',
                        'Reason Withdrawn': '',
                        'Content Files': '',
                        'Attachment Files': row.get('Attachment Files', ''),
                        'Display Properties (Name, Label, Tooltip)': 'pageCount, Page Count, Number of pages In the content file'
                    }
                    
                    writer.writerow(cms_row)
                    records_written += 1
                    
                    if records_processed % 100 == 0:
                        print(f"Processed {records_processed} records...")
        
        print(f"\nTranslation complete!")
        print(f"Records processed: {records_processed}")
        print(f"Records written: {records_written}")
        print(f"Output saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during translation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main function to run the translation."""
    
    # Default file paths (no docket-specific defaults — user must provide paths)
    input_file = None
    output_file = None
    
    # Check if file paths provided as arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    if not input_file:
        print("Synthetic Comments to CMS Format Translator")
        print("=" * 50)
        print("\nUsage: python shuffler/translate_to_cms_format.py <input_file> [output_file]")
        print("\nExample:")
        print("  python shuffler/translate_to_cms_format.py CMS-2025-0050/synthetic_comments/comments.txt CMS-2025-0050/synthetic_comments/comments_cms.csv")
        sys.exit(1)
    
    # Default output file based on input
    if not output_file:
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix('.csv').with_name(input_path.stem + '_cms.csv'))
    
    print("Synthetic Comments to CMS Format Translator")
    print("=" * 50)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print("=" * 50)
    print()
    
    # Verify input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        print("\nUsage: python shuffler/translate_to_cms_format.py <input_file> [output_file]")
        sys.exit(1)
    
    # Run the translation
    translate_synthetic_to_cms(input_file, output_file)


if __name__ == '__main__':
    main()
