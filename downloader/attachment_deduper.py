"""attachment_deduper.py — Remove duplicate attachment PDFs within each comment directory.

This utility is intended for the docket-centric attachment layout produced by
`downloader/download_attachments.py`:

    {DOCKET_ID}/comment_attachments/{DOCUMENT_ID}/attachment_1.pdf

Duplicates occasionally occur when regulations.gov exports contain repeated URLs
or when the same PDF is attached multiple times to a single comment.

Rules implemented
-----------------
* Each *document subdirectory* is treated independently.
* Only PDF files are deduplicated.
* Duplicates are detected by SHA-256 hash (size is used as a pre-filter).
* For each hash group, one "keeper" PDF is retained and the rest are deleted.
  The keeper is chosen deterministically (prefer attachment_<N>.pdf with the
  lowest N, else lexical filename order).
* When a PDF named e.g. ``attachment_5.pdf`` is deleted, a sidecar
  ``attachment_5.txt`` (same stem) is also deleted if it exists.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable


_ATTACHMENT_NUM_RE = re.compile(r"^attachment_(\d+)$", re.IGNORECASE)


def _iter_pdf_files(dir_path: Path) -> list[Path]:
    return sorted(
        [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"],
        key=_keeper_sort_key,
    )


def _keeper_sort_key(p: Path) -> tuple[int, int, str]:
    """Sort key used to pick the deterministic keeper within a doc directory.

    Prefer well-formed attachment_<N>.pdf names, with the smallest N, then fall
    back to lexical ordering.
    """
    m = _ATTACHMENT_NUM_RE.match(p.stem)
    if m:
        try:
            n = int(m.group(1))
        except ValueError:
            n = 10**9
        return (0, n, p.name.lower())
    return (1, 10**9, p.name.lower())


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def delete_duplicate_attachments(
    attachments_root: str | Path,
    *,
    dry_run: bool = True,
    delete_sidecar_txt: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Delete duplicate PDFs within each subdirectory of an attachments root.

    Parameters
    ----------
    attachments_root:
        Path to a directory containing per-document subdirectories
        (e.g. ``CMS-2025-0050/comment_attachments``).
    dry_run:
        If True, do not delete anything; only report what *would* be deleted.
    delete_sidecar_txt:
        If True (default), also delete ``<stem>.txt`` when deleting ``<stem>.pdf``.
    verbose:
        If True, print per-directory actions.

    Returns
    -------
    dict
        Summary statistics and per-directory details.
    """
    root = Path(attachments_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Attachments root not found or not a directory: {root}")

    doc_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

    summary: dict[str, Any] = {
        "root": str(root),
        "dry_run": dry_run,
        "delete_sidecar_txt": delete_sidecar_txt,
        "document_dirs_scanned": 0,
        "pdfs_scanned": 0,
        "duplicate_pdfs_found": 0,
        "pdfs_deleted": 0,
        "sidecar_txt_deleted": 0,
        "bytes_freed": 0,
        "per_dir": {},  # doc_id -> details
    }

    for doc_dir in doc_dirs:
        pdfs = _iter_pdf_files(doc_dir)
        if not pdfs:
            continue

        summary["document_dirs_scanned"] += 1
        summary["pdfs_scanned"] += len(pdfs)

        # Pre-filter by file size to minimize hashing.
        size_groups: dict[int, list[Path]] = {}
        for p in pdfs:
            try:
                size_groups.setdefault(p.stat().st_size, []).append(p)
            except OSError:
                # Skip unreadable files, but keep going.
                continue

        hash_to_keeper: dict[str, Path] = {}
        to_delete: list[Path] = []
        groups: list[dict[str, Any]] = []

        for size, group in size_groups.items():
            if len(group) < 2:
                continue

            # Stable ordering so the chosen keeper is deterministic.
            group = sorted(group, key=_keeper_sort_key)
            for p in group:
                try:
                    digest = _sha256(p)
                except OSError:
                    continue

                if digest not in hash_to_keeper:
                    hash_to_keeper[digest] = p
                else:
                    keeper = hash_to_keeper[digest]
                    to_delete.append(p)
                    groups.append(
                        {
                            "sha256": digest,
                            "size": size,
                            "keeper": str(keeper),
                            "duplicate": str(p),
                        }
                    )

        if not to_delete:
            continue

        doc_details: dict[str, Any] = {
            "duplicates": [str(p) for p in sorted(to_delete, key=lambda x: x.name.lower())],
            "groups": groups,
            "pdfs_deleted": 0,
            "sidecar_txt_deleted": 0,
            "bytes_freed": 0,
        }

        if verbose:
            print(f"[dedupe] {doc_dir.name}: {len(to_delete)} duplicate PDF(s)")

        for dup_pdf in to_delete:
            try:
                size = dup_pdf.stat().st_size
            except OSError:
                size = 0

            if verbose:
                action = "WOULD DELETE" if dry_run else "DELETE"
                print(f"  {action}: {dup_pdf.name}")

            if not dry_run:
                try:
                    dup_pdf.unlink()
                except OSError:
                    # If deletion fails, continue with others.
                    continue

            summary["duplicate_pdfs_found"] += 1
            doc_details["bytes_freed"] += size
            summary["bytes_freed"] += size
            if not dry_run:
                summary["pdfs_deleted"] += 1
                doc_details["pdfs_deleted"] += 1

            if delete_sidecar_txt:
                sidecar = dup_pdf.with_suffix(".txt")
                if sidecar.exists():
                    if verbose:
                        action = "WOULD DELETE" if dry_run else "DELETE"
                        print(f"    {action}: {sidecar.name}")
                    if not dry_run:
                        try:
                            sidecar.unlink()
                            summary["sidecar_txt_deleted"] += 1
                            doc_details["sidecar_txt_deleted"] += 1
                        except OSError:
                            pass

        summary["per_dir"][doc_dir.name] = doc_details

    if verbose:
        print("\n" + "=" * 60)
        print("DUPLICATE ATTACHMENT SUMMARY")
        print("=" * 60)
        print(f"Root:                 {summary['root']}")
        print(f"Dry run:              {summary['dry_run']}")
        print(f"Doc dirs scanned:     {summary['document_dirs_scanned']}")
        print(f"PDFs scanned:         {summary['pdfs_scanned']}")
        print(f"Duplicate PDFs found: {summary['duplicate_pdfs_found']}")
        if not dry_run:
            print(f"PDFs deleted:         {summary['pdfs_deleted']}")
            print(f"Sidecar .txt deleted: {summary['sidecar_txt_deleted']}")
        print(f"Bytes freed:          {summary['bytes_freed']:,}")

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Delete duplicate PDFs within each subdirectory of an attachments root",
    )
    p.add_argument(
        "attachments_root",
        help="Root directory containing per-document attachment subdirectories",
    )
    p.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete duplicates (default is dry-run).",
    )
    p.add_argument(
        "--keep-sidecar-txt",
        action="store_true",
        help="Do not delete <stem>.txt when deleting <stem>.pdf.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-directory output (still prints summary).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    delete_duplicate_attachments(
        args.attachments_root,
        dry_run=not args.delete,
        delete_sidecar_txt=not args.keep_sidecar_txt,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
