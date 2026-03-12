"""downloader/attachment_ai_classifier.py

AI-based classifier for downloaded PDF attachments.

This module walks a regulations.gov-style attachment tree:

  {DOCKET_ID}/comment_attachments/{DOCUMENT_ID}/attachment_N.pdf

For each PDF attachment, it sends rendered page images to an
OpenAI-compatible chat-completions endpoint configured by environment
variables:
  - SLOP_CLASSIFER_API_BASE_URL
  - SLOP_CLASSIFER_API_KEY
  - SLOP_CLASSIFER_MODEL

The AI classification (comment vs not_comment) is written to a CSV file
(default: ``attachment_classification.csv``) which downstream tools—such as
``text_converter.py``—read to determine which PDFs to convert to text.

Workflow
--------
1. ``download_attachments.py`` downloads all PDFs for a docket.
2. **This module** classifies each PDF via AI → ``attachment_classification.csv``.
3. ``text_converter.py`` reads the CSV and converts only *comment* PDFs to text.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import requests
from dotenv import load_dotenv

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:  # pragma: no cover
    _HAS_TQDM = False


# Load .env so classifier vars are available when running as a standalone script.
load_dotenv()


# ── Prompting ───────────────────────────────────────────────────────────────

PROMPT_VERSION = "v2"

_SYSTEM_PROMPT = """\
You are a document triage system for a U.S. federal regulatory docket.

You will be given a single PDF attachment.

Task: classify whether THIS PDF contains the substantive public comment
response to the rule/RFI (i.e., the actual arguments/recommendations submitted),
as opposed to non-comment content.

Definitions:
- label=comment
  The PDF contains the substantive comment/response in the form of a letter, memo, table, 
  or multi-part narrative, containing arguments, requests, recommendations, critiques, or
  specific feedback aimed at the agency. Comments may be preceded by a cover sheet, 
  a cover letter, and/or a table of contents.

- label=not_comment
  The PDF does NOT contain the substantive comment. Examples include:
    * cover letter / transmittal letter (if standalone)
    * academic paper / article / white paper / supporting evidence
    * presentation / slide deck
    * marketing / brochure / company description / product description / datasheet
    * other (forms / receipts / signatures pages / unrelated materials)

Borderline rules (must follow):
* If the PDF is a standalone cover letter (the file has no additional content), label=not_comment
* If the PDF has more than one page, starts with a cover letter, and is then followed by comments, label=comment
* Graphics appearing in a letterhead are non-determinative

CRITICAL OUTPUT RULES — you MUST follow all of these:
1. Return ONLY a single JSON object. Do NOT output anything before or after it.
2. Do NOT output markdown, code fences, or commentary.
3. The "rationale" value MUST be <=100 characters.
4. Use exactly these keys: "label", "confidence", "doc_type", "rationale".

Example of a valid response (your output must look exactly like this):
{"label": "comment", "confidence": 0.95, "doc_type": "substantive_comment", "rationale": "Contains policy recommendations and arguments directed at the agency."}
"""

_USER_PROMPT = """\
Classify the attached PDF.

Filename: {filename}
Document ID: {document_id}
"""


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClassifierConfig:
    chat_completions_url: str
    api_key: str
    model: str
    timeout_s: float = 120.0


def _env(name: str, default: str = "") -> str:
    if name == "SLOP_CLASSIFER_API_BASE_URL":
        return os.getenv("SLOP_CLASSIFER_API_BASE_URL") or os.getenv("SLOP_CLASSIFIER_API_BASE_URL", default)
    if name == "SLOP_CLASSIFER_API_KEY":
        return os.getenv("SLOP_CLASSIFER_API_KEY") or os.getenv("SLOP_CLASSIFIER_API_KEY", default)
    if name == "SLOP_CLASSIFER_MODEL":
        return os.getenv("SLOP_CLASSIFER_MODEL") or os.getenv("SLOP_CLASSIFIER_MODEL", default)
    return os.getenv(name, default)


def load_classifier_config() -> ClassifierConfig:
    base = (_env("SLOP_CLASSIFER_API_BASE_URL") or "").strip()
    key = (_env("SLOP_CLASSIFER_API_KEY") or "").strip()
    model = (_env("SLOP_CLASSIFER_MODEL") or "").strip()

    if not base:
        raise ValueError("Missing SLOP_CLASSIFER_API_BASE_URL env var")
    if not key:
        raise ValueError("Missing SLOP_CLASSIFER_API_KEY env var")
    if not model:
        raise ValueError("Missing SLOP_CLASSIFER_MODEL env var")

    # Allow env to be either the API root (.../v1) or the full chat-completions URL.
    # Many OpenAI-compatible servers use /v1/chat/completions.
    if base.rstrip("/").endswith("/chat/completions"):
        chat_url = base
    else:
        chat_url = base.rstrip("/") + "/chat/completions"

    return ClassifierConfig(chat_completions_url=chat_url, api_key=key, model=model)


# ── JSON parsing helpers ────────────────────────────────────────────────────


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Parse a JSON object from a model response.

    Uses a layered strategy to maximise data recovery from malformed LLM output:

    1. Direct ``json.loads`` on the (cleaned) string.
    2. Brace-matching to extract the first complete ``{…}`` object, handling
       trailing garbage, concatenated objects, and encoding artefacts.
    3. Truncated-JSON repair: if the response was cut off mid-value, close any
       open strings/braces and attempt to parse.
    4. Regex field extraction as a last resort — individually pull out
       ``label``, ``confidence``, ``doc_type``, and ``rationale`` even when
       the overall JSON is mangled.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}

    # ── Pre-clean: strip markdown code fences ──────────────────────────
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

    # ── Strategy 1: direct json.loads ──────────────────────────────────
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # ── Strategy 2: brace-matching extraction ──────────────────────────
    # Find the first '{' and walk to its matching '}', ignoring trailing
    # garbage, concatenated objects, encoding artefacts, etc.
    start = raw.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = raw[start : i + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        # If json.loads failed on the brace-matched block,
                        # don't keep scanning — fall through to repair.
                        break

    # ── Strategy 3: truncated-JSON repair ──────────────────────────────
    # The response may have been cut off by max_tokens. Try to close any
    # open string literal and open braces, then parse.
    if start is not None and start >= 0:
        fragment = raw[start:]
        # Strip trailing non-ASCII garbage (encoding artefacts)
        fragment = fragment.encode("ascii", errors="ignore").decode("ascii").rstrip()
        # If we're inside a string value that was never closed, close it.
        # Count unescaped quotes after the opening brace.
        quote_count = 0
        esc = False
        for ch in fragment[1:]:  # skip the leading '{'
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                quote_count += 1
        if quote_count % 2 == 1:
            # Odd number of quotes → an unclosed string. Close it.
            fragment = fragment.rstrip()
            # Remove trailing partial escape sequences
            while fragment.endswith("\\"):
                fragment = fragment[:-1]
            fragment += '"'
        # Now close any open braces.
        depth = 0
        for ch in fragment:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        while depth > 0:
            fragment += "}"
            depth -= 1
        try:
            parsed = json.loads(fragment)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # ── Strategy 4: regex field extraction (last resort) ───────────────
    # Even severely mangled output often contains recognisable key-value
    # fragments. Pull them out individually.
    result: dict[str, Any] = {}

    # label
    m = re.search(r'"label"\s*:\s*"(comment|not_comment|uncertain)"', raw)
    if m:
        result["label"] = m.group(1)

    # confidence
    m = re.search(r'"confidence"\s*:\s*([\d.]+)', raw)
    if m:
        try:
            result["confidence"] = float(m.group(1))
        except ValueError:
            pass

    # doc_type
    m = re.search(r'"doc_type"\s*:\s*"([^"]+)"', raw)
    if m:
        result["doc_type"] = m.group(1)

    # rationale — value may be truncated (no closing quote), so we use a
    # greedy-but-bounded capture that stops at a quote or end-of-string.
    m = re.search(r'"rationale"\s*:\s*"([^"]{0,300})', raw)
    if m:
        result["rationale"] = m.group(1)

    return result


def _clean_rationale(s: str, max_len: int = 200) -> str:
    s = (s or "").strip().replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


# ── Attachment discovery ────────────────────────────────────────────────────


_ATTACHMENT_RE = re.compile(r"^attachment_(\d+)\.pdf$", re.IGNORECASE)


@dataclass(frozen=True)
class AttachmentItem:
    document_id: str
    pdf_path: Path
    attachment_num: int | None

    @property
    def filename(self) -> str:
        return self.pdf_path.name

    @property
    def txt_path(self) -> Path:
        return self.pdf_path.with_suffix(".txt")


def iter_pdf_attachments(attachments_root: Path) -> Iterable[AttachmentItem]:
    """Yield AttachmentItems under `attachments_root`.

    Expected layout:
      attachments_root/<DOCUMENT_ID>/attachment_N.pdf
    """
    if not attachments_root.is_dir():
        raise FileNotFoundError(f"attachments_root is not a directory: {attachments_root}")

    # Full tree walk: tolerate nested structures, but derive document_id as the
    # first path segment under attachments_root when possible.
    for pdf in sorted(attachments_root.rglob("*.pdf")):
        try:
            rel = pdf.relative_to(attachments_root)
            document_id = rel.parts[0] if len(rel.parts) >= 2 else pdf.parent.name
        except Exception:
            document_id = pdf.parent.name

        m = _ATTACHMENT_RE.match(pdf.name)
        n = int(m.group(1)) if m else None
        yield AttachmentItem(document_id=document_id, pdf_path=pdf, attachment_num=n)


# ── AI call ─────────────────────────────────────────────────────────────────


def _build_payload_content_parts(filename: str, document_id: str, pdf_bytes: bytes) -> dict[str, Any]:
    """Build a chat.completions payload using OpenAI-style multipart content.

    Many OpenAI-compatible servers accept `messages[].content` as a list of
    structured parts (commonly used for images). For PDF attachments, there is
    no single universal schema. We implement a reasonable best-effort format
    and allow operators to adjust the server-side parser if needed.
    """
    user_text = _USER_PROMPT.format(filename=filename, document_id=document_id)

    return {
        "model": None,  # filled by caller
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    # image_url parts added by caller (PDF rendered to images)
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 400,
    }


def _pdf_first_pages_as_png_data_uris(
    pdf_path: Path,
    *,
    max_pages: int = 3,
    dpi: int = 150,
) -> list[str]:
    """Render the first N PDF pages to PNG data URIs.

    This is the most broadly compatible way to "send a PDF" to OpenAI-compatible
    endpoints, because chat-completions generally supports image inputs via
    `image_url` but not raw PDF blobs.

    For multi-page PDFs, at least 3 pages are always rendered regardless of the
    ``max_pages`` argument so the AI has enough context to distinguish standalone
    cover letters from substantive comments that begin with a cover page.
    """
    if max_pages <= 0:
        return []

    try:
        from pdf2image import convert_from_path  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "pdf2image is required for PDF->image rendering (pip install pdf2image)"
        ) from e

    # For multi-page PDFs, guarantee at least 2 pages so the AI can see past an
    # opening cover/transmittal page and make an accurate comment vs. not-comment
    # determination.
    effective_max_pages = max_pages
    try:
        import pypdf  # type: ignore

        page_count = len(pypdf.PdfReader(str(pdf_path)).pages)
        if page_count > 1:
            effective_max_pages = max(max_pages, 2)
    except Exception:  # noqa: BLE001
        # If pypdf is unavailable or the page count can't be read, fall back to
        # whatever max_pages was requested.
        pass

    # convert_from_path can restrict page ranges.
    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=1,
        last_page=effective_max_pages,
    )

    uris: list[str] = []
    for img in images:
        # PIL Image -> PNG bytes
        from io import BytesIO

        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        uris.append(f"data:image/png;base64,{b64}")
    return uris


def _build_payload_attachments_field(filename: str, document_id: str, pdf_bytes: bytes) -> dict[str, Any]:
    """Alternate payload shape using a top-level `attachments` field.

    Some OpenAI-compatible gateways accept attachments out-of-band from
    messages, e.g. as a list of base64-encoded blobs.
    """
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    user_text = _USER_PROMPT.format(filename=filename, document_id=document_id)
    return {
        "model": None,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "attachments": [
            {
                "filename": filename,
                "mime_type": "application/pdf",
                "data": b64,
            }
        ],
        "temperature": 0.0,
        "max_tokens": 400,
    }


def classify_pdf_via_ai(
    *,
    item: AttachmentItem,
    config: ClassifierConfig,
    request_format: Literal["content_parts", "attachments_field", "multipart_form"] = "content_parts",
    max_mb: float = 20.0,
    sleep_s: float = 0.0,
    max_pages: int = 3,
    dpi: int = 150,
) -> dict[str, Any]:
    """Send one PDF to the AI endpoint and return a parsed classification dict."""

    pdf_path = item.pdf_path
    try:
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
    except OSError:
        size_mb = 0.0

    if max_mb and size_mb > max_mb:
        return {
            "label": "uncertain",
            "confidence": 0.0,
            "doc_type": "other",
            "rationale": f"skipped: file too large ({size_mb:.1f} MB > {max_mb:.1f} MB)",
            "error": "file_too_large",
        }

    try:
        pdf_bytes = pdf_path.read_bytes()
    except Exception as e:  # noqa: BLE001
        return {
            "label": "uncertain",
            "confidence": 0.0,
            "doc_type": "other",
            "rationale": "failed to read pdf",
            "error": f"read_failed: {e}",
        }

    if sleep_s:
        time.sleep(sleep_s)

    payload: dict[str, Any] | None = None
    if request_format == "content_parts":
        # Render PDF pages to images and send as image_url parts.
        try:
            page_uris = _pdf_first_pages_as_png_data_uris(
                item.pdf_path,
                max_pages=max_pages,
                dpi=dpi,
            )
        except Exception as e:  # noqa: BLE001
            return {
                "label": "2",
                "confidence": 0.0,
                "doc_type": "other",
                "rationale": "pdf render failed",
                "error": f"pdf_render_failed: {e}",
            }

        payload = _build_payload_content_parts(item.filename, item.document_id, pdf_bytes)

        # Append image parts
        user_msg = payload["messages"][1]
        content_parts: list[dict[str, Any]] = user_msg["content"]
        for uri in page_uris:
            content_parts.append({"type": "image_url", "image_url": {"url": uri}})
    elif request_format == "attachments_field":
        payload = _build_payload_attachments_field(item.filename, item.document_id, pdf_bytes)

    headers = {
        "Authorization": f"Bearer {config.api_key}",
    }

    # Note: multipart_form is not part of the OpenAI chat.completions spec.
    # It's provided as a pragmatic escape hatch for gateways that accept
    # multipart uploads.
    if request_format == "multipart_form":
        files = {
            "file": (item.filename, pdf_bytes, "application/pdf"),
        }
        form = {
            "model": config.model,
            "system": _SYSTEM_PROMPT,
            "user": _USER_PROMPT.format(filename=item.filename, document_id=item.document_id),
            "temperature": "0.0",
            "max_tokens": "400",
        }
        try:
            resp = requests.post(
                config.chat_completions_url,
                headers=headers,
                data=form,
                files=files,
                timeout=config.timeout_s,
            )
        except Exception as e:  # noqa: BLE001
            return {
                "label": "uncertain",
                "confidence": 0.0,
                "doc_type": "other",
                "rationale": "request failed",
                "error": f"request_failed: {e}",
            }
    else:
        assert payload is not None
        payload["model"] = config.model
        headers["Content-Type"] = "application/json"

        try:
            resp = requests.post(
                config.chat_completions_url,
                headers=headers,
                json=payload,
                timeout=config.timeout_s,
            )
        except Exception as e:  # noqa: BLE001
            return {
                "label": "uncertain",
                "confidence": 0.0,
                "doc_type": "other",
                "rationale": "request failed",
                "error": f"request_failed: {e}",
            }

    if resp.status_code >= 400:
        # Avoid dumping response text in full (could be large); keep first ~400 chars.
        t = (resp.text or "")
        return {
            "label": "uncertain",
            "confidence": 0.0,
            "doc_type": "other",
            "rationale": f"http_{resp.status_code}",
            "error": f"http_{resp.status_code}: {t[:400]}",
        }

    try:
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        return {
            "label": "uncertain",
            "confidence": 0.0,
            "doc_type": "other",
            "rationale": "invalid json response",
            "error": f"invalid_json: {e}",
        }

    # OpenAI chat.completions format: choices[0].message.content
    content = ""
    try:
        choices = data.get("choices") or []
        content = choices[0]["message"]["content"]
    except Exception:
        content = ""

    parsed = _parse_json_response(content)

    label = str(parsed.get("label", "uncertain"))
    if label not in {"comment", "not_comment", "uncertain"}:
        label = "uncertain"

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    doc_type = str(parsed.get("doc_type", "other"))
    rationale = _clean_rationale(str(parsed.get("rationale", "")))

    return {
        "label": label,
        "confidence": confidence,
        "doc_type": doc_type,
        "rationale": rationale,
        "raw": content.strip(),
    }


# ── CSV output ──────────────────────────────────────────────────────────────


CSV_COLUMNS = [
    "document_id",
    "attachment_filename",
    "attachment_path",
    "size_bytes",
    "ai_label",
    "ai_confidence",
    "ai_doc_type",
    "ai_rationale",
    "ai_raw",
    "model",
    "prompt_version",
    "request_format",
    "error",
]


def _load_existing_keys(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    try:
        with output_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            keys = set()
            for row in reader:
                p = (row.get("attachment_path") or "").strip()
                if p:
                    keys.add(p)
            return keys
    except Exception:
        return set()


def _ensure_parent_dir(p: Path) -> None:
    if p.parent:
        p.parent.mkdir(parents=True, exist_ok=True)


def _classify_one_item(
    item: AttachmentItem,
    config: ClassifierConfig,
    request_format: str,
    max_mb: float,
    sleep_s: float,
    max_pages: int,
    dpi: int,
) -> tuple[AttachmentItem, dict[str, Any]]:
    """Classify a single attachment via the AI endpoint.

    Designed to be called from a thread pool worker.

    Returns (item, ai_result).
    """
    ai = classify_pdf_via_ai(
        item=item,
        config=config,
        request_format=request_format,
        max_mb=max_mb,
        sleep_s=sleep_s,
        max_pages=max_pages,
        dpi=dpi,
    )

    return item, ai


def classify_attachment_tree(
    *,
    attachments_root: str | Path,
    output_csv: str | Path,
    force: bool = False,
    limit: int | None = None,
    request_format: str = "content_parts",
    max_mb: float = 20.0,
    sleep_s: float = 0.0,
    max_pages: int = 3,
    dpi: int = 150,
    verbose: bool = True,
    progress_every: int = 10,
    concurrency: int = 10,
) -> dict[str, Any]:
    """Classify all PDF attachments under a directory tree.

    Up to ``concurrency`` attachments (default 10) are sent to the AI endpoint
    in parallel using a :class:`~concurrent.futures.ThreadPoolExecutor`, which
    dramatically reduces wall-clock time when the bottleneck is network I/O.

    Returns a small stats dict.
    """
    root = Path(attachments_root)
    out = Path(output_csv)
    _ensure_parent_dir(out)

    config = load_classifier_config()

    existing = set() if force else _load_existing_keys(out)
    wrote_header = out.exists() and out.stat().st_size > 0

    stats: dict[str, Any] = {
        "total_seen": 0,
        "skipped_existing": 0,
        "classified": 0,
        "errors": 0,
        "output_csv": str(out),
    }

    all_items = list(iter_pdf_attachments(root))
    stats["total_seen"] = len(all_items)

    # Split into items to skip (already done) and items to classify.
    skip_items: list[AttachmentItem] = []
    work_items: list[AttachmentItem] = []
    for item in all_items:
        if (not force) and str(item.pdf_path) in existing:
            skip_items.append(item)
        else:
            work_items.append(item)

    stats["skipped_existing"] = len(skip_items)

    # Apply limit: only classify up to `limit` new items.
    if limit is not None:
        work_items = work_items[:limit]

    # Progress bar wraps the work list when tqdm is available.
    progress: Any = (
        tqdm(total=len(work_items), desc="Classifying PDFs", unit="pdf")
        if (_HAS_TQDM and verbose)
        else None
    )

    csv_lock = threading.Lock()

    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not wrote_header:
            writer.writeheader()

        # Submit all work items to the thread pool and process results as they
        # complete so the CSV is written incrementally (crash-safe).
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            future_to_item: dict[Future[Any], AttachmentItem] = {
                pool.submit(
                    _classify_one_item,
                    item,
                    config,
                    request_format,
                    max_mb,
                    sleep_s,
                    max_pages,
                    dpi,
                ): item
                for item in work_items
            }

            completed = 0
            for future in as_completed(future_to_item):
                completed += 1
                item = future_to_item[future]
                err = ""

                try:
                    result_item, ai = future.result()
                except Exception as exc:  # noqa: BLE001
                    # Unexpected worker crash – record as error row.
                    err = f"worker_exception: {exc}"
                    ai = {
                        "label": "uncertain",
                        "confidence": 0.0,
                        "doc_type": "other",
                        "rationale": "worker crashed",
                        "error": err,
                    }

                if ai.get("error"):
                    if err:
                        err = err + "; " + str(ai.get("error"))
                    else:
                        err = str(ai.get("error"))
                    stats["errors"] += 1

                try:
                    size_bytes = item.pdf_path.stat().st_size
                except OSError:
                    size_bytes = 0

                row = {
                    "document_id": item.document_id,
                    "attachment_filename": item.filename,
                    "attachment_path": str(item.pdf_path),
                    "size_bytes": size_bytes,
                    "ai_label": ai.get("label", ""),
                    "ai_confidence": ai.get("confidence", ""),
                    "ai_doc_type": ai.get("doc_type", ""),
                    "ai_rationale": ai.get("rationale", ""),
                    "ai_raw": ai.get("raw", ""),
                    "model": config.model,
                    "prompt_version": PROMPT_VERSION,
                    "request_format": request_format,
                    "error": err,
                }

                with csv_lock:
                    writer.writerow(row)
                    f.flush()

                existing.add(str(item.pdf_path))
                stats["classified"] += 1

                if progress is not None:
                    progress.update(1)
                elif verbose and progress_every and (completed % progress_every == 0):
                    print(
                        f"[{completed}/{len(work_items)}] classified={stats['classified']}"
                        f" skipped_existing={stats['skipped_existing']}"
                        f" errors={stats['errors']}"
                        f" last={item.document_id}/{item.filename}"
                        f" ai={row['ai_label']}",
                        flush=True,
                    )

    if progress is not None:
        progress.close()

    return stats


# ── Reparse existing CSV ───────────────────────────────────────────────────


def reparse_csv(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Re-read an existing classification CSV and re-parse all ``ai_raw`` values.

    This uses the improved :func:`_parse_json_response` parser to recover
    ``label``, ``confidence``, ``doc_type``, and ``rationale`` from rows that
    were previously marked ``uncertain`` due to malformed LLM output — **without
    re-calling the AI endpoint**.

    Parameters
    ----------
    input_csv:
        Path to the existing classification CSV.
    output_csv:
        Where to write the reparsed CSV.  If *None*, overwrites *input_csv*.
    verbose:
        Print a summary when done.

    Returns
    -------
    dict with keys ``total_rows``, ``reparsed``, ``still_uncertain``.
    """
    inp = Path(input_csv)
    out = Path(output_csv) if output_csv else inp

    if not inp.exists():
        raise FileNotFoundError(f"Input CSV not found: {inp}")

    rows: list[dict[str, str]] = []
    with inp.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or CSV_COLUMNS
        for row in reader:
            rows.append(dict(row))

    stats = {"total_rows": len(rows), "reparsed": 0, "still_uncertain": 0}

    for row in rows:
        old_label = (row.get("ai_label") or "").strip()
        raw = (row.get("ai_raw") or "").strip()

        # Only attempt reparse if the label is uncertain/empty and there is
        # raw content to work with.
        if old_label not in ("uncertain", "") or not raw:
            continue

        parsed = _parse_json_response(raw)

        new_label = str(parsed.get("label", ""))
        if new_label not in ("comment", "not_comment"):
            stats["still_uncertain"] += 1
            continue

        # Update fields from the newly parsed data.
        row["ai_label"] = new_label

        if "confidence" in parsed:
            try:
                row["ai_confidence"] = str(max(0.0, min(1.0, float(parsed["confidence"]))))
            except (ValueError, TypeError):
                pass

        if "doc_type" in parsed:
            row["ai_doc_type"] = str(parsed["doc_type"])

        if "rationale" in parsed:
            row["ai_rationale"] = _clean_rationale(str(parsed["rationale"]))

        stats["reparsed"] += 1

    # Write output
    _ensure_parent_dir(out)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"Reparse complete: {stats['reparsed']} rows recovered, "
              f"{stats['still_uncertain']} still uncertain, "
              f"{stats['total_rows']} total rows → {out}")

    return stats
