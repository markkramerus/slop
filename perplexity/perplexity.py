"""
perplexity.py

Standalone Python function that computes per-sentence perplexity for a document
using the MITRE-hosted LLM API (OpenAI-compatible /v1/completions endpoint with
echo=True and logprobs=1).

Algorithm extracted from perplexity.html.

Usage
-----
    from perplexity import get_sentence_perplexities

    results = get_sentence_perplexities(
        document="Your document text goes here...",
        api_key="YOUR_MITRE_API_KEY",
        model="devstral",   # optional
        parallel=20,                   # optional – concurrent API calls
    )

    for r in results:
        print(r["sentence"][:60], "->", r["perplexity"])

Return value
------------
A list of dicts, one per narrative sentence, each containing:
    {
        "sentence":    str,           # sentence text
        "paragraph":   int,           # 0-based paragraph index
        "perplexity":  float | None,  # perplexity score (None on error)
        "token_count": int,           # number of tokens scored
        "error":       str | None,    # error message if applicable
    }
"""

import asyncio
import math
import re
import time
from typing import List, Dict, Optional

import httpx  # pip install httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "https://models.k8s.aip.mitre.org"
# nemotron-3-nano is a base completion model that supports echo+logprobs.
# openai/gpt-oss-120b is chat-tuned and returns logprobs=null for this endpoint.
DEFAULT_MODEL = "devstral"
DEFAULT_PARALLEL = 20
BATCH_DELAY_S = 0.1   # seconds between batches
REQUEST_TIMEOUT_S = 60.0


# ===========================================================================
# STEP 1 – Text cleaning
# ===========================================================================

def clean_text(text: str) -> str:
    """
    Remove non-narrative elements from raw text.

    Mirrors cleanText() in perplexity.html:
    - Strips figure/table captions, inline citations, email addresses,
      standalone page numbers, common header/footer patterns.
    - Removes the References/Bibliography section entirely.
    - Normalises whitespace.
    """
    cleaned = text

    # Figure/table captions
    cleaned = re.sub(r'\b(Figure|Fig\.|Table|Tbl\.)\s*\d+[^.]*\.', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b(Figure|Fig\.|Table|Tbl\.)\s*\d+[^.]*:', '', cleaned, flags=re.IGNORECASE)

    # Inline bracketed citations [1], [1-3], [1,2,3]
    cleaned = re.sub(r'\[\d+(?:[-,\s]*\d+)*\]', '', cleaned)

    # Email addresses
    cleaned = re.sub(r'[\w.\-]+@[\w.\-]+\.\w+', '', cleaned, flags=re.IGNORECASE)

    # Standalone page numbers
    cleaned = re.sub(r'^\s*\d+\s*$', '', cleaned, flags=re.MULTILINE)

    # Common header/footer patterns
    cleaned = re.sub(r'^(page|chapter|section)\s*\d+.*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)

    # References/Bibliography section (remove everything from the heading onward)
    ref_patterns = [
        r'\n\s*(References|Bibliography|Works Cited|Literature Cited|Reference List|Citations)\s*\n[\s\S]*$',
        r'\n\s*\d+\.?\s*(References|Bibliography|Works Cited)\s*\n[\s\S]*$',
        r'\n\s*(REFERENCES|BIBLIOGRAPHY|WORKS CITED)\s*\n[\s\S]*$',
        r'\n\s*Appendix\s+[A-Z0-9]+\.?\s*(References|Bibliography|Works Cited|Citations)\s*\n[\s\S]*$',
        r'\n\s*[A-Z]\.?\s*(References|Bibliography)\s*\n[\s\S]*$',
        r'\n\s*\bReferences\b\s*$',
    ]
    for pattern in ref_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Normalise paragraph breaks
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)

    # Collapse intra-paragraph whitespace
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)

    # Merge single newlines within a paragraph
    cleaned = re.sub(r'(?<!\n)\n(?!\n)', ' ', cleaned)

    return cleaned.strip()


# ===========================================================================
# STEP 2 – Sentence-level filters
# ===========================================================================

def _count_words(sentence: str) -> int:
    return len([w for w in sentence.strip().split() if w])


def _is_heading(sentence: str) -> bool:
    """Mirrors isHeading() in perplexity.html."""
    s = sentence.strip()

    # Numbered section headings
    if re.match(r'^\d+(\.\d+)*\.?\s+[A-Z]', s) and len(s) < 100:
        return True

    # All-caps headings
    if s == s.upper() and len(s) > 3 and len(s) < 100:
        return True

    # Title-case short lines without ending punctuation
    if len(s) < 80 and not re.search(r'[.!?]$', s):
        words = s.split()
        caps = sum(1 for w in words if re.match(r'^[A-Z]', w))
        if len(words) <= 10 and len(words) > 0 and caps / len(words) >= 0.6:
            return True

    # Common heading patterns
    heading_patterns = [
        r'^(Abstract|Introduction|Background|Methods?|Results?|Discussion|Conclusion|Summary|References|Bibliography|Acknowledgements?|Appendix|Table of Contents|List of Figures|List of Tables)',
        r'^(Chapter|Section|Part)\s+\d+',
        r'^[IVXLCDM]+\.\s+',
        r'^[A-Z]\.\s+[A-Z]',
    ]
    for pat in heading_patterns:
        if re.match(pat, s, re.IGNORECASE) and len(s) < 100:
            return True

    return False


def _is_table_entry(sentence: str) -> bool:
    """Mirrors isTableEntry() in perplexity.html."""
    s = sentence.strip()
    if '\t' in s:
        return True
    if '|' in s:
        return True
    numbers = re.findall(r'\d+\.?\d*', s)
    words = re.findall(r'[a-zA-Z]+', s)
    if len(numbers) > 3 and len(numbers) > len(words):
        return True
    if len(s) < 30 and len(numbers) >= 1 and len(words) <= 3:
        return True
    if re.search(r'\s{3,}', s):
        return True
    return False


def _is_reference(sentence: str) -> bool:
    """Mirrors isReference() in perplexity.html."""
    s = sentence.strip()
    if re.match(r'^\[\d+\]', s):
        return True
    if re.match(r'^[A-Z][a-záàâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ\'-]+,\s*[A-Z]\.', s):
        return True
    if re.match(r'^\d+\.\s*[A-Z]', s):
        return True
    if re.search(r'\bdoi[:\s]', s, re.IGNORECASE):
        return True
    if re.search(r'\b(Journal|Proceedings|Conference|Trans\.|IEEE|ACM|NIPS|ICML|Springer|Elsevier|Cambridge|Oxford|Press|Publishing|arXiv|bioRxiv|medRxiv)\b', s, re.IGNORECASE):
        if re.search(r'\d+[:\(]\d+', s) or re.search(r'pp?\.\s*\d+', s, re.IGNORECASE) or re.search(r'vol\.', s, re.IGNORECASE):
            return True
    if re.search(r'https?://', s):
        return True
    if re.search(r'\bet\s+al\.', s, re.IGNORECASE):
        return True
    if re.search(r'\(\d{4}[a-z]?\)', s):
        if re.match(r'^[A-Z][^.]*,\s*[A-Z]', s) or re.search(r'\d+[-\u2013]\d+', s):
            return True
    if re.search(r'\d+[-\u2013]\d+\.?\s*$', s):
        return True
    if re.search(r'\bISBN', s, re.IGNORECASE):
        return True
    if re.search(r'\b(Retrieved|Accessed|Available)\s+(from|at|on)', s, re.IGNORECASE):
        return True
    return False


def _is_narrative(sentence: str) -> bool:
    """Mirrors isNarrativeSentence() in perplexity.html."""
    s = sentence.strip()
    if _count_words(s) <= 5:
        return False
    if _is_heading(s):
        return False
    if _is_table_entry(s):
        return False
    if _is_reference(s):
        return False
    return True


# ===========================================================================
# STEP 3 – Sentence tokenisation
# ===========================================================================

ABBREVIATIONS = [
    'Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Sr', 'Jr', 'vs', 'etc', 'al',
    'Inc', 'Ltd', 'Corp', 'Co', 'St', 'Ave', 'Blvd', 'Rd',
    'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    r'U\.S\.A', r'U\.S', r'U\.K', r'E\.U', r'i\.e', r'e\.g', 'cf', 'viz',
    'Fig', 'Figs', 'Eq', 'Eqs', 'No', 'Nos', 'Vol', 'pp', 'ed', 'eds',
    r'Ph\.D', r'M\.D', r'B\.A', r'M\.A', r'B\.S', r'M\.S',
]

_ABBREV_RE = re.compile(r'\b(' + '|'.join(ABBREVIATIONS) + r')\.', re.IGNORECASE)
_DECIMAL_RE = re.compile(r'(\d)\.(\d)')
_INITIAL_RE = re.compile(r'\b([A-Z])\.(?=\s*[A-Z])')
_ELLIPSIS_RE = re.compile(r'\.{3}')


def tokenize_sentences(text: str) -> List[Dict]:
    """
    Split *text* into narrative sentences.

    Returns a list of dicts::
        {"text": str, "paragraph": int}

    Mirrors tokenizeSentences() in perplexity.html.
    """
    paragraphs = re.split(r'\n\n+', text)
    sentences = []

    for para_idx, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue

        processed = paragraph

        # Protect abbreviations
        processed = _ABBREV_RE.sub(r'\1<<<DOT>>>', processed)
        # Protect decimals
        processed = _DECIMAL_RE.sub(r'\1<<<DOT>>>\2', processed)
        # Protect initials
        processed = _INITIAL_RE.sub(r'\1<<<DOT>>>', processed)
        # Protect ellipsis
        processed = _ELLIPSIS_RE.sub('<<<ELLIPSIS>>>', processed)

        current = ''
        chars = list(processed)

        for i, ch in enumerate(chars):
            current += ch

            if ch in '.!?':
                next_ch = chars[i + 1] if i + 1 < len(chars) else None
                next_next = chars[i + 2] if i + 2 < len(chars) else None

                is_end = (
                    next_ch is None
                    or (next_ch == ' ' and (next_next is None or re.match(r'[A-Z"\'\(]', next_next)))
                    or next_ch == '\n'
                )

                if is_end:
                    sentence = current.strip()
                    sentence = sentence.replace('<<<DOT>>>', '.')
                    sentence = sentence.replace('<<<ELLIPSIS>>>', '...')

                    if len(sentence) > 10:
                        sentences.append({'text': sentence, 'paragraph': para_idx})
                    current = ''

        # Remainder of paragraph
        if len(current.strip()) > 10:
            sentence = current.strip()
            sentence = sentence.replace('<<<DOT>>>', '.')
            sentence = sentence.replace('<<<ELLIPSIS>>>', '...')
            sentences.append({'text': sentence, 'paragraph': para_idx})

    # Filter non-narrative sentences
    return [s for s in sentences if _is_narrative(s['text'])]


# ===========================================================================
# STEP 4 – LLM API call (async)
# ===========================================================================

async def _get_logprobs(
    text: str,
    api_key: str,
    model: str,
    client: httpx.AsyncClient,
) -> dict:
    """
    Call the /v1/completions endpoint and return the raw JSON response.

    Mirrors getLogprobs() in perplexity.html.
    """
    response = await client.post(
        f"{API_BASE_URL}/v1/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "prompt": text,
            "max_tokens": 1,
            "echo": True,
            "logprobs": 1,
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


# ===========================================================================
# STEP 5 – Perplexity calculation
# ===========================================================================

def _calculate_perplexity(logprobs: List[Optional[float]]) -> Optional[float]:
    """
    perplexity = exp(-mean(log_probs))

    Mirrors calculatePerplexity() in perplexity.html.
    """
    if not logprobs:
        return None
    valid = [lp for lp in logprobs if lp is not None and not math.isnan(lp)]
    if not valid:
        return None
    avg = sum(valid) / len(valid)
    return math.exp(-avg)


# ===========================================================================
# STEP 6 – Per-job async processing
# ===========================================================================

async def _process_job(job: dict, api_key: str, model: str, client: httpx.AsyncClient) -> dict:
    """
    Process a single sentence job and return a result dict.

    Mirrors processJob() in perplexity.html.
    The prompt is: [previous sentence] + ' ' + [current sentence] (or just
    the current sentence for the first one).  Raw logprobs are stored so that
    context tokens can be stripped in the post-processing step.
    """
    result = {
        "id": job["id"],
        "sentence": job["sentence"],
        "paragraph": job["paragraph"],
        "status": "complete",
        "logprobs": None,
        "total_tokens": 0,
        "perplexity": None,
        "token_count": 0,
        "error": None,
    }

    try:
        prompt = (
            job["previous_sentence"] + ' ' + job["sentence"]
            if job.get("previous_sentence")
            else job["sentence"]
        )
        api_result = await _get_logprobs(prompt, api_key, model, client)

        choices = api_result.get("choices", [])
        if choices and choices[0].get("logprobs"):
            token_logprobs = choices[0]["logprobs"].get("token_logprobs", [])
            result["logprobs"] = token_logprobs
            result["total_tokens"] = len(token_logprobs) if token_logprobs else 0
        else:
            # Surface what the API actually returned so the caller can diagnose.
            # Most common cause: the model is chat-tuned and ignores echo/logprobs.
            # Try --model nemotron-3-nano or --model devstral (base completion models).
            result["status"] = "error"
            result["error"] = (
                f"logprobs=null from API — model may not support echo+logprobs. "
                f"Try a base completion model (e.g. nemotron-3-nano). "
                f"Response: {str(api_result)[:200]}"
            )
            if job["id"] == 0:   # print once per batch, not for every sentence
                print(
                    f"\n    ⚠  logprobs=null returned for this model.\n"
                    f"       The model '{model}' appears to be chat-tuned and does not\n"
                    f"       support echo+logprobs via the completions endpoint.\n"
                    f"       Try: --model nemotron-3-nano  or  --model devstral\n",
                    flush=True,
                )

    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:500]
        result["status"] = "error"
        result["error"] = f"HTTP {exc.response.status_code}: {body}"
        print(f"    [job {job['id']}] HTTP error {exc.response.status_code}: {body}", flush=True)

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        print(f"    [job {job['id']}] Exception: {exc}", flush=True)

    return result


# ===========================================================================
# STEP 7 – Context-aware perplexity assignment
# ===========================================================================

def _calculate_perplexities_with_context(results: List[dict]) -> None:
    """
    Strip context tokens and compute perplexity for each sentence in place.

    Mirrors calculatePerplexitiesWithContext() in perplexity.html:
    - Sentence 0: use all tokens (no context).
    - Sentence i>0: the previous sentence's token count tells us how many
      leading tokens to skip (those belong to the context sentence).
    """
    prev_token_count = 0

    for i, result in enumerate(results):
        if result["status"] == "error" or result["logprobs"] is None:
            result["perplexity"] = None
            result["token_count"] = 0
            prev_token_count = 0
            continue

        total_tokens = result["total_tokens"]

        if i == 0:
            result["token_count"] = total_tokens
            result["perplexity"] = _calculate_perplexity(result["logprobs"])
        else:
            current_count = total_tokens - prev_token_count
            if 0 < current_count <= total_tokens:
                current_logprobs = result["logprobs"][prev_token_count:]
                result["token_count"] = current_count
                result["perplexity"] = _calculate_perplexity(current_logprobs)
            else:
                # Fallback: use all tokens
                result["token_count"] = total_tokens
                result["perplexity"] = _calculate_perplexity(result["logprobs"])

        prev_token_count = result["token_count"]


# ===========================================================================
# STEP 8 – Parallel batch processing
# ===========================================================================

async def _process_jobs_in_parallel(
    jobs: List[dict],
    api_key: str,
    model: str,
    parallel: int,
) -> List[dict]:
    """
    Process all jobs in parallel batches.

    Mirrors processJobsInParallel() in perplexity.html.
    """
    results = []
    total = len(jobs)

    # verify=False: the MITRE internal API uses an enterprise CA certificate
    # that Python's ssl module may not trust out of the box, even though
    # browsers accept it via the OS certificate store.
    async with httpx.AsyncClient(verify=False) as client:
        for batch_start in range(0, total, parallel):
            batch = jobs[batch_start: batch_start + parallel]
            batch_end = batch_start + len(batch)
            print(f"  Batch {batch_start + 1}-{batch_end} of {total} …")

            tasks = [_process_job(job, api_key, model, client) for job in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
            results.extend(batch_results)

            if batch_end < total:
                await asyncio.sleep(BATCH_DELAY_S)

    # Sort by original ID to restore sentence order
    results.sort(key=lambda r: r["id"])

    # Post-process: strip context tokens and compute perplexity
    _calculate_perplexities_with_context(results)

    return results


# ===========================================================================
# PUBLIC API
# ===========================================================================

def get_sentence_perplexities(
    document: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    parallel: int = DEFAULT_PARALLEL,
) -> List[Dict]:
    """
    Compute per-sentence perplexity for *document*.

    Parameters
    ----------
    document : str
        The raw document text (plain text; not DOCX).
    api_key : str
        MITRE API key for the LLM endpoint.
    model : str
        LLM model identifier (default: "devstral").
    parallel : int
        Number of concurrent API calls (default: 20).

    Returns
    -------
    list of dict
        One dict per narrative sentence::

            {
                "sentence":    str,
                "paragraph":   int,    # 0-based paragraph index
                "perplexity":  float | None,
                "token_count": int,
                "error":       str | None,
            }
    """
    # 1. Clean text
    cleaned = clean_text(document)

    # 2. Tokenise into narrative sentences
    sentences = tokenize_sentences(cleaned)
    if not sentences:
        return []

    print(f"Found {len(sentences)} narrative sentences.  Analysing…")

    # 3. Build jobs (each job carries the previous sentence as context)
    jobs = []
    for idx, sent in enumerate(sentences):
        prev = sentences[idx - 1]["text"] if idx > 0 else None
        jobs.append({
            "id": idx,
            "sentence": sent["text"],
            "paragraph": sent["paragraph"],
            "previous_sentence": prev,
        })

    # 4. Run async batch processing
    raw_results = asyncio.run(
        _process_jobs_in_parallel(jobs, api_key, model, parallel)
    )

    # 5. Return clean, public-facing dicts
    return [
        {
            "sentence":    r["sentence"],
            "paragraph":   r["paragraph"],
            "perplexity":  r["perplexity"],
            "token_count": r["token_count"],
            "error":       r["error"],
        }
        for r in raw_results
    ]


# ===========================================================================
# CLI convenience entry-point
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Compute per-sentence perplexity for a plain-text document."
    )
    parser.add_argument("document", help="Path to the plain-text file to analyse.")
    parser.add_argument("--api-key", required=True, help="MITRE API key.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model identifier.")
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL,
                        help="Number of parallel API calls.")
    parser.add_argument("--output", default="-",
                        help="Output JSON file path (default: stdout).")
    args = parser.parse_args()

    with open(args.document, encoding="utf-8") as fh:
        text = fh.read()

    results = get_sentence_perplexities(
        document=text,
        api_key=args.api_key,
        model=args.model,
        parallel=args.parallel,
    )

    out = json.dumps(results, indent=2, ensure_ascii=False)

    if args.output == "-":
        sys.stdout.write(out + "\n")
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out + "\n")
        print(f"Results written to {args.output}")
