"""
pipeline/rubric_generator.py
=============================
Generates a structured rubric from a question paper PDF using Gemini VLM.

Changes in this version:
  1. Overlapping chunks         — pages 0-4, 3-7, 6-10 so cross-page questions aren't split
  2. Math-operator fallback     — semantic validation accepts "x²+5x+6=0" even without hint words
  3. Best-extraction dedup      — keeps longer question_text + more steps, not just first-seen
  4. Single shared Gemini client — created once, reused across all chunks and retries
  5. Multi-hint AK detection    — page needs score ≥ 2 to be treated as answer-key page
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import fitz          # PyMuPDF
import pdfplumber
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHUNK_SIZE         = 4      # pages per Gemini call
CHUNK_OVERLAP      = 1      # pages of overlap between consecutive chunks
API_RETRIES        = 3
API_BACKOFF_BASE   = 5      # seconds; wait = base * 2^attempt
COMPLETENESS_RATIO = 0.70
AK_MIN_HINT_SCORE  = 2      # Fix 5: page needs ≥ 2 AK hints to be filtered out

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

RUBRIC_EXTRACTION_PROMPT = """
You are extracting a structured marking rubric from a CBSE mathematics exam paper.

Extract ONLY actual student-answerable exam questions.

DO NOT extract:
- instructions or general directions
- blueprint tables or weightage tables
- marking schemes or answer keys
- solved examples or worked solutions
- evaluation schemes or difficulty tables
- page headers, page footers, section descriptions
- lines like "all questions are compulsory"

For every REAL question extract:

  question_number  — string e.g. "1", "12", "3a"
  section          — A | B | C | D
  question_type    — mcq | short | long | descriptive
  marks            — integer as printed on the paper
  question_text    — complete question text verbatim
  keywords         — mathematical terms/concepts required in a correct answer
  steps            — GRANULAR ordered list of markable solution steps (rules below)

Rules for steps (CRITICAL):
- Every step must be a concrete independently markable action.
- BAD:  ["solve the equation"]
- GOOD: ["Write LHS = RHS statement", "Substitute x=2", "Compute both sides", "Conclude equality"]
- BAD:  ["draw the triangle"]
- GOOD: ["Draw base BC = 6 cm with ruler", "Construct 60 degree angle at B", "Arc from B and C to find A", "Label all vertices"]
- Short answer (<=2 marks): minimum 2 steps.
- Long / descriptive (>2 marks): minimum 3 steps.
- MCQ: leave steps as [].

DO NOT guess MCQ correct answers. Set correct_answer to "" for every question.

Return ONLY a valid JSON array. No explanation. No markdown fences. No extra text.

Schema for each element:
{
  "question_number": "11",
  "section":         "B",
  "question_type":   "short",
  "marks":           2,
  "question_text":   "Verify that x = 2 is a zero of p(x) = x^3 - 8.",
  "correct_answer":  "",
  "keywords":        ["zero of polynomial", "substitution", "p(2)=0"],
  "steps":           ["Substitute x=2 into p(x)", "Compute 2^3 - 8 = 8 - 8 = 0", "Conclude x=2 is a zero"]
}
"""

# ---------------------------------------------------------------------------
# AK hint scoring  (Fix 5: need ≥ 2 hits, not just any 1)
# ---------------------------------------------------------------------------

ANSWER_KEY_HINTS = [
    "answer key",
    "answers",
    "marking scheme",
    "marking guidelines",
    "solution",
    "answer sheet",
    "key answers",
    "correct option",
]

def _ak_hint_score(text: str) -> int:
    """Count how many AK hint phrases appear in the page text."""
    lower = text.lower()
    return sum(1 for hint in ANSWER_KEY_HINTS if hint in lower)

def _is_answer_key_page(text: str) -> bool:
    """Fix 5: require at least AK_MIN_HINT_SCORE matches to avoid false positives."""
    return _ak_hint_score(text) >= AK_MIN_HINT_SCORE


# ---------------------------------------------------------------------------
# Semantic filters
# ---------------------------------------------------------------------------

BAD_QTEXT_PATTERNS = [
    "weightage", "difficulty", "all questions are compulsory",
    "marks allotted", "forms of questions", "scheme of options",
    "general instructions", "time allowed", "internal choice",
    "blueprint", "evaluation scheme",
]

QUESTION_HINTS = [
    "find", "solve", "construct", "draw", "prove", "evaluate",
    "show", "verify", "calculate", "simplify", "factorise", "factorize",
    "expand", "express", "write", "define", "state", "which", "what",
    "how many", "if ", "the value", "is equal", "are equal",
]

# Fix 3: math-operator pattern — if present, treat as valid even without hint words
_MATH_OP_RE = re.compile(r"[=+\-×÷√∫∑π²³⁴]|\\frac|\\sqrt|\^|\d+x")

_VALID_SECTIONS = {"A", "B", "C", "D"}

# ---------------------------------------------------------------------------
# Shared Gemini client  (Fix 4: single instance)
# ---------------------------------------------------------------------------

_client: genai.Client | None = None

def _get_client(api_key: str) -> genai.Client:
    """Fix 4: create client once and reuse across all chunks and retries."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_rubric(qp_pdf_path: str | Path, api_key: str = "") -> dict:
    """
    Main entry point.

    Returns:
    {
        "mcq_questions":     [{question_number, section, max_marks, correct_answer, keywords}],
        "written_questions": [{question_number, section, max_marks, question_text, expected_steps, keywords}],
        "page_index":        {"11": [2], "12": [2, 3], ...}
    }
    """
    global _client
    _client = None   # reset so a fresh client is built with the right key each run

    api_key     = api_key or os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    qp_pdf_path = Path(qp_pdf_path)
    if not qp_pdf_path.exists():
        raise FileNotFoundError(f"Question paper not found: {qp_pdf_path}")

    logger.info("Generating rubric from: %s", qp_pdf_path)

    # Step 1: chunked VLM extraction with overlapping windows
    flat = _generate_via_vlm_chunked(qp_pdf_path, api_key)
    if not flat:
        logger.warning("VLM extraction produced nothing — falling back to regex.")
        flat = _generate_via_regex(qp_pdf_path)

    # Step 2: best-extraction dedup (keeps richer entry, not just first-seen)
    flat = _deduplicate_best(flat)

    # Step 3: semantic + structural validation
    flat = _validate_questions(flat)

    # Step 4: completeness check
    _check_completeness(flat, qp_pdf_path)

    # Step 5: extract MCQ answers locally
    mcq_answer_key = _extract_mcq_answers_locally(qp_pdf_path)
    logger.info("Local MCQ answer key: %s", mcq_answer_key)

    for q in flat:
        qnum = str(q.get("question_number", "")).strip()
        if q.get("question_type") == "mcq" and qnum in mcq_answer_key:
            q["correct_answer"] = mcq_answer_key[qnum]

    # Step 6: build page index
    page_index = _build_page_index(qp_pdf_path)
    logger.info("Page index built for %d questions.", len(page_index))

    result = _split_and_normalise(flat)
    result["page_index"] = page_index
    return result


# ---------------------------------------------------------------------------
# Fix 1: Overlapping chunked VLM extraction
# ---------------------------------------------------------------------------

def _overlapping_chunks(lst: list, size: int, overlap: int):
    """
    Yield overlapping windows of `size` with step = size - overlap.

    Example with size=4, overlap=1:
      pages 0-3, 3-6, 6-9, 9-12 ...

    This ensures a question spanning a chunk boundary (e.g. starts page 3,
    ends page 4) appears fully in at least one chunk.
    """
    step  = max(1, size - overlap)
    start = 0
    while start < len(lst):
        yield lst[start : start + size]
        start += step


def _generate_via_vlm_chunked(qp_pdf_path: Path, api_key: str) -> list[dict]:
    page_bytes = _pdf_to_png_bytes(qp_pdf_path)
    page_texts = _extract_page_texts(qp_pdf_path)

    # Strip answer-key pages before chunking
    filtered_bytes = [
        png for png, txt in zip(page_bytes, page_texts)
        if not _is_answer_key_page(txt)
    ]
    logger.info(
        "Chunked VLM: %d total pages → %d question pages after AK filter.",
        len(page_bytes), len(filtered_bytes),
    )

    all_questions: list[dict] = []
    chunks = list(_overlapping_chunks(filtered_bytes, CHUNK_SIZE, CHUNK_OVERLAP))
    logger.info("Chunks: %d (size=%d, overlap=%d).", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)

    for chunk_idx, chunk in enumerate(chunks, start=1):
        logger.info("  Chunk %d/%d (%d pages)…", chunk_idx, len(chunks), len(chunk))
        try:
            questions = _call_vlm_for_chunk(chunk, api_key)
            logger.info("  Chunk %d → %d questions.", chunk_idx, len(questions))
            all_questions.extend(questions)
        except Exception as exc:
            logger.error("  Chunk %d failed: %s — skipping.", chunk_idx, exc)

    logger.info("VLM total before dedup: %d.", len(all_questions))
    return all_questions


def _call_vlm_for_chunk(chunk_pngs: list[bytes], api_key: str) -> list[dict]:
    """Send one chunk to Gemini; retry on API/network errors."""
    client = _get_client(api_key)               # Fix 4: reuse shared client
    model  = os.environ.get("VLM_MODEL", "gemini-2.5-flash")

    parts: list[Any] = [types.Part.from_text(text=RUBRIC_EXTRACTION_PROMPT)]
    for png in chunk_pngs:
        parts.append(types.Part.from_bytes(data=png, mime_type="image/png"))

    last_exc: Exception | None = None
    for attempt in range(API_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=types.Content(role="user", parts=parts),
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=8192,
                ),
            )
            raw = (response.text or "").strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$",          "", raw)
            return [_normalise_flat(q) for q in _parse_json_balanced(raw)]

        except Exception as exc:
            last_exc = exc
            if attempt < API_RETRIES - 1:
                wait = API_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "  Attempt %d/%d failed: %s — retrying in %ds.",
                    attempt + 1, API_RETRIES, exc, wait,
                )
                time.sleep(wait)

    raise RuntimeError(f"All {API_RETRIES} API attempts failed.") from last_exc


# ---------------------------------------------------------------------------
# JSON repair: bracket-balanced parser
# ---------------------------------------------------------------------------

def _parse_json_balanced(raw: str) -> list[dict]:
    """
    Locate the outermost [...] by bracket-walking, not by regex.
    Handles nesting and extra trailing text.
    On truncated strings: attempts to close the array and retry.
    """
    # Fast path
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    start = raw.find("[")
    if start == -1:
        raise ValueError("No JSON array found in Gemini response.")

    depth  = 0
    in_str = False
    escape = False
    end    = -1

    for i, ch in enumerate(raw[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        logger.warning("Truncated JSON — attempting to close array.")
        candidate = raw[start:] + "]"
    else:
        candidate = raw[start : end + 1]

    # Strip non-printable control chars (leave \n and space)
    candidate = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", candidate)

    try:
        result = json.loads(candidate)
        if isinstance(result, list):
            logger.info("Bracket-balanced repair: %d items.", len(result))
            return result
    except json.JSONDecodeError as e:
        raise ValueError(f"Bracket-balanced repair failed: {e}") from e

    raise ValueError("Repair produced non-list result.")


# ---------------------------------------------------------------------------
# Fix 2: Best-extraction deduplication (richer entry wins)
# ---------------------------------------------------------------------------

def _extraction_score(q: dict) -> int:
    """
    Score an extracted entry by richness.
    Higher = more complete extraction.
    Used to pick the best version when a question appears in multiple chunks.
    """
    return (
        len(str(q.get("question_text", "")))      # longer text is better
        + len(q.get("steps",    [])) * 10         # steps matter most
        + len(q.get("keywords", [])) * 5
    )


def _deduplicate_best(flat: list[dict]) -> list[dict]:
    """
    Fix 2: When a question_number appears multiple times (from overlapping chunks),
    keep the entry with the highest _extraction_score instead of first-seen.
    """
    best: dict[str, dict] = {}
    for q in flat:
        qnum = str(q.get("question_number", "")).strip()
        if not qnum:
            continue
        if qnum not in best or _extraction_score(q) > _extraction_score(best[qnum]):
            best[qnum] = q

    deduped = sorted(best.values(), key=lambda q: _safe_qnum(q))
    logger.info("After best-dedup: %d questions.", len(deduped))
    return deduped


def _safe_qnum(q: dict) -> int:
    raw = str(q.get("question_number", "999"))
    m   = re.match(r"(\d+)", raw)
    return int(m.group(1)) if m else 999


# ---------------------------------------------------------------------------
# Fix 3: Semantic + structural validation with math-operator fallback
# ---------------------------------------------------------------------------

def _validate_questions(flat: list[dict]) -> list[dict]:
    valid   : list[dict] = []
    dropped : int        = 0

    for q in flat:
        qnum  = str(q.get("question_number", "")).strip()
        sec   = str(q.get("section",         "")).strip().upper()
        marks = int(q.get("marks",           0))
        qtext = str(q.get("question_text",   "")).strip()
        qtype = str(q.get("question_type",   "")).strip().lower()
        qlow  = qtext.lower()

        def drop(reason: str) -> None:
            nonlocal dropped
            logger.debug("Dropped Q%s — %s.", qnum, reason)
            dropped += 1

        # Structural
        num_match = re.match(r"^(\d+)", qnum)
        if not num_match or int(num_match.group(1)) > 50:
            drop("bad question number"); continue
        if sec not in _VALID_SECTIONS:
            drop(f"bad section {sec!r}"); continue
        if marks <= 0:
            drop(f"marks={marks}"); continue
        if qtype not in {"mcq", "short", "long", "descriptive"}:
            drop(f"unknown type {qtype!r}"); continue

        # Semantic (non-MCQ only)
        if qtype != "mcq":
            if len(qtext) < 8:
                drop("text too short"); continue
            if any(bad in qlow for bad in BAD_QTEXT_PATTERNS):
                drop("metadata text"); continue
            # Fix 3: accept if math operators present OR hint word present
            has_hint = any(h in qlow for h in QUESTION_HINTS)
            has_math = bool(_MATH_OP_RE.search(qtext))
            if not has_hint and not has_math:
                drop("no question-hint word or math operator"); continue

        valid.append(q)

    if dropped:
        logger.warning("Validation dropped %d entries.", dropped)
    logger.info("After validation: %d questions.", len(valid))
    return valid


# ---------------------------------------------------------------------------
# Completeness check
# ---------------------------------------------------------------------------

def _count_expected_questions(qp_pdf_path: Path) -> int:
    found: set[str] = set()
    pat = re.compile(r'^\s*Q?(\d{1,2}[a-z]?)[.)]\s', re.IGNORECASE | re.MULTILINE)
    with pdfplumber.open(str(qp_pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if _is_answer_key_page(text):
                continue
            for m in pat.finditer(text):
                found.add(m.group(1).strip())
    return len(found)


def _check_completeness(flat: list[dict], qp_pdf_path: Path) -> None:
    expected = _count_expected_questions(qp_pdf_path)
    if expected == 0:
        logger.warning("Completeness: could not estimate expected question count.")
        return
    ratio = len(flat) / expected
    if ratio < COMPLETENESS_RATIO:
        logger.warning(
            "Completeness WARNING: extracted %d / estimated %d (%.0f%% < %.0f%%). "
            "Re-run with --regen-rubric or decrease CHUNK_OVERLAP.",
            len(flat), expected, ratio * 100, COMPLETENESS_RATIO * 100,
        )
    else:
        logger.info("Completeness OK: %d/%d (%.0f%%).", len(flat), expected, ratio * 100)


# ---------------------------------------------------------------------------
# MCQ extraction — marking-scheme pages only
# ---------------------------------------------------------------------------

_MCQ_PATTERNS = [
    re.compile(r'(\d+)\.\s*\(([A-D])\)',                         re.IGNORECASE),
    re.compile(r'(\d+)\)\s*\(([A-D])\)',                         re.IGNORECASE),
    re.compile(r'(\d+)[\.\)]\s*[Aa]ns(?:wer)?[:\s]+([A-D])\b', re.IGNORECASE),
    re.compile(r'^\s*(\d+)\.\s+([A-D])\s*$',                    re.MULTILINE),
]

def _extract_mcq_answers_locally(qp_pdf_path: Path) -> dict[str, str]:
    page_texts = _extract_page_texts(qp_pdf_path)
    scheme_pages = [t for t in page_texts if _is_answer_key_page(t)]
    scan_text = "\n".join(scheme_pages) if scheme_pages else "\n".join(page_texts)

    if scheme_pages:
        logger.info("MCQ extraction: %d AK page(s).", len(scheme_pages))
    else:
        logger.warning("MCQ extraction: no AK page found — scanning full PDF.")

    answer_key: dict[str, str] = {}
    for pattern in _MCQ_PATTERNS:
        for qnum, ans in pattern.findall(scan_text):
            ans = ans.upper()
            if qnum in answer_key:
                if answer_key[qnum] != ans:
                    logger.warning(
                        "MCQ conflict Q%s: have %s, found %s — keeping %s.",
                        qnum, answer_key[qnum], ans, answer_key[qnum],
                    )
            else:
                answer_key[qnum] = ans
    return answer_key


# ---------------------------------------------------------------------------
# Page index — skips AK pages
# ---------------------------------------------------------------------------

_QNUM_RE = re.compile(r'^\s*Q?(\d{1,2}[a-z]?)[.)]\s', re.IGNORECASE | re.MULTILINE)

def _build_page_index(qp_pdf_path: Path) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    with pdfplumber.open(str(qp_pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if _is_answer_key_page(text):
                continue
            for m in _QNUM_RE.finditer(text):
                qnum = m.group(1).strip()
                index.setdefault(qnum, [])
                if page_num not in index[qnum]:
                    index[qnum].append(page_num)
    return index


# ---------------------------------------------------------------------------
# Regex / pdfplumber fallback
# ---------------------------------------------------------------------------

_SECTION_RE  = re.compile(r"\bSECTION\s*[-:]?\s*([ABCD])\b", re.IGNORECASE)
_MCQ_OPT_RE  = re.compile(r"^\s*\(?\s*([A-D])\s*\)?\s*\.?\s+", re.IGNORECASE)
_MARKS_RE    = re.compile(r"\[(\d+)\s*[Mm]arks?\]|\((\d+)\s*[Mm]arks?\)")
_QUESTION_RE = re.compile(r"^\s*Q?\.?\s*(\d+[a-z]?)\s*[.)]\s+", re.IGNORECASE)

_BAD_LINE_PATTERNS = [
    "weightage", "difficulty", "all questions are compulsory",
    "forms of questions", "scheme of options", "total marks",
    "time allowed", "general instructions", "internal choice",
    "marks allotted", "blueprint",
]

def _is_bad_line(line: str) -> bool:
    return any(b in line.lower() for b in _BAD_LINE_PATTERNS)


def _generate_via_regex(qp_pdf_path: Path) -> list[dict]:
    questions:       list[dict] = []
    current_section: list[str]  = ["A"]
    with pdfplumber.open(str(qp_pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if _is_answer_key_page(text):
                continue
            _process_lines(text.splitlines(), questions, current_section)
    if not questions:
        logger.warning("Regex fallback produced no questions.")
    return questions


def _process_lines(lines: list[str], questions: list[dict], sec_ref: list[str]) -> None:
    for i, line in enumerate(lines):
        m = _SECTION_RE.search(line)
        if m:
            sec_ref[0] = m.group(1).upper()
        if _is_bad_line(line):
            continue
        q_match = _QUESTION_RE.match(line)
        if not q_match:
            continue
        rest = line[q_match.end():].lower()
        # Fix 3: require hint word OR math operator in the line text
        if len(rest) < 5:
            continue
        if not any(h in rest for h in QUESTION_HINTS) and not _MATH_OP_RE.search(rest):
            continue
        qnum  = q_match.group(1)
        marks = _extract_marks(line)
        opts  = [
            lines[j].strip()
            for j in range(i + 1, min(i + 6, len(lines)))
            if _MCQ_OPT_RE.match(lines[j])
        ]
        questions.append({
            "question_number": qnum,
            "section":         sec_ref[0],
            "question_type":   "mcq" if len(opts) >= 2 else _guess_type(marks),
            "marks":           marks,
            "question_text":   line.strip(),
            "correct_answer":  "",
            "keywords":        [],
            "steps":           [],
        })


def _extract_marks(line: str) -> int:
    m = _MARKS_RE.search(line)
    return int(m.group(1) or m.group(2)) if m else 1


def _guess_type(marks: int) -> str:
    if marks <= 2: return "short"
    if marks <= 5: return "long"
    return "descriptive"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalise_flat(q: dict) -> dict:
    return {
        "question_number": str(q.get("question_number", "")).strip(),
        "section":         str(q.get("section",         "A")).strip().upper(),
        "question_type":   str(q.get("question_type",   "short")).strip().lower(),
        "marks":           int(q.get("marks",           1)),
        "question_text":   str(q.get("question_text",   "")),
        "correct_answer":  str(q.get("correct_answer",  "")).strip().upper(),
        "keywords":        list(q.get("keywords",        [])),
        "steps":           list(q.get("steps",           [])),
    }


def _split_and_normalise(flat: list[dict]) -> dict:
    mcq:     list[dict] = []
    written: list[dict] = []
    for q in flat:
        base = {
            "question_number": q["question_number"],
            "section":         q["section"],
            "max_marks":       q["marks"],
            "keywords":        q["keywords"],
        }
        if q["question_type"] == "mcq":
            mcq.append({**base, "correct_answer": q["correct_answer"]})
        else:
            written.append({
                **base,
                "question_text":  q["question_text"],
                "expected_steps": q["steps"],
            })
    return {"mcq_questions": mcq, "written_questions": written}


def _pdf_to_png_bytes(pdf_path: Path, dpi: int = 150) -> list[bytes]:
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    out = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        out.append(pix.tobytes("png"))
    doc.close()
    return out


def _extract_page_texts(pdf_path: Path) -> list[str]:
    texts: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return texts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python rubric_generator.py <question_paper.pdf>")
        sys.exit(1)

    result = generate_rubric(sys.argv[1])
    print(json.dumps(result, indent=2))

    output_path = Path("schemas/rubric.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved rubric to: {output_path.resolve()}")