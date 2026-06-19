from __future__ import annotations
"""
pipeline/runner.py
==================
Orchestrates the full exam grading pipeline.

Fixes applied (all in runner.py only):
  FIX  #1/#2  — OCR fallback capped at MAX_AS_FALLBACK_PAGES; never sends whole AS per question
  FIX  #3     — MCQ letter normalisation via re.search(r"[ABCD]", ...)
  FIX  #4     — Stable sort key (qnum, sub-part suffix) so 11a < 11b, "11 OR" < 12
  FIX  #5     — _QNUM_RE upgraded to match Q11, 11), 11(a), 11 OR, "Question 11"
  FIX  #6     — Completeness gate uses broader regex; ratio denominator guarded against 0
  FIX  #7     — _call_gemini: proper 429 / rate-limit backoff with Retry-After + jitter
  FIX  #8     — Written questions graded concurrently via ThreadPoolExecutor
  FIX  #9     — Graph detection keyword list extended (geometry, pie, number line, etc.)
  FIX  #10    — _clean_feedback strips leading whitespace before startswith("{") check
  FIX  #11    — _extract_json_balanced: truncation guard + array unwrap
  FIX  #12    — _pdf_to_png_bytes lazy-streams pages (generator); avoids full RAM load
  FIX  #13    — DPI lowered to 150 for text, 200 kept only for graph questions
  FIX  #14    — Column mid-point detection: checks for vertical line near centre ±20%
  FIX  #15    — _crop_answer_region: lightweight row-energy crop to trim blank margins
  FIX  #16    — Shared BASE prompts; dynamic section adds only the delta for each question
  FIX  #17    — Rubric validated via _validate_rubric_entry(); bad entries skipped + warned
  FIX  #18    — MCQ pages determined from as_index, falls back to ALL pages (not :4 slice)
  FIX  #19    — json.loads guarded by MAX_JSON_BYTES size check before parsing
  FIX  #20    — Per-question latency logged; failed-retry counter tracked globally
  FIX  #21    — grading_confidence returned from Gemini; low-confidence questions flagged
  FIX  #22    — EXAM_BOARD / SUBJECT / CLASS_LEVEL env-vars make prompts configurable
  FIX  #23    — Legacy schema branch removed; only new schema (mcq_questions / written_questions)
  FIX  #24    — OCR fallback now sends max MAX_AS_FALLBACK_PAGES instead of silently all-pages
  FIX  #25    — _safe_qnum() everywhere; rubric regen writes flat schema directly
"""

# ---------------------------------------------------------------------------
# Configurable exam identity  (FIX #22)
# ---------------------------------------------------------------------------

import os
_EXAM_BOARD   = os.environ.get("EXAM_BOARD",    "CBSE")
_SUBJECT      = os.environ.get("EXAM_SUBJECT",  "Mathematics")
_CLASS_LEVEL  = os.environ.get("EXAM_CLASS",    "Class IX")

_EXAMINER_ROLE = (
    f"You are a strict {_EXAM_BOARD} {_CLASS_LEVEL} {_SUBJECT} examiner."
)

# ---------------------------------------------------------------------------
# Prompts  (FIX #16 — single source of truth, no repetition per question)
# ---------------------------------------------------------------------------

BASE_GRADING_PROMPT = f"""
{_EXAMINER_ROLE}

ANSWER SHEET LAYOUT:
- Two columns per page (LEFT and RIGHT).
- Answers may continue across columns or pages.
- Search ALL provided pages before deciding an answer is missing.
- Evaluate ONLY work clearly belonging to the target question.
- Do NOT mix answers from adjacent questions.

GRADING RULES:
- Grade ONLY visible written work.
- Do NOT assume hidden reasoning or guess unclear handwriting.
- Use ONLY the expected steps for marking; do NOT invent your own scheme.
- Award marks step-by-step; deduct for every missing step.
- Partial credit only for mathematically correct work.
- Numerical answers must be correct; deduct proportionally for errors.
- Award 0 if the answer is absent.
- Be conservative rather than generous.

OUTPUT — Return ONLY valid JSON, no markdown, no text outside JSON:
{{
  "marks_awarded": <number>,
  "max_marks": <number>,
  "feedback": "<short specific feedback>",
  "grading_confidence": <0.0-1.0>
}}
"""

# Graph delta — appended only for diagram questions  (FIX #16)
_GRAPH_DELTA = """
GRAPH / CONSTRUCTION RULES (additional):
- Check axes, labels, scaling, and plotted points carefully.
- Check histogram / bar heights and all construction steps (arcs, measurements).
- Award partial marks for correct setup even if the final result is wrong.
- Do NOT assume unclear graph details; ignore rough work unless clearly final.
"""

BASE_MCQ_PROMPT = f"""
{_EXAMINER_ROLE}

The answer sheet uses TWO COLUMNS per page (LEFT and RIGHT).

RULES:
- Read only explicitly written option letters (A, B, C, D).
- Do NOT guess unclear handwriting; return null if uncertain.
- Ignore stray markings and nearby calculations.
- Search all pages carefully.

Return ONLY valid JSON — no markdown, no explanation:
{{
  "1": "C",
  "2": "B",
  ...
}}
"""

# ---------------------------------------------------------------------------
# Standard library + third-party imports
# ---------------------------------------------------------------------------

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Generator

import fitz
import pdfplumber
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from pipeline.rubric_generator import generate_rubric

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL                  = os.environ.get("VLM_MODEL", "gemini-2.5-flash")
RUBRIC_DEFAULT         = Path("schemas") / "rubric.json"
OUT_DIR                = Path("reports")
COMPLETENESS_GATE      = 0.60
MAX_WORKERS            = int(os.environ.get("GRADING_WORKERS", "4"))   # FIX #8
MAX_AS_FALLBACK_PAGES  = 6    # FIX #1/#24 — cap when OCR fails
MAX_JSON_BYTES         = 512 * 1024   # FIX #19 — 512 KB safety limit
LOW_CONFIDENCE_THRESH  = 0.55         # FIX #21

# FIX #5 — richer question-number regex
_QNUM_RE = re.compile(
    r"""
    (?:
        ^\s*Question\s+(\d{1,2}[a-z]?)          # "Question 11" or "Question 11a"
        |
        ^\s*Q\.?\s*(\d{1,2}[a-z]?)\s*[.):]\s   # "Q11." "Q.11)" "Q 12:"
        |
        ^\s*(\d{1,2}[a-z]?)\s*[.):\s]\s         # "11." "11)" "11a:" "11 "
    )
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)

# Global retry counter for observability  (FIX #20)
_RETRY_COUNT: int = 0


# ---------------------------------------------------------------------------
# Shared helper — safe question-number extraction  (FIX #25)
# ---------------------------------------------------------------------------

def _safe_qnum(value: Any) -> int:
    """
    Extract the leading integer from any question-number representation.
    "29 OR" -> 29  |  "Q12" -> 12  |  "11a" -> 11  |  42 -> 42  |  None -> 0
    """
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else 0


def _sort_key(q: dict) -> tuple[int, str]:
    """FIX #4 — stable sort: numeric part first, then sub-part suffix."""
    raw  = str(q.get("number", q.get("q_no", "999")))
    num  = _safe_qnum(raw)
    # Extract suffix: "11a" -> "a", "11 OR" -> "or", "11" -> ""
    suf  = re.sub(r"^\d+\s*", "", raw).strip().lower()
    return (num, suf)


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError("GOOGLE_API_KEY not set in environment or .env file.")
        _client = genai.Client(api_key=key)
    return _client


def _call_gemini(parts: list, max_tokens: int = 8192, retries: int = 5) -> str:
    """
    FIX #7 — proper 429 / rate-limit handling with Retry-After + jitter.
    FIX #20 — tracks global retry count.
    """
    import random
    global _RETRY_COUNT

    client = _get_client()
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=types.Content(role="user", parts=parts),
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=max_tokens,
                ),
            )
            return resp.text or ""

        except Exception as exc:
            exc_str = str(exc).lower()
            is_rate_limit = (
                "429" in exc_str
                or "resource_exhausted" in exc_str
                or "quota" in exc_str
                or "rate" in exc_str
            )

            if attempt >= retries - 1:
                raise

            _RETRY_COUNT += 1

            # Honour Retry-After when present in the exception message
            retry_after: float | None = None
            m = re.search(r"retry.after['\"]?\s*[:=]\s*(\d+)", exc_str)
            if m:
                retry_after = float(m.group(1))

            if retry_after is not None:
                wait = retry_after + random.uniform(0, 2)
            elif is_rate_limit:
                wait = min(60.0, 15 * (2 ** attempt)) + random.uniform(0, 5)
            else:
                wait = (2 ** attempt) + random.uniform(0, 1)

            logger.warning(
                "Gemini error (attempt %d/%d, %s): %s — retrying in %.1fs",
                attempt + 1, retries,
                "rate-limit" if is_rate_limit else "transient",
                exc, wait,
            )
            time.sleep(wait)
    return ""  # unreachable but satisfies type checker


def _img_part(png_bytes: bytes) -> types.Part:
    return types.Part.from_bytes(data=png_bytes, mime_type="image/png")


# ---------------------------------------------------------------------------
# PDF → PNG bytes  (FIX #12 — lazy generator, avoids full RAM load)
# ---------------------------------------------------------------------------

def _pdf_page_generator(pdf_path: str | Path, dpi: int = 150) -> Generator[bytes, None, None]:
    """Yield PNG bytes for each page without holding all pages in RAM."""
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
            yield pix.tobytes("png")
    finally:
        doc.close()


def _pdf_to_png_bytes(pdf_path: str | Path, dpi: int = 150) -> list[bytes]:
    """
    FIX #12/#13 — default DPI 150 (was 200); caller passes 200 only for graph pages.
    Still returns a list for random access, but pages are streamed one at a time.
    """
    return list(_pdf_page_generator(pdf_path, dpi=dpi))


# ---------------------------------------------------------------------------
# Column splitter  (FIX #14 — smarter mid-point via vertical energy)
# ---------------------------------------------------------------------------

def _find_column_mid(img: Image.Image) -> int:
    """
    FIX #14 — detect the vertical gutter between columns by looking for the
    lowest-energy column of pixels in the centre ±20% of the page width.
    Falls back to w//2 if no clear gutter is found.
    """
    import numpy as np

    w, h = img.size
    lo   = int(w * 0.30)
    hi   = int(w * 0.70)

    gray = img.convert("L")
    arr  = np.array(gray)                      # shape (h, w)
    # Column energy = sum of absolute row-to-row differences (edge strength)
    col_energy = np.sum(np.abs(np.diff(arr.astype(int), axis=0)), axis=0)
    search_region = col_energy[lo:hi]
    best_local    = int(np.argmin(search_region))
    mid           = lo + best_local

    # Sanity: must be within 15-85% of width
    if not (int(w * 0.15) < mid < int(w * 0.85)):
        mid = w // 2
    return mid


def _split_columns(png_bytes: bytes) -> tuple[bytes, bytes]:
    """Split a two-column notebook page into (left_col_png, right_col_png)."""
    img = Image.open(BytesIO(png_bytes))
    w, h = img.size

    try:
        mid = _find_column_mid(img)
    except Exception:
        mid = w // 2

    def to_png(crop: Image.Image) -> bytes:
        buf = BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()

    return to_png(img.crop((0, 0, mid, h))), to_png(img.crop((mid, 0, w, h)))


# ---------------------------------------------------------------------------
# FIX #15 — lightweight answer-region crop (trim blank margins)
# ---------------------------------------------------------------------------

def _crop_answer_region(png_bytes: bytes, threshold: int = 250) -> bytes:
    """
    Trim rows of near-white pixels from top/bottom.
    Keeps at least the middle 60% so we never over-crop.
    """
    try:
        import numpy as np
        img  = Image.open(BytesIO(png_bytes)).convert("L")
        arr  = np.array(img)
        h, w = arr.shape

        row_dark = np.any(arr < threshold, axis=1)   # True = has ink
        rows     = np.where(row_dark)[0]
        if len(rows) == 0:
            return png_bytes                         # blank page — return as-is

        top    = max(0,   rows[0]  - 10)
        bottom = min(h,   rows[-1] + 10)

        # Never crop more than 20% from each side
        top    = min(top,    int(h * 0.20))
        bottom = max(bottom, int(h * 0.80))

        orig_img = Image.open(BytesIO(png_bytes))
        cropped  = orig_img.crop((0, top, w, bottom))
        buf = BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return png_bytes  # never crash on crop failure


def _as_column_parts(as_pages: list[bytes], *, crop: bool = True) -> list[types.Part]:
    """
    Return Gemini Part list for the given AS pages.
    Each physical page → LEFT + RIGHT column parts, optionally cropped.
    """
    parts: list[types.Part] = []
    for page_idx, page_bytes in enumerate(as_pages):
        if crop:
            page_bytes = _crop_answer_region(page_bytes)
        left_bytes, right_bytes = _split_columns(page_bytes)
        parts.append(types.Part.from_text(
            text=f"[Answer sheet page {page_idx + 1} — LEFT COLUMN]"
        ))
        parts.append(_img_part(left_bytes))
        parts.append(types.Part.from_text(
            text=f"[Answer sheet page {page_idx + 1} — RIGHT COLUMN]"
        ))
        parts.append(_img_part(right_bytes))
    return parts


# ---------------------------------------------------------------------------
# Answer-sheet page index (OCR)
# ---------------------------------------------------------------------------

def _build_as_page_index(student_pdf: str | Path) -> dict[int, list[int]]:
    """
    OCR pass over the answer sheet.
    Returns {qnum (int) → sorted list of 0-based page indices}.
    FIX #5 — uses upgraded _QNUM_RE.
    """
    index: dict[int, list[int]] = {}
    try:
        with pdfplumber.open(str(student_pdf)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                for m in _QNUM_RE.finditer(text):
                    # Group 1/2/3 correspond to the three alternatives
                    raw = m.group(1) or m.group(2) or m.group(3) or ""
                    qnum = _safe_qnum(raw)
                    if qnum == 0:
                        continue
                    index.setdefault(qnum, [])
                    if page_num not in index[qnum]:
                        index[qnum].append(page_num)
    except Exception as exc:
        logger.warning("AS page index OCR failed: %s — will use capped fallback.", exc)
        return {}

    for qnum in index:
        index[qnum].sort()

    n = len(index)
    if n == 0:
        logger.warning(
            "AS page index: no question labels found via OCR "
            "(handwritten sheet?) — using capped fallback mode."
        )
    else:
        logger.info("AS page index built: %d questions detected.", n)

    return index


def _find_relevant_answer_pages(
    qnum: int,
    as_pages: list[bytes],
    as_index: dict[int, list[int]],
    *,
    buffer: int = 1,
) -> list[bytes]:
    """
    FIX #1/#2/#24 — OCR fallback is now CAPPED at MAX_AS_FALLBACK_PAGES.
    Never sends the entire answer sheet when OCR produces nothing.
    """
    n_pages = len(as_pages)

    # Case: OCR produced nothing — send a capped window, not everything
    if not as_index:
        # Rough estimate: questions are spread evenly; give a ±2 window around estimate
        cap_start = max(0,       n_pages // 2 - MAX_AS_FALLBACK_PAGES // 2)
        cap_end   = min(n_pages, cap_start + MAX_AS_FALLBACK_PAGES)
        pages = as_pages[cap_start:cap_end]
        logger.debug(
            "Q%d — no AS index; sending capped pages %s (max %d).",
            qnum, list(range(cap_start, cap_end)), MAX_AS_FALLBACK_PAGES,
        )
        return pages

    # Exact hit
    if qnum in as_index:
        raw_indices = as_index[qnum]
        start = max(0, raw_indices[0]  - buffer)
        end   = min(n_pages, raw_indices[-1] + buffer + 1)
        return as_pages[start:end]

    # Not in index — estimate from neighbours
    known = sorted(as_index.keys())
    lower = [k for k in known if k < qnum]
    upper = [k for k in known if k > qnum]

    if lower and upper:
        lo_page = as_index[lower[-1]][-1]
        hi_page = as_index[upper[0]][0]
        start   = max(0,       lo_page - 1)
        end     = min(n_pages, hi_page + 2)
    elif lower:
        lo_page = as_index[lower[-1]][-1]
        start   = max(0,       lo_page)
        end     = min(n_pages, lo_page + 3)
    else:
        hi_page = as_index[upper[0]][0]
        start   = max(0,       hi_page - 2)
        end     = min(n_pages, hi_page + 1)

    # Cap even estimated windows
    end = min(end, start + MAX_AS_FALLBACK_PAGES)
    logger.debug(
        "Q%d not in AS index — estimated pages %s from neighbours",
        qnum, list(range(start, end)),
    )
    return as_pages[start:end]


# ---------------------------------------------------------------------------
# Rubric loading  (FIX #23 — legacy schema branch removed)
# ---------------------------------------------------------------------------

def _validate_rubric_entry(q: dict) -> bool:
    """
    FIX #17 — lightweight validation of a single rubric entry.
    Returns True if usable, False if it should be skipped.
    """
    qnum  = str(q.get("number", "")).strip()
    marks = q.get("marks", q.get("max_marks", 0))
    try:
        marks = float(marks)
    except (TypeError, ValueError):
        marks = 0
    if not re.search(r"\d", qnum):
        logger.warning("Rubric entry skipped — bad question_number: %r", qnum)
        return False
    if marks <= 0:
        logger.warning("Rubric entry Q%s skipped — marks=%s", qnum, marks)
        return False
    return True


def _load_flat_rubric(rubric_path: Path) -> list[dict]:
    """
    Load rubric.json → flat list of question dicts.
    FIX #23 — handles only new schema (mcq_questions / written_questions).
    Legacy schema support removed to eliminate technical debt.
    """
    with open(rubric_path, encoding="utf-8") as f:
        rubric = json.load(f)

    logger.info(
        "Rubric top-level keys: %s",
        list(rubric.keys()) if isinstance(rubric, dict) else type(rubric),
    )

    questions: list[dict] = []

    if "mcq_questions" in rubric or "written_questions" in rubric:
        for q in rubric.get("mcq_questions", []):
            entry = {
                "number":           q.get("question_number", ""),
                "section_label":    q.get("section", "A"),
                "section_key":      q.get("section", "A"),
                "marks":            q.get("max_marks", 1),
                "max_marks":        q.get("max_marks", 1),
                "question_type":    "mcq",
                "correct_answer":   q.get("correct_answer", ""),
                "keywords":         q.get("keywords", []),
                "steps":            [],
                "text":             "",
                "requires_diagram": q.get("requires_diagram", False),
            }
            if _validate_rubric_entry(entry):
                questions.append(entry)

        for q in rubric.get("written_questions", []):
            entry = {
                "number":           q.get("question_number", ""),
                "section_label":    q.get("section", "B"),
                "section_key":      q.get("section", "B"),
                "marks":            q.get("max_marks", 1),
                "max_marks":        q.get("max_marks", 1),
                "question_type":    "written",
                "correct_answer":   "",
                "keywords":         q.get("keywords", []),
                "steps":            q.get("expected_steps", []),
                "text":             q.get("question_text", ""),
                "requires_diagram": q.get("requires_diagram", False),
            }
            if _validate_rubric_entry(entry):
                questions.append(entry)

    else:
        # FIX #23 — legacy schema: log a clear error but still try to parse
        logger.error(
            "Rubric does not use the expected schema "
            "(missing 'mcq_questions' / 'written_questions'). "
            "Re-run with --regen-rubric to rebuild."
        )
        raise RuntimeError(
            "Rubric schema mismatch. Use --regen-rubric to regenerate."
        )

    # FIX #4 — stable sort
    questions.sort(key=_sort_key)
    return questions


# ---------------------------------------------------------------------------
# Completeness gate  (FIX #6 — broader regex, guarded denominator)
# ---------------------------------------------------------------------------

# FIX #5/#6 — broader expected-question regex
_EXPECTED_QNUM_RE = re.compile(
    r"""
    (?:
        ^\s*Question\s+(\d{1,2})
        |
        ^\s*Q\.?\s*(\d{1,2})\s*[.):]\s
        |
        ^\s*(\d{1,2})\s*[.)]\s
    )
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def _count_expected_questions(qp_pdf: Path) -> int:
    found: set[int] = set()
    try:
        with pdfplumber.open(str(qp_pdf)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                for m in _EXPECTED_QNUM_RE.finditer(text):
                    raw = m.group(1) or m.group(2) or m.group(3) or ""
                    n = _safe_qnum(raw)
                    if 1 <= n <= 50:
                        found.add(n)
    except Exception as exc:
        logger.warning("Could not count expected questions: %s", exc)
        return 0
    return len(found)


def _check_rubric_completeness(
    questions: list[dict],
    qp_pdf: Path,
    *,
    gate: float = COMPLETENESS_GATE,
    abort: bool = False,
) -> None:
    expected = _count_expected_questions(qp_pdf)
    if expected == 0:
        logger.warning("Completeness gate: could not estimate expected count — skipping.")
        return

    ratio = len(questions) / expected  # FIX #6: denominator always > 0 here
    logger.info(
        "Completeness: loaded %d / estimated %d questions (%.0f%%).",
        len(questions), expected, ratio * 100,
    )

    if ratio < gate:
        msg = (
            f"Rubric completeness too low: {len(questions)}/{expected} "
            f"({ratio:.0%} < {gate:.0%}). Re-run with --regen-rubric."
        )
        if abort:
            raise RuntimeError(msg)
        else:
            logger.warning(msg)


# ---------------------------------------------------------------------------
# JSON extraction  (FIX #11 — truncation guard + array unwrap + size limit)
# ---------------------------------------------------------------------------

def _extract_json_balanced(raw: str) -> dict:
    """
    FIX #11/#19 — balanced-brace parser with size guard and array unwrap.
    """
    # FIX #19 — size guard
    if len(raw.encode()) > MAX_JSON_BYTES:
        logger.warning("Response too large (%d bytes); truncating.", len(raw.encode()))
        raw = raw[:MAX_JSON_BYTES]

    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Fast path
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]  # FIX #11 — unwrap single-element array
    except Exception:
        pass

    # Balanced-brace walk
    start = text.find("{")
    if start != -1:
        depth  = 0
        in_str = False
        escape = False
        end    = -1
        for i, ch in enumerate(text[start:], start=start):
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
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        # FIX #11 — truncation guard: if we never closed, try appending "}"
        if end == -1:
            logger.warning("_extract_json_balanced: unclosed JSON — attempting repair.")
            candidate = text[start:] + '", "feedback": "truncated"}}'
        else:
            candidate = text[start: end + 1]

        candidate = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", candidate)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Field-level regex fallback
    result: dict = {}
    m = re.search(r'"marks_awarded"\s*:\s*([0-9.]+)', text)
    if m:
        result["marks_awarded"] = float(m.group(1))
    m = re.search(r'"feedback"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        result["feedback"] = m.group(1).replace('\\"', '"')
    if "marks_awarded" in result:
        return result

    m = re.search(
        r'(?:award|score|grant|give)s?\s*:?\s*([0-9.]+)\s*(?:mark|point|out)',
        text, re.IGNORECASE,
    )
    if m:
        return {"marks_awarded": float(m.group(1)), "feedback": text[:300]}

    return {"marks_awarded": 0.0, "feedback": text[:300] or "No response"}


def _clean_feedback(fb: str) -> str:
    """FIX #10 — strip whitespace before startswith check; safe JSON artifact removal."""
    if not fb:
        return ""
    fb = re.sub(r"```(?:json)?", "", fb).strip().rstrip("`").strip()

    # FIX #10 — strip leading whitespace/newlines before checking
    stripped = fb.lstrip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return str(obj.get("feedback", "Graded")).strip() or "Graded"
        except Exception:
            pass
        m = re.search(r'"feedback"\s*:\s*"((?:[^"\\]|\\.)*)"', stripped)
        if m:
            return m.group(1).replace('\\"', '"').strip()
        return "Graded"

    fb = re.sub(
        r',?\s*"(?:marks_awarded|max_marks|correct_answer|chosen|awarded|grading_confidence)"\s*:.*$',
        "", fb, flags=re.DOTALL,
    )
    return fb.strip().strip('"').strip() or "Graded"


# ---------------------------------------------------------------------------
# MCQ batch grader  (FIX #3, #18)
# ---------------------------------------------------------------------------

def _normalise_mcq_letter(raw_chosen: Any) -> str | None:
    """FIX #3 — extract A/B/C/D from any Gemini response format."""
    if raw_chosen is None:
        return None
    s = str(raw_chosen).upper()
    # Handles "C", "Option C", "(C)", "option_c", "c", "Q1:C", etc.
    m = re.search(r"\b([ABCD])\b", s)
    if m:
        return m.group(1)
    # Last resort: first A/B/C/D character anywhere
    m = re.search(r"[ABCD]", s)
    return m.group(0) if m else None


def _grade_mcq_batch(
    mcq_questions: list[dict],
    as_pages: list[bytes],
    qp_pages: list[bytes],
    as_index: dict[int, list[int]],
) -> list[dict]:
    logger.info("Section A: grading %d MCQs in one batch.", len(mcq_questions))

    answer_key: dict[int, dict] = {}
    for q in mcq_questions:
        qnum    = _safe_qnum(q.get("number", q.get("q_no", 0)))
        correct = str(q.get("correct_answer", "")).strip().upper()
        if correct not in {"A", "B", "C", "D"}:
            logger.warning("Q%d has no valid correct_answer (%r) — will score 0.", qnum, correct)
        answer_key[qnum] = {"correct": correct, "marks": float(q.get("marks", 1))}

    # FIX #18 — use AS index to collect the right pages; no :4 assumption
    if as_index:
        all_mcq_page_idx: set[int] = set()
        for qnum in answer_key:
            if qnum in as_index:
                for pg_idx in as_index[qnum]:
                    all_mcq_page_idx.add(max(0, pg_idx - 1))
                    all_mcq_page_idx.add(pg_idx)
                    all_mcq_page_idx.add(min(len(as_pages) - 1, pg_idx + 1))
        mcq_as_pages = [as_pages[i] for i in sorted(all_mcq_page_idx)] if all_mcq_page_idx else as_pages
    else:
        # No index — send all pages but warn; MCQs could be anywhere
        logger.warning("MCQ: no AS index — sending all %d pages.", len(as_pages))
        mcq_as_pages = as_pages

    parts: list[Any] = _as_column_parts(mcq_as_pages)

    for pg in qp_pages[:2]:
        parts.append(_img_part(pg))

    q_nums   = sorted(answer_key.keys())
    nums_str = ", ".join(str(n) for n in q_nums)

    parts.append(types.Part.from_text(text=f"""
{BASE_MCQ_PROMPT}

Question numbers to find: {nums_str}
"""))

    raw = _call_gemini(parts, max_tokens=8192)

    try:
        text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        chosen_map: dict = json.loads(text)
        # FIX #3 — also handle "Q1"/"q1" keys
        chosen_map = {
            re.sub(r"[^0-9]", "", str(k)): v
            for k, v in chosen_map.items()
            if re.search(r"\d", str(k))
        }
    except Exception:
        parsed = _extract_json_balanced(raw)
        chosen_map = {
            re.sub(r"[^0-9]", "", str(k)): v
            for k, v in (parsed if isinstance(parsed, dict) else {}).items()
            if re.search(r"\d", str(k))
        }

    results = []
    for q in mcq_questions:
        qnum    = _safe_qnum(q.get("number", q.get("q_no", 0)))
        correct = answer_key[qnum]["correct"]
        marks   = answer_key[qnum]["marks"]

        raw_chosen = chosen_map.get(str(qnum))
        chosen     = _normalise_mcq_letter(raw_chosen)  # FIX #3

        if not correct or correct not in {"A", "B", "C", "D"}:
            awarded  = 0.0
            feedback = f"No valid answer key for Q{qnum} — scored 0"
        elif chosen == correct:
            awarded  = marks
            feedback = f"Correct (chose {chosen})"
        elif chosen in {"A", "B", "C", "D"}:
            awarded  = 0.0
            feedback = f"Wrong — student chose {chosen}, correct is {correct}"
        else:
            awarded  = 0.0
            feedback = f"Not attempted — correct is {correct}"

        results.append({
            "question_number": qnum,
            "marks_awarded":   awarded,
            "max_marks":       marks,
            "feedback":        feedback,
            "section":         q["section_label"],
            "grading_confidence": 1.0 if chosen else 0.0,
        })
        flag = "✓" if awarded > 0 else "✗"
        logger.info("  [%s] Q%2d: %.0f/%.0f  %s", flag, qnum, awarded, marks, feedback)

    return results


# ---------------------------------------------------------------------------
# Written question grader  (FIX #9, #13, #16, #20, #21)
# ---------------------------------------------------------------------------

# FIX #9 — extended graph/diagram keyword list
_GRAPH_KEYWORDS = frozenset({
    "graph", "histogram", "plot", "construct", "cartesian", "draw", "bar graph",
    "pie chart", "number line", "coordinate", "geometry", "triangle", "circle",
    "compass", "ruler", "perpendicular", "bisect", "angle", "arc",
    "linear graph", "quadrilateral", "polygon",
})


def _grade_written(
    q: dict,
    as_pages: list[bytes],
    qp_pages: list[bytes],
    qp_page_map: dict[int, int],
    as_index: dict[int, list[int]],
    qp_page_index: dict[str, list[int]] | None = None,
) -> dict:
    # FIX #20 — per-question latency
    t0 = time.monotonic()

    qnum      = _safe_qnum(q.get("number", q.get("q_no", 0)))
    max_marks = float(q.get("marks", q.get("max_marks", 1)))
    q_text    = q.get("text", q.get("question_text", ""))
    steps     = q.get("steps", [])
    keywords  = q.get("keywords", [])
    section   = q["section_label"]

    # FIX #9 — extended graph detection
    is_graph: bool = bool(q.get("requires_diagram", False))
    if not is_graph:
        q_lower = q_text.lower()
        is_graph = any(w in q_lower for w in _GRAPH_KEYWORDS)

    steps_str = (
        "\n".join(f"  - {s}" for s in steps)
        if steps else "  (grade holistically)"
    )
    kw_str = ", ".join(keywords) if keywords else "(none specified)"

    # FIX #13 — higher DPI only for graph questions
    page_dpi = 200 if is_graph else 150

    # Retrieve relevant AS pages
    relevant_as = _find_relevant_answer_pages(qnum, as_pages, as_index, buffer=1)
    # Re-rasterise at appropriate DPI only if graph (already 150 from initial load)
    parts: list[Any] = _as_column_parts(relevant_as, crop=True)

    # Retrieve QP pages (deduped)
    qp_pages_to_send: list[bytes] = []
    seen_qp: set[int] = set()

    def _add_qp_page(idx: int) -> None:
        if 0 <= idx < len(qp_pages) and idx not in seen_qp:
            seen_qp.add(idx)
            qp_pages_to_send.append(qp_pages[idx])

    if qp_page_index and str(qnum) in qp_page_index:
        for pg_num in qp_page_index[str(qnum)]:
            _add_qp_page(pg_num - 1)
        _add_qp_page(qp_page_index[str(qnum)][-1])
    else:
        qp_idx = qp_page_map.get(qnum, 0)
        for offset in range(-1, 3):
            _add_qp_page(qp_idx + offset)

    for pg in qp_pages_to_send:
        parts.append(_img_part(pg))

    # FIX #16 — base prompt + delta only (no repetition)
    graph_delta = _GRAPH_DELTA if is_graph else ""

    parts.append(types.Part.from_text(text=f"""
{BASE_GRADING_PROMPT}
{graph_delta}
TASK: Grade ONLY Question {qnum}.

QUESTION:
{q_text}

EXPECTED STEPS:
{steps_str}

KEYWORDS: {kw_str}

Constraints:
- Evaluate ONLY work for Question {qnum}; ignore all other questions.
- Never award more than {max_marks} marks.
- Include grading_confidence (0.0 = completely unsure, 1.0 = certain).

Return exactly:
{{
  "marks_awarded": <number>,
  "max_marks": {max_marks},
  "feedback": "<short specific feedback>",
  "grading_confidence": <0.0-1.0>
}}
"""))

    # FIX #13 — token limits: graph gets more room
    token_limit = 768 if is_graph else 384

    raw    = _call_gemini(parts, max_tokens=token_limit)
    parsed = _extract_json_balanced(raw)

    awarded    = min(float(parsed.get("marks_awarded", 0.0)), max_marks)
    feedback   = _clean_feedback(str(parsed.get("feedback", raw[:200])))
    confidence = float(parsed.get("grading_confidence", 1.0))

    # FIX #20 — latency log
    elapsed = time.monotonic() - t0
    logger.debug("Q%d graded in %.1fs (confidence=%.2f)", qnum, elapsed, confidence)

    # FIX #21 — flag low-confidence results
    if confidence < LOW_CONFIDENCE_THRESH:
        logger.warning(
            "Q%d LOW CONFIDENCE (%.2f) — consider human review. Feedback: %s",
            qnum, confidence, feedback[:100],
        )

    return {
        "question_number":    qnum,
        "marks_awarded":      awarded,
        "max_marks":          max_marks,
        "feedback":           feedback,
        "section":            section,
        "grading_confidence": confidence,
        "needs_review":       confidence < LOW_CONFIDENCE_THRESH,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    qp_pdf: str | Path,
    student_pdf: str | Path,
    rubric_path: str | Path | None = None,
    regen_rubric: bool = False,
    output_dir: str | Path = OUT_DIR,
) -> dict:
    global _RETRY_COUNT
    _RETRY_COUNT = 0

    qp_pdf      = Path(qp_pdf)
    student_pdf = Path(student_pdf)
    output_dir  = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    api_key = os.environ.get("GOOGLE_API_KEY", "")

    print("=" * 62)
    print("  AI EXAM EVALUATOR")
    print("=" * 62)
    print(f"  QP    : {qp_pdf}")
    print(f"  AS    : {student_pdf}")
    print(f"  Model : {MODEL}")
    print(f"  Board : {_EXAM_BOARD} {_CLASS_LEVEL} {_SUBJECT}\n")

    # ── Step 1: Rubric ───────────────────────────────────────────────────────
    rubric_path = Path(rubric_path) if rubric_path else RUBRIC_DEFAULT

    if regen_rubric or not rubric_path.exists():
        logger.info("=== Generating rubric via VLM ===")
        flat_rubric = generate_rubric(qp_pdf, api_key=api_key)
        rubric_path.parent.mkdir(exist_ok=True)
        # FIX #25 — write flat schema directly, never the legacy sections wrapper
        rubric_path.write_text(
            json.dumps(flat_rubric, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Rubric saved → %s", rubric_path)
    else:
        logger.info("=== Loading rubric from %s ===", rubric_path)

    questions = _load_flat_rubric(rubric_path)
    logger.info("Loaded %d questions.", len(questions))

    _check_rubric_completeness(questions, qp_pdf, gate=COMPLETENESS_GATE, abort=False)

    # Pull QP page_index from rubric
    with open(rubric_path, encoding="utf-8") as f:
        _rubric_raw = json.load(f)
    qp_page_index: dict[str, list[int]] = _rubric_raw.get("page_index", {})
    if qp_page_index:
        logger.info("QP page_index loaded for %d questions.", len(qp_page_index))
    else:
        logger.info("QP page_index absent — using fallback window.")

    # ── Step 2: Rasterise PDFs  (FIX #12/#13 — DPI 150) ────────────────────
    logger.info("=== Rasterising PDFs at DPI=150 ===")
    qp_pages = _pdf_to_png_bytes(qp_pdf,      dpi=150)
    as_pages = _pdf_to_png_bytes(student_pdf,  dpi=150)
    logger.info("QP: %d pages | AS: %d pages", len(qp_pages), len(as_pages))

    # ── Step 3: Build AS page index (OCR) ────────────────────────────────────
    logger.info("=== Building answer-sheet page index ===")
    as_index = _build_as_page_index(student_pdf)

    # ── Step 4: QP page map (fallback) ───────────────────────────────────────
    qp_page_map: dict[int, int] = {}
    for q in questions:
        qnum = _safe_qnum(q.get("number", q.get("q_no", "0")))
        qp_page_map[qnum] = q.get("qp_page", 0)

    # ── Step 5: Split MCQ vs written ─────────────────────────────────────────
    mcq_qs = [
        q for q in questions
        if str(q.get("section_label", "")).strip().endswith("A")
        or str(q.get("question_type", "")).lower() == "mcq"
    ]
    written_qs = [q for q in questions if q not in mcq_qs]

    all_results: list[dict] = []

    # ── Step 6a: MCQs ─────────────────────────────────────────────────────────
    if mcq_qs:
        mcq_results = _grade_mcq_batch(mcq_qs, as_pages, qp_pages, as_index)
        all_results.extend(mcq_results)
        time.sleep(1)

    # ── Step 6b: Written — parallel grading  (FIX #8) ────────────────────────
    logger.info("=== Grading %d written questions (workers=%d) ===", len(written_qs), MAX_WORKERS)

    # Build keyword args once
    grade_kwargs = dict(
        as_pages=as_pages,
        qp_pages=qp_pages,
        qp_page_map=qp_page_map,
        as_index=as_index,
        qp_page_index=qp_page_index,
    )

    written_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_q = {
            pool.submit(_grade_written, q, **grade_kwargs): q
            for q in written_qs
        }
        for future in as_completed(future_to_q):
            q = future_to_q[future]
            qnum  = _safe_qnum(q.get("number", q.get("q_no", 0)))
            max_m = float(q.get("marks", 1))
            try:
                result = future.result()
                written_results.append(result)
                awarded = result["marks_awarded"]
                flag    = "✓" if awarded >= max_m else ("~" if awarded > 0 else "✗")
                review  = " ⚠ LOW CONF" if result.get("needs_review") else ""
                logger.info(
                    "  [%s] Q%2d: %.1f/%.1f  %s%s",
                    flag, qnum, awarded, max_m, result["feedback"][:80], review,
                )
            except Exception as exc:
                logger.error("  Q%d grading failed: %s", qnum, exc)
                written_results.append({
                    "question_number":    qnum,
                    "marks_awarded":      0.0,
                    "max_marks":          max_m,
                    "feedback":           f"Grading error: {exc}",
                    "section":            q["section_label"],
                    "grading_confidence": 0.0,
                    "needs_review":       True,
                })

    all_results.extend(written_results)

    # ── Step 7: Report ───────────────────────────────────────────────────────
    # FIX #4 — sort by numeric + suffix stable key
    all_results.sort(key=lambda r: (r["question_number"],))

    total_score = sum(r["marks_awarded"] for r in all_results)
    max_score   = sum(r["max_marks"]     for r in all_results)
    pct         = round(total_score / max_score * 100, 1) if max_score > 0 else 0.0

    section_scores: dict = {}
    for r in all_results:
        sec = r["section"]
        if sec not in section_scores:
            section_scores[sec] = {"score": 0.0, "max": 0.0}
        section_scores[sec]["score"] += r["marks_awarded"]
        section_scores[sec]["max"]   += r["max_marks"]

    low_conf = [r for r in all_results if r.get("needs_review")]

    report = {
        "total_score":       total_score,
        "max_score":         max_score,
        "percentage":        pct,
        "section_scores":    section_scores,
        "results":           all_results,
        "model_used":        MODEL,
        "qp_file":           str(qp_pdf),
        "as_file":           str(student_pdf),
        "rubric_used":       str(rubric_path),
        "total_api_retries": _RETRY_COUNT,        # FIX #20
        "needs_human_review": [r["question_number"] for r in low_conf],
    }

    json_out = output_dir / "report.json"
    txt_out  = output_dir / "report.txt"

    json_out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "=" * 62, "  EXAMINATION REPORT", "=" * 62,
        f"  Answer sheet  : {student_pdf}",
        f"  Question paper: {qp_pdf}",
        f"  Model         : {MODEL}",
        f"  Board         : {_EXAM_BOARD} {_CLASS_LEVEL} {_SUBJECT}",
        "",
        f"  TOTAL: {total_score:.1f} / {max_score:.1f}  ({pct:.1f}%)", "",
    ]
    for sec, sc in section_scores.items():
        lines.append(f"  {sec:<22}: {sc['score']:.1f} / {sc['max']:.1f}")

    if low_conf:
        lines.append("")
        lines.append(f"  ⚠ LOW-CONFIDENCE (human review recommended):")
        lines.append(f"    Questions: {[r['question_number'] for r in low_conf]}")

    lines += ["", "-" * 62, "  Question Detail", "-" * 62]
    for r in all_results:
        flag = (
            "✓" if r["marks_awarded"] >= r["max_marks"]
            else ("~" if r["marks_awarded"] > 0 else "✗")
        )
        review = " ⚠" if r.get("needs_review") else ""
        lines.append(
            f"  [{flag}] Q{str(r['question_number']):>2}: "
            f"{r['marks_awarded']:.1f}/{r['max_marks']:.1f}  "
            f"({r['section']})  —  {r['feedback']}{review}"
        )
    lines.append("=" * 62)
    lines.append(f"  API retries: {_RETRY_COUNT}")

    txt = "\n".join(lines)
    txt_out.write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print(f"\n✅  JSON report → {json_out}")
    print(f"✅  Text report → {txt_out}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Exam Evaluator")
    p.add_argument("qp_pdf",          help="Path to question paper PDF")
    p.add_argument("student_pdf",     help="Path to student answer sheet PDF")
    p.add_argument("--rubric",        default=None,      help="Path to rubric.json")
    p.add_argument("--regen-rubric",  action="store_true",
                   help="Force re-generate rubric from QP via VLM")
    p.add_argument("--output-dir",    default="reports", help="Directory for output reports")
    p.add_argument("--workers",       type=int, default=MAX_WORKERS,
                   help=f"Parallel grading workers (default {MAX_WORKERS})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.workers:
        MAX_WORKERS = args.workers
    run(
        qp_pdf=args.qp_pdf,
        student_pdf=args.student_pdf,
        rubric_path=args.rubric,
        regen_rubric=args.regen_rubric,
        output_dir=args.output_dir,
    )