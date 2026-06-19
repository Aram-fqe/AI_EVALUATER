"""
pipeline/input_handler.py
==========================
Handles:
  - PDF / PNG / ZIP → list of page dicts  (load_answer_images)
  - Rubric loading and validation          (load_rubric, _validate_rubric)
  - Rubric helper queries                  (get_question, mark_budget)
  - Question → page image mapping          (build_question_image_map)
  - QP page detection                      (build_qp_page_map, get_marking_scheme_pages)

All imports used by test_pipeline.py are exported from this module.
"""

from __future__ import annotations

import base64
import io
import json
import re
import zipfile
from pathlib import Path

import fitz          # PyMuPDF
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SECTIONS = set("ABCDEFGHIJ")
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# ---------------------------------------------------------------------------
# PDF / image loading
# ---------------------------------------------------------------------------

def load_answer_images(path: str | Path, dpi: int = 200) -> list[dict]:
    """
    Load an answer sheet from a PDF, PNG/JPG image, or ZIP of images.

    Returns a list of page dicts:
        [{"b64_image": "<base64 string>", "mime_type": "image/png"}, ...]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Answer sheet not found: {path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        return _load_pdf(path, dpi)
    elif ext in SUPPORTED_IMAGE_EXTS:
        return _load_single_image(path)
    elif ext == ".zip":
        return _load_zip(path, dpi)
    else:
        raise ValueError(f"Unsupported file type: {ext!r}. Use PDF, PNG, JPG, or ZIP.")


def _load_pdf(pdf_path: Path, dpi: int) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pages = []
    for page in doc:
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        png  = pix.tobytes("png")
        pages.append({
            "b64_image": base64.standard_b64encode(png).decode(),
            "mime_type": "image/png",
        })
    doc.close()
    return pages


def _load_single_image(img_path: Path) -> list[dict]:
    raw  = img_path.read_bytes()
    mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
    return [{"b64_image": base64.standard_b64encode(raw).decode(), "mime_type": mime}]


def _load_zip(zip_path: Path, dpi: int) -> list[dict]:
    pages = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(
            n for n in zf.namelist()
            if Path(n).suffix.lower() in SUPPORTED_IMAGE_EXTS | {".pdf"}
        )
        if not names:
            raise ValueError(f"ZIP {zip_path.name} contains no PDF or image files.")
        for name in names:
            data = zf.read(name)
            ext  = Path(name).suffix.lower()
            if ext == ".pdf":
                tmp = Path(name).name
                tmp_path = Path("/tmp") / tmp
                tmp_path.write_bytes(data)
                pages.extend(_load_pdf(tmp_path, dpi))
            else:
                mime = "image/png" if ext == ".png" else "image/jpeg"
                pages.append({
                    "b64_image": base64.standard_b64encode(data).decode(),
                    "mime_type": mime,
                })
    return pages


# ---------------------------------------------------------------------------
# Rubric loading and validation
# ---------------------------------------------------------------------------

def load_rubric(path: str | Path) -> dict:
    """
    Load and validate a rubric JSON file.
    Raises FileNotFoundError, ValueError on problems.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rubric not found: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError(f"Rubric must be a .json file, got: {path.suffix!r}")

    with open(path, encoding="utf-8") as f:
        rubric = json.load(f)

    _validate_rubric(rubric)
    return rubric


def _validate_rubric(rubric: dict) -> None:
    """
    In-place validation + auto-correction of the rubric dict.
    Raises ValueError on unrecoverable problems.
    """
    if "assessment_meta" not in rubric:
        raise ValueError("Rubric missing 'assessment_meta' key.")
    if "sections" not in rubric:
        raise ValueError("Rubric missing 'sections' key.")

    for sec_id, sec_data in rubric["sections"].items():
        if sec_id not in VALID_SECTIONS:
            raise ValueError(f"Invalid section id: {sec_id!r}. Must be A–J.")

        is_mcq = sec_data.get("question_subtype", "") == "mcq"

        # Auto-correct: MCQ sections cannot have partial credit
        if is_mcq and sec_data.get("partial_credit_allowed", False):
            sec_data["partial_credit_allowed"] = False

        for q in sec_data.get("questions", []):
            # MCQ must have correct option
            if is_mcq and "mcq_correct_option" not in q:
                raise ValueError(
                    f"Section {sec_id} Q{q.get('q_no')} is MCQ but missing 'mcq_correct_option'."
                )
            # Step marks must be multiples of 0.5
            for step in q.get("steps", []):
                m = step.get("marks", 1)
                if round(m * 2) != m * 2:
                    raise ValueError(
                        f"Step mark {m} in Q{q.get('q_no')} is not a multiple of 0.5."
                    )


# ---------------------------------------------------------------------------
# Rubric helpers
# ---------------------------------------------------------------------------

def get_question(rubric: dict, q_no: str) -> tuple[dict | None, dict | None]:
    """
    Find a question by q_no string across all sections.
    Returns (question_dict, section_dict) or (None, None) if not found.
    """
    for sec_data in rubric.get("sections", {}).values():
        for q in sec_data.get("questions", []):
            if str(q.get("q_no", "")) == str(q_no):
                return q, sec_data
    return None, None


def mark_budget(rubric: dict) -> float:
    """Return the total marks across all questions in the rubric."""
    total = 0.0
    for sec_data in rubric.get("sections", {}).values():
        for q in sec_data.get("questions", []):
            total += float(q.get("marks", 1))
    return total


def build_question_image_map(
    rubric: dict,
    pages: list[dict],
    mode: str = "broadcast",
) -> dict[str, str]:
    """
    Map each question number to a base64 page image.

    mode="broadcast" : every question gets page 0's image (fast, less accurate).
    mode="per_page"  : questions are distributed across pages proportionally.

    Returns {q_no: b64_image_string}
    """
    all_q_nos = [
        str(q.get("q_no", ""))
        for sec in rubric.get("sections", {}).values()
        for q in sec.get("questions", [])
    ]

    if not pages:
        return {}

    if mode == "broadcast":
        img = pages[0]["b64_image"]
        return {qno: img for qno in all_q_nos}

    # per_page: distribute questions across pages
    n_pages = len(pages)
    n_qs    = len(all_q_nos)
    mapping: dict[str, str] = {}
    for i, qno in enumerate(all_q_nos):
        page_idx = min(int(i * n_pages / n_qs), n_pages - 1)
        mapping[qno] = pages[page_idx]["b64_image"]
    return mapping


# ---------------------------------------------------------------------------
# QP-specific helpers (used by runner.py)
# ---------------------------------------------------------------------------

_Q_PATTERNS = [
    re.compile(r'\bQ\.?\s*(\d{1,2})\b', re.IGNORECASE),
    re.compile(r'\bQuestion\s+(\d{1,2})\b', re.IGNORECASE),
    re.compile(r'^\s*(\d{1,2})[.)]\s', re.MULTILINE),
]


def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
    """Rasterise every page of a PDF; return list of PIL Images."""
    doc    = fitz.open(pdf_path)
    mat    = fitz.Matrix(dpi / 72, dpi / 72)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    doc.close()
    return images


def pdf_page_to_image(pdf_path: str, page_index: int, dpi: int = 200) -> Image.Image:
    """Return a single page as PIL Image (0-indexed)."""
    doc  = fitz.open(pdf_path)
    page = doc[page_index]
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    img  = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()
    return img


def extract_text_per_page(pdf_path: str) -> list[str]:
    doc   = fitz.open(pdf_path)
    texts = [page.get_text() for page in doc]
    doc.close()
    return texts


def detect_questions_on_page(text: str) -> set[int]:
    found = set()
    for pat in _Q_PATTERNS:
        for m in pat.finditer(text):
            n = int(m.group(1))
            if 1 <= n <= 60:
                found.add(n)
    return found


def build_answer_page_map(pdf_path: str) -> dict[int, list[int]]:
    """question_number → [page_indices] (0-based) in the answer sheet."""
    texts      = extract_text_per_page(pdf_path)
    q_to_pages: dict[int, list[int]] = {}
    for page_idx, text in enumerate(texts):
        for q_num in detect_questions_on_page(text):
            q_to_pages.setdefault(q_num, [])
            if page_idx not in q_to_pages[q_num]:
                q_to_pages[q_num].append(page_idx)
    return q_to_pages


def build_qp_page_map(pdf_path: str) -> dict[int, int]:
    """question_number → first page index (0-based) in the question paper."""
    texts      = extract_text_per_page(pdf_path)
    q_to_page: dict[int, int] = {}
    for page_idx, text in enumerate(texts):
        low = text.lower()
        if any(kw in low for kw in ("marking scheme", "answer key", "mark scheme")):
            break
        for q_num in detect_questions_on_page(text):
            if q_num not in q_to_page:
                q_to_page[q_num] = page_idx
    return q_to_page


def get_marking_scheme_pages(pdf_path: str) -> list[int]:
    texts        = extract_text_per_page(pdf_path)
    scheme_pages = []
    in_scheme    = False
    for page_idx, text in enumerate(texts):
        low = text.lower()
        if not in_scheme and any(kw in low for kw in ("marking scheme", "answer key", "mark scheme")):
            in_scheme = True
        if in_scheme:
            scheme_pages.append(page_idx)
    return scheme_pages