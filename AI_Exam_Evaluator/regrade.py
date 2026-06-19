"""
regrade.py  —  AI Exam Evaluator (accuracy-first rewrite)
==========================================================
Fixes:
1. Sends ALL answer-sheet pages to Gemini for every question
   → grader finds the answer itself; wrong-page errors eliminated
2. MCQs graded in one batch call (entire page 0 at once)
3. Robust JSON extraction — never returns "See answer image" due to parse failure
4. Clean feedback — no raw JSON fragments in output
5. Works dynamically for any question paper / answer sheet

Usage:
    python regrade.py
    python regrade.py path/to/qp.pdf path/to/answers.pdf
"""

import os
import sys
import json
import re
import io
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import base64
from openai import OpenAI
import fitz  # PyMuPDF

# ── Config ────────────────────────────────────────────────────────────────────
QP_PATH   = sys.argv[1] if len(sys.argv) > 1 else r"data\question_papers\question_paper.pdf"
AS_PATH   = sys.argv[2] if len(sys.argv) > 2 else r"data\sample_sheets\student_answers.pdf"
RUBRIC    = sys.argv[3] if len(sys.argv) > 3 else r"schemas\rubric.json"
OUT_JSON  = r"reports\report_accurate.json"
OUT_TXT   = r"reports\report_accurate.txt"
MODEL     = os.environ.get("MODEL_NAME") or os.environ.get("VLM_MODEL", "qwen/qwen3.5-397b-a17b")

# ── Qwen client ───────────────────────────────────────────────────────────────
_client = None
def get_client():
    global _client
    if _client is None:
        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise ValueError("NVIDIA_API_KEY not set in .env")
        base_url = os.environ.get("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1"
        _client = OpenAI(api_key=key, base_url=base_url)
    return _client

# ── PDF → PNG bytes ───────────────────────────────────────────────────────────
def pdf_to_pages(path: str, dpi: int = 150) -> list[bytes]:
    doc = fitz.open(path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        pages.append(pix.tobytes("png"))
    doc.close()
    return pages

def img_part(png: bytes):
    return {"type": "image", "data": png}

# ── Qwen call with retry ──────────────────────────────────────────────────────
def call_qwen(parts: list, max_tokens: int = 2048, retries: int = 3) -> str:
    client = get_client()
    messages = [{"role": "user", "content": []}]
    for p in parts:
        if isinstance(p, str):
            messages[0]["content"].append({"type": "text", "text": p})
        elif isinstance(p, dict) and p.get("type") == "image":
            b64 = base64.b64encode(p["data"]).decode('utf-8')
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"    ⚠ Qwen error (attempt {attempt+1}): {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                raise

# ── JSON extraction — handles any Gemini response format ─────────────────────
def extract_json(raw: str) -> dict:
    """
    Extract a JSON object from Gemini's response.
    Tries multiple strategies so partial/prose responses still yield usable data.
    """
    # Remove markdown fences
    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Strategy 1: direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Strategy 2: find first {...} block
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass

    # Strategy 3: field-by-field regex extraction
    result = {}

    m = re.search(r'"marks_awarded"\s*:\s*([0-9.]+)', text)
    if m:
        result["marks_awarded"] = float(m.group(1))

    m = re.search(r'"max_marks"\s*:\s*([0-9.]+)', text)
    if m:
        result["max_marks"] = float(m.group(1))

    # Feedback: capture until end of JSON string (stop at next key or end)
    m = re.search(r'"feedback"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        result["feedback"] = m.group(1).replace('\\"', '"')

    if "marks_awarded" in result:
        return result

    # Strategy 4: look for numbers after common phrasings
    m = re.search(r'(?:award|score|grant|give)[s]?\s*:?\s*([0-9.]+)\s*(?:mark|point|out)', text, re.IGNORECASE)
    if m:
        result["marks_awarded"] = float(m.group(1))
        result["feedback"] = text[:300].strip()
        return result

    # Give up — return 0 with raw text as feedback
    return {
        "marks_awarded": 0.0,
        "feedback": text[:300].strip() or "No response from model"
    }

def clean_feedback(fb: str) -> str:
    """Remove any leaked JSON fragments from feedback text."""
    if not fb:
        return ""
    # If feedback IS json, extract just the feedback field
    if fb.strip().startswith("{"):
        m = re.search(r'"feedback"\s*:\s*"((?:[^"\\]|\\.)*)"', fb)
        if m:
            return m.group(1).replace('\\"', '"')
        return "See grader output"
    # Remove trailing JSON fragments like ', "max_marks": 1.0...'
    fb = re.sub(r',?\s*"(?:marks_awarded|max_marks|score|correct_answer)".*$', "", fb, flags=re.DOTALL)
    return fb.strip().strip('"').strip()

# ── Load rubric ───────────────────────────────────────────────────────────────
def load_rubric(path: str) -> list[dict]:
    """Load rubric and return flat list of question dicts."""
    with open(path, encoding="utf-8") as f:
        rubric = json.load(f)

    questions = []
    sections = rubric.get("sections", {})
    for sec_key, sec_data in sections.items():
        label = sec_data.get("label", f"Section {sec_key}")
        for q in sec_data.get("questions", []):
            q["section_label"] = label
            q.setdefault("marks", q.get("max_marks", 1))
            questions.append(q)

    questions.sort(key=lambda q: int(str(q.get("number", q.get("q_no", 999)))))
    return questions

# ── Grade Section A (MCQ) in one batch ───────────────────────────────────────
def grade_mcq_batch(mcq_questions: list[dict], as_pages: list[bytes], qp_pages: list[bytes]) -> list[dict]:
    """
    Grade all MCQ questions in one Gemini call.
    Sends the entire answer sheet so Gemini can find MCQ answers anywhere.
    """
    print(f"\n  ── Section A: grading {len(mcq_questions)} MCQs in one batch ──")

    # Build answer key from rubric
    answer_key = {}
    for q in mcq_questions:
        qnum = int(str(q.get("number", q.get("q_no", 0))))
        answer_key[qnum] = {
            "correct": str(q.get("correct_answer", "")).strip().upper(),
            "marks":   float(q.get("marks", 1))
        }

    parts = []
    # Add all answer sheet pages
    for i, pg in enumerate(as_pages):
        parts.append(img_part(pg))
    # Add first QP page for question context
    if qp_pages:
        parts.append(img_part(qp_pages[0]))

    answer_key_str = json.dumps(answer_key, indent=2)

    parts.append(f"""You are an exam grader. The images above are the student's answer sheet (all pages) followed by the question paper.

The MCQ answer key is:
{answer_key_str}

Find the student's chosen answer (a/b/c/d or A/B/C/D) for each MCQ question 1 through {len(mcq_questions)}.
Look carefully through ALL pages — MCQs are usually on the first page but may be elsewhere.

Return ONLY a JSON object like this (no markdown, no explanation):
{{
  "1": {{"chosen": "C", "awarded": 1.0, "feedback": "Correct — student chose C"}},
  "2": {{"chosen": "B", "awarded": 0.0, "feedback": "Wrong — student chose B, correct is D"}},
  "3": {{"chosen": null, "awarded": 0.0, "feedback": "Not attempted"}}
}}

For each question:
- "chosen": the letter the student circled/wrote, or null if not found
- "awarded": marks awarded (1.0 if correct, 0.0 if wrong or not attempted)
- "feedback": brief one-line explanation

Be strict: if the student's chosen letter does not match the correct answer, award 0.""")

    raw = call_qwen(parts, max_tokens=2048)

    # Parse the batch response
    try:
        text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
        batch = json.loads(text)
    except Exception:
        m = re.search(r'\{[\s\S]*\}', raw)
        try:
            batch = json.loads(m.group()) if m else {}
        except Exception:
            batch = {}

    results = []
    for q in mcq_questions:
        qnum = int(str(q.get("number", q.get("q_no", 0))))
        qdata = batch.get(str(qnum), {})
        awarded  = float(qdata.get("awarded", 0.0))
        feedback = clean_feedback(str(qdata.get("feedback", "No response")))
        results.append({
            "question_number": qnum,
            "marks_awarded":   awarded,
            "max_marks":       float(q.get("marks", 1)),
            "feedback":        feedback,
            "section":         q["section_label"],
        })
        flag = "✓" if awarded > 0 else "✗"
        print(f"    [{flag}] Q{qnum:>2}: {awarded:.0f}/{q.get('marks',1):.0f}  {feedback[:70]}")

    return results

# ── Grade a single written question ──────────────────────────────────────────
def grade_written_question(q: dict, as_pages: list[bytes], qp_pages: list[bytes]) -> dict:
    """
    Grade one written question by sending ALL answer sheet pages.
    Gemini finds the answer itself.
    """
    qnum      = int(str(q.get("number", q.get("q_no", 0))))
    max_marks = float(q.get("marks", q.get("max_marks", 1)))
    q_text    = q.get("text", "")
    steps     = q.get("steps", [])
    keywords  = q.get("keywords", [])
    correct   = q.get("correct_answer", "")
    section   = q["section_label"]

    steps_str = "\n".join(f"  - {s}" for s in steps) if steps else "  (grade holistically)"
    kw_str    = ", ".join(keywords) if keywords else "(none specified)"

    parts = []
    # All answer sheet pages — let Gemini find the answer
    for pg in as_pages:
        parts.append(img_part(pg))
    # Relevant QP page for question context
    if qp_pages:
        # Use first 2 QP pages for context (question text may span pages)
        for pg in qp_pages[:2]:
            parts.append(img_part(pg))

    parts.append(f"""You are an expert CBSE exam grader.

QUESTION {qnum} ({section}, {max_marks} marks):
{q_text}

Expected answer / key steps:
{steps_str}

Keywords to look for: {kw_str}
{f"Correct answer: {correct}" if correct else ""}

The images above are ALL pages of the student's answer sheet.

TASK:
1. Find the student's answer to Question {qnum} — look carefully through every page
2. If Question {qnum} is not answered anywhere, award 0 marks
3. Grade strictly against CBSE marking scheme:
   - Award marks for each correct step / key point shown
   - Partial credit where steps are partially correct
   - Deduct for wrong statements, missing steps, or calculation errors
   - Maximum marks: {max_marks}

Return ONLY this JSON (no markdown, no extra text):
{{"marks_awarded": <number>, "max_marks": {max_marks}, "feedback": "<specific feedback about what student did right/wrong, 1-3 sentences>"}}""")

    raw = call_qwen(parts, max_tokens=1024)
    parsed = extract_json(raw)

    awarded  = min(float(parsed.get("marks_awarded", 0.0)), max_marks)
    feedback = clean_feedback(str(parsed.get("feedback", raw[:200])))

    return {
        "question_number": qnum,
        "marks_awarded":   awarded,
        "max_marks":       max_marks,
        "feedback":        feedback,
        "section":         section,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  AI EXAM EVALUATOR  —  Accuracy-First Mode")
    print("=" * 62)
    print(f"  QP     : {QP_PATH}")
    print(f"  AS     : {AS_PATH}")
    print(f"  Rubric : {RUBRIC}")
    print(f"  Model  : {MODEL}")
    print()

    # Load rubric
    if not Path(RUBRIC).exists():
        print(f"ERROR: Rubric not found at {RUBRIC}")
        print("Run:  python -m pipeline.runner <qp.pdf> <as.pdf> --regen-rubric  first")
        sys.exit(1)

    questions = load_rubric(RUBRIC)
    print(f"  Loaded {len(questions)} questions from rubric")

    # Rasterize PDFs
    print("\n📄 Rasterizing PDFs…")
    qp_pages = pdf_to_pages(QP_PATH, dpi=150)
    as_pages = pdf_to_pages(AS_PATH, dpi=150)
    print(f"  QP: {len(qp_pages)} pages  |  AS: {len(as_pages)} pages")
    print(f"\n  Strategy: sending ALL {len(as_pages)} answer-sheet pages for each question")
    print(f"  (slower but eliminates wrong-page errors)\n")

    # Split MCQ vs written
    mcq_qs     = [q for q in questions if q.get("section_label", "").endswith("A")
                  or float(q.get("marks", 1)) == 1.0]
    written_qs = [q for q in questions if q not in mcq_qs]

    all_results = []

    # Grade MCQs in one batch
    if mcq_qs:
        mcq_results = grade_mcq_batch(mcq_qs, as_pages, qp_pages)
        all_results.extend(mcq_results)
        time.sleep(1)  # rate limit buffer

    # Grade written questions one by one
    sections_seen = set()
    for i, q in enumerate(written_qs):
        qnum    = int(str(q.get("number", q.get("q_no", 0))))
        section = q["section_label"]
        max_m   = float(q.get("marks", 1))

        if section not in sections_seen:
            print(f"\n  ── {section} ──")
            sections_seen.add(section)

        print(f"    Grading Q{qnum}…", end="", flush=True)
        result = grade_written_question(q, as_pages, qp_pages)
        all_results.append(result)

        awarded = result["marks_awarded"]
        flag = "✓" if awarded >= max_m else ("~" if awarded > 0 else "✗")
        fb   = result["feedback"][:70]
        print(f"\r    [{flag}] Q{qnum:>2}: {awarded:.1f}/{max_m:.1f}  {fb}")

        # Rate limit buffer between calls
        time.sleep(0.5)

    # Sort results by question number
    all_results.sort(key=lambda r: r["question_number"])

    # Compute totals
    total_score = sum(r["marks_awarded"] for r in all_results)
    max_score   = sum(r["max_marks"]     for r in all_results)

    section_scores: dict = {}
    for r in all_results:
        sec = r["section"]
        if sec not in section_scores:
            section_scores[sec] = {"score": 0.0, "max": 0.0}
        section_scores[sec]["score"] += r["marks_awarded"]
        section_scores[sec]["max"]   += r["max_marks"]

    pct = (total_score / max_score * 100) if max_score > 0 else 0

    # Build report
    report = {
        "total_score":     total_score,
        "max_score":       max_score,
        "percentage":      round(pct, 1),
        "section_scores":  section_scores,
        "results":         all_results,
        "model_used":      MODEL,
        "qp_file":         QP_PATH,
        "as_file":         AS_PATH,
        "rubric_used":     RUBRIC,
    }

    Path("reports").mkdir(exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Text report
    lines = [
        "=" * 62,
        "  EXAMINATION REPORT",
        "=" * 62,
        f"  Answer sheet : {AS_PATH}",
        f"  Question paper: {QP_PATH}",
        f"  Model used   : {MODEL}",
        "",
        f"  TOTAL SCORE : {total_score:.1f} / {max_score:.1f}  ({pct:.1f}%)",
        "",
    ]
    for sec, sc in section_scores.items():
        lines.append(f"  {sec:<20}: {sc['score']:.1f} / {sc['max']:.1f}")
    lines += ["", "-" * 62, "  Question Detail", "-" * 62]

    for r in all_results:
        qn   = r["question_number"]
        aw   = r["marks_awarded"]
        mx   = r["max_marks"]
        fb   = r["feedback"]
        sec  = r["section"]
        flag = "✓" if aw >= mx else ("~" if aw > 0 else "✗")
        lines.append(f"  [{flag}] Q{str(qn):>2}: {aw:.1f}/{mx:.1f}  ({sec})  —  {fb}")

    lines += ["=" * 62]
    txt = "\n".join(lines)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(txt)

    print()
    print(txt)
    print(f"\n✅  JSON report → {OUT_JSON}")
    print(f"✅  Text report → {OUT_TXT}")

if __name__ == "__main__":
    main()