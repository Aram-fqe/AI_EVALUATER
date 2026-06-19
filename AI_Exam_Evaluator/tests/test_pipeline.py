"""
tests/test_pipeline.py

Comprehensive test suite for the VLM grading pipeline.
All API calls are mocked — no real Gemini/Claude/OpenAI needed.

Run:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -v -k "Rubric"
    pytest tests/test_pipeline.py -v -k "Grader"
    pytest tests/test_pipeline.py -v -m e2e
"""

import os
import sys
import json
import base64
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.input_handler import (
    load_rubric, _validate_rubric, get_question, mark_budget, build_question_image_map
)
from intelligence.grader import (
    _clamp_and_round, _grading_prompt, _parse_response, grade_question, grade_submission
)


# ════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════
@pytest.fixture
def minimal_rubric():
    return {
        "assessment_meta": {
            "title": "Test Exam", "board": "CBSE", "class": 9,
            "max_marks": 10, "half_mark_allowed": True,
            "vlm_provider": "gemini", "vlm_model": "gemini-1.5-pro"
        },
        "sections": {
            "A": {
                "label": "MCQ", "marks_per_question": 1,
                "partial_credit_allowed": False,
                "question_subtype": "mcq",
                "vlm_instructions": "Award 1 for correct, 0 for wrong.",
                "questions": [{
                    "q_no": "1",
                    "question_text": "What is 2+2?",
                    "mcq_options": {"A": "3", "B": "4", "C": "5", "D": "6"},
                    "mcq_correct_option": "B",
                    "topic": "Arithmetic",
                    "marks": 1
                }]
            },
            "B": {
                "label": "Short Answer", "marks_per_question": 2,
                "partial_credit_allowed": True,
                "question_subtype": "short_answer",
                "vlm_instructions": "Award marks per step.",
                "questions": [{
                    "q_no": "21",
                    "question_text": "Find remainder when x^3+1 divided by x+1",
                    "topic": "Polynomials",
                    "marks": 2,
                    "steps": [
                        {"id": "S1", "description": "Applies remainder theorem", "marks": 1, "is_mandatory": True},
                        {"id": "S2", "description": "Correct answer: 0", "marks": 1, "is_mandatory": True}
                    ],
                    "acceptable_alternatives": [],
                    "common_errors": []
                }]
            }
        },
        "skill_topics": ["Arithmetic", "Polynomials"]
    }


@pytest.fixture
def full_rubric_path():
    return Path(__file__).parent.parent / "schemas" / "rubric.json"


@pytest.fixture
def tiny_png_bytes():
    """Minimal 1×1 white PNG in raw bytes."""
    import struct, zlib
    def _chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
           + _chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
           + _chunk(b"IEND", b""))
    return png


@pytest.fixture
def tiny_png_b64(tiny_png_bytes):
    return base64.standard_b64encode(tiny_png_bytes).decode()


# ════════════════════════════════════════════════════════════
# 1. Rubric validation
# ════════════════════════════════════════════════════════════
class TestRubricValidation:

    def test_valid_rubric_passes(self, minimal_rubric):
        _validate_rubric(minimal_rubric)  # no exception

    def test_missing_assessment_meta_raises(self):
        with pytest.raises(ValueError, match="assessment_meta"):
            _validate_rubric({"sections": {}})

    def test_missing_sections_raises(self):
        with pytest.raises(ValueError, match="sections"):
            _validate_rubric({"assessment_meta": {}})

    def test_invalid_section_id_raises(self, minimal_rubric):
        bad = dict(minimal_rubric)
        bad["sections"]["Z"] = bad["sections"]["A"].copy()
        with pytest.raises(ValueError, match="Invalid section"):
            _validate_rubric(bad)

    def test_non_half_mark_step_raises(self, minimal_rubric):
        bad = dict(minimal_rubric)
        bad["sections"]["B"]["questions"][0]["steps"][0]["marks"] = 0.3
        with pytest.raises(ValueError, match="multiple of 0.5"):
            _validate_rubric(bad)

    def test_mcq_missing_correct_option_raises(self, minimal_rubric):
        bad = dict(minimal_rubric)
        del bad["sections"]["A"]["questions"][0]["mcq_correct_option"]
        with pytest.raises(ValueError, match="mcq_correct_option"):
            _validate_rubric(bad)

    def test_mcq_partial_credit_auto_corrected(self, minimal_rubric):
        bad = dict(minimal_rubric)
        bad["sections"]["A"]["partial_credit_allowed"] = True
        _validate_rubric(bad)
        assert bad["sections"]["A"]["partial_credit_allowed"] is False

    def test_load_rubric_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_rubric("/nonexistent/path/rubric.json")

    def test_load_rubric_wrong_extension(self, tmp_path):
        f = tmp_path / "rubric.txt"
        f.write_text("{}")
        with pytest.raises(ValueError, match=".json"):
            load_rubric(f)

    def test_full_rubric_json_is_valid(self, full_rubric_path):
        if not full_rubric_path.exists():
            pytest.skip("schemas/rubric.json not found")
        rubric = load_rubric(full_rubric_path)
        assert "assessment_meta" in rubric
        assert "sections" in rubric


# ════════════════════════════════════════════════════════════
# 2. Rubric helpers
# ════════════════════════════════════════════════════════════
class TestRubricHelpers:

    def test_get_question_found(self, minimal_rubric):
        q, sec = get_question(minimal_rubric, "21")
        assert q["question_text"].startswith("Find remainder")
        assert sec["label"] == "Short Answer"

    def test_get_question_not_found(self, minimal_rubric):
        q, sec = get_question(minimal_rubric, "999")
        assert q is None and sec is None

    def test_mark_budget(self, minimal_rubric):
        assert mark_budget(minimal_rubric) == 3.0  # 1 MCQ + 2 SA

    def test_build_image_map_broadcast(self, minimal_rubric, tiny_png_b64):
        pages = [{"b64_image": tiny_png_b64}]
        q_map = build_question_image_map(minimal_rubric, pages, mode="broadcast")
        assert set(q_map.keys()) == {"1", "21"}
        assert q_map["1"] == q_map["21"] == tiny_png_b64

    def test_build_image_map_per_page(self, minimal_rubric, tiny_png_b64):
        b64_a = tiny_png_b64
        b64_b = tiny_png_b64  # same image — just testing mapping
        pages = [{"b64_image": b64_a}, {"b64_image": b64_b}]
        q_map = build_question_image_map(minimal_rubric, pages, mode="per_page")
        assert "1"  in q_map
        assert "21" in q_map

    def test_build_image_map_auto_single_page_broadcasts(self, minimal_rubric, tiny_png_b64):
        pages = [{"b64_image": tiny_png_b64}]
        q_map = build_question_image_map(minimal_rubric, pages, mode="auto")
        assert len(q_map) == 2
        assert q_map["1"] == q_map["21"]


# ════════════════════════════════════════════════════════════
# 3. Half-mark clamping  (NCERT rules)
# ════════════════════════════════════════════════════════════
class TestClampAndRound:

    def test_exact_half_unchanged(self):
        assert _clamp_and_round(1.5, 2.0) == 1.5

    def test_rounds_up_to_half(self):
        assert _clamp_and_round(1.6, 2.0) == 1.5   # 1.6 → nearest 0.5 = 1.5
        assert _clamp_and_round(1.8, 2.0) == 2.0

    def test_rounds_down_to_half(self):
        assert _clamp_and_round(1.3, 2.0) == 1.5
        assert _clamp_and_round(1.2, 2.0) == 1.0

    def test_clamp_above_max(self):
        assert _clamp_and_round(10.0, 6.0) == 6.0

    def test_clamp_below_zero(self):
        assert _clamp_and_round(-1.0, 6.0) == 0.0

    def test_mcq_all_or_nothing_correct(self):
        assert _clamp_and_round(1.0, 1.0, is_mcq=True) == 1.0

    def test_mcq_all_or_nothing_partial(self):
        assert _clamp_and_round(0.5, 1.0, is_mcq=True) == 0.0

    def test_mcq_zero(self):
        assert _clamp_and_round(0.0, 1.0, is_mcq=True) == 0.0


# ════════════════════════════════════════════════════════════
# 4. Prompt builder
# ════════════════════════════════════════════════════════════
class TestPromptBuilder:

    def test_mcq_prompt_includes_correct_option(self, minimal_rubric):
        q   = minimal_rubric["sections"]["A"]["questions"][0]
        sec = minimal_rubric["sections"]["A"]
        prompt = _grading_prompt(q, sec)
        assert "Correct option: B" in prompt
        assert "MCQ" in prompt

    def test_sa_prompt_includes_step_ids(self, minimal_rubric):
        q   = minimal_rubric["sections"]["B"]["questions"][0]
        sec = minimal_rubric["sections"]["B"]
        prompt = _grading_prompt(q, sec)
        assert "S1" in prompt
        assert "S2" in prompt

    def test_or_variant_text_present(self, minimal_rubric):
        q = {
            "q_no": "28", "question_text": "OR question", "marks": 3,
            "or_variant": True, "paired_with": "27",
            "question_subtype": "short_answer", "steps": [], "acceptable_alternatives": [],
            "common_errors": []
        }
        sec = minimal_rubric["sections"]["B"]
        prompt = _grading_prompt(q, sec)
        assert "OR QUESTION" in prompt

    def test_construction_question_shows_vlm_instructions(self, minimal_rubric):
        q = {
            "q_no": "35", "question_text": "Draw triangle.", "marks": 6,
            "question_subtype": "construction",
            "vlm_instructions": "Check diagram accuracy.",
            "steps": [], "acceptable_alternatives": [], "common_errors": []
        }
        sec = minimal_rubric["sections"]["D"] if "D" in minimal_rubric["sections"] else minimal_rubric["sections"]["B"]
        prompt = _grading_prompt(q, sec)
        assert "Check diagram accuracy" in prompt


# ════════════════════════════════════════════════════════════
# 5. Parse VLM response
# ════════════════════════════════════════════════════════════
class TestParseResponse:

    def test_clean_json_parsed(self):
        raw = '{"score": 1.5, "criterion_scores": {"S1": 1.0, "S2": 0.5}, "feedback": "Good.", "confidence": 0.9, "needs_review": false}'
        r = _parse_response(raw, 2.0, is_mcq=False)
        assert r["score"] == 1.5
        assert r["confidence"] == 0.9

    def test_markdown_fences_stripped(self):
        raw = '```json\n{"score": 2.0, "feedback": "Perfect."}\n```'
        r = _parse_response(raw, 2.0, is_mcq=False)
        assert r["score"] == 2.0

    def test_garbage_returns_parse_error(self):
        r = _parse_response("Sorry I cannot grade this.", 2.0, is_mcq=False)
        assert r["score"] == 0.0
        assert r.get("flag") == "PARSE_ERROR"
        assert r["needs_review"] is True

    def test_mcq_partial_score_zeroed(self):
        raw = '{"score": 0.5, "feedback": "Almost."}'
        r = _parse_response(raw, 1.0, is_mcq=True)
        assert r["score"] == 0.0  # all-or-nothing

    def test_score_clamped_to_max(self):
        raw = '{"score": 99, "feedback": "Way too high."}'
        r = _parse_response(raw, 3.0, is_mcq=False)
        assert r["score"] == 3.0

    def test_missing_score_defaults_to_zero(self):
        raw = '{"feedback": "No score field."}'
        r = _parse_response(raw, 2.0, is_mcq=False)
        assert r["score"] == 0.0


# ════════════════════════════════════════════════════════════
# 6. grade_question — mocked VLM
# ════════════════════════════════════════════════════════════
class TestGradeQuestionMocked:

    def _mock_vlm_response(self, payload: dict):
        """Return a mock that makes _call_vlm return JSON."""
        return patch(
            "intelligence.grader._call_vlm",
            return_value=json.dumps(payload)
        )

    def test_mcq_correct(self, minimal_rubric, tiny_png_b64):
        q   = minimal_rubric["sections"]["A"]["questions"][0]
        sec = minimal_rubric["sections"]["A"]
        payload = {"score": 1.0, "feedback": "Correct.", "criterion_scores": {}, "confidence": 0.99, "needs_review": False}
        with self._mock_vlm_response(payload):
            r = grade_question(q, sec, tiny_png_b64)
        assert r["score"] == 1.0
        assert r["needs_review"] is False

    def test_mcq_wrong_gives_zero(self, minimal_rubric, tiny_png_b64):
        q   = minimal_rubric["sections"]["A"]["questions"][0]
        sec = minimal_rubric["sections"]["A"]
        payload = {"score": 0.0, "feedback": "Wrong option.", "criterion_scores": {}, "confidence": 0.95, "needs_review": False}
        with self._mock_vlm_response(payload):
            r = grade_question(q, sec, tiny_png_b64)
        assert r["score"] == 0.0

    def test_sa_partial_credit(self, minimal_rubric, tiny_png_b64):
        q   = minimal_rubric["sections"]["B"]["questions"][0]
        sec = minimal_rubric["sections"]["B"]
        payload = {"score": 1.0, "feedback": "Partial.", "criterion_scores": {"S1": 1.0, "S2": 0.0}, "confidence": 0.8, "needs_review": False}
        with self._mock_vlm_response(payload):
            r = grade_question(q, sec, tiny_png_b64)
        assert r["score"] == 1.0
        assert r["criterion_scores"]["S1"] == 1.0

    def test_low_confidence_sets_needs_review(self, minimal_rubric, tiny_png_b64):
        q   = minimal_rubric["sections"]["B"]["questions"][0]
        sec = minimal_rubric["sections"]["B"]
        payload = {"score": 1.5, "feedback": "Unclear.", "criterion_scores": {}, "confidence": 0.3, "needs_review": True}
        with self._mock_vlm_response(payload):
            r = grade_question(q, sec, tiny_png_b64)
        assert r["needs_review"] is True

    def test_construction_always_needs_review(self, tiny_png_b64):
        q = {
            "q_no": "35", "question_text": "Draw triangle.", "marks": 6,
            "question_subtype": "construction", "requires_visual_check": True,
            "steps": [], "acceptable_alternatives": [], "common_errors": []
        }
        sec = {"question_subtype": "long_answer", "marks_per_question": 6, "vlm_instructions": ""}
        payload = {"score": 5.0, "feedback": "Good diagram.", "criterion_scores": {}, "confidence": 0.9, "needs_review": False}
        with patch("intelligence.grader._call_vlm", return_value=json.dumps(payload)):
            r = grade_question(q, sec, tiny_png_b64)
        assert r["needs_review"] is True  # always True for construction

    def test_vlm_exception_returns_error_dict(self, minimal_rubric, tiny_png_b64):
        q   = minimal_rubric["sections"]["A"]["questions"][0]
        sec = minimal_rubric["sections"]["A"]
        with patch("intelligence.grader._call_vlm", side_effect=RuntimeError("API down")):
            r = grade_question(q, sec, tiny_png_b64)
        assert r["score"] == 0.0
        assert r["flag"] == "ERROR"
        assert "API down" in r["feedback"]

    def test_hallucinated_score_clamped(self, minimal_rubric, tiny_png_b64):
        q   = minimal_rubric["sections"]["B"]["questions"][0]
        sec = minimal_rubric["sections"]["B"]
        payload = {"score": 99, "feedback": "Excellent!", "criterion_scores": {}, "confidence": 1.0, "needs_review": False}
        with patch("intelligence.grader._call_vlm", return_value=json.dumps(payload)):
            r = grade_question(q, sec, tiny_png_b64)
        assert r["score"] == 2.0  # clamped to max_marks


# ════════════════════════════════════════════════════════════
# 7. grade_submission — mocked
# ════════════════════════════════════════════════════════════
class TestGradeSubmissionMocked:

    def test_total_score_accumulates(self, minimal_rubric, tiny_png_b64):
        q_image_map = {"1": tiny_png_b64, "21": tiny_png_b64}

        def fake_grade(question, section, image):
            marks = float(question.get("marks", 1))
            return {"q_no": question["q_no"], "score": marks, "max_marks": marks,
                    "feedback": "Full marks.", "criterion_scores": {},
                    "confidence": 1.0, "needs_review": False}

        with patch("intelligence.grader.grade_question", side_effect=fake_grade):
            result = grade_submission(minimal_rubric, q_image_map)

        assert result["total_score"] == 3.0
        assert result["max_score"]   == 3.0
        assert result["percentage"]  == 100.0

    def test_skill_scores_aggregated(self, minimal_rubric, tiny_png_b64):
        q_image_map = {"1": tiny_png_b64, "21": tiny_png_b64}

        def fake_grade(question, section, image):
            return {"q_no": question["q_no"], "score": 0.5, "max_marks": float(question["marks"]),
                    "feedback": "", "criterion_scores": {}, "confidence": 1.0, "needs_review": False}

        with patch("intelligence.grader.grade_question", side_effect=fake_grade):
            result = grade_submission(minimal_rubric, q_image_map)

        assert "Arithmetic"  in result["skill_scores"]
        assert "Polynomials" in result["skill_scores"]

    def test_needs_review_false_when_all_confident(self, minimal_rubric, tiny_png_b64):
        q_image_map = {"1": tiny_png_b64, "21": tiny_png_b64}

        def fake_grade(question, section, image):
            return {"q_no": question["q_no"], "score": 1.0, "max_marks": float(question["marks"]),
                    "feedback": "", "criterion_scores": {}, "confidence": 1.0, "needs_review": False}

        with patch("intelligence.grader.grade_question", side_effect=fake_grade):
            result = grade_submission(minimal_rubric, q_image_map)

        assert result["needs_review"] is False

    def test_needs_review_true_when_any_flagged(self, minimal_rubric, tiny_png_b64):
        q_image_map = {"1": tiny_png_b64, "21": tiny_png_b64}
        call_count = {"n": 0}

        def fake_grade(question, section, image):
            call_count["n"] += 1
            review = call_count["n"] == 1  # first question flags review
            return {"q_no": question["q_no"], "score": 1.0, "max_marks": float(question["marks"]),
                    "feedback": "", "criterion_scores": {}, "confidence": 0.4, "needs_review": review}

        with patch("intelligence.grader.grade_question", side_effect=fake_grade):
            result = grade_submission(minimal_rubric, q_image_map)

        assert result["needs_review"] is True


# ════════════════════════════════════════════════════════════
# 8. Image loading unit tests
# ════════════════════════════════════════════════════════════
class TestImageLoading:

    def test_single_png_loads(self, tmp_path, tiny_png_bytes):
        from pipeline.input_handler import load_answer_images
        img = tmp_path / "sheet.png"
        img.write_bytes(tiny_png_bytes)
        pages = load_answer_images(img)
        assert len(pages) == 1
        assert pages[0]["mime_type"] == "image/png"
        assert pages[0]["b64_image"]  # non-empty

    def test_file_not_found(self):
        from pipeline.input_handler import load_answer_images
        with pytest.raises(FileNotFoundError):
            load_answer_images("/nope/not/here.pdf")

    def test_unsupported_extension(self, tmp_path):
        from pipeline.input_handler import load_answer_images
        f = tmp_path / "answer.docx"
        f.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported"):
            load_answer_images(f)

    def test_zip_with_no_images_raises(self, tmp_path):
        import zipfile
        from pipeline.input_handler import load_answer_images
        z = tmp_path / "empty.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("readme.txt", "nothing here")
        with pytest.raises(ValueError, match="no PDF or image"):
            load_answer_images(z)


# ════════════════════════════════════════════════════════════
# 9. End-to-end pipeline (mocked VLM)
# ════════════════════════════════════════════════════════════
@pytest.mark.e2e
class TestEndToEndPipeline:

    def _fake_grade(self, question, section, image):
        return {
            "q_no":             question["q_no"],
            "score":            float(question.get("marks", 1)),
            "max_marks":        float(question.get("marks", 1)),
            "feedback":         "Full marks.",
            "criterion_scores": {},
            "confidence":       0.95,
            "needs_review":     False
        }

    def test_png_broadcast_pipeline(self, tmp_path, minimal_rubric, tiny_png_bytes):
        img = tmp_path / "answers.png"
        img.write_bytes(tiny_png_bytes)

        rubric_file = tmp_path / "rubric.json"
        rubric_file.write_text(json.dumps(minimal_rubric))

        from pipeline.input_handler import load_answer_images, build_question_image_map
        pages   = load_answer_images(img)
        q_map   = build_question_image_map(minimal_rubric, pages, mode="broadcast")

        with patch("intelligence.grader.grade_question", side_effect=self._fake_grade):
            result = grade_submission(minimal_rubric, q_map)

        assert result["total_score"] == 3.0
        assert result["percentage"]  == 100.0
        assert len(result["results"]) == 2

    def test_zip_pipeline(self, tmp_path, minimal_rubric, tiny_png_bytes):
        import zipfile
        # Create ZIP with two PNGs
        png1 = tmp_path / "p1.png"
        png1.write_bytes(tiny_png_bytes)
        z = tmp_path / "sheets.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.write(png1, "p1.png")

        from pipeline.input_handler import load_answer_images, build_question_image_map
        pages = load_answer_images(z)
        assert len(pages) >= 1

        q_map = build_question_image_map(minimal_rubric, pages, mode="broadcast")
        with patch("intelligence.grader.grade_question", side_effect=self._fake_grade):
            result = grade_submission(minimal_rubric, q_map)

        assert result["total_score"] > 0


# ════════════════════════════════════════════════════════════
# 10. Edge cases
# ════════════════════════════════════════════════════════════
class TestEdgeCases:

    def test_empty_question_image_map_gives_zero(self, minimal_rubric):
        with patch("intelligence.grader.grade_question") as mock_gq:
            result = grade_submission(minimal_rubric, {})  # no images
        mock_gq.assert_not_called()
        assert result["total_score"] == 0.0
        assert result["max_score"]   == 0.0

    def test_question_not_in_map_skipped(self, minimal_rubric, tiny_png_b64):
        # Only provide image for q_no "1", not "21"
        q_image_map = {"1": tiny_png_b64}

        def fake_grade(question, section, image):
            return {"q_no": question["q_no"], "score": 1.0, "max_marks": 1.0,
                    "feedback": "", "criterion_scores": {}, "confidence": 1.0, "needs_review": False}

        with patch("intelligence.grader.grade_question", side_effect=fake_grade):
            result = grade_submission(minimal_rubric, q_image_map)

        # Only Q1 graded (1 mark); Q21 skipped (max 2 marks not counted in max_total)
        assert result["total_score"] == 1.0
        assert len(result["results"]) == 1

    def test_score_zero_when_all_wrong(self, minimal_rubric, tiny_png_b64):
        q_image_map = {"1": tiny_png_b64, "21": tiny_png_b64}

        def fake_grade(question, section, image):
            return {"q_no": question["q_no"], "score": 0.0, "max_marks": float(question["marks"]),
                    "feedback": "Wrong.", "criterion_scores": {}, "confidence": 0.9, "needs_review": False}

        with patch("intelligence.grader.grade_question", side_effect=fake_grade):
            result = grade_submission(minimal_rubric, q_image_map)

        assert result["total_score"] == 0.0
        assert result["percentage"]  == 0.0