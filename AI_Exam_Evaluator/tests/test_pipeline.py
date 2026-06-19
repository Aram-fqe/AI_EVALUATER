import os
import pytest
import numpy as np
from PIL import Image
from io import BytesIO

from pipeline.runner import (
    _safe_qnum,
    _sort_key,
    _find_column_mid,
    _check_rubric_completeness,
)


def test_safe_qnum():
    assert _safe_qnum("29 OR") == 29
    assert _safe_qnum("Q12") == 12
    assert _safe_qnum("11a") == 11
    assert _safe_qnum(42) == 42
    assert _safe_qnum(None) == 0


def test_sort_key():
    q1 = {"number": "11a", "section_label": "B"}
    q2 = {"number": "11b", "section_label": "B"}
    q3 = {"number": "11 OR", "section_label": "B"}
    q4 = {"number": "12", "section_label": "B"}

    assert _sort_key(q1) == (11, "a")
    assert _sort_key(q2) == (11, "b")
    assert _sort_key(q3) == (11, "or")
    assert _sort_key(q4) == (12, "")

    sorted_list = sorted([q4, q1, q3, q2], key=_sort_key)
    assert sorted_list == [q1, q2, q3, q4]


def test_find_column_mid():
    # Create a dummy image with two columns separated by a low-ink gutter (high pixel value/white)
    # in the middle, and dark columns (low pixel value/black) on the sides.
    w, h = 200, 100
    img_data = np.zeros((h, w), dtype=np.uint8)
    
    # Left column: draw some horizontal lines (ink/black/0)
    # Right column: draw some horizontal lines
    # Middle gutter: keep it blank/white (255)
    img_data[:, :] = 255
    img_data[20:80, 0:80] = 50
    img_data[20:80, 120:200] = 50
    
    img = Image.fromarray(img_data)
    mid = _find_column_mid(img)
    # The mid-point should be detected within the search range (60 to 140)
    assert 80 <= mid <= 120


def test_check_rubric_completeness(caplog):
    import logging
    # Test that it prints / logs warning but doesn't raise error if abort=False
    questions = [{"number": "1", "marks": 1}]
    # We pass a nonexistent file path to trigger fallback/warning
    with caplog.at_level(logging.WARNING):
        _check_rubric_completeness(questions, "nonexistent.pdf", gate=0.5, abort=False)
    # It should not fail, just log a warning
    assert any("Could not count expected questions" in rec.message for rec in caplog.records)
