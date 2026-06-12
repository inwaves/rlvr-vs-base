"""End-to-end grading checks (requires math-verify; skipped if missing)."""

import pytest

pytest.importorskip("math_verify")

from rlvr_vs_base.grade import grade_text


@pytest.mark.parametrize(
    "text,gold,expected",
    [
        ("reasoning...\nAnswer: 204", "204", True),
        ("steps here\nAnswer: 1/2", "\\frac{1}{2}", True),
        ("the result is \\boxed{0.5}", "\\frac{1}{2}", True),
        ("reasoning...\nAnswer: 205", "204", False),
        ("I cannot solve this.", "204", False),
    ],
)
def test_grading_cases(text, gold, expected):
    assert grade_text(text, gold)["correct"] is expected, (text, gold)
