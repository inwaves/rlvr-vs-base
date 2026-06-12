"""Answer extraction tests (no math-verify needed)."""

from rlvr_vs_base.grade import extract_candidate, extract_last_boxed


def test_answer_line_takes_last():
    text = "thinking...\nAnswer: 42\nmore thoughts\nAnswer: 56"
    assert extract_candidate(text) == ("56", "answer_line")


def test_answer_line_case_and_whitespace():
    cand, method = extract_candidate("  answer :  x+1  ")
    assert cand == "x+1"
    assert method == "answer_line"


def test_decorated_answer_line():
    assert extract_candidate("**Answer:** 7")[0] == "7"
    assert extract_candidate("**Answer**: 7")[0] == "7"
    assert extract_candidate("> Answer: 7")[0] == "7"


def test_boxed_fallback():
    cand, method = extract_candidate("the result is \\boxed{\\frac{1}{2}} qed")
    assert cand == "\\frac{1}{2}"
    assert method == "boxed"


def test_boxed_nested_and_last():
    assert extract_last_boxed("\\boxed{a{b}c}") == "a{b}c"
    assert extract_last_boxed("\\boxed{1} then \\boxed{2}") == "2"
    assert extract_last_boxed("\\boxed{unclosed") is None
    assert extract_last_boxed("nothing") is None


def test_answer_line_wins_over_boxed():
    text = "\\boxed{1}\nAnswer: 2"
    assert extract_candidate(text) == ("2", "answer_line")


def test_none():
    assert extract_candidate("no final result here") == (None, "none")
