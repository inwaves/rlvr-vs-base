"""Golden test: our static prompt rendering must match the HF chat template.

The vendored template is the chat_template.jinja shipped with
allenai/Olmo-3.1-7B-RL-Zero-Math (fetched 2026-06-12). transformers renders
chat templates with trim_blocks=True, lstrip_blocks=True; we mirror that.
"""

from pathlib import Path

import pytest

from rlvr_vs_base.prompts import render

ASSET = Path(__file__).resolve().parent.parent / "assets" / "olmo31_rlzero_math.chat_template.jinja"


def test_static_render_matches_hf_template():
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    tpl = env.from_string(ASSET.read_text())
    problem = "Compute $1+1$.\nReport the result as an integer."
    expected = tpl.render(
        messages=[{"role": "user", "content": problem}],
        add_generation_prompt=True,
    )
    assert render("rlzero_math", problem) == expected


def test_boxed_variant_contains_problem():
    out = render("boxed", "What is 2+2?")
    assert "What is 2+2?" in out
    assert "\\boxed{}" in out
