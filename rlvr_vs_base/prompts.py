"""Prompt construction.

Parity-critical. The primary template reproduces, byte for byte, the
rendering of the chat template shipped with allenai/Olmo-3.1-7B-RL-Zero-Math
(chat_template.jinja, fetched 2026-06-12; vendored at
assets/olmo31_rlzero_math.chat_template.jinja and golden-tested in
tests/test_prompts.py). This is the prompt the RL Zero model was trained
with, and — following Yue et al.'s protocol — the base model receives the
IDENTICAL string, zero-shot, no few-shot examples.

Note the format: plain text, no chat special tokens, and the answer is
requested on a final 'Answer: ...' line (not \\boxed{}). Grading must match
this (see grade.py).
"""

RLZERO_HEADER = (
    "Solve the following problem step by step. The last line of your response "
    "should be the answer to the problem in form Answer: $Answer (without "
    "quotes) where $Answer is the answer to the problem."
)

RLZERO_SUFFIX = 'Remember to put your answer on its own line after "Answer:"'


def render_rlzero_math(problem: str) -> str:
    # chat_template.jinja with one user message and add_generation_prompt=True:
    #   HEADER + "\n\n" + content + "\n" + "\n" + SUFFIX
    return f"{RLZERO_HEADER}\n\n{problem}\n\n{RLZERO_SUFFIX}"


def render_boxed(problem: str) -> str:
    # Prompt-variant ablation: the qwen-boxed instruction used by the original
    # paper's harness, minus Qwen chat tokens (Olmo has no chat template).
    return (
        f"{problem}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )


TEMPLATES = {
    "rlzero_math": render_rlzero_math,
    "boxed": render_boxed,
}


def render(template: str, problem: str) -> str:
    return TEMPLATES[template](problem)
