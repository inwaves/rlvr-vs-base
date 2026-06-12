# rlvr-vs-base

Budget-constrained replication of the central claim of Yue et al., *Does
Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond
the Base Model?* ([arXiv:2504.13837](https://arxiv.org/abs/2504.13837)),
Figure 2: **RLVR-trained models beat their base models at pass@1, but as k
grows into the tens or hundreds, the base model catches up and overtakes.**

As of June 2026 there is no independent published replication of that
crossover on a post-Qwen2.5 model pair. This repo provides one, on a fully
open pair with verified zero-RL lineage (HF metadata, checked 2026-06-12):

- **Base**: [`allenai/Olmo-3-1025-7B`](https://huggingface.co/allenai/Olmo-3-1025-7B)
- **RLVR**: [`allenai/Olmo-3.1-7B-RL-Zero-Math`](https://huggingface.co/allenai/Olmo-3.1-7B-RL-Zero-Math)
  — RLVR directly from that base on
  [`allenai/Dolci-RL-Zero-Math-7B`](https://huggingface.co/datasets/allenai/Dolci-RL-Zero-Math-7B),
  no SFT in between. The 3.1 refresh is Ai2's longer, more stable RL Zero
  run, which gives the boundary-expansion hypothesis its best shot.

Everything that can run without a GPU is already done in this repo: code,
unit tests, prompt-parity goldens, dataset checks, pre-registered selection
rules. **The GPU box only runs inference.**

Cost is a declared design constraint: the whole experiment is designed to an
all-in cap of $100 (one H100 at $2-3/hr spot; ~12-16 GPU-hours of
generation). The write-up discusses this openly — what was dropped for
budget and what a funded version adds back.

## Design

Three stages; every stage reuses all previous samples (same sampling
distribution, engine seed varies per run — the same mechanism as the
original repo). Selection rules are pre-registered in
`configs/experiment.yaml` and symmetric in the two models.

| Stage | What runs | New samples (both models) | Output tokens (est.) |
|-------|-----------|---------------------------|----------------------|
| 0 pilot | all 500 MATH500 at N=16; all 60 AIME24+25 at N=32 | ~20k | ~70M |
| 1 main | MATH500-hard subset (~250) to N=128; AIME24 (30) to N=256; + prompt ablation at N=32 | ~70k | ~250-300M |
| 2 extension | only unresolved problems (capped at 12/benchmark): AIME24 to N=1024, MATH500 stragglers to N=256 | ~23k | ~115M |

- **MATH500-hard subset**: every problem where either model's pilot accuracy
  is below 50%, topped up to 250 stratified by MATH level. Results are
  labelled MATH500-hard, never passed off as full MATH500.
- **Stage-2 rule**: a problem is unresolved if one model solves it while the
  other has fewer than 5 successes at the current N (curve not saturated).
- AIME25 runs in the pilot only; full AIME25, the code leg, the MiMo-7B
  cross-family pair, and k=1024-everywhere are the first things a funded
  follow-up adds back.

## Parity with the original

| Knob | Paper (2504.13837) | This repo |
|------|--------------------|-----------|
| Sampling | temperature 0.6, top-p 0.95 | identical |
| Max new tokens | 16,384 | 16,384 on AIME; 8,192 on MATH500 (budget deviation, gate G2 reverts it if cap-hits exceed 3%) |
| Prompting | zero-shot, no few-shot for base; both models get the RLVR training prompt | identical: the RL Zero chat template rendered to plain text, byte-identical string to both models (golden-tested) |
| pass@k | unbiased Codex estimator from N samples per problem | identical formula (`rlvr_vs_base/passk.py`) |
| Sampling diversity | vLLM engine seed per run, automatic intra-run diversity | identical mechanism |
| Verifier | latex2sympy pipeline (SimpleRL-Zoo lineage) | math-verify; extraction follows the RL Zero `Answer:` training format with boxed fallback; same policy for both models |
| Guessing control | manual CoT inspection of solved problems with accuracy < 5% | identical (`spotcheck.py`); extra-important on AIME at N=1024, where integer answers in 0-999 make lucky guessing non-negligible |
| Benchmarks | GSM8K, MATH500, Minerva, Olympiad, AIME24, AMC23 | MATH500-hard + AIME24 (+ AIME25 pilot) — budget subset, symmetric selection |
| k range | up to 1024 | 128 on MATH500-hard; 256, adaptively extended to 1024, on AIME24 |

## Prompt format (parity-critical)

The RL Zero models were trained with a plain-text prompt — no chat special
tokens — defined by the `chat_template.jinja` shipped with the checkpoint
(vendored at `assets/olmo31_rlzero_math.chat_template.jinja`, golden-tested
in `tests/test_prompts.py`). Rendered:

```text
Solve the following problem step by step. The last line of your response should be the answer to the problem in form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:"
```

Note the answer format is a final `Answer:` line, not `\boxed{}` — grading
(`grade.py`) extracts accordingly, with boxed and full-text fallbacks, and
applies the identical policy to both models.

## Layout

```
configs/experiment.yaml   # single source of truth: models, stages, rules, caps
rlvr_vs_base/
  data.py        # benchmark loading, hard count/field assertions
  prompts.py     # parity-critical templates (+ boxed ablation variant)
  generate.py    # resumable vLLM batch generation -> runs/.../gens/
  grade.py       # Answer-line extraction + math-verify, per-item timeouts
  aggregate.py   # graded shards -> per-problem (n, c) counts
  passk.py       # unbiased estimator, curves, bootstrap CIs, crossover-k
  analyze.py     # subset selection, stage-2 candidates, pilot/final reports
  spotcheck.py   # dumps low-accuracy solves for manual CoT review
scripts/         # setup_box.sh, stage0.sh, stage1.sh, stage2.sh
tests/           # estimator, extraction, prompt goldens, selection, alignment
```

## Runbook — exact order of operations

**Phase A (no GPU) — done in this repo.** Code, tests, prompt goldens,
dataset verification, pre-registered rules.

**Phase B (GPU box). Stages are serial; analysis gates between them. If a
gate fails: stop — GPU time costs money, debugging is free on CPU.**

```bash
# 0. Setup + preflight (~15 min, <$1)
git clone https://github.com/inwaves/rlvr-vs-base && cd rlvr-vs-base
bash scripts/setup_box.sh
#    Installs deps, runs pytest, checks datasets, prints the rendered prompt
#    (compare to the golden above), prefetches both models, runs a small
#    throughput probe.
#    GATE G0: tests pass; prompt matches; probe shows sane tokens/s.

# 1. Pilot (~70M tokens, ~2-3 H100-hours, ~$5-8)
bash scripts/stage0.sh
#    GATE G1 (sanity): RL pass@1 > base pass@1 on aime24. If not, debug
#    prompts/grading before spending more.
#    GATE G2 (caps): MATH500 cap-hit rate < 3% for both models; else set
#    math500 max_tokens: 16384 in the config and rerun the math500 pilot.
#    GATE G3 (budget): pilot-report token forecast within budget; else trim
#    stage-1 targets in the config.

# 2. Main run + prompt ablation (~250-300M tokens, ~8-10 H100-hours, ~$20-30)
bash scripts/stage1.sh
#    Produces interim pass@k curves (runs/report/) and stage-2 candidate
#    lists (runs/stage2/). Review both.

# 3. Adaptive extension (~115M tokens, ~3-4 H100-hours, ~$10)
bash scripts/stage2.sh
#    Deep sampling only on pre-registered unresolved problems; final
#    reports; spot-check trace export.

# 4. Wrap up
#    Pull runs/aggregates/, runs/report/, runs/spotcheck/ (small);
#    archive or discard raw runs/*/gens/ shards (~30-60 GB compressed).
#    Tear down the box. Manual CoT spot check happens off-box.
```

Every stage is resumable: generation counts existing samples per problem
and only fills the deficit, so a preempted box loses at most one group of
samples. Run stages inside tmux/screen on the box; poll progress from a
second shell with `bash scripts/watch.sh`.

## Observability

Designed so the operator can tell at a glance whether the run is healthy,
stuck, or producing garbage — without interrupting it:

- **Heartbeat** (`runs/status.json`): rewritten atomically after every
  generation group (~5-10 min at stage-1 scale; `prompts_per_call` bounds
  this) and every graded shard. Fields: phase, benchmark/model, samples
  done/total, group tok/s, cap-hit %, near-empty %, ETA, disk free.
  `python -m rlvr_vs_base.status` pretty-prints it and flags a stale
  heartbeat (>20 min in an active phase = engine wedged; kill and re-run,
  resume handles the rest).
- **Quality canaries**: each group prints WARN lines when cap-hit rate
  exceeds 10% (max_tokens too low — gate G2 territory) or near-empty
  completions exceed 2% (prompt/template breakage). These catch a broken
  run within minutes, before a stage budget is burned.
- **Logs**: every stage script tees all output (including vLLM progress
  bars and engine errors) to `runs/logs/<stage>-<timestamp>.log`, so
  postmortems survive terminal scrollback.
- **GPU telemetry**: stage scripts background a 30s `nvidia-smi` CSV logger
  into `runs/logs/gpu.csv` — utilization/memory/power history for
  diagnosing throughput regressions.
- **`bash scripts/watch.sh`**: heartbeat + latest log tail + GPU snapshot
  in one command.

Failure playbook: engine hang → stale heartbeat → kill, re-run stage
(resumes). CUDA OOM at startup → lower `--gpu-memory-utilization`. OOM
mid-run → kill, lower, re-run (resumes). Persistent WARN canaries → stop,
inspect a shard with `zcat runs/<bench>/<model>/gens/*/shard-*.jsonl.gz | head`,
fix, resume.

## Estimator note (adaptive N)

pass@k for a problem with n < k samples is reported as pass@n (conservative
clamp). The stage-2 rule extends exactly the problems where the clamp could
bite — unsaturated curves — so the residual bias is confined to saturated
problems where pass@k is pinned near its asymptote. Disclosed in the
write-up.

## Decision criteria

- **Replicated**: the base curve crosses the RL curve at some k within
  budget on the majority of benchmarks (or the k=1 gap shrinks to within CI
  noise at k_max), and the CoT spot check does not attribute the base
  model's high-k solves to guessing.
- **Qualified/contradicted**: the RL curve stays above the base with a
  stable or growing gap through k_max — the ProRL-shaped outcome, notable
  on a true zero-RL pair.

## Related

- Official code for the original paper: [LeapLabTHU/limit-of-RLVR](https://github.com/LeapLabTHU/limit-of-RLVR)
  (protocol source; this harness lifts its sampling protocol and estimator,
  not its Qwen-specific plumbing).
- Paper: [arXiv:2504.13837](https://arxiv.org/abs/2504.13837) (v5 adds the
  Magistral-Medium vs Mistral-Medium-3 extension).
