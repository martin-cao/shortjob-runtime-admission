# Reproducibility Guide

This document describes the public artifact workflow. Unless noted otherwise, run all commands from the repository root.

The independent single-GPU supplemental protocol is documented in [SUPPLEMENTAL_EXPERIMENTS.md](SUPPLEMENTAL_EXPERIMENTS.md). It has separate inputs and outputs; historical results below retain their original protocol.

## 1. CPU-Side Setup

Install the locked dependencies and run the CPU-side test suite:

```bash
uv sync
uv run python -m unittest discover -s tests
```

Inspect CLI help:

```bash
uv run python src/run_shortjob_isolated.py --help
uv run python src/summarize_shortjob.py --help
uv run python src/evaluate_baselines.py --help
uv run python src/replay_worker_queue.py --help
```

CPU-side tests and CLI inspection do not require an NVIDIA GPU.

## 2. GPU Preflight

On a Linux machine with an NVIDIA GPU, verify the runtime environment, then run a minimal correctness check:

```bash
uv sync
uv run python src/check_runtime_env.py
uv run python src/check_correctness.py \
  --workload fixed_shape_infer_small \
  --action eager graphs_only \
  --batch-size 4 \
  --num-steps 3 \
  --out data/raw/smoke_correctness_gate.jsonl
```

`check_correctness.py` may return a nonzero exit code when some actions fail or are infeasible. That is not necessarily a runner crash. 

In the full experiment pipeline, failed and infeasible action records are used to construct eligibility gates and blocklists. The preflight command above is an independent execution check and is not consumed by the minimal smoke analysis in the next section.

## 3. Minimal Smoke Run

Run a small GPU measurement:

```bash
uv run python src/run_shortjob_isolated.py \
  --workload fixed_shape_infer_small \
  --action eager graphs_only \
  --batch-size 16 \
  --num-steps 50 \
  --repeats 1 \
  --seed 42 \
  --device cuda \
  --time-accounting-mode cold_total \
  --cache-state cold_cache \
  --out data/raw/smoke_shortjob.jsonl
```

Generate a summary, evaluate the baseline policies, and run a small worker-queue replay:

```bash
uv run python src/summarize_shortjob.py \
  --input data/raw/smoke_shortjob.jsonl \
  --out results/tables/smoke_action_summary.jsonl \
  --allow-unconstrained

uv run python src/evaluate_baselines.py \
  --summary results/tables/smoke_action_summary.jsonl \
  --out results/tables/smoke_policy_eval.jsonl \
  --aggregate-out results/tables/smoke_policy_aggregate.jsonl \
  --allow-unconstrained

uv run python src/replay_worker_queue.py \
  --summary results/tables/smoke_action_summary.jsonl \
  --policy-eval results/tables/smoke_policy_eval.jsonl \
  --out results/tables/smoke_queue_replay.jsonl \
  --num-jobs 100 \
  --seeds 1 2 3 \
  --job-mix-id smoke \
  --trace-id-prefix smoke \
  --allow-unconstrained
```

## 4. Full GPU Core Suite

`scripts/run_official_core.sh` runs the GPU-side phases:

1. correctness gate
2. cold-core sweep
3. warm compile-cache sensitivity
4. graph-reuse sensitivity

Smoke mode:

```bash
SHORTJOB_RUN_MODE=smoke bash scripts/run_official_core.sh
```

Official mode:

```bash
SHORTJOB_RUN_MODE=official bash scripts/run_official_core.sh
```

In this repository, official denotes the full experiment configuration used for the paper-facing result package, as opposed to the reduced smoke configuration.

Useful environment variables:

- `DEVICE_LABEL`: optional human-readable GPU label.
- `DEVICE_TAG`: output tag; defaults to `official_<device>_<YYYYMMDD>`.
- `CUDA_DEVICE`: default `cuda`.
- `SHORTJOB_DRY_RUN=1`: print commands without executing.
- `SHORTJOB_SKIP_UV_SYNC=1`: skip `uv sync`.
- `SHORTJOB_RUN_GATE`, `SHORTJOB_RUN_COLD`, `SHORTJOB_RUN_WARM`, `SHORTJOB_RUN_GRAPH`: phase toggles.

Raw outputs are written under:

```text
data/raw/<DEVICE_TAG>/
data/env/<DEVICE_TAG>_env.txt
```

Preserve the environment snapshot alongside each run. Measurements collected on different GPU or software configurations should remain explicitly labeled.

## 5. Released Results and Raw Measurements

The repository includes the paper-facing derived result package:

```text
results/tables/paper_official_20260627/
results/figures/paper_official_20260627/
```

These directories contain the derived tables and figures used to support the paper’s reported results. They can be inspected without re-running the GPU measurement suite.

The full per-condition GPU measurement logs are distributed separately as the following GitHub Release asset:

```text
shortjob_paper_official_20260627_raw.tar.gz
```

Its expected SHA-256 checksum and extraction commands are documented in the repository `README.md`. The archive contains the raw JSONL measurements and environment snapshots used to generate the bundled result package.

After verifying and extracting the archive, place its contents under:

```text
data/raw/
data/env/
```

Use the artifact manifest and analysis commands in this repository to regenerate the derived summaries and figures from the released measurements.