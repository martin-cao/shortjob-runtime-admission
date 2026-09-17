# Admit or Avoid: Characterizing Runtime Optimization Decisions for Short-Lived GPU Jobs

This repository is the public companion artifact for a study of **worker-local runtime optimization admission** for short-lived GPU AI jobs.

The question is not "which PyTorch switch is faster overall." The question is:

> Can a GPU worker decide whether to admit or avoid heavyweight runtime optimizations such as `torch.compile` and CUDA Graphs before paying their upfront compile or capture cost?

The artifact is a measurement and decision-support codebase. `torch.compile` and CUDA Graphs are existing runtime mechanisms in the action space; this repository does not introduce a new compiler backend or a production scheduler.

## Repository Contents

- `src/`: experiment runners, correctness gate, summarization, policy evaluation, worker-queue replay, admission analyzer, and plotting scripts.
- `tests/`: CPU-side unit and regression tests for policies, queue replay, correctness-gate summarization, rank stability, and graph-reuse summaries.
- `scripts/`: shell entry points for the GPU core experiment suite.
- `docker/gpu/`: optional GPU container definition.
- `results/tables/paper_official_20260627/`: public paper-facing derived summary tables.
- `results/figures/paper_official_20260627/`: public paper-facing figures.

## Environment

Use `uv` from the repository root:

```bash
uv sync
```

The Linux x86_64 GPU environment is pinned to:

- Python `3.11.x`
- `torch==2.12.0+cu126`
- PyTorch CUDA runtime `12.6`

CPU-side tests and CLI help work without a GPU. Actual GPU measurements require a Linux machine with an NVIDIA GPU.

## Quick Checks

```bash
uv run python src/check_runtime_env.py
uv run python src/run_shortjob_isolated.py --help
uv run python src/summarize_shortjob.py --help
uv run python src/evaluate_baselines.py --help
uv run python src/replay_worker_queue.py --help
uv run python src/analyze_admission.py --help
```

CPU-side tests:

```bash
uv run python -m unittest discover -s tests
```

## Minimal GPU Smoke Run

The following command launches CUDA work:

```bash
uv run python src/run_shortjob_isolated.py \
  --workload fixed_shape_infer_small \
  --action eager graphs_only \
  --batch-size 16 \
  --num-steps 50 \
  --repeats 1 \
  --seed 42 \
  --out data/raw/smoke_shortjob.jsonl
```

Then generate an action summary:

```bash
uv run python src/summarize_shortjob.py \
  --input data/raw/smoke_shortjob.jsonl \
  --out results/tables/smoke_action_summary.jsonl \
  --allow-unconstrained
```

For the full workflow, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Core Pipeline

```text
raw GPU runs
  -> action summary
  -> policy evaluation
  -> worker queue replay
  -> figures / admission analyzer
```

The main evaluation scope is single-machine, single-GPU, `cold_total` timing, and worker-local decisions. CUDA Graphs are candidates only when capture is feasible and shapes are stable. In the current measured short-job matrix, `compile_only` is treated primarily as a negative control and wrong-admit stress test.

## Public Result Artifacts

The bundled public result package is:

```text
results/tables/paper_official_20260627/
results/figures/paper_official_20260627/
```

These files are the derived summaries and figures that the paper audits directly,
so the tables and figures can be reviewed without re-running any GPU workload.

## Data Availability (raw measurements)

The full per-condition raw GPU JSONL is **not** committed to Git. It is published as a GitHub Release asset:

- **Archive:** `shortjob_paper_official_20260627_raw.tar.gz`
- **SHA-256:** `2d50112588a16ed992e0b9ce6797544f9a86cf7b338d3508518c7d6437cb5e63`
- **Download:** see this repository's [Releases](https://github.com/martin-cao/shortjob-runtime-admission/releases) page.

It contains the four device runs that back `paper_official_20260627`
(`official_4060_20260625`, `official_v100_20260625`, `official_a100_20260626`,
`official_h100_20260627`), each with `cold_core` / `warm_reuse_core`
(+ `_prime`) / `graph_reuse_core` / `correctness_gate` JSONL plus the per-run
environment snapshot, and an inner `CHECKSUMS.sha256`.

To replay from raw, verify and unpack into the local data tree:

```bash
shasum -a 256 -c <(echo "2d50112588a16ed992e0b9ce6797544f9a86cf7b338d3508518c7d6437cb5e63  shortjob_paper_official_20260627_raw.tar.gz")
tar -xzf shortjob_paper_official_20260627_raw.tar.gz
cp -R shortjob_paper_official_20260627_raw/raw/* data/raw/
cp -R shortjob_paper_official_20260627_raw/env/* data/env/
```

Then follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) to regenerate the
derived tables and figures.

## Citation

If you use this repository, please cite the associated paper, *"Admit or Avoid:
Characterizing Runtime Optimization Decisions for Short-Lived GPU Jobs."*
Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). Result tables
and figures are provided for research reproducibility and review.
