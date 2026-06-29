# Paper-Facing Result Artifact Manifest (4-device, n=5)

This directory contains the distilled tables/figures used by the current paper draft.
It supersedes `paper_official_20260616`, which was the earlier two-device n=3 package.
All four devices in this package use the unified rerun protocol: **5 cold-core repeats**,
identical 13-workload / 8-action / 4-batch / 4-step matrix, and the corrected
correctness gate. The Python/PyTorch runtime is aligned across devices
(Python 3.11.15, torch 2.12.0+cu126, torch CUDA runtime 12.6), while driver
build and host/container stack are recorded per device rather than treated as
identical.

Headline cross-device finding: no compilation action is the oracle under the
measured finite-job objective on any device (cold or warm finite-job settings),
while the CUDA-Graph regime is **non-monotonic** across measured host/device packages:
graph-oracle conditions are 62 (4060), 60 (V100), 78 (A100), 49 (H100), and the
repeated-shape reuse speedup is 2.08x / 2.09x / 3.15x / 1.11x. A100 is widest,
H100 narrowest. This is per-device-calibration evidence, not an isolated
GPU-hardware mechanism claim.

## Source Raw Data

- RTX 4060 Laptop cold/warm/graph-reuse: `data/raw/official_4060_20260625/`
- V100 cold/warm/graph-reuse:             `data/raw/official_v100_20260625/`
- A100-SXM4-80GB cold/warm/graph-reuse:   `data/raw/official_a100_20260626/`
- H100 80GB HBM3 cold/warm/graph-reuse:   `data/raw/official_h100_20260627/`

Each device ships `cold_core.jsonl` (8320 rows = 1664 conditions x 5 repeats),
`warm_reuse_core.jsonl` (360 rows) + `warm_reuse_core_prime.jsonl` (48 rows),
`graph_reuse_core.jsonl` (288 rows), and `correctness_gate.jsonl` (104 rows).

## Completeness (cold-core)

- `completeness_4060_cold_core.json`, `completeness_v100_cold_core.json`,
  `completeness_a100_cold_core.json`, `completeness_h100_cold_core.json`
- All four observe 8320/8320 expected rows, 0 missing/duplicate/unexpected,
  6320 completed, 1040 precheck-infeasible, 960 runtime (graph-capture) failures.
- The correctness gate is identical across the four devices
  (76 passed / 15 failed / 13 infeasible of 104 audit rows).

## Objective / Horizon Semantics

- Cold finite-job evaluation: includes model/context initialization, compile,
  graph capture, warm-up, input-copy setup, and measured execution. It answers
  whether the worker should admit an action before paying startup cost.
- Warm finite-job sensitivity: primes compile caches first, but still measures
  finite end-to-end job time for 10/50/100-step jobs. It tests whether cache
  warming changes the admission ranking.
- Steady-state benchmark: warms runtime/cache state and excludes or separately
  reports one-time setup to measure sustained throughput. This is not the
  primary objective of the paper package.

## Distilled Paper Tables / Figures

Same file set and semantics as `paper_official_20260616`, regenerated for four devices:
`core_oracle_counts.csv`, `oracle_action_mix_by_workload.csv`,
`action_validity_audit.csv`, `action_validity_reasons.csv`,
`action_failure_mechanisms.csv`, `ACTION_VALIDITY.md`,
`device_runtime_context_table.csv`,
`selected_runtime_boundaries*.csv`,
`rank_stability_*.csv`, `graph_reuse_sensitivity_*.csv`,
`warm_compile_cache_*.csv`, `policy_regret_and_queue_summary.csv`,
`policy_harder_split_summary.csv`, `queue_replay_*_heavy.jsonl` +
`queue_replay_job_mix_sensitivity_heavy_summary.csv`.

Figures are in `results/figures/paper_official_20260627/`
(`fig_oracle_action_mix`, `fig_selected_runtime_boundaries`,
`fig_policy_regret_and_queue`).

## Regeneration

```bash
# Per-device derived tables (cold + warm), for each TAG/LABEL, use the
# GPU core outputs described above: cold_core.jsonl, warm_reuse_core.jsonl,
# graph_reuse_core.jsonl, and correctness_gate.jsonl.

# Four-device distilled figures + core/oracle/policy/queue tables:
uv run python src/plot_paper_official_results.py

# Correctness/action validity reason artifacts:
uv run python src/summarize_action_validity_reasons.py

# Warm compile-cache sensitivity (4 devices):
uv run python src/summarize_warm_cache_sensitivity.py

# Rank stability + graph reuse (4 devices): pass the per-device action summaries,
# graph-reuse raw files, and correctness gates for the *_20260625 / *_20260626 /
# *_20260627 tags.
```
