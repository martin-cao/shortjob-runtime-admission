# src Experiment Code Guide

`src/` is the code entry point for the public artifact. It implements the minimum reproducible experiment pipeline:

```text
raw GPU runs -> action summary -> policy evaluation -> worker queue replay -> figures / admission analyzer
```

This is not a general benchmark platform or a production scheduler. The main scope is single-machine, single-GPU, `cold_total` timing, and worker-local admission decisions.

## Environment

From the repository root:

```bash
uv sync
```

Check the runtime environment:

```bash
uv run python src/check_runtime_env.py
```

On a machine without CUDA, this check may report that the GPU is unavailable. CPU-side tests can still run, but GPU measurements cannot.

## Common Entry Points

The following help commands do not launch GPU workloads:

```bash
uv run python src/run_shortjob_isolated.py --help
uv run python src/check_correctness.py --help
uv run python src/summarize_shortjob.py --help
uv run python src/evaluate_baselines.py --help
uv run python src/replay_worker_queue.py --help
uv run python src/analyze_admission.py --help
uv run python src/run_graph_reuse_sensitivity.py --help
```

Installed console scripts are also available:

```bash
uv run shortjob-runner --help
uv run shortjob-summarize --help
uv run shortjob-evaluate --help
uv run shortjob-admit --help
uv run shortjob-check-env
uv run shortjob-check-correctness --help
uv run shortjob-profile --help
```

## Workloads

The workload suite covers:

- Pilot mechanisms: `fixed_shape_infer_small`, `short_train_small`, `dynamic_shape_infer_small`, `prefill_toy`, `decode_toy`
- CV proxy: `cv_online_infer`
- Short Transformer: `short_text_transformer_padded`, `short_text_transformer_dynamic`
- Decode proxy: `llm_decode_proxy`
- Recommendation proxy: `dlrm_recommendation`
- Irregular graph proxy: `gnn_irregular`
- CPU-bound boundary: `rl_policy_infer`
- Mechanism/fusion proxy: `synthetic_kernel_chain`

## Actions

The action suite covers:

- Baselines: `eager`, `eager_inference_mode`, `best_eager`, `cpu_eager`
- Compile modes: `compile_only`, `compile_reduce_overhead`, `compile_max_autotune`
- CUDA Graphs: `graphs_only`, `graphs_input_copy`, `compile_plus_graphs`, `compile_reduce_overhead_plus_graphs`
- Precision modes: `amp_fp16`, `amp_bf16`, `tf32_eager`
- Batching variants: `micro_batch_2`, `micro_batch_4`

Adding an action does not mean it should be treated as a symmetric positive candidate. Paper-facing analysis must go through the correctness gate, eligibility matrix, negative-result table, and queue replay.

## Minimal GPU Smoke Run

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

`run_shortjob_isolated.py` launches each condition in a separate Python process to reduce cross-condition contamination from `torch.compile`, Inductor, and CUDA Graph state.

## Derived Analysis

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

## Public Result Package

The public summary/figure package is:

```text
results/tables/paper_official_20260627/
results/figures/paper_official_20260627/
```

These are derived public tables and figures generated from complete GPU raw JSONL runs. The raw runs are larger and are not committed to Git.
