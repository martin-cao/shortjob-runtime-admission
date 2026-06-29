# Action Validity and Correctness Gate

This artifact is generated from the saved four-device `correctness_gate.jsonl` files by `src/summarize_action_validity_reasons.py`. It does not rerun GPU workloads or change performance numbers.

## Checker Definition

- Reference action: literal `eager`, not `a_base`.
- Official gate condition: one workload-action row at `batch_size=4`, `num_steps=3`, `seed=42` per device.
- Inputs and initial model state are reproducible because the checker seeds model/input construction before building both the eager reference and candidate workload.
- For `graphs_input_copy`, refreshed request inputs use an explicitly seeded generator so the eager reference and graph candidate see the same refreshed-input sequence.
- Graph-action references run the same logical trajectory as the candidate: three graph warm-up steps, one capture-equivalent step, then the requested replay steps.
- Inference workloads compare `output` from `correctness_state()`.
- The training workload compares `loss` and model `param:*` tensors from `correctness_state()`; gradients and optimizer state are not separately saved by the current checker.
- Pass uses `torch.testing.assert_close(candidate, eager_reference, rtol=1e-4, atol=1e-5)` for every compared state tensor.

## Outcome Definitions

- `pass`: the action executes and all compared output/state tensors match the eager reference within tolerance.
- `infeasible`: precheck rejects the action before execution, for example because inference mode is illegal for training or the workload declares graph capture ineligible.
- `fail`: execution/capture raises after admission, or execution completes but the output/state equivalence check fails.
- Table I counts workload-action validation cells. The current official gate has one tested condition per workload-action cell, so no multi-condition vote is used.

## Four-Device Summary

- Each device has 76 pass, 13 infeasible, and 15 fail cells out of 104.
- The pass/infeasible/fail pattern is identical on RTX 4060, V100, A100, and H100.

## Fail Distribution

- `graphs_only`: 2 fail cells.
- `graphs_input_copy`: 2 fail cells.
- `compile_plus_graphs`: 1 fail cells.
- `compile_reduce_overhead_plus_graphs`: 10 fail cells.

Failure categories on the representative device:
- `padded_text_explicit_graph_capture_failure`: 2 cells.
- `reduce_overhead_plus_explicit_graph_capture_failure`: 10 cells.
- `short_train_loss_mismatch`: 3 cells.

The 15 fail cells are concentrated in three observed categories rather than 15 unrelated defects. `compile_reduce_overhead_plus_graphs` is an explicit manual capture wrapped around a `torch.compile(mode='reduce-overhead')` callable; PyTorch documents `reduce-overhead` as a CUDA-graphs-based mode for reducing Python overhead, so this row is treated as a nested/manual-plus-framework capture anti-pattern rather than generic CUDA Graph fragility. The short-train loss mismatch is a strict-tolerance state rejection; the current gate does not prove semantic training divergence.

## Failure Mechanism Classes

| Class | Cells | Actions | Mechanism / cause | Divergence class | Treatment |
|---|---:|---|---|---|---|
| `reduce_overhead_plus_explicit_graph_capture_failure` | 10 | `compile_reduce_overhead_plus_graphs` | explicit torch.cuda.CUDAGraph capture wraps a torch.compile(mode='reduce-overhead') callable; PyTorch reduce-overhead itself uses CUDA graphs to reduce Python overhead, so the observed errors are a nested/manual-plus-framework capture anti-pattern | feasibility rejection, not numerical divergence | avoid-target / negative-control stress case; excluded from positive graph evidence |
| `padded_text_explicit_graph_capture_failure` | 2 | `graphs_only; graphs_input_copy` | manual graph capture around the padded Transformer path hits stream-capture-invalidated runtime errors despite fixed tensor shapes | capture feasibility rejection, not completed-output divergence | graph eligibility caveat for this workload family |
| `short_train_loss_mismatch` | 3 | `graphs_only; graphs_input_copy; compile_plus_graphs` | stateful training graph replay compares final loss and parameters after matched warm-up, capture, and replay trajectory; only the loss scalar trips the strict gate | strict-tolerance state rejection; possible tolerance artifact, not proved semantic failure | exclude short-train graph paths from positive graph evidence |

## Failed Workload-Action Cells

| Workload | Action | Stage | Category | Reason |
|---|---|---|---|---|
| `fixed_shape_infer_small` | `compile_reduce_overhead_plus_graphs` | `execution` | `reduce_overhead_plus_explicit_graph_capture_failure` | CUDA error: operation failed due to a previous error during capture |
| `short_train_small` | `graphs_only` | `correctness` | `short_train_loss_mismatch` | Scalars are not close! |
| `short_train_small` | `graphs_input_copy` | `correctness` | `short_train_loss_mismatch` | Scalars are not close! |
| `short_train_small` | `compile_plus_graphs` | `correctness` | `short_train_loss_mismatch` | Scalars are not close! |
| `short_train_small` | `compile_reduce_overhead_plus_graphs` | `execution` | `reduce_overhead_plus_explicit_graph_capture_failure` | CUDA error: operation failed due to a previous error during capture |
| `prefill_toy` | `compile_reduce_overhead_plus_graphs` | `execution` | `reduce_overhead_plus_explicit_graph_capture_failure` | CUDA error: operation failed due to a previous error during capture |
| `decode_toy` | `compile_reduce_overhead_plus_graphs` | `execution` | `reduce_overhead_plus_explicit_graph_capture_failure` | CUDA error: operation failed due to a previous error during capture |
| `cv_online_infer` | `compile_reduce_overhead_plus_graphs` | `execution` | `reduce_overhead_plus_explicit_graph_capture_failure` | CUDA error: operation failed due to a previous error during capture |
| `short_text_transformer_padded` | `graphs_only` | `execution` | `padded_text_explicit_graph_capture_failure` | CUDA error: operation failed due to a previous error during capture |
| `short_text_transformer_padded` | `graphs_input_copy` | `execution` | `padded_text_explicit_graph_capture_failure` | CUDA error: operation failed due to a previous error during capture |
| `short_text_transformer_padded` | `compile_reduce_overhead_plus_graphs` | `execution` | `reduce_overhead_plus_explicit_graph_capture_failure` | CUDA error: operation failed due to a previous error during capture |
| `llm_decode_proxy` | `compile_reduce_overhead_plus_graphs` | `execution` | `reduce_overhead_plus_explicit_graph_capture_failure` | CUDA error: operation failed due to a previous error during capture |
| `dlrm_recommendation` | `compile_reduce_overhead_plus_graphs` | `execution` | `reduce_overhead_plus_explicit_graph_capture_failure` | CUDA error: operation failed due to a previous error during capture |
| `rl_policy_infer` | `compile_reduce_overhead_plus_graphs` | `execution` | `reduce_overhead_plus_explicit_graph_capture_failure` | CUDA error: operation failed due to a previous error during capture |
| `synthetic_kernel_chain` | `compile_reduce_overhead_plus_graphs` | `execution` | `reduce_overhead_plus_explicit_graph_capture_failure` | CUDA error: operation failed due to a previous error during capture |

## Infeasible Categories

- `precheck_declared_graph_ineligible`: 8 cells.
- `precheck_training_inference_mode`: 1 cells.
- `precheck_unstable_shape`: 4 cells.

## Evaluation Treatment

- `eager` and legal `best_eager` are eager-family baselines.
- `graphs_only` and `graphs_input_copy` are eligible candidates only for workload families that pass the gate and runtime feasibility checks.
- `compile_only` and `compile_reduce_overhead` pass correctness but are retained as performance stress baselines because their startup cost is not amortized in the cold-total matrix.
- `compile_plus_graphs` is a diagnostic combined action.
- `compile_reduce_overhead_plus_graphs` is excluded from positive evidence.
