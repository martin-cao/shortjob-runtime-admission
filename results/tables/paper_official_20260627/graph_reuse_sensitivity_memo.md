# Graph-Reuse Sensitivity Memo

Status: Derived / bounded repeated-shape sensitivity

## 0. Interpretation Boundary

This analysis measures a bounded setting where same-shape jobs in one worker process capture a CUDA Graph once and replay it many times.

It is not a production graph cache and does not implement persistent captured-graph artifacts across processes.

It should be described as bounded graph-reuse or repeated-shape sensitivity, not as full cache-state robustness.

## 1. Device Summary

See `graph_reuse_sensitivity_device_summary.csv`.

## 2. Paper Wording Guidance

- Acceptable: repeated same-shape jobs can change graph amortization because capture is paid once per group in this bounded worker-local sensitivity.
- Do not write: the paper implements a production graph cache or persistent captured-graph store.
- Oracle counts, graph wins, and speedups include only actions that pass the corresponding device correctness gate.
- If `graphs_input_copy` or the `short_train_small` graph path does not pass the gate, do not treat it as positive graph evidence.
