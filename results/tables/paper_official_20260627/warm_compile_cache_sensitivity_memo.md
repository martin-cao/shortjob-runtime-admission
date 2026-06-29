# Warm Compile-Cache Sensitivity Memo

Status: Completed / derived from measured warm-cache runs

## 0. Conclusion

The warm compile-cache mini-run is complete on all four devices. Prime rows and measured rows are separated, and the measured run has no runtime failures.

In the current four-workload mini matrix, warm compile-cache substantially reduces compile-family total time, but it does not make `compile_only` or `compile_reduce_overhead` the oracle action and does not make the compile family beat `eager`.

Therefore, the paper can treat warm-cache as a bounded sensitivity rather than a missing blocker. It supports `compile` remaining an avoid-target / negative control in the tested warm compile-cache regime, but it does not establish a fully cache-state-robust admission boundary.

## 1. Device Summary

See `warm_compile_cache_sensitivity_summary.csv`.

## 2. Paper Wording Guidance

- Acceptable: warm compile-cache sensitivity across RTX 4060 Laptop, V100, A100, and H100 reduces compile-family cost but does not change the tested avoid-target conclusion.
- Do not write: cache-state-robust admission boundary has been fully established.
- Do not treat `short_train_small` graph-oracle rows as uncaveated positive graph evidence unless the correctness gate passes separately.
