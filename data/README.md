# data

This directory is used for local experiment outputs.

- `raw/`: per-condition JSONL emitted by GPU runners.
- `env/`: environment snapshots such as `nvidia-smi`, host information, and runtime-check output.

The public Git repository does not include the full raw GPU runs. To reproduce paper-facing summary tables, rerun the GPU core workflow in `docs/REPRODUCIBILITY.md` or download raw JSONL from the separately published release/data artifact once available.
