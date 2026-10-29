# Measured environments

This table describes the formal real-model measurement sessions. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for general commands. New runs record the device actually used.

| Sessions | GPU | NVIDIA driver |
|---|---|---|
| laptop_a, laptop_b | RTX 4060 Laptop | 610.57.04 |
| v100_a, v100_b, v100_c | V100 | 580.159.04 |
| a100_a | A100 | 580.159.04 |
| a100_b | A100 | 580.126.16 |
| h100_a, h100_b | H100 | 580.126.09 |

Each session's `environment.json` in the release data records the GPU name, memory capacity, compute capability, Python, PyTorch, CUDA runtime, TorchVision, CPU model, visible CPU count, affinity, and available resource-quota records. Driver versions above were transcribed from NVIDIA query records captured during measurement.

Historical controlled environment records are retained in `shortjob_paper_official_20260627_raw/env/` and the original measurement rows. Sessions sharing a GPU model remain separate; timing comparisons use each session's own matched baseline.

Frozen measurement source is stored in this repository's `measurement_source/` and verified against the release stage manifests. Public entry-point adaptations for paths and device discovery have their own code identity, recorded by each new campaign.
