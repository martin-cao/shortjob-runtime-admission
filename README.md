# Admit or Avoid: Characterizing Runtime Optimization Decisions for Short-Lived GPU Jobs

This repository provides the experiment, analysis, plotting, and verification code for the IEEE ICPADS 2026 paper. The study examines worker-local runtime optimization admission on single-machine, single-GPU configurations.

## Code and experimental data

| Location | Contents |
|---|---|
| `src/` | Experiment runners, correctness screening, policies, queue replay, analysis, and plotting |
| `scripts/` | GPU experiment entry points and verification of values displayed in the submitted paper |
| `tests/` | CPU test code |
| `measurement_source/` | Frozen source files, dependencies, and identity manifest from the formal measurements |
| `docker/` | GPU environment definition |
| Root documents and dependency files | Protocols, metric definitions, reproduction instructions, license, and citation metadata |
| [GitHub Releases](https://github.com/martin-cao/shortjob-runtime-admission/releases) | Measurements, environment records, derived results, CSV tables, figures, and checksums |

The data and results are distributed together as `shortjob_camera_ready_v0.2.0_data.tar.gz`. Download this asset and its accompanying `SHA256SUMS`. The archive root is `shortjob_camera_ready_v0.2.0/`, containing `data/camera_ready_v0.2.0/`, `results/`, `RELEASE_MANIFEST.json`, and checksum files.

## Complete CPU reproduction

Use Python 3.11. Download the release assets into `../shortjob-artifacts/`, outside this repository, and run `shasum -a 256 -c SHA256SUMS` from that directory. From the repository root, run:

```sh
python -m pip install -r requirements-analysis.txt
mkdir -p ../shortjob-artifacts/release-input
tar -xzf ../shortjob-artifacts/shortjob_camera_ready_v0.2.0_data.tar.gz -C ../shortjob-artifacts/release-input
python src/verify_release.py --release-root ../shortjob-artifacts/release-input/shortjob_camera_ready_v0.2.0 --out ../shortjob-artifacts/release-validation.json
python src/reproduce_camera_ready.py --data-root ../shortjob-artifacts/release-input/shortjob_camera_ready_v0.2.0/data/camera_ready_v0.2.0 --out ../shortjob-artifacts/reproduced-results
python src/verify_camera_ready_results.py --data-root ../shortjob-artifacts/release-input/shortjob_camera_ready_v0.2.0/data/camera_ready_v0.2.0 --results ../shortjob-artifacts/reproduced-results
python scripts/verify_submitted_paper.py --data-root ../shortjob-artifacts/release-input/shortjob_camera_ready_v0.2.0/data/camera_ready_v0.2.0 --results ../shortjob-artifacts/reproduced-results
python src/verify_reproduced_results.py --reference ../shortjob-artifacts/release-input/shortjob_camera_ready_v0.2.0/results/camera_ready_v0.2.0 --reproduced ../shortjob-artifacts/reproduced-results --out ../shortjob-artifacts/result-comparison.json
python src/plot_camera_ready.py --results ../shortjob-artifacts/reproduced-results --out ../shortjob-artifacts/reproduced-figures
```

These commands reconstruct the four original controlled configurations, all nine real-model sessions, and the complete independent V100 replication from recorded measurements. Reconstruction requires no CUDA, PyTorch, model weights, or network access after dependencies and data have been obtained. Output directories must not already exist.

## GPU experiments

The formal measurements use Linux, one visible NVIDIA GPU, Python 3.11, PyTorch 2.12.0, and CUDA runtime 12.6. The consumer GPU configuration is an **NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)**. Historical task identifiers containing `4060` retain their original spelling.

```sh
uv sync --frozen --extra real-models
uv run --frozen python src/check_runtime_env.py
env CUDA_VISIBLE_DEVICES=0 SHORTJOB_RUN_MODE=official SHORTJOB_DRY_RUN=1 SHORTJOB_NOTIFY_DISABLED=1 DEVICE_TAG=controlled_session01 bash scripts/run_official_core.sh
```

`SHORTJOB_DRY_RUN=1` prints the complete controlled-suite command plan. Use `SHORTJOB_DRY_RUN=0` for execution. Measurements, environment snapshots, and compilation caches default to `../shortjob-artifacts/`, outside the repository; set `ARTIFACT_DIR` to choose another external directory.

See the [reproduction guide](REPRODUCIBILITY.md), [supplemental protocol](SUPPLEMENTAL_EXPERIMENTS.md), and [metric definitions](METRIC_DEFINITIONS.md) for details.

## Citation and license

The associated paper is *Admit or Avoid: Characterizing Runtime Optimization Decisions for Short-Lived GPU Jobs*. Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff). Code is distributed under the [MIT License](LICENSE).
