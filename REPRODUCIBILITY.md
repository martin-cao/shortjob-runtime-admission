# Reproduction guide

Run all commands from the repository root. This repository contains code, scripts, dependencies, and reproduction instructions. Release assets contain experimental data and results.

## Release asset and external working directory

Download `shortjob_camera_ready_v0.2.0_data.tar.gz` and `SHA256SUMS` into `../shortjob-artifacts/`. Run `shasum -a 256 -c SHA256SUMS` from that external directory, then extract the archive:

```sh
mkdir -p ../shortjob-artifacts/release-input
tar -xzf ../shortjob-artifacts/shortjob_camera_ready_v0.2.0_data.tar.gz -C ../shortjob-artifacts/release-input
```

| Path within the archive | Contents |
|---|---|
| `data/camera_ready_v0.2.0/` | 49,134 formal records, environments, input asset identities, and paper reference values |
| `results/camera_ready_v0.2.0/` | Reconstructed results, CSV tables, queue traces, and verification records |
| `results/figures/camera_ready_v0.2.0/` | Graph boundary and real-model effect figures |
| `results/historical_20260627/` | Historical derived results with their original version scope |
| `RELEASE_MANIFEST.json` | Data and result hashes, record count, and corresponding measurement-code identity |
| `CHECKSUMS.sha256` | Per-file checksums within the archive |

Keep measurements, extracted data, build outputs, reconstructed results, and review records outside the code repository. `measurement_source/` retains the original measurement source bytes and identity manifest, including the relative paths used during measurement.

## CPU environment and release verification

Use Python 3.11:

```sh
python -m pip install -r requirements-analysis.txt
python src/verify_release.py --release-root ../shortjob-artifacts/release-input/shortjob_camera_ready_v0.2.0 --out ../shortjob-artifacts/release-validation.json
```

Verification checks the complete file set, SHA-256 hashes, experiment matrices, task uniqueness, gate associations, precision settings, record counts, frozen measurement code, and verification status of packaged results. Input files are read-only; write verification reports outside the extracted release directory.

## Complete reconstruction and independent verification

```sh
python src/reproduce_camera_ready.py --data-root ../shortjob-artifacts/release-input/shortjob_camera_ready_v0.2.0/data/camera_ready_v0.2.0 --out ../shortjob-artifacts/reproduced-results
python src/verify_camera_ready_results.py --data-root ../shortjob-artifacts/release-input/shortjob_camera_ready_v0.2.0/data/camera_ready_v0.2.0 --results ../shortjob-artifacts/reproduced-results
python scripts/verify_submitted_paper.py --data-root ../shortjob-artifacts/release-input/shortjob_camera_ready_v0.2.0/data/camera_ready_v0.2.0 --results ../shortjob-artifacts/reproduced-results
python src/verify_reproduced_results.py --reference ../shortjob-artifacts/release-input/shortjob_camera_ready_v0.2.0/results/camera_ready_v0.2.0 --reproduced ../shortjob-artifacts/reproduced-results --out ../shortjob-artifacts/result-comparison.json
python src/plot_camera_ready.py --results ../shortjob-artifacts/reproduced-results --out ../shortjob-artifacts/reproduced-figures
```

Reconstruction covers cold execution, warm compilation caches, eight-job graph reuse, policies, queue replay, repeat omission, and prediction diagnostics for the four original controlled configurations and the full independent V100 replication. It also reconstructs all nine real-model sessions.

`reference/expected.json` is consulted only after the measurement-derived calculations are complete. Floating-point comparisons use a tolerance of `1e-11 * max(1, abs(expected))`. The independent verifier reads measurements directly and recomputes medians, valid policy regret, FCFS recurrences, and real-model statistics. The submitted-paper check covers Tables I-VI, Figure 2 cells, and the Figure 3 MobileNet estimates; complete plotting values are checked during reconstruction.

The result-comparison command checks every JSON, JSONL, and CSV file in two independent output directories using SHA-256. Figures are written to a new directory; PDF creation-time metadata is handled separately. Reconstruction and figure output directories must not already exist.

## Repeating GPU measurements

GPU measurement requires Linux, a compatible NVIDIA driver, and one visible GPU. The runner records the actual device, driver, and resource configuration:

```sh
uv sync --frozen --extra real-models
uv run --frozen python src/check_runtime_env.py
env CUDA_VISIBLE_DEVICES=0 SHORTJOB_RUN_MODE=official SHORTJOB_DRY_RUN=1 SHORTJOB_NOTIFY_DISABLED=1 DEVICE_TAG=controlled_session01 bash scripts/run_official_core.sh
```

Inspect the command plan, then use `SHORTJOB_DRY_RUN=0` for measurement. The suite runs reference-cell correctness checks, the cold matrix, separate cache priming, warm-cache measurements, and graph reuse. Give each new campaign a new session name and separate caches.

See [SUPPLEMENTAL_EXPERIMENTS.md](SUPPLEMENTAL_EXPERIMENTS.md) for real-model and expanded correctness experiments. Prepare images and weights from the release asset manifest while preserving precision, input refresh, effective-update counts, and timing boundaries.

## Mapping to the paper

| Paper content | Output relative to `results/camera_ready_v0.2.0/` |
|---|---|
| Table I: reference-cell screening | Original gates and `data_validation.json` |
| Table II: expanded correctness | Supplemental task records and `real/` |
| Figure 2: graph boundaries | `fig_oracle_action_mix` in `plotted_values.json` |
| Table III: boundary stability and reuse | `tables/boundaries.csv` and `controlled/` |
| Figure 3: real-model effects and envelopes | `real_model_results` in `plotted_values.json` and `tables/real_models.csv` |
| Table IV: policy regret | `tables/policies.csv` |
| Table V: uniform heavy-load queue | `queue_statistics` in `controlled/<device>/analysis.json` |
| Table VI: job-mix sensitivity | `tables/queue_improvements.csv` |
| Independent V100 replication | `replication/` and `replication_comparison.json` |

See [METRIC_DEFINITIONS.md](METRIC_DEFINITIONS.md) for timing contracts, eligible candidates, and metric denominators. CPU reconstruction uses completed measurements; new GPU campaigns must be analyzed using their own observed values.
