# Single-GPU supplemental protocol

The common entry point runs expanded correctness checks, eager characterization, and real-model timings. Select one GPU with `CUDA_VISIBLE_DEVICES`; `--platform auto` records its actual model. Manifests record the models, inputs, precision settings, task matrix, and code identity.

## Environment and task plans

```sh
uv sync --frozen --extra real-models
uv run --frozen python src/check_runtime_env.py
uv run --frozen python src/run_supplement.py run --suite core
uv run --frozen python src/run_supplement.py run --suite real \
  --precision-policy strict_fp32 \
  --models resnet50 vit_b_16 mobilenet_v3_large \
  --batch 1 --lengths 10 50 100 500 --repeats 5
```

These commands print task plans by default. Add `--execute` to launch CUDA subprocesses. The core suite contains 175 checks: 169 expected passes and six dynamic/irregular-shape precheck rejections. Each real-model session contains 72 correctness gates and 180 formal timings.

## Fixed images and weights

Use the 64 Imagenette validation images specified by `assets/manifest.json` and `assets/dataset-selection.json` in the externally extracted data. Preserve their relative paths and verify every image's SHA-256. Put exactly this subset in `../shortjob-artifacts/prepared/selected-images/`, outside the repository.

```sh
uv run --frozen --extra real-models python src/run_supplement.py prepare \
  --images ../shortjob-artifacts/prepared/selected-images \
  --source 'EXACT dataset_source STRING FROM assets/manifest.json' \
  --models resnet50 vit_b_16 mobilenet_v3_large \
  --out ../shortjob-artifacts/prepared/imagenette64
```

Use `IMAGENET1K_V2` for ResNet-50 and MobileNetV3-Large, and `IMAGENET1K_V1` for ViT-B/16. Keep the locked TorchVision version and the transforms defined by those weights. The preparation command orders images by relative path; its input directory must contain only the specified 64 images.

Compare the generated manifest's SHA-256 with the release asset manifest. Prepare images, weights, and preprocessed input pools before formal timing; measurement reads local assets.

## Expanded correctness

```sh
env CUDA_VISIBLE_DEVICES=0 uv run --frozen --extra real-models \
  python src/run_supplement.py run --platform auto --suite core \
  --precision-policy legacy_cudnn_tf32 \
  --run-dir ../shortjob-artifacts/data/raw/session01_core --execute
```

Checks cover inference trajectories, exact effective training updates, independent jobs, and eight-job reuse. Training restores model state after warm-up/capture, then compares loss, parameters, gradients, and buffers after the requested effective updates. Relative and absolute tolerances are `1e-4` and `1e-5`.

`legacy_cudnn_tf32` preserves the controlled protocol's settings: CUDA matmul TF32 disabled and cuDNN TF32 enabled. Real-model experiments use the separate `strict_fp32` protocol, with both TF32 switches disabled. Protocol identity is included in task IDs and gate matching.

## Eager characterization

```sh
env CUDA_VISIBLE_DEVICES=0 uv run --frozen --extra real-models \
  python src/run_supplement.py run --platform auto --suite characterize \
  --precision-policy strict_fp32 \
  --models resnet50 vit_b_16 mobilenet_v3_large \
  --repeats 1 \
  --assets ../shortjob-artifacts/prepared/imagenette64 \
  --run-dir ../shortjob-artifacts/data/raw/session01_characterize --execute
```

This stage records 24 agreement checks and 12 duration probes, separately from formal performance repeats.

## Formal real-model timing

```sh
env CUDA_VISIBLE_DEVICES=0 uv run --frozen --extra real-models \
  python src/run_supplement.py run --platform auto --suite real \
  --precision-policy strict_fp32 \
  --models resnet50 vit_b_16 mobilenet_v3_large \
  --batch 1 --lengths 10 50 100 500 --repeats 5 \
  --assets ../shortjob-artifacts/prepared/imagenette64 \
  --run-dir ../shortjob-artifacts/data/raw/session01_real --execute
```

The three actions are eager, graphs with per-step input refresh, and default compilation, with matched inference mode. Each condition passes correctness checks with seeds 17 and 42 before five separate-process, cold-cache timings using seed 42.

Each job cycles through the same input pool. Timing starts after CUDA context creation and includes model/weight initialization, input-pool loading, optimization setup, and exactly N input-refresh/execution steps. First-use compilation may occur during execution. Network acquisition, Python import, offline preprocessing, asset verification, and post-timing correctness checks are outside this boundary.

## Results, failures, and resume

Each session records `manifest.json`, environment information, `rows/<task_id>.json`, logs, and `summary.json`. Formal records preserve protocol-defined success, failure, and infeasibility outcomes. Performance repeats require the corresponding gates.

Repeat the same command to resume unfinished tasks; existing terminal records are retained. Start a new session directory when code, environment, assets, or matrix changes. The default timeout is 900 seconds per task. Timeout records support failure analysis; formal performance statistics use timings that satisfy the applicable protocol.

After execution, check completion counts, status distributions, and task-to-gate associations in `summary.json`. Store and analyze sessions separately, preserving their timing contracts and precision policies.
