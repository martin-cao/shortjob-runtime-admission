#!/usr/bin/env bash
set -euo pipefail

# All-in-one GPU "core" experiment runner. Runs every part of the study that
# must execute inside the GPU environment, in one shot:
#
#   1. correctness gate      -> <tag>/correctness_gate.jsonl   (characterization)
#   2. cold-core sweep       -> <tag>/cold_core.jsonl
#   3. warm compile-cache    -> <tag>/warm_reuse_core.jsonl (+ _prime)
#   4. graph-reuse / shape   -> <tag>/graph_reuse_core.jsonl
#
# CPU-only statistics / derived analysis (summaries, policy eval, queue replay,
# figures, bootstrap CIs) are intentionally NOT run here; they belong to the
# no-GPU pipeline and can run anywhere afterwards.
#
# Resumable: every measurement phase uses --skip-existing, and the gate is
# regenerated fresh each run. Re-running after an interruption (the 4060
# breakpoint case) skips completed conditions and continues where it stopped.
#
# Non-fatal phases: a phase that ends with failing conditions records what it
# can, logs a warning, and the next phase still runs. The script exits non-zero
# at the end if any phase failed, so the bootstrap still reports a problem.
#
# Works on single-GPU Linux machines; device label and tag auto-detect from
# nvidia-smi when available.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SHORTJOB_ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

source "${SCRIPT_DIR}/shortjob_shell_lib.sh"

usage() {
  cat <<'USAGE'
Usage:
  SHORTJOB_RUN_MODE=official bash scripts/run_official_core.sh
  SHORTJOB_RUN_MODE=smoke    bash scripts/run_official_core.sh

Runs the full GPU core suite: correctness gate + cold-core + warm compile-cache
+ graph-reuse. Resumable (--skip-existing) and safe to re-run.

Key environment variables:
  SHORTJOB_RUN_MODE     official (default) or smoke.
  DEVICE_LABEL          Optional; auto-detected from nvidia-smi.
  DEVICE_TAG            Optional; defaults to official_<device>_<YYYYMMDD>.
  CUDA_DEVICE           Default: cuda.
  SHORTJOB_DRY_RUN      If 1, print commands without executing.
  SHORTJOB_SKIP_UV_SYNC If 1, skip uv sync.
  SHORTJOB_HEARTBEAT_S  Heartbeat notification interval (default 1800s).

Per-phase toggles (all default 1; set 0 to skip a phase):
  SHORTJOB_RUN_GATE  SHORTJOB_RUN_COLD  SHORTJOB_RUN_WARM  SHORTJOB_RUN_GRAPH

Per-phase repeat overrides:
  SHORTJOB_COLD_REPEATS  SHORTJOB_WARM_REPEATS  SHORTJOB_GRAPH_REPEATS
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

cd "$ROOT_DIR"

RUN_MODE="${SHORTJOB_RUN_MODE:-official}"
if [[ "$RUN_MODE" != "official" && "$RUN_MODE" != "smoke" ]]; then
  echo "SHORTJOB_RUN_MODE must be official or smoke, got ${RUN_MODE}" >&2
  exit 2
fi

export DEVICE_LABEL="${DEVICE_LABEL:-$(shortjob_infer_device_label)}"
export DEVICE_TAG="${DEVICE_TAG:-$(shortjob_default_device_tag "$DEVICE_LABEL" "$RUN_MODE")}"

CUDA_DEVICE="${CUDA_DEVICE:-cuda}"
RAW_DIR="${RAW_DIR:-data/raw/${DEVICE_TAG}}"
GATE="${GATE:-${RAW_DIR}/correctness_gate.jsonl}"
COLD_RAW="${COLD_RAW:-${RAW_DIR}/cold_core.jsonl}"
WARM_RAW="${WARM_RAW:-${RAW_DIR}/warm_reuse_core.jsonl}"
GRAPH_RAW="${GRAPH_RAW:-${RAW_DIR}/graph_reuse_core.jsonl}"
ENV_OUT="${ENV_OUT:-data/env/${DEVICE_TAG}_env.txt}"
CACHE_BASE="${CACHE_BASE:-/tmp/shortjob_runtime_cache/${DEVICE_TAG}}"
DRY_RUN="${SHORTJOB_DRY_RUN:-0}"
HEARTBEAT_S="${SHORTJOB_HEARTBEAT_S:-1800}"

RUN_GATE="${SHORTJOB_RUN_GATE:-1}"
RUN_COLD="${SHORTJOB_RUN_COLD:-1}"
RUN_WARM="${SHORTJOB_RUN_WARM:-1}"
RUN_GRAPH="${SHORTJOB_RUN_GRAPH:-1}"

# --- experiment matrices -----------------------------------------------------
# Cold-core: full action space; the gate characterizes this same set.
COLD_ACTIONS=(
  eager
  best_eager
  compile_only
  compile_reduce_overhead
  graphs_only
  graphs_input_copy
  compile_plus_graphs
  compile_reduce_overhead_plus_graphs
)
# Warm compile-cache and graph-reuse use the focused paper-facing subsets.
WARM_WORKLOADS=(cv_online_infer llm_decode_proxy dynamic_shape_infer_small short_train_small)
WARM_ACTIONS=(eager best_eager graphs_only compile_only compile_reduce_overhead)
GRAPH_WORKLOADS=(cv_online_infer llm_decode_proxy dynamic_shape_infer_small short_train_small)
GRAPH_ACTIONS=(eager best_eager graphs_only graphs_input_copy)

if [[ "$RUN_MODE" == "smoke" ]]; then
  COLD_WORKLOADS=(cv_online_infer short_train_small)
  COLD_ACTIONS=(eager graphs_only compile_only)
  COLD_BATCH=(4); COLD_STEPS=(10 100); COLD_REPEATS_DEFAULT=2

  WARM_WORKLOADS=(cv_online_infer short_train_small)
  WARM_ACTIONS=(eager graphs_only compile_only)
  WARM_BATCH=(4); WARM_STEPS=(50); WARM_REPEATS_DEFAULT=2

  GRAPH_WORKLOADS=(cv_online_infer)
  GRAPH_ACTIONS=(eager graphs_only graphs_input_copy)
  GRAPH_BATCH=(4); GRAPH_STEPS=(50); GRAPH_REPEATS_DEFAULT=2; GRAPH_REUSE_JOBS=4
else
  COLD_WORKLOADS=(all)
  COLD_BATCH=(1 4 16 32); COLD_STEPS=(10 50 100 500); COLD_REPEATS_DEFAULT=5

  WARM_BATCH=(4 16); WARM_STEPS=(50 100 500); WARM_REPEATS_DEFAULT=3

  GRAPH_BATCH=(4 16); GRAPH_STEPS=(50 100 500); GRAPH_REPEATS_DEFAULT=3; GRAPH_REUSE_JOBS=8
fi

COLD_REPEATS="${SHORTJOB_COLD_REPEATS:-$COLD_REPEATS_DEFAULT}"
WARM_REPEATS="${SHORTJOB_WARM_REPEATS:-$WARM_REPEATS_DEFAULT}"
GRAPH_REPEATS="${SHORTJOB_GRAPH_REPEATS:-$GRAPH_REPEATS_DEFAULT}"

# --- helpers -----------------------------------------------------------------
OVERALL_STATUS=0

run_cmd() {
  shortjob_log "+ $*"
  [[ "$DRY_RUN" == "1" ]] && return 0
  "$@"
}

run_shell() {
  shortjob_log "+ $*"
  [[ "$DRY_RUN" == "1" ]] && return 0
  bash -lc "$*"
}

# Run a long phase with a heartbeat; failure is recorded but non-fatal so the
# remaining phases still run. Returns 0 always (status tracked in OVERALL_STATUS).
run_phase() {
  local label="$1"; shift
  shortjob_log "+ $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  if shortjob_run_with_heartbeat "$HEARTBEAT_S" "$label" -- "$@"; then
    return 0
  fi
  OVERALL_STATUS=1
  shortjob_notify "warn" "${label} ended with failures" "device=${DEVICE_LABEL} tag=${DEVICE_TAG}; resumable, re-run to fill gaps"
  return 0
}

assert_cuda_visible() {
  [[ "$CUDA_DEVICE" != cuda* ]] && return 0
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    shortjob_notify "critical" "GPU preflight failed" "nvidia-smi not found; pass GPU devices into the container before running experiments"
    echo "GPU preflight failed: nvidia-smi not found. For Podman, try --device nvidia.com/gpu=all --security-opt=label=disable." >&2
    return 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    shortjob_notify "critical" "GPU preflight failed" "nvidia-smi exists but cannot access the NVIDIA driver"
    echo "GPU preflight failed: nvidia-smi exists but cannot access the NVIDIA driver." >&2
    return 1
  fi
}

on_exit() {
  local status=$?
  if [[ $status -eq 0 ]]; then
    shortjob_notify "success" "shortjob core ${DEVICE_TAG} complete" "device=${DEVICE_LABEL} mode=${RUN_MODE}"
  else
    shortjob_notify "failed" "shortjob core ${DEVICE_TAG} failed" "device=${DEVICE_LABEL} mode=${RUN_MODE} exit=${status}"
  fi
}
trap on_exit EXIT

# --- preflight ---------------------------------------------------------------
shortjob_log "device_label=${DEVICE_LABEL}"
shortjob_log "device_tag=${DEVICE_TAG}"
shortjob_log "run_mode=${RUN_MODE}"
shortjob_log "phases: gate=${RUN_GATE} cold=${RUN_COLD} warm=${RUN_WARM} graph=${RUN_GRAPH}"
shortjob_notify "progress" "core experiment started" "device=${DEVICE_LABEL} mode=${RUN_MODE}"

run_cmd mkdir -p "$RAW_DIR" "$(dirname "$ENV_OUT")"

assert_cuda_visible

if [[ "${SHORTJOB_SKIP_UV_SYNC:-0}" != "1" ]]; then
  shortjob_notify "progress" "uv sync" "starting"
  run_cmd uv sync
fi

shortjob_notify "progress" "environment check" "starting"
run_shell "( date; hostname; uname -a; nvidia-smi; uv run python src/check_runtime_env.py ) | tee '${ENV_OUT}'"

# --- phase 1: correctness gate (regenerated fresh; non-fatal) -----------------
if [[ "$RUN_GATE" == "1" ]]; then
  shortjob_notify "progress" "correctness gate" "starting"
  # Remove first so a resumed run does not append duplicate gate rows.
  run_cmd rm -f "$GATE"
  if ! run_cmd uv run python src/check_correctness.py \
      --workload "${COLD_WORKLOADS[@]}" \
      --action "${COLD_ACTIONS[@]}" \
      --batch-size 4 \
      --num-steps 3 \
      --out "$GATE"; then
    shortjob_notify "warn" "correctness gate recorded failing actions" "gate=${GATE} (expected: amp/tf32/input_copy/reduce_overhead+graphs)"
  fi
fi

# --- phase 2: cold-core ------------------------------------------------------
if [[ "$RUN_COLD" == "1" ]]; then
  shortjob_notify "progress" "cold-core run" "starting repeats=${COLD_REPEATS} heartbeat=${HEARTBEAT_S}s"
  run_phase "cold-core ${DEVICE_TAG}" \
    uv run python src/run_shortjob_isolated.py \
    --workload "${COLD_WORKLOADS[@]}" \
    --action "${COLD_ACTIONS[@]}" \
    --batch-size "${COLD_BATCH[@]}" \
    --num-steps "${COLD_STEPS[@]}" \
    --first-k-steps 5 \
    --repeats "$COLD_REPEATS" \
    --seed 42 \
    --device "$CUDA_DEVICE" \
    --time-accounting-mode cold_total \
    --cache-state cold_cache \
    --cache-dir-root "${CACHE_BASE}_cold_core" \
    --skip-existing \
    --out "$COLD_RAW"
fi

# --- phase 3: warm compile-cache ---------------------------------------------
if [[ "$RUN_WARM" == "1" ]]; then
  shortjob_notify "progress" "warm compile-cache run" "starting repeats=${WARM_REPEATS}"
  run_phase "warm compile-cache ${DEVICE_TAG}" \
    uv run python src/run_shortjob_isolated.py \
    --workload "${WARM_WORKLOADS[@]}" \
    --action "${WARM_ACTIONS[@]}" \
    --batch-size "${WARM_BATCH[@]}" \
    --num-steps "${WARM_STEPS[@]}" \
    --first-k-steps 5 \
    --repeats "$WARM_REPEATS" \
    --seed 42 \
    --device "$CUDA_DEVICE" \
    --time-accounting-mode warm_cached_total \
    --cache-state warm_compile_cache \
    --cache-dir-root "${CACHE_BASE}_warm_reuse_core" \
    --prime-compile-cache \
    --skip-existing \
    --out "$WARM_RAW"
fi

# --- phase 4: graph-reuse / repeated-shape -----------------------------------
if [[ "$RUN_GRAPH" == "1" ]]; then
  shortjob_notify "progress" "graph-reuse run" "starting repeats=${GRAPH_REPEATS} reuse_jobs=${GRAPH_REUSE_JOBS}"
  run_phase "graph-reuse ${DEVICE_TAG}" \
    uv run python src/run_graph_reuse_sensitivity.py \
    --device-label "$DEVICE_LABEL" \
    --workload "${GRAPH_WORKLOADS[@]}" \
    --action "${GRAPH_ACTIONS[@]}" \
    --batch-size "${GRAPH_BATCH[@]}" \
    --num-steps "${GRAPH_STEPS[@]}" \
    --reuse-jobs "$GRAPH_REUSE_JOBS" \
    --repeats "$GRAPH_REPEATS" \
    --seed 42 \
    --device "$CUDA_DEVICE" \
    --skip-existing \
    --out "$GRAPH_RAW"
fi

if [[ "$OVERALL_STATUS" -eq 0 ]]; then
  shortjob_notify "progress" "core experiment complete" "GPU core done; statistics/derived run separately"
  shortjob_log "GPU core complete: cold=${COLD_RAW} warm=${WARM_RAW} graph=${GRAPH_RAW}"
else
  shortjob_log "GPU core finished with failing conditions in one or more phases; re-run to fill gaps (resumable)"
fi

exit "$OVERALL_STATUS"
