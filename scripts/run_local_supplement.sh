#!/usr/bin/env bash
# Execute on the selected Linux GPU host; no SSH or rental actions are performed.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_local_supplement.sh {4060|v100} {core|characterize|real} [runner options]" >&2
  exit 2
fi
platform=$1
suite=$2
shift 2
case "$platform" in 4060|v100) ;; *) echo "Local profile must be 4060 or v100" >&2; exit 2 ;; esac
case "$suite" in core|characterize|real) ;; *) echo "Invalid suite" >&2; exit 2 ;; esac
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
entry=03_src/run_supplement.py
[[ -f "$entry" ]] || entry=src/run_supplement.py
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec uv run --frozen --extra real-models python "$entry" run --platform "$platform" --suite "$suite" "$@"
