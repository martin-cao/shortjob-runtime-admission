#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible shim.
#
# The GPU core suite is now run by run_official_core.sh, which runs the
# correctness gate + cold-core + warm compile-cache + graph-reuse phases (all
# resumable). This shim is retained so that older baked bootstrap images that
# invoke `scripts/run_official_cold_core.sh` still launch the full suite without
# needing a rebuild. New callers should use run_official_core.sh directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_official_core.sh" "$@"
