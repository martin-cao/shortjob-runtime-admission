#!/usr/bin/env bash

shortjob_log() {
  printf '[shortjob] %s\n' "$*" >&2
}

shortjob_urlencode() {
  local value="${1:-}"
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$value"
  elif command -v python >/dev/null 2>&1; then
    python -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$value"
  else
    printf '%s' "$value"
  fi
}

shortjob_notify() {
  # No-op unless SHORTJOB_NOTIFY_URL is set by the caller.
  local status="${1:-info}"
  local summary="${2:-shortjob}"
  local detail="${3:-}"
  local host
  local level="timeSensitive"
  local title body group
  host="$(hostname 2>/dev/null || printf 'unknown-host')"

  if [[ "${SHORTJOB_NOTIFY_DISABLED:-0}" == "1" ]]; then
    return 0
  fi

  case "$status" in
    failed|failure|error|critical)
      level="critical"
      ;;
  esac

  title="${DEVICE_TAG:-shortjob}"
  if [[ -n "$detail" ]]; then
    body="${summary}: ${detail}"
  else
    body="$summary"
  fi
  group="${DEVICE_LABEL:-shortjob}"

  if [[ -n "${SHORTJOB_NOTIFY_URL:-}" ]]; then
    curl -fsS --max-time "${SHORTJOB_NOTIFY_TIMEOUT_S:-10}" \
      --get \
      --data-urlencode "status=${status}" \
      --data-urlencode "title=${title}" \
      --data-urlencode "body=${body}" \
      --data-urlencode "level=${level}" \
      --data-urlencode "group=${group}" \
      --data-urlencode "host=${host}" \
      --data-urlencode "device_tag=${DEVICE_TAG:-}" \
      --data-urlencode "device_label=${DEVICE_LABEL:-}" \
      "$SHORTJOB_NOTIFY_URL" >/dev/null || true
  fi
}

shortjob_run_with_heartbeat() {
  # Run a long command in the background and emit a periodic notify
  # heartbeat while it is active. Usage:
  #   shortjob_run_with_heartbeat <interval_seconds> <label> [--] command args...
  # The command's exit status is preserved and returned. Heartbeat is a pure
  # shell wrapper; it never alters the wrapped command or its semantics.
  local interval="${1:-1800}"; shift || true
  local label="${1:-long-running step}"; shift || true
  if [[ "${1:-}" == "--" ]]; then
    shift || true
  fi

  if (( $# == 0 )); then
    shortjob_log "shortjob_run_with_heartbeat: no command given"
    return 2
  fi

  "$@" &
  local cmd_pid=$!

  (
    local elapsed=0
    while kill -0 "$cmd_pid" 2>/dev/null; do
      sleep "$interval" || exit 0
      kill -0 "$cmd_pid" 2>/dev/null || exit 0
      elapsed=$(( elapsed + interval ))
      shortjob_notify "progress" "${label} heartbeat" "still running after ~$(( elapsed / 60 )) min"
    done
  ) &
  local hb_pid=$!

  local status=0
  wait "$cmd_pid" || status=$?

  kill "$hb_pid" 2>/dev/null || true
  wait "$hb_pid" 2>/dev/null || true

  return "$status"
}

shortjob_infer_device_label() {
  if [[ -n "${DEVICE_LABEL:-}" ]]; then
    printf '%s\n' "$DEVICE_LABEL"
    return 0
  fi

  local gpu_name=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '\r' || true)"
  fi

  case "$gpu_name" in
    *H100*) printf 'H100\n' ;;
    *A100*) printf 'A100\n' ;;
    *A40*)  printf 'A40\n' ;;
    *V100*) printf 'V100\n' ;;
    *3080*) printf 'RTX 3080\n' ;;
    *4060*) printf 'RTX 4060\n' ;;
    "") printf 'unknown_gpu\n' ;;
    *) printf '%s\n' "$gpu_name" ;;
  esac
}

shortjob_device_slug() {
  local label="${1:-unknown_gpu}"
  case "$label" in
    H100*) printf 'h100\n' ;;
    A100*) printf 'a100\n' ;;
    A40*)  printf 'a40\n' ;;
    V100*) printf 'v100\n' ;;
    *3080*) printf '3080\n' ;;
    *4060*) printf '4060\n' ;;
    *)
      printf '%s\n' "$label" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
      ;;
  esac
}

shortjob_default_device_tag() {
  local label="${1:-$(shortjob_infer_device_label)}"
  local mode="${2:-official}"
  local date_stamp="${SHORTJOB_DATE_STAMP:-$(date +%Y%m%d)}"
  local slug suffix
  slug="$(shortjob_device_slug "$label")"
  suffix="${SHORTJOB_RUN_ID_SUFFIX:-}"

  if [[ -n "$suffix" ]]; then
    printf 'official_%s_%s_%s\n' "$slug" "$date_stamp" "$suffix"
  elif [[ "$mode" == "smoke" ]]; then
    printf 'official_%s_%s_smoke\n' "$slug" "$date_stamp"
  else
    printf 'official_%s_%s\n' "$slug" "$date_stamp"
  fi
}
