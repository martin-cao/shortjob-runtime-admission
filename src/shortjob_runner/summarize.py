from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from shortjob_runner.io import read_jsonl, write_jsonl

# Student-t two-tailed 95 % critical values by degrees of freedom.
# df = n - 1; falls back to 1.960 (normal) for df >= 30.
_T_CRIT_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    20: 2.086,
    29: 2.045,
}


def _t_crit_95(n: int) -> float:
    df = n - 1
    if df < 1:
        return float("inf")
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    # Linear search for nearest larger df with a known entry, else use 1.960.
    for threshold in sorted(_T_CRIT_95):
        if df <= threshold:
            return _T_CRIT_95[threshold]
    return 1.960


def _is_correctness_failure(row: dict[str, Any]) -> bool:
    """True iff the gate row is a numerical/state CORRECTNESS failure.

    A correctness failure means the action ran to completion but produced output
    that disagreed with eager.  This is distinct from runtime infeasibility
    (precheck rejection or an execution exception), which is captured separately
    by ``action_feasible`` in the measurement data and must NOT land in the
    correctness blocklist (reviewer item 5).

    Schema-aware with legacy fallback:
    - new gates carry ``failure_stage`` / ``correctness_failed``;
    - legacy gates only carry ``status`` / ``action_feasible``, where a
      correctness failure is exactly ``status == "failed"`` with the action still
      feasible (it ran), versus an execution failure which has
      ``action_feasible == False``.
    """
    if "failure_stage" in row:
        return row.get("failure_stage") == "correctness"
    if "correctness_failed" in row:
        return bool(row.get("correctness_failed"))
    return row.get("status") == "failed" and row.get("action_feasible") is True


def load_correctness_blocklist(path: Path) -> frozenset[tuple[str, str]]:
    """Return a frozenset of (workload_id, action) pairs that fail correctness.

    The gate file is a JSONL produced by check_correctness.py.  Only correctness
    failures (ran but produced wrong output) are blocked from oracle / policy
    candidate selection.  Runtime execution failures and precheck-infeasible
    actions are NOT blocked here; their infeasibility is represented separately by
    ``action_feasible`` in measurement data.  See ``_is_correctness_failure``.
    """
    blocked: set[tuple[str, str]] = set()
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        if _is_correctness_failure(row):
            wid = row.get("workload_id")
            act = row.get("action")
            if wid and act:
                blocked.add((wid, act))
    return frozenset(blocked)


GROUP_FIELDS = (
    "workload_id",
    "task_family",
    "workload_class",
    "shape_stability",
    "application",
    "batch_size",
    "num_steps",
    "input_mode",
    "time_accounting_mode",
    "cache_state",
    "execution_device",
)

ACTION_TIE_ORDER = (
    "eager",
    "best_eager",
    "graphs_only",
    "graphs_input_copy",
    "compile_only",
    "compile_reduce_overhead",
    "compile_plus_graphs",
    "compile_reduce_overhead_plus_graphs",
)
ACTION_TIE_RANK = {action: rank for rank, action in enumerate(ACTION_TIE_ORDER)}
EAGER_FAMILY_ACTIONS = {"eager", "best_eager", "eager_inference_mode"}


def median_or_none(values: list[float]) -> float | None:
    return round(float(statistics.median(values)), 6) if values else None


def mean_or_none(values: list[float]) -> float | None:
    return round(float(statistics.mean(values)), 6) if values else None


def stdev_or_none(values: list[float]) -> float | None:
    return round(float(statistics.stdev(values)), 6) if len(values) >= 2 else None


def quantile_or_none(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 6)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    value = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(float(value), 6)


def ci95(values: list[float]) -> tuple[float | None, float | None]:
    """Two-sided 95 % confidence interval using Student-t critical value.

    With n < 30 the normal approximation (1.96) under-covers substantially.
    For the typical n=3 case, df=2 gives t=4.303 rather than 1.96.
    """
    n = len(values)
    if n < 2:
        return None, None
    mean_value = statistics.mean(values)
    stderr = statistics.stdev(values) / (n ** 0.5)
    half_width = _t_crit_95(n) * stderr
    return round(float(mean_value - half_width), 6), round(float(mean_value + half_width), 6)


def cv_or_none(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean_value = statistics.mean(values)
    if mean_value == 0:
        return None
    return round(float(statistics.stdev(values) / mean_value), 6)


def group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in GROUP_FIELDS)


def group_dict(key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(GROUP_FIELDS, key, strict=True))


def numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row.get(field) is not None]


def is_valid_action_row(
    row: dict[str, Any] | None,
    *,
    blocked_actions: frozenset[tuple[str, str]] = frozenset(),
) -> bool:
    if row is None:
        return False
    workload_id = row.get("workload_id", "")
    action = row.get("action", "")
    return bool(
        row.get("action_feasible")
        and row.get("median_total_runtime_s") is not None
        and (workload_id, action) not in blocked_actions
    )


def eager_baseline_row(
    action_rows: list[dict[str, Any]],
    *,
    blocked_actions: frozenset[tuple[str, str]] = frozenset(),
) -> dict[str, Any] | None:
    """Return the per-condition eager-family reference baseline.

    Paper-facing metrics use ``best_eager`` when that action is legal for the
    workload/condition, otherwise literal ``eager``.  This keeps gain,
    wrong-admit, missed-opportunity, waste, and queue fallback on the same
    denominator while preserving a separate literal-eager field for audits.
    """
    by_action = {row.get("action"): row for row in action_rows}
    for action in ("best_eager", "eager"):
        row = by_action.get(action)
        if is_valid_action_row(row, blocked_actions=blocked_actions):
            return row
    return None


def is_precheck_infeasible(row: dict[str, Any]) -> bool:
    reason = row.get("failure_reason") or ""
    return bool(
        row.get("precheck_failed")
        or row.get("status") == "infeasible"
        or "unstable_shape_not_graph_capturable" in reason
        or "graph_capture_not_feasible" in reason
    )


def summarize_action(key: tuple[Any, ...], action: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed" and row.get("action_feasible") is True]
    failed = [row for row in rows if row not in completed]
    precheck_infeasible = [row for row in failed if is_precheck_infeasible(row)]
    runtime_failed = [row for row in failed if row not in precheck_infeasible]
    total_times = numeric_values(completed, "total_runtime_s")
    measured_times = numeric_values(completed, "measured_runtime_s")
    ci_low, ci_high = ci95(total_times)
    q25 = quantile_or_none(total_times, 0.25)
    q75 = quantile_or_none(total_times, 0.75)
    summary = {
        **group_dict(key),
        "action": action,
        "n_repeats": len(rows),
        "n_completed": len(completed),
        "n_failed": len(failed),
        "n_infeasible_precheck": len(precheck_infeasible),
        "n_runtime_failed": len(runtime_failed),
        "failure_rate": round(len(failed) / len(rows), 6) if rows else None,
        "infeasible_precheck_rate": round(len(precheck_infeasible) / len(rows), 6) if rows else None,
        "runtime_failure_rate": round(len(runtime_failed) / len(rows), 6) if rows else None,
        "action_feasible": bool(completed),
        "mean_total_runtime_s": mean_or_none(total_times),
        "std_total_runtime_s": stdev_or_none(total_times),
        "cv_total_runtime": cv_or_none(total_times),
        "p50_total_runtime_s": quantile_or_none(total_times, 0.50),
        "p90_total_runtime_s": quantile_or_none(total_times, 0.90),
        "p95_total_runtime_s": quantile_or_none(total_times, 0.95),
        "p99_total_runtime_s": quantile_or_none(total_times, 0.99),
        "iqr_total_runtime_s": round(q75 - q25, 6) if q25 is not None and q75 is not None else None,
        "ci95_total_runtime_low_s": ci_low,
        "ci95_total_runtime_high_s": ci_high,
        "median_total_runtime_s": median_or_none(total_times),
        "median_cold_total_runtime_s": median_or_none(numeric_values(completed, "cold_total_runtime_s")),
        "median_warm_cached_total_runtime_s": median_or_none(numeric_values(completed, "warm_cached_total_runtime_s")),
        "median_warmup_runtime_s": median_or_none(numeric_values(completed, "warmup_runtime_s")),
        "median_measured_runtime_s": median_or_none(measured_times),
        "p95_measured_runtime_s": quantile_or_none(measured_times, 0.95),
        "median_device_event_runtime_s": median_or_none(numeric_values(completed, "device_event_runtime_s")),
        "median_compile_call_overhead_s": median_or_none(numeric_values(completed, "compile_call_overhead_s")),
        "median_compile_overhead_s": median_or_none(numeric_values(completed, "compile_overhead_s")),
        "median_graph_capture_overhead_s": median_or_none(numeric_values(completed, "graph_capture_overhead_s")),
        "median_graph_warmup_runtime_s": median_or_none(numeric_values(completed, "graph_warmup_runtime_s")),
        "median_graph_instantiate_runtime_s": median_or_none(numeric_values(completed, "graph_instantiate_runtime_s")),
        "median_input_copy_overhead_s": median_or_none(numeric_values(completed, "input_copy_overhead_s")),
        "median_input_copy_bytes": median_or_none(numeric_values(completed, "input_copy_bytes")),
        "median_avg_step_time_ms": median_or_none(numeric_values(completed, "avg_step_time_ms")),
        "median_step_time_ms": median_or_none(numeric_values(completed, "median_step_time_ms")),
        "median_p50_step_time_ms": median_or_none(numeric_values(completed, "p50_step_time_ms")),
        "median_p90_step_time_ms": median_or_none(numeric_values(completed, "p90_step_time_ms")),
        "median_p95_step_time_ms": median_or_none(numeric_values(completed, "p95_step_time_ms")),
        "median_p99_step_time_ms": median_or_none(numeric_values(completed, "p99_step_time_ms")),
        "median_throughput_per_s": median_or_none(numeric_values(completed, "throughput_per_s")),
        "median_first_k_steps": median_or_none(numeric_values(completed, "first_k_steps")),
        "median_first_k_eager_step_time_ms": median_or_none(
            numeric_values(completed, "first_k_eager_step_time_ms")
        ),
        "median_first_k_step_time_variance_ms": median_or_none(
            numeric_values(completed, "first_k_step_time_variance_ms")
        ),
        "median_peak_gpu_memory_mb": median_or_none(numeric_values(completed, "peak_gpu_memory_mb")),
        "median_model_init_runtime_s": median_or_none(numeric_values(completed, "model_init_runtime_s")),
        "median_cuda_context_init_s": median_or_none(numeric_values(completed, "cuda_context_init_s")),
        "compile_modes": sorted({row.get("compile_mode") for row in completed if row.get("compile_mode")}),
        "precision_modes": sorted({row.get("precision_mode") for row in completed if row.get("precision_mode")}),
        "failure_reasons": sorted({row.get("failure_reason") for row in failed if row.get("failure_reason")}),
    }
    return summary


def attach_oracle_and_regret(
    rows: list[dict[str, Any]],
    blocked_actions: frozenset[tuple[str, str]] = frozenset(),
) -> list[dict[str, Any]]:
    """Attach oracle_action, regret, and per-action wrong_enable/missed_opportunity.

    ``blocked_actions`` is a frozenset of (workload_id, action) pairs that
    fail the correctness gate.  Blocked actions are excluded from oracle
    selection so a fast-but-incorrect action cannot become the reference
    baseline.  They still appear in the output with regret computed relative
    to the correctness-constrained oracle.
    """
    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[group_key(row)].append(row)

    output: list[dict[str, Any]] = []
    for _, action_rows in by_group.items():
        workload_id = action_rows[0].get("workload_id", "") if action_rows else ""
        feasible = [
            row
            for row in action_rows
            if row["action_feasible"]
            and row["median_total_runtime_s"] is not None
            and (workload_id, row["action"]) not in blocked_actions
        ]
        oracle = (
            min(
                feasible,
                key=lambda row: (
                    float(row["median_total_runtime_s"]),
                    ACTION_TIE_RANK.get(str(row["action"]), len(ACTION_TIE_RANK)),
                ),
            )
            if feasible
            else None
        )
        eager = next((row for row in action_rows if row["action"] == "eager"), None)
        literal_eager_time = eager.get("median_total_runtime_s") if eager else None
        baseline = eager_baseline_row(action_rows, blocked_actions=blocked_actions)
        baseline_action = baseline.get("action") if baseline else None
        baseline_time = baseline.get("median_total_runtime_s") if baseline else None
        non_eager_oracle = bool(oracle and oracle["action"] not in EAGER_FAMILY_ACTIONS)
        opportunity_s = (
            round(baseline_time - oracle["median_total_runtime_s"], 6)
            if non_eager_oracle
            and baseline_time is not None
            and oracle
            and oracle.get("median_total_runtime_s") is not None
            else None
        )

        for row in action_rows:
            enriched = dict(row)
            current_time = row["median_total_runtime_s"]
            enriched["oracle_action"] = oracle["action"] if oracle else None
            enriched["oracle_total_runtime_s"] = oracle["median_total_runtime_s"] if oracle else None
            enriched["is_best_action"] = bool(oracle and row["action"] == oracle["action"])
            enriched["eager_baseline_action"] = baseline_action
            enriched["eager_baseline_total_runtime_s"] = baseline_time
            enriched["literal_eager_total_runtime_s"] = literal_eager_time
            enriched["eager_baseline_is_best_eager"] = baseline_action == "best_eager"
            enriched["relative_gain_vs_eager"] = (
                round(baseline_time / current_time, 6)
                if baseline_time is not None and current_time not in (None, 0)
                else None
            )
            enriched["relative_gain_vs_eager_baseline"] = enriched["relative_gain_vs_eager"]
            enriched["relative_gain_vs_literal_eager"] = (
                round(literal_eager_time / current_time, 6)
                if literal_eager_time is not None and current_time not in (None, 0)
                else None
            )
            enriched["regret_vs_oracle_s"] = (
                round(current_time - oracle["median_total_runtime_s"], 6)
                if oracle and current_time is not None
                else None
            )
            enriched["normalized_regret_vs_oracle"] = (
                round((current_time - oracle["median_total_runtime_s"]) / oracle["median_total_runtime_s"], 6)
                if oracle and current_time is not None and oracle["median_total_runtime_s"] != 0
                else None
            )
            enriched["excess_vs_eager_baseline_s"] = (
                round(current_time - baseline_time, 6)
                if baseline_time is not None and current_time is not None
                else None
            )
            enriched["normalized_excess_vs_eager_baseline"] = (
                round((current_time - baseline_time) / baseline_time, 6)
                if baseline_time not in (None, 0) and current_time is not None
                else None
            )
            enriched["missed_benefit_vs_eager_baseline_s"] = opportunity_s
            enriched["wrong_enable"] = bool(
                row["action"] not in EAGER_FAMILY_ACTIONS
                and baseline_time is not None
                and current_time is not None
                and current_time > baseline_time
            )
            enriched["missed_opportunity"] = bool(
                row["action"] in EAGER_FAMILY_ACTIONS
                and oracle is not None
                and oracle["action"] not in EAGER_FAMILY_ACTIONS
                and baseline_time is not None
                and current_time is not None
                and current_time > oracle["median_total_runtime_s"]
            )
            output.append(enriched)
    return output


def summarize_raw(
    rows: list[dict[str, Any]],
    blocked_actions: frozenset[tuple[str, str]] = frozenset(),
) -> list[dict[str, Any]]:
    buckets: dict[tuple[tuple[Any, ...], str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(group_key(row), row["action"])].append(row)

    action_summaries = [
        summarize_action(key, action, action_rows)
        for (key, action), action_rows in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    return attach_oracle_and_regret(action_summaries, blocked_actions=blocked_actions)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize raw shortjob JSONL and compute oracle/regret metrics")
    parser.add_argument("--input", type=Path, default=Path("data/raw/shortjob_pilot.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/tables/shortjob_action_summary.jsonl"))
    parser.add_argument(
        "--correctness-gate",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "This device's correctness gate JSONL from check_correctness.py. "
            "Correctness failures are excluded from oracle selection. Required "
            "unless --allow-unconstrained is set."
        ),
    )
    parser.add_argument(
        "--allow-unconstrained",
        action="store_true",
        help="Explicitly run the speed-only oracle with no correctness gate.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.correctness_gate is None and not args.allow_unconstrained:
        raise SystemExit(
            "error: --correctness-gate PATH is required (this device's own gate) "
            "or pass --allow-unconstrained to run the speed-only oracle explicitly."
        )
    blocked: frozenset[tuple[str, str]] = frozenset()
    if args.correctness_gate is not None:
        blocked = load_correctness_blocklist(args.correctness_gate)
    write_jsonl(args.out, summarize_raw(read_jsonl(args.input), blocked_actions=blocked))
    return 0
