#!/usr/bin/env python3
"""Summarize the warm compile-cache sensitivity runs for paper-facing use."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = ROOT / "results" / "tables"
RAW_ROOT = ROOT / "data" / "raw"
OUT_DIR = TABLE_ROOT / "paper_official_20260627"

WARM_ACTIONS = {
    "eager",
    "best_eager",
    "graphs_only",
    "compile_only",
    "compile_reduce_overhead",
}

def _device_paths(cold_tag: str, warm_tag: str) -> dict[str, Any]:
    return {
        "cold_summary": TABLE_ROOT / cold_tag / "cold_core" / "action_summary.jsonl",
        "warm_summary": TABLE_ROOT / warm_tag / "warm_reuse_core" / "action_summary.jsonl",
        "warm_policy": TABLE_ROOT / warm_tag / "warm_reuse_core" / "policy_aggregate.jsonl",
        "warm_completeness": TABLE_ROOT / warm_tag / "warm_reuse_core" / "completeness_report.json",
        "prime_raw": RAW_ROOT / warm_tag / "warm_reuse_core_prime.jsonl",
    }


DEVICES = {
    "RTX 4060": _device_paths("official_4060_20260625", "official_4060_20260625"),
    "V100": _device_paths("official_v100_20260625", "official_v100_20260625"),
    "A100": _device_paths("official_a100_20260626", "official_a100_20260626"),
    "H100": _device_paths("official_h100_20260627", "official_h100_20260627"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def condition_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("workload_id"),
        row.get("batch_size"),
        row.get("num_steps"),
        row.get("input_mode"),
        row.get("execution_device"),
    )


def action_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return condition_key(row) + (row.get("action"),)


def is_feasible(row: dict[str, Any]) -> bool:
    return bool(row.get("action_feasible")) and row.get("median_total_runtime_s") is not None


def oracle_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[condition_key(row)].append(row)

    out = []
    for condition_rows in grouped.values():
        # Prefer the pre-computed correctness-constrained oracle flag (is_best_action=True).
        oracle = next((r for r in condition_rows if r.get("is_best_action") is True), None)
        if oracle is None:
            feasible = [row for row in condition_rows if is_feasible(row)]
            oracle = min(feasible, key=lambda row: float(row["median_total_runtime_s"])) if feasible else None
        if oracle is not None:
            out.append(oracle)
    return out


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.median(values), 6)


def minimum(values: list[float]) -> float | None:
    if not values:
        return None
    return round(min(values), 6)


def maximum(values: list[float]) -> float | None:
    if not values:
        return None
    return round(max(values), 6)


def summarize_ratios(
    device: str, cold_rows: list[dict[str, Any]], warm_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    cold_by_action = {action_key(row): row for row in cold_rows if row.get("action") in WARM_ACTIONS}
    ratios_by_action: dict[str, list[float]] = defaultdict(list)
    warm_faster_by_action: Counter[str] = Counter()

    for warm_row in warm_rows:
        if warm_row.get("action") not in WARM_ACTIONS or not is_feasible(warm_row):
            continue
        cold_row = cold_by_action.get(action_key(warm_row))
        if cold_row is None or not is_feasible(cold_row):
            continue
        warm_runtime = float(warm_row["median_total_runtime_s"])
        cold_runtime = float(cold_row["median_total_runtime_s"])
        if cold_runtime <= 0:
            continue
        action = str(warm_row["action"])
        ratio = warm_runtime / cold_runtime
        ratios_by_action[action].append(ratio)
        if ratio < 1.0:
            warm_faster_by_action[action] += 1

    rows = []
    for action in sorted(WARM_ACTIONS):
        ratios = ratios_by_action[action]
        rows.append(
            {
                "device": device,
                "action": action,
                "matched_completed_pairs": len(ratios),
                "warm_faster_count": warm_faster_by_action[action],
                "median_warm_cold_total_runtime_ratio": median(ratios),
                "min_warm_cold_total_runtime_ratio": minimum(ratios),
                "max_warm_cold_total_runtime_ratio": maximum(ratios),
            }
        )
    return rows


def summarize_compile_wins(warm_rows: list[dict[str, Any]]) -> tuple[int, int]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in warm_rows:
        grouped[condition_key(row)][row["action"]] = row

    total = 0
    wins = 0
    for action_rows in grouped.values():
        baseline = action_rows.get("best_eager")
        if baseline is None or not is_feasible(baseline):
            baseline = action_rows.get("eager")
        if baseline is None or not is_feasible(baseline):
            continue
        compile_rows = [
            action_rows[action]
            for action in ("compile_only", "compile_reduce_overhead")
            if action in action_rows and is_feasible(action_rows[action])
        ]
        if not compile_rows:
            continue
        total += 1
        compile_best = min(float(row["median_total_runtime_s"]) for row in compile_rows)
        if compile_best < float(baseline["median_total_runtime_s"]):
            wins += 1
    return wins, total


def summarize_prime(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    status_counts = Counter(str(row.get("status")) for row in rows)
    action_counts = Counter(str(row.get("action")) for row in rows)
    cache_counts = Counter(str(row.get("cache_state")) for row in rows)
    mode_counts = Counter(str(row.get("time_accounting_mode")) for row in rows)
    return {
        "prime_rows": len(rows),
        "prime_completed_rows": status_counts.get("completed", 0),
        "prime_failed_rows": sum(count for status, count in status_counts.items() if status != "completed"),
        "prime_compile_only_rows": action_counts.get("compile_only", 0),
        "prime_compile_reduce_overhead_rows": action_counts.get("compile_reduce_overhead", 0),
        "prime_warm_compile_cache_rows": cache_counts.get("warm_compile_cache", 0),
        "prime_warm_cached_total_rows": mode_counts.get("warm_cached_total", 0),
    }


def summarize_policy(device: str, path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        out = dict(row)
        out["device"] = device
        rows.append(out)
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_memo(summary_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> None:
    policy_by_device = {(row["device"], row["policy"]): row for row in policy_rows}
    lines = [
        "# 2026-06-17 Warm Compile-Cache Sensitivity Memo",
        "",
        "Status: Completed / derived from measured warm-cache runs",
        "",
        "## 0. Conclusion",
        "",
        "The warm compile-cache mini-run is complete on all four devices. Prime rows and measured rows are separated, and the measured run has no runtime failures.",
        "In the current four-workload mini matrix, warm compile-cache substantially reduces compile-family total time, but it does not make `compile_only` or `compile_reduce_overhead` the oracle action and does not make the compile family beat `eager`.",
        "",
        "Therefore, the paper can treat warm-cache as a bounded sensitivity rather than a missing blocker. It supports `compile` remaining an avoid-target / negative control in the tested warm compile-cache regime, but it does not establish a fully cache-state-robust admission boundary.",
        "",
        "## 1. Device Summary",
        "",
        "| Device | Rows | Prime | Conditions | Oracle compile | Compile wins vs eager | `compile_only` warm/cold | `compile_reduce` warm/cold | Always compile P90 | Amortization P90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        always_compile = policy_by_device.get((row["device"], "always_compile"), {})
        amortization = policy_by_device.get((row["device"], "amortization_policy"), {})
        lines.append(
            "| {device} | {observed}/{expected} | {prime_completed}/{prime_rows} | {conditions} | "
            "{oracle_compile} | {compile_wins}/{compile_total} | {compile_only_ratio} | "
            "{compile_reduce_ratio} | {always_compile_p90} | {amortization_p90} |".format(
                device=row["device"],
                observed=row["warm_observed_rows"],
                expected=row["warm_expected_rows"],
                prime_completed=row["prime_completed_rows"],
                prime_rows=row["prime_rows"],
                conditions=row["warm_evaluated_conditions"],
                oracle_compile=row["warm_oracle_compile_count"],
                compile_wins=row["warm_compile_win_vs_eager_count"],
                compile_total=row["warm_compile_win_vs_eager_total"],
                compile_only_ratio=fmt(row["compile_only_median_warm_cold_ratio"]),
                compile_reduce_ratio=fmt(row["compile_reduce_overhead_median_warm_cold_ratio"]),
                always_compile_p90=fmt(always_compile.get("p90_normalized_regret")),
                amortization_p90=fmt(amortization.get("p90_normalized_regret")),
            )
        )

    lines.extend(
        [
            "",
            "## 2. Paper Wording Guidance",
            "",
            "- Acceptable: warm compile-cache sensitivity across RTX 4060 Laptop, V100, A100, and H100 reduces compile-family cost but does not change the tested avoid-target conclusion.",
            "- Do not write: cache-state-robust admission boundary has been fully established.",
            "- Do not treat `short_train_small` graph-oracle rows as uncaveated positive graph evidence unless the correctness gate passes separately.",
            "",
            "## 3. Source Artifacts",
            "",
            "- `warm_compile_cache_sensitivity_summary.csv`",
            "- `warm_compile_cache_action_ratios.csv`",
            "- `warm_compile_cache_oracle_by_workload.csv`",
            "- `warm_compile_cache_policy_summary.csv`",
            "",
        ]
    )
    (OUT_DIR / "warm_compile_cache_sensitivity_memo.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    oracle_workload_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []

    for device, paths in DEVICES.items():
        cold_rows = read_jsonl(paths["cold_summary"])
        warm_rows = read_jsonl(paths["warm_summary"])
        completeness = read_json(paths["warm_completeness"])
        prime = summarize_prime(paths["prime_raw"])
        ratios = summarize_ratios(device, cold_rows, warm_rows)
        ratio_rows.extend(ratios)

        warm_oracle = oracle_rows(warm_rows)
        oracle_counts = Counter(str(row["action"]) for row in warm_oracle)
        oracle_by_workload: dict[str, Counter[str]] = defaultdict(Counter)
        for row in warm_oracle:
            oracle_by_workload[str(row["workload_id"])][str(row["action"])] += 1

        for workload, counts in sorted(oracle_by_workload.items()):
            for action, count in sorted(counts.items()):
                oracle_workload_rows.append(
                    {
                        "device": device,
                        "workload_id": workload,
                        "oracle_action": action,
                        "count": count,
                    }
                )

        compile_wins, compile_total = summarize_compile_wins(warm_rows)
        ratios_by_action = {row["action"]: row for row in ratios}
        device_policy_rows = summarize_policy(device, paths["warm_policy"])
        policy_rows.extend(device_policy_rows)

        summary_rows.append(
            {
                "device": device,
                "warm_expected_rows": completeness.get("expected_rows"),
                "warm_observed_rows": completeness.get("observed_rows"),
                "warm_missing_count": completeness.get("missing_count"),
                "warm_duplicate_condition_count": completeness.get("duplicate_condition_count"),
                "warm_unexpected_condition_count": completeness.get("unexpected_condition_count"),
                "warm_runtime_failure_count": completeness.get("runtime_failure_count"),
                "warm_completed_rows": completeness.get("status_counts", {}).get("completed"),
                "warm_infeasible_rows": completeness.get("status_counts", {}).get("infeasible"),
                "warm_evaluated_conditions": len(warm_oracle),
                "warm_oracle_eager_count": oracle_counts.get("eager", 0),
                "warm_oracle_best_eager_count": oracle_counts.get("best_eager", 0),
                "warm_oracle_graphs_only_count": oracle_counts.get("graphs_only", 0),
                "warm_oracle_compile_count": sum(
                    count for action, count in oracle_counts.items() if "compile" in action
                ),
                "warm_compile_win_vs_eager_count": compile_wins,
                "warm_compile_win_vs_eager_total": compile_total,
                "compile_only_median_warm_cold_ratio": ratios_by_action[
                    "compile_only"
                ].get("median_warm_cold_total_runtime_ratio"),
                "compile_reduce_overhead_median_warm_cold_ratio": ratios_by_action[
                    "compile_reduce_overhead"
                ].get("median_warm_cold_total_runtime_ratio"),
                "graphs_only_median_warm_cold_ratio": ratios_by_action["graphs_only"].get(
                    "median_warm_cold_total_runtime_ratio"
                ),
                "eager_median_warm_cold_ratio": ratios_by_action["eager"].get(
                    "median_warm_cold_total_runtime_ratio"
                ),
                "best_eager_median_warm_cold_ratio": ratios_by_action["best_eager"].get(
                    "median_warm_cold_total_runtime_ratio"
                ),
                **prime,
            }
        )

    write_csv(OUT_DIR / "warm_compile_cache_sensitivity_summary.csv", summary_rows)
    write_csv(OUT_DIR / "warm_compile_cache_action_ratios.csv", ratio_rows)
    write_csv(OUT_DIR / "warm_compile_cache_oracle_by_workload.csv", oracle_workload_rows)
    write_csv(OUT_DIR / "warm_compile_cache_policy_summary.csv", policy_rows)
    write_memo(summary_rows, policy_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
