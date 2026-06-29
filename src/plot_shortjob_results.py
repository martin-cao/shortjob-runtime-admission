#!/usr/bin/env python3
"""Plot and tabulate the expanded short-job optimisation study results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_matplotlib() -> Any:
    import matplotlib.pyplot as plt

    return plt


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def workload_taxonomy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        workload = row.get("workload_id")
        if workload and workload not in seen:
            seen[workload] = {
                "workload_id": workload,
                "application": row.get("application"),
                "task_family": row.get("task_family"),
                "workload_class": row.get("workload_class"),
                "shape_stability": row.get("shape_stability"),
                "input_mode": row.get("input_mode"),
                "bottleneck_hypothesis": row.get("bottleneck_hypothesis"),
                "execution_device": row.get("execution_device"),
            }
    return sorted(seen.values(), key=lambda item: item["workload_id"])


def method_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("workload_id")), str(row.get("action")))].append(row)
    output: list[dict[str, Any]] = []
    for (workload, action), action_rows in sorted(grouped.items()):
        completed = sum(1 for row in action_rows if row.get("action_feasible") is True)
        failed = len(action_rows) - completed
        reasons = sorted({row.get("failure_reason") for row in action_rows if row.get("failure_reason")})
        output.append(
            {
                "workload_id": workload,
                "action": action,
                "n_rows": len(action_rows),
                "n_completed": completed,
                "n_failed_or_infeasible": failed,
                "applicability": "Y" if completed else "N/A",
                "reason": "; ".join(reasons[:3]),
            }
        )
    return output


def plot_latency_quantiles(rows: list[dict[str, Any]], out: Path) -> None:
    plt = ensure_matplotlib()
    completed = [row for row in rows if row.get("action_feasible") and row.get("p50_total_runtime_s") is not None]
    if not completed:
        return
    labels = [f"{row['workload_id']}\n{row['action']}" for row in completed]
    p50 = [float(row.get("p50_total_runtime_s") or row.get("median_total_runtime_s")) for row in completed]
    p95 = [float(row.get("p95_total_runtime_s") or row.get("median_total_runtime_s")) for row in completed]
    p99 = [float(row.get("p99_total_runtime_s") or row.get("median_total_runtime_s")) for row in completed]
    x = list(range(len(completed)))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 4.8))
    ax.bar([idx - width for idx in x], p50, width, label="P50")
    ax.bar(x, p95, width, label="P95")
    ax.bar([idx + width for idx in x], p99, width, label="P99")
    ax.set_ylabel("Runtime (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_break_even(rows: list[dict[str, Any]], out: Path) -> None:
    plt = ensure_matplotlib()
    candidates = [
        row
        for row in rows
        if row.get("action_feasible")
        and row.get("median_total_runtime_s") is not None
        and row.get("median_avg_step_time_ms") is not None
    ]
    if not candidates:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for row in candidates:
        startup = float(row.get("median_compile_overhead_s") or 0.0) + float(row.get("median_graph_capture_overhead_s") or 0.0)
        step = max(float(row["median_avg_step_time_ms"]) / 1000.0, 1e-9)
        counts = [1, 5, 10, 25, 50, 100, 250, 500]
        totals = [startup + count * step for count in counts]
        label = f"{row['workload_id']}:{row['action']}"
        ax.plot(counts, totals, marker="o", linewidth=1, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Request / step count")
    ax.set_ylabel("Modeled total runtime (s)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def negative_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row.get("wrong_enable") or row.get("action_feasible") is False or row.get("n_failed"):
            output.append(
                {
                    "workload_id": row.get("workload_id"),
                    "action": row.get("action"),
                    "wrong_enable": row.get("wrong_enable"),
                    "failure_rate": row.get("failure_rate"),
                    "infeasible_precheck_rate": row.get("infeasible_precheck_rate"),
                    "normalized_regret_vs_oracle": row.get("normalized_regret_vs_oracle"),
                    "failure_reasons": row.get("failure_reasons"),
                }
            )
    return output


def copy_overhead_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row.get("median_input_copy_overhead_s") is not None:
            replay = float(row.get("median_measured_runtime_s") or math.nan)
            copy = float(row.get("median_input_copy_overhead_s") or 0.0)
            output.append(
                {
                    "workload_id": row.get("workload_id"),
                    "action": row.get("action"),
                    "median_input_copy_overhead_s": copy,
                    "median_measured_runtime_s": replay,
                    "copy_fraction_of_replay": round(copy / replay, 6) if replay and not math.isnan(replay) else None,
                    "median_input_copy_bytes": row.get("median_input_copy_bytes"),
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build expanded shortjob figures and tables from summaries")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/expanded"))
    parser.add_argument("--prefix", default="shortjob")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{args.prefix}_workload_taxonomy.csv", workload_taxonomy(rows))
    write_csv(args.out_dir / f"{args.prefix}_method_matrix.csv", method_matrix(rows))
    write_csv(args.out_dir / f"{args.prefix}_negative_results.csv", negative_results(rows))
    write_csv(args.out_dir / f"{args.prefix}_input_copy_overhead.csv", copy_overhead_table(rows))
    plot_latency_quantiles(rows, args.out_dir / f"{args.prefix}_latency_quantiles.pdf")
    plot_break_even(rows, args.out_dir / f"{args.prefix}_break_even_curves.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
