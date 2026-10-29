from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Set
from typing import Any, Iterable


GRAPH_ACTIONS = {"graphs_only", "graphs_input_copy"}


def condition_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("device_label"),
        row.get("workload_id"),
        row.get("batch_size"),
        row.get("num_steps"),
        row.get("reuse_jobs"),
    )


def is_feasible(row: dict[str, Any], metric: str) -> bool:
    return bool(row.get("action_feasible")) and row.get(metric) is not None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.median(values), 6)


def aggregate_action_rows(action: str, rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    feasible_metric_values = [
        float(row[metric])
        for row in rows
        if bool(row.get("action_feasible")) and row.get(metric) is not None
    ]
    first_job_values = [
        float(row["first_job_total_runtime_s"])
        for row in rows
        if bool(row.get("action_feasible")) and row.get("first_job_total_runtime_s") is not None
    ]
    return {
        "action": action,
        "action_feasible": bool(feasible_metric_values),
        "status": "completed" if feasible_metric_values else str(rows[0].get("status")),
        metric: median(feasible_metric_values),
        "first_job_total_runtime_s": median(first_job_values),
        "repeat_count": len(rows),
        "feasible_repeat_count": len(feasible_metric_values),
    }


def action_rows_by_condition(
    rows: Iterable[dict[str, Any]], metric: str = "amortized_runtime_per_job_s"
) -> dict[tuple[Any, ...], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[condition_key(row)][str(row.get("action"))].append(row)
    return {
        key: {
            action: aggregate_action_rows(action, action_rows, metric)
            for action, action_rows in action_map.items()
        }
        for key, action_map in grouped.items()
    }


def speedup(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def summarize_conditions(
    rows: Iterable[dict[str, Any]],
    *,
    metric: str,
    blocked_actions_by_device: Mapping[str, Set[tuple[str, str]]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, action_rows in sorted(action_rows_by_condition(rows, metric).items()):
        device_label = str(key[0])
        workload_id = str(key[1])
        blocked_actions = (
            blocked_actions_by_device.get(device_label, frozenset())
            if blocked_actions_by_device is not None
            else frozenset()
        )

        def correctness_passes(action: str) -> bool:
            return (workload_id, action) not in blocked_actions

        feasible = [
            row
            for row in action_rows.values()
            if is_feasible(row, metric) and correctness_passes(str(row.get("action")))
        ]
        if not feasible:
            continue
        oracle = min(feasible, key=lambda row: float(row[metric]))
        eager = action_rows.get("eager") or action_rows.get("best_eager")
        graphs_only = action_rows.get("graphs_only")
        graphs_input_copy = action_rows.get("graphs_input_copy")
        eager_runtime = float(eager[metric]) if eager is not None and is_feasible(eager, metric) else None
        graphs_only_runtime_feasible = bool(graphs_only and is_feasible(graphs_only, metric))
        graphs_only_correctness_pass = correctness_passes("graphs_only")
        graphs_only_eligible = graphs_only_runtime_feasible and graphs_only_correctness_pass
        graphs_input_copy_runtime_feasible = bool(
            graphs_input_copy and is_feasible(graphs_input_copy, metric)
        )
        graphs_input_copy_correctness_pass = correctness_passes("graphs_input_copy")
        graphs_input_copy_eligible = (
            graphs_input_copy_runtime_feasible and graphs_input_copy_correctness_pass
        )
        graph_runtime = (
            float(graphs_only[metric])
            if graphs_only is not None and graphs_only_eligible
            else None
        )
        first_job_graph = (
            float(graphs_only["first_job_total_runtime_s"])
            if graphs_only is not None
            and graphs_only_eligible
            and graphs_only.get("first_job_total_runtime_s") is not None
            else None
        )
        first_job_eager = (
            float(eager["first_job_total_runtime_s"])
            if eager is not None and eager.get("first_job_total_runtime_s") is not None
            else eager_runtime
        )

        out.append(
            {
                "device_label": device_label,
                "workload_id": workload_id,
                "batch_size": key[2],
                "num_steps": key[3],
                "reuse_jobs": key[4],
                "oracle_action": oracle.get("action"),
                "oracle_runtime_s": round(float(oracle[metric]), 6),
                "eager_runtime_s": round(eager_runtime, 6) if eager_runtime is not None else None,
                "graphs_only_runtime_s": round(graph_runtime, 6) if graph_runtime is not None else None,
                "graphs_only_runtime_feasible": graphs_only_runtime_feasible,
                "graphs_only_correctness_pass": graphs_only_correctness_pass,
                "graphs_only_feasible": graphs_only_eligible,
                "graphs_input_copy_runtime_feasible": graphs_input_copy_runtime_feasible,
                "graphs_input_copy_correctness_pass": graphs_input_copy_correctness_pass,
                "graphs_input_copy_feasible": graphs_input_copy_eligible,
                "graphs_only_speedup_vs_eager": speedup(eager_runtime, graph_runtime),
                "graphs_only_win_vs_eager": (
                    graph_runtime is not None and eager_runtime is not None and graph_runtime < eager_runtime
                ),
                "graphs_only_first_job_slower_than_eager": (
                    first_job_graph is not None
                    and first_job_eager is not None
                    and first_job_graph > first_job_eager
                ),
            }
        )
    return out


def summarize_device(device_label: str, condition_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in condition_rows if row.get("device_label") == device_label]
    speedups = [
        float(row["graphs_only_speedup_vs_eager"])
        for row in rows
        if row.get("graphs_only_speedup_vs_eager") is not None
    ]
    return {
        "device_label": device_label,
        "n_conditions": len(rows),
        "graph_oracle_count": sum(
            1 for row in rows if str(row.get("oracle_action")) in GRAPH_ACTIONS
        ),
        "graphs_only_win_vs_eager_count": sum(
            1 for row in rows if row.get("graphs_only_win_vs_eager")
        ),
        "graphs_only_infeasible_count": sum(
            1 for row in rows if not row.get("graphs_only_runtime_feasible")
        ),
        "graphs_only_correctness_blocked_count": sum(
            1
            for row in rows
            if row.get("graphs_only_runtime_feasible")
            and not row.get("graphs_only_correctness_pass")
        ),
        "graphs_only_first_job_slower_count": sum(
            1 for row in rows if row.get("graphs_only_first_job_slower_than_eager")
        ),
        "median_graphs_only_speedup_vs_eager": median(speedups),
    }
