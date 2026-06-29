#!/usr/bin/env python3
"""Summarize oracle rank margin and CI overlap from action summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from shortjob_runner.io import read_jsonl
from shortjob_runner.summarize import load_correctness_blocklist


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "tables" / "paper_official_20260627"


def condition_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("workload_id"),
        row.get("batch_size"),
        row.get("num_steps"),
        row.get("input_mode"),
        row.get("time_accounting_mode"),
        row.get("cache_state"),
        row.get("execution_device"),
    )


def is_feasible(row: dict[str, Any]) -> bool:
    return bool(row.get("action_feasible")) and row.get("median_total_runtime_s") is not None


def ci_overlap(first: dict[str, Any], second: dict[str, Any]) -> bool | None:
    first_low = first.get("ci95_total_runtime_low_s")
    first_high = first.get("ci95_total_runtime_high_s")
    second_low = second.get("ci95_total_runtime_low_s")
    second_high = second.get("ci95_total_runtime_high_s")
    if None in (first_low, first_high, second_low, second_high):
        return None
    return max(float(first_low), float(second_low)) <= min(float(first_high), float(second_high))


def summarize_conditions(
    rows: Iterable[dict[str, Any]],
    *,
    device_label: str,
    marginal_threshold: float,
    blocked_actions: frozenset[tuple[str, str]] = frozenset(),
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[condition_key(row)].append(row)

    out: list[dict[str, Any]] = []
    for key, condition_rows in sorted(grouped.items()):
        # Prefer the pre-computed correctness-constrained oracle flag set by summarize.py.
        # Fall back to speed-based oracle (filtered by blocked_actions) when the flag is
        # absent, so the test fixture and legacy data still work.
        oracle = next((row for row in condition_rows if row.get("is_best_action") is True), None)
        if oracle is None:
            workload_id = condition_rows[0].get("workload_id", "") if condition_rows else ""
            feasible_unblocked = [
                row for row in condition_rows
                if is_feasible(row)
                and (workload_id, row.get("action", "")) not in blocked_actions
            ]
            oracle = min(feasible_unblocked, key=lambda r: float(r["median_total_runtime_s"])) if feasible_unblocked else None
        if oracle is None:
            continue
        oracle_action = oracle.get("action")
        workload_id = oracle.get("workload_id", "")
        # Runner-up: fastest feasible non-oracle action that is not correctness-blocked
        # and has runtime >= oracle runtime (prevents a blocked faster action from showing
        # a negative margin).
        runner_up_candidates = sorted(
            [
                row for row in condition_rows
                if is_feasible(row)
                and row.get("action") != oracle_action
                and (workload_id, row.get("action", "")) not in blocked_actions
                and float(row["median_total_runtime_s"]) >= float(oracle["median_total_runtime_s"])
            ],
            key=lambda row: float(row["median_total_runtime_s"]),
        )
        runner_up = runner_up_candidates[0] if runner_up_candidates else None
        oracle_runtime = float(oracle["median_total_runtime_s"])
        runner_runtime = float(runner_up["median_total_runtime_s"]) if runner_up is not None else None
        relative_margin = (
            (runner_runtime - oracle_runtime) / oracle_runtime
            if runner_runtime is not None and oracle_runtime > 0
            else None
        )
        overlap = ci_overlap(oracle, runner_up) if runner_up is not None else None
        out.append(
            {
                "device_label": device_label,
                "workload_id": key[0],
                "batch_size": key[1],
                "num_steps": key[2],
                "time_accounting_mode": key[4],
                "cache_state": key[5],
                "oracle_action": oracle.get("action"),
                "oracle_runtime_s": round(oracle_runtime, 6),
                "runner_up_action": runner_up.get("action") if runner_up is not None else None,
                "runner_up_runtime_s": round(runner_runtime, 6) if runner_runtime is not None else None,
                "relative_margin": round(relative_margin, 6) if relative_margin is not None else None,
                "ci95_overlap": overlap,
                "marginal_rank": (
                    relative_margin is not None and relative_margin < marginal_threshold
                ),
            }
        )
    return out


def summarize_device(device_label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    device_rows = [row for row in rows if row["device_label"] == device_label]
    # Count only audit-passing graphs_only as positive graph evidence; graphs_input_copy
    # is a diagnostic action and excluded from the positive graph oracle count.
    graph_oracle = [row for row in device_rows if row.get("oracle_action") == "graphs_only"]
    return {
        "device_label": device_label,
        "n_conditions": len(device_rows),
        "marginal_rank_count": sum(1 for row in device_rows if row["marginal_rank"]),
        "ci95_overlap_count": sum(1 for row in device_rows if row["ci95_overlap"] is True),
        "graph_oracle_count": len(graph_oracle),
        "graph_oracle_marginal_count": sum(1 for row in graph_oracle if row["marginal_rank"]),
        "graph_oracle_ci95_overlap_count": sum(1 for row in graph_oracle if row["ci95_overlap"] is True),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.parent.name, path
    label, path = value.split("=", 1)
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summary", nargs="+", required=True, help="Inputs as DEVICE_LABEL=action_summary.jsonl.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="rank_stability")
    parser.add_argument("--marginal-threshold", type=float, default=0.05)
    parser.add_argument(
        "--correctness-gate",
        nargs="+",
        default=None,
        metavar="DEVICE_LABEL=PATH",
        help=(
            "Per-device correctness gates as DEVICE_LABEL=correctness_gate.jsonl. "
            "Each device in --summary must have a matching gate (its OWN gate; do "
            "not reuse one device's blocklist for another). Required unless "
            "--allow-unconstrained is set."
        ),
    )
    parser.add_argument(
        "--allow-unconstrained",
        action="store_true",
        help="Explicitly run the speed-only (unconstrained) analysis with no correctness gate.",
    )
    return parser.parse_args()


def resolve_device_gates(
    devices: list[str],
    gate_entries: list[str] | None,
    *,
    allow_unconstrained: bool,
) -> dict[str, frozenset[tuple[str, str]]]:
    """Map each device to its OWN correctness blocklist.

    Enforces provenance: a per-device gate is required for every device unless
    the caller explicitly opts into unconstrained analysis.  Refuses to silently
    apply one device's blocklist to another.
    """
    if gate_entries is None:
        if allow_unconstrained:
            return {device: frozenset() for device in devices}
        raise SystemExit(
            "error: --correctness-gate is required (one DEVICE_LABEL=path per device) "
            "or pass --allow-unconstrained to run the speed-only analysis explicitly."
        )
    gates: dict[str, Path] = {}
    for entry in gate_entries:
        if "=" not in entry:
            raise SystemExit(
                f"error: --correctness-gate entry {entry!r} must be DEVICE_LABEL=path"
            )
        label, path = entry.split("=", 1)
        gates[label] = Path(path)
    missing = [device for device in devices if device not in gates]
    if missing and not allow_unconstrained:
        raise SystemExit(
            f"error: no correctness gate provided for device(s) {missing}; "
            "provide DEVICE_LABEL=path for each or pass --allow-unconstrained."
        )
    resolved: dict[str, frozenset[tuple[str, str]]] = {}
    for device in devices:
        if device in gates:
            resolved[device] = load_correctness_blocklist(gates[device])
        else:
            resolved[device] = frozenset()
    return resolved


def main() -> int:
    args = parse_args()
    devices: list[str] = [parse_input(item)[0] for item in args.summary]
    device_gates = resolve_device_gates(
        devices, args.correctness_gate, allow_unconstrained=args.allow_unconstrained
    )
    per_condition: list[dict[str, Any]] = []
    for item in args.summary:
        device_label, path = parse_input(item)
        per_condition.extend(
            summarize_conditions(
                read_jsonl(path),
                device_label=device_label,
                marginal_threshold=args.marginal_threshold,
                blocked_actions=device_gates[device_label],
            )
        )
    device_rows = [summarize_device(device, per_condition) for device in devices]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{args.prefix}_per_condition.csv", per_condition)
    write_csv(args.out_dir / f"{args.prefix}_device_summary.csv", device_rows)
    print(json.dumps({"devices": devices, "conditions": len(per_condition)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
