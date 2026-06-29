#!/usr/bin/env python3
"""Summarize bounded repeated-shape CUDA Graph reuse sensitivity outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from shortjob_runner.graph_reuse import summarize_conditions, summarize_device
from shortjob_runner.io import read_jsonl
from shortjob_runner.summarize import load_correctness_blocklist


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "tables" / "paper_official_20260627"


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_input(value: str) -> tuple[str | None, Path]:
    if "=" not in value:
        return None, Path(value)
    label, path = value.split("=", 1)
    return label, Path(path)


def rows_from_inputs(inputs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in inputs:
        label, path = parse_input(item)
        for row in read_jsonl(path):
            if label is not None:
                row = {**row, "device_label": label}
            rows.append(row)
    return rows


def write_memo(path: Path, device_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Graph Reuse Sensitivity Memo",
        "",
        "Status: Derived / bounded repeated-shape sensitivity",
        "",
        "## 0. Interpretation Boundary",
        "",
        "This analysis measures a bounded setting where same-shape jobs in one worker process capture a CUDA Graph once and replay it many times.",
        "It is not a production graph cache and does not implement persistent captured-graph artifacts across processes.",
        "It should be described as bounded graph-reuse or repeated-shape sensitivity, not as full cache-state robustness.",
        "",
        "## 1. Device Summary",
        "",
        "| Device | Conditions | Graph oracle | Graph wins vs eager | Runtime-infeasible | Correctness-blocked | First-job graph slower | Median graph speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in device_rows:
        lines.append(
            "| {device} | {conditions} | {oracle} | {wins} | {infeasible} | {blocked} | {first_slow} | {speedup} |".format(
                device=row["device_label"],
                conditions=row["n_conditions"],
                oracle=row["graph_oracle_count"],
                wins=row["graphs_only_win_vs_eager_count"],
                infeasible=row["graphs_only_infeasible_count"],
                blocked=row["graphs_only_correctness_blocked_count"],
                first_slow=row["graphs_only_first_job_slower_count"],
                speedup=row["median_graphs_only_speedup_vs_eager"],
            )
        )
    lines.extend(
        [
            "",
            "## 2. Paper Wording Guidance",
            "",
            "- Acceptable: repeated same-shape jobs can change graph amortization because capture is paid once per group in this bounded worker-local sensitivity.",
            "- Do not write: the paper implements a production graph cache or persistent captured-graph store.",
            "- Oracle counts, graph wins, and speedups include only actions that pass the corresponding device correctness gate.",
            "- If `graphs_input_copy` or the `short_train_small` graph path does not pass the gate, do not treat it as positive graph evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="JSONL inputs, optionally as DEVICE_LABEL=path.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metric", default="amortized_runtime_per_job_s")
    parser.add_argument("--prefix", default="graph_reuse_sensitivity")
    parser.add_argument(
        "--correctness-gate",
        nargs="+",
        default=[],
        metavar="DEVICE=PATH",
        help="Per-device correctness gate JSONL; failed workload-action pairs are excluded.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = rows_from_inputs(args.input)
    blocked_actions_by_device = {}
    for item in args.correctness_gate:
        label, path = parse_input(item)
        if label is None:
            raise ValueError("--correctness-gate requires DEVICE=PATH")
        blocked_actions_by_device[label] = load_correctness_blocklist(path)
    per_condition = summarize_conditions(
        rows,
        metric=args.metric,
        blocked_actions_by_device=blocked_actions_by_device,
    )
    devices = sorted({str(row.get("device_label")) for row in per_condition})
    device_rows = [summarize_device(device, per_condition) for device in devices]

    status_counts = Counter(str(row.get("status")) for row in rows)
    metadata_rows = [
        {
            "raw_rows": len(rows),
            "completed_rows": status_counts.get("completed", 0),
            "infeasible_rows": status_counts.get("infeasible", 0),
            "failed_rows": sum(count for status, count in status_counts.items() if status not in {"completed", "infeasible"}),
            "metric": args.metric,
        }
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{args.prefix}_per_condition.csv", per_condition)
    write_csv(args.out_dir / f"{args.prefix}_device_summary.csv", device_rows)
    write_csv(args.out_dir / f"{args.prefix}_metadata.csv", metadata_rows)
    write_memo(args.out_dir / f"{args.prefix}_memo.md", device_rows)
    print(json.dumps({"devices": devices, "conditions": len(per_condition), **metadata_rows[0]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
