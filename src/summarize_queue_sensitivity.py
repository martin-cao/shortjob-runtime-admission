#!/usr/bin/env python3
"""Summarize queue replay job-mix sensitivity outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from shortjob_runner.io import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "tables" / "paper_official_20260627" / "queue_replay_job_mix_sensitivity_heavy_summary.csv"
METRICS = (
    "mean_completion_time_s",
    "p95_completion_time_s",
    "throughput_jobs_per_s",
    "wasted_optimization_overhead_s",
    "wrong_admit_count",
    "missed_opportunity_count",
)


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    label, path = value.split("=", 1)
    return label, Path(path)


def mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return round(statistics.fmean(numeric), 6) if numeric else None


def summarize_rows(source_file: str, rows: Iterable[dict[str, Any]], *, load_level: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("load_level") != load_level:
            continue
        grouped[(str(row.get("job_mix_id")), str(row.get("policy")))].append(row)

    out: list[dict[str, Any]] = []
    for (job_mix_id, policy), group in sorted(grouped.items()):
        summary = {
            "source_file": source_file,
            "job_mix_id": job_mix_id,
            "policy": policy,
        }
        for metric in METRICS:
            summary[f"{load_level}_{metric}"] = mean(row.get(metric) for row in group)
        out.append(summary)
    return out


def write_csv(path: Path, rows: Iterable[dict[str, Any]], *, load_level: str) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_file", "job_mix_id", "policy"] + [
        f"{load_level}_{metric}" for metric in METRICS
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--queue", nargs="+", required=True, help="Queue replay JSONL, optionally as LABEL=path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--load-level", default="heavy", choices=("light", "medium", "heavy"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for item in args.queue:
        source_file, path = parse_input(item)
        rows.extend(summarize_rows(source_file, read_jsonl(path), load_level=args.load_level))
    write_csv(args.out, rows, load_level=args.load_level)
    print(json.dumps({"rows": len(rows), "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
