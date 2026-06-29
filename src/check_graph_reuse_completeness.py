#!/usr/bin/env python3
"""Check graph-reuse raw JSONL coverage against the intended experiment grid."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from shortjob_runner.cli import expand_choice
from shortjob_runner.io import read_jsonl


DEFAULT_WORKLOADS = (
    "cv_online_infer",
    "llm_decode_proxy",
    "dynamic_shape_infer_small",
    "short_train_small",
)
DEFAULT_ACTIONS = ("eager", "best_eager", "graphs_only", "graphs_input_copy")


Key = tuple[str, str, int, int, int, int, str]


def expected_keys(args: argparse.Namespace) -> set[Key]:
    return {
        (workload, action, batch, steps, reuse_jobs, repeat, args.device_label)
        for workload in expand_choice(args.workload, DEFAULT_WORKLOADS)
        for action in expand_choice(args.action, DEFAULT_ACTIONS)
        for batch in args.batch_size
        for steps in args.num_steps
        for reuse_jobs in args.reuse_jobs
        for repeat in range(1, args.repeats + 1)
    }


def row_key(row: dict[str, Any]) -> Key:
    return (
        str(row.get("workload_id")),
        str(row.get("action")),
        int(row.get("batch_size")),
        int(row.get("num_steps")),
        int(row.get("reuse_jobs")),
        int(row.get("repeat")),
        str(row.get("device_label")),
    )


def sample(keys: set[Key], limit: int) -> list[dict[str, Any]]:
    return [
        {
            "workload_id": key[0],
            "action": key[1],
            "batch_size": key[2],
            "num_steps": key[3],
            "reuse_jobs": key[4],
            "repeat": key[5],
            "device_label": key[6],
        }
        for key in sorted(keys)[:limit]
    ]


def build_report(rows: list[dict[str, Any]], expected: set[Key], max_examples: int) -> dict[str, Any]:
    observed_counter = Counter(row_key(row) for row in rows)
    observed = set(observed_counter)
    duplicate = {key for key, count in observed_counter.items() if count > 1}
    missing = expected - observed
    unexpected = observed - expected
    status_counts = Counter(str(row.get("status")) for row in rows)
    return {
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "observed_unique_conditions": len(observed),
        "missing_count": len(missing),
        "duplicate_condition_count": len(duplicate),
        "unexpected_condition_count": len(unexpected),
        "status_counts": dict(sorted(status_counts.items())),
        "missing_examples": sample(missing, max_examples),
        "duplicate_examples": sample(duplicate, max_examples),
        "unexpected_examples": sample(unexpected, max_examples),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--device-label", required=True)
    parser.add_argument("--workload", nargs="+", choices=("all", *DEFAULT_WORKLOADS), default=list(DEFAULT_WORKLOADS))
    parser.add_argument("--action", nargs="+", choices=DEFAULT_ACTIONS, default=list(DEFAULT_ACTIONS))
    parser.add_argument("--batch-size", nargs="+", type=int, required=True)
    parser.add_argument("--num-steps", nargs="+", type=int, required=True)
    parser.add_argument("--reuse-jobs", nargs="+", type=int, default=[8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.raw)
    report = {"raw": str(args.raw), **build_report(rows, expected_keys(args), args.max_examples)}
    output = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output, end="")
    return 1 if report["missing_count"] or report["duplicate_condition_count"] or report["unexpected_condition_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
