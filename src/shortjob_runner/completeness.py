from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from shortjob_runner.cli import expand_choice
from shortjob_runner.io import read_jsonl
from shortjob_runner.types import ACTIONS, WORKLOADS


Key = tuple[str, str, int, int, int, str, str]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether raw shortjob JSONL covers the expected experiment grid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--workload", nargs="+", choices=("all", *WORKLOADS), default=["all"])
    parser.add_argument("--action", nargs="+", choices=("all", *ACTIONS), default=["all"])
    parser.add_argument("--batch-size", nargs="+", type=int, required=True)
    parser.add_argument("--num-steps", nargs="+", type=int, required=True)
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Expected repeats per condition. Must match the run --repeats (default 5).",
    )
    parser.add_argument("--repeat-start", type=int, default=1)
    parser.add_argument("--time-accounting-mode", required=True)
    parser.add_argument("--cache-state", required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--fail-on-status",
        action="store_true",
        help="Treat non-completed / non-infeasible rows as check failures.",
    )
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.repeat_start < 1:
        raise ValueError("--repeat-start must be >= 1")
    if any(value < 1 for value in args.batch_size):
        raise ValueError("--batch-size values must be >= 1")
    if any(value < 1 for value in args.num_steps):
        raise ValueError("--num-steps values must be >= 1")


def expected_keys(args: argparse.Namespace) -> set[Key]:
    return {
        (
            workload_id,
            action,
            batch_size,
            num_steps,
            repeat,
            args.time_accounting_mode,
            args.cache_state,
        )
        for workload_id in expand_choice(args.workload, WORKLOADS)
        for action in expand_choice(args.action, ACTIONS)
        for batch_size in args.batch_size
        for num_steps in args.num_steps
        for repeat in range(args.repeat_start, args.repeat_start + args.repeats)
    }


def row_key(row: dict[str, Any]) -> Key:
    return (
        str(row.get("workload_id")),
        str(row.get("action")),
        int(row.get("batch_size")),
        int(row.get("num_steps")),
        int(row.get("repeat")),
        str(row.get("time_accounting_mode")),
        str(row.get("cache_state")),
    )


def sample_keys(keys: set[Key], limit: int) -> list[dict[str, object]]:
    return [
        {
            "workload_id": key[0],
            "action": key[1],
            "batch_size": key[2],
            "num_steps": key[3],
            "repeat": key[4],
            "time_accounting_mode": key[5],
            "cache_state": key[6],
        }
        for key in sorted(keys)[:limit]
    ]


def build_report(rows: list[dict[str, Any]], expected: set[Key], max_examples: int) -> dict[str, Any]:
    observed_counter = Counter(row_key(row) for row in rows)
    observed = set(observed_counter)
    duplicate_keys = {key for key, count in observed_counter.items() if count > 1}
    missing = expected - observed
    unexpected = observed - expected
    status_counts = Counter(str(row.get("status")) for row in rows)
    failure_reason_counts = Counter(str(row.get("failure_reason")) for row in rows if row.get("failure_reason"))
    runtime_failure_rows = [
        row
        for row in rows
        if row.get("status") not in {"completed", "infeasible"}
        and not row.get("precheck_failed")
    ]

    return {
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "observed_unique_conditions": len(observed),
        "missing_count": len(missing),
        "duplicate_condition_count": len(duplicate_keys),
        "unexpected_condition_count": len(unexpected),
        "completed_count": status_counts.get("completed", 0),
        "infeasible_count": status_counts.get("infeasible", 0),
        "status_counts": dict(sorted(status_counts.items())),
        "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        "missing_examples": sample_keys(missing, max_examples),
        "duplicate_examples": sample_keys(duplicate_keys, max_examples),
        "unexpected_examples": sample_keys(unexpected, max_examples),
        "runtime_failure_count": len(runtime_failure_rows),
        "runtime_failure_examples": [
            {
                "workload_id": row.get("workload_id"),
                "action": row.get("action"),
                "batch_size": row.get("batch_size"),
                "num_steps": row.get("num_steps"),
                "repeat": row.get("repeat"),
                "status": row.get("status"),
                "failure_reason": row.get("failure_reason"),
            }
            for row in runtime_failure_rows[:max_examples]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    validate_args(args)

    rows = read_jsonl(args.raw)
    expected = expected_keys(args)
    report = {"raw": str(args.raw), **build_report(rows, expected, args.max_examples)}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))

    failed = bool(
        report["missing_count"]
        or report["duplicate_condition_count"]
        or report["unexpected_condition_count"]
    )
    if args.fail_on_status and report["runtime_failure_count"]:
        failed = True
    return 1 if failed else 0
