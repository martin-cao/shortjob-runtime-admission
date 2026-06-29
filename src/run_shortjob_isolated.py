#!/usr/bin/env python3
"""Run each condition in a separate Python process to avoid cold-run contamination."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from shortjob_runner.cli import expand_choice
from shortjob_runner.sharding import select_shard
from shortjob_runner.types import ACTIONS, WORKLOADS, Condition

# Each condition runs in its own subprocess. A subprocess can die at import time
# from the transient PyTorch 2.12 TSC assert ("getCount is non-monotonic"); a
# single such crash must NOT abort a multi-hour sweep. Retry the launch a few
# times (the crash is random per process, so a retry almost always succeeds),
# then, if it still fails, record the condition and continue. Set
# SHORTJOB_SUBPROCESS_RETRIES=0 to disable retries.
SUBPROCESS_RETRIES = max(0, int(os.environ.get("SHORTJOB_SUBPROCESS_RETRIES", "3")))
SUBPROCESS_RETRY_BACKOFF_S = float(os.environ.get("SHORTJOB_SUBPROCESS_RETRY_BACKOFF_S", "1.0"))


COMPILE_CACHE_ACTIONS = {
    "compile_only",
    "compile_reduce_overhead",
    "compile_max_autotune",
    "compile_plus_graphs",
    "compile_reduce_overhead_plus_graphs",
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the shortjob runner with one subprocess per condition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workload", nargs="+", choices=("all", *WORKLOADS), default=["all"])
    parser.add_argument("--action", nargs="+", choices=("all", *ACTIONS), default=["all"])
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1, 16, 64])
    parser.add_argument("--num-steps", nargs="+", type=int, default=[10, 50, 100, 500])
    parser.add_argument("--first-k-steps", type=int, default=5)
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help=(
            "Measurement repeats per condition. Default 5 (df=4, t=2.776) for new "
            "runs; committed 4060/V100 cold-core artifacts stay at the n=3 they were "
            "collected with, and the Student-t CI selects the critical value per n."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("data/raw/shortjob_pilot.jsonl"))
    parser.add_argument("--time-accounting-mode", choices=("cold_total", "warm_cached_total"), default="cold_total")
    parser.add_argument("--cache-state", default="cold_cache")
    parser.add_argument(
        "--cache-dir-root",
        type=Path,
        default=Path("/tmp/shortjob_runtime_cache"),
        help="Root for per-condition Inductor/Triton cache dirs.",
    )
    parser.add_argument(
        "--prime-compile-cache",
        action="store_true",
        help=(
            "Before measuring warm cache conditions, run each compile-family "
            "condition key once into --prime-out using the same cache dirs."
        ),
    )
    parser.add_argument(
        "--prime-out",
        type=Path,
        default=None,
        help="JSONL file for cache priming rows; defaults to <out stem>_prime.jsonl.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip condition rows already present in --out for the same mode/cache label.",
    )
    parser.add_argument("--power-limit-watts", type=float, default=None)
    parser.add_argument("--no-randomized-order", action="store_false", dest="randomized_order")
    parser.set_defaults(randomized_order=True)
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help=(
            "Total number of deterministic condition shards. Each condition is "
            "assigned to exactly one shard by a stable hash, so shards are disjoint "
            "and their union is the full matrix; merged shard outputs pass the "
            "normal completeness checker. Run AT MOST ONE process per GPU."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard this process runs, 0 <= shard-index < shard-count.",
    )
    return parser.parse_args(argv)


def build_conditions(args: argparse.Namespace) -> list[Condition]:
    conditions = [
        Condition(workload_id, action, batch_size, num_steps, repeat)
        for workload_id in expand_choice(args.workload, WORKLOADS)
        for action in expand_choice(args.action, ACTIONS)
        for batch_size in args.batch_size
        for num_steps in args.num_steps
        for repeat in range(1, args.repeats + 1)
    ]
    if args.randomized_order:
        random.Random(args.seed).shuffle(conditions)
    return conditions


def validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.first_k_steps < 0:
        raise ValueError("--first-k-steps must be >= 0")
    if args.prime_compile_cache and args.time_accounting_mode != "warm_cached_total":
        raise ValueError("--prime-compile-cache requires --time-accounting-mode warm_cached_total")
    if args.prime_compile_cache and args.cache_state == "cold_cache":
        raise ValueError("--prime-compile-cache requires a non-cold --cache-state label")
    if args.prime_out is None:
        args.prime_out = args.out.with_name(f"{args.out.stem}_prime{args.out.suffix or '.jsonl'}")
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count")


def cache_key(condition: Condition, args: argparse.Namespace) -> str:
    if args.cache_state == "cold_cache":
        return (
            f"{condition.workload_id}_{condition.action}"
            f"_b{condition.batch_size}_s{condition.num_steps}_r{condition.repeat}_seed{args.seed}"
        )
    return (
        f"{args.cache_state}_{condition.workload_id}_{condition.action}"
        f"_b{condition.batch_size}_s{condition.num_steps}_seed{args.seed}"
    )


def condition_key(condition: Condition, args: argparse.Namespace) -> tuple[object, ...]:
    return (
        condition.workload_id,
        condition.action,
        condition.batch_size,
        condition.num_steps,
        condition.repeat,
        args.time_accounting_mode,
        args.cache_state,
        args.device,
    )


def existing_condition_keys(path: Path) -> set[tuple[object, ...]]:
    keys: set[tuple[object, ...]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            keys.add(
                (
                    row.get("workload_id"),
                    row.get("action"),
                    row.get("batch_size"),
                    row.get("num_steps"),
                    row.get("repeat"),
                    row.get("time_accounting_mode"),
                    row.get("cache_state"),
                    row.get("requested_device") or row.get("execution_device"),
                )
            )
    return keys


def set_cache_env(condition: Condition, args: argparse.Namespace, env: dict[str, str]) -> None:
    key = cache_key(condition, args)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(args.cache_dir_root / "inductor" / key)
    env["TRITON_CACHE_DIR"] = str(args.cache_dir_root / "triton" / key)


def command_for(condition: Condition, args: argparse.Namespace, out: Path) -> list[str]:
    return [
        sys.executable,
        "src/run_shortjob.py",
        "--workload",
        condition.workload_id,
        "--action",
        condition.action,
        "--batch-size",
        str(condition.batch_size),
        "--num-steps",
        str(condition.num_steps),
        "--first-k-steps",
        str(args.first_k_steps),
        "--repeats",
        "1",
        "--repeat-start",
        str(condition.repeat),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--out",
        str(out),
        "--time-accounting-mode",
        args.time_accounting_mode,
        "--cache-state",
        args.cache_state,
        "--no-randomized-order",
    ]


def run_one(condition: Condition, args: argparse.Namespace, out: Path, *, purpose: str) -> bool:
    """Run a single condition subprocess with retries.

    Returns True on success. A condition crashes at import (before any output is
    written), so a retry is a clean fresh attempt with no risk of duplicate rows.
    Returns False only if every attempt failed.
    """
    env = os.environ.copy()
    set_cache_env(condition, args, env)
    command = command_for(condition, args, out)
    if args.power_limit_watts is not None:
        command.extend(["--power-limit-watts", str(args.power_limit_watts)])
    label = (
        f"{condition.workload_id} {condition.action} "
        f"b={condition.batch_size} steps={condition.num_steps} repeat={condition.repeat}"
    )
    attempts = SUBPROCESS_RETRIES + 1
    for attempt in range(1, attempts + 1):
        suffix = "" if attempt == 1 else f" (attempt {attempt}/{attempts})"
        print(f"[{purpose}] {label}{suffix}", flush=True)
        result = subprocess.run(command, env=env)
        if result.returncode == 0:
            return True
        print(
            f"[warn] {purpose} subprocess for [{label}] exited code={result.returncode} "
            f"(attempt {attempt}/{attempts})",
            flush=True,
        )
        if attempt < attempts and SUBPROCESS_RETRY_BACKOFF_S > 0:
            time.sleep(SUBPROCESS_RETRY_BACKOFF_S)
    print(f"[error] {purpose} condition failed after {attempts} attempt(s): {label}", flush=True)
    return False


def should_prime(condition: Condition, args: argparse.Namespace, primed: set[str]) -> bool:
    if not args.prime_compile_cache or condition.action not in COMPILE_CACHE_ACTIONS:
        return False
    key = cache_key(condition, args)
    if key in primed:
        return False
    primed.add(key)
    return True


def iter_conditions_to_run(args: argparse.Namespace, conditions: Iterable[Condition]) -> list[Condition]:
    all_conditions = list(conditions)
    if not args.skip_existing:
        return all_conditions
    existing = existing_condition_keys(args.out)
    selected = [condition for condition in all_conditions if condition_key(condition, args) not in existing]
    skipped = len(all_conditions) - len(selected)
    print(f"[resume] skip-existing enabled; skipped {skipped} already-recorded condition(s)", flush=True)
    return selected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    validate_args(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.prime_out.parent.mkdir(parents=True, exist_ok=True)
    sharded = select_shard(build_conditions(args), args.shard_count, args.shard_index)
    if args.shard_count > 1:
        print(
            f"[shard] shard {args.shard_index}/{args.shard_count}: "
            f"{len(sharded)} condition(s) selected",
            flush=True,
        )
    conditions = iter_conditions_to_run(args, sharded)
    primed: set[str] = set()
    total = len(conditions)
    failures: list[Condition] = []
    for index, condition in enumerate(conditions, start=1):
        if should_prime(condition, args, primed):
            run_one(condition, args, args.prime_out, purpose="prime")
        print(f"[measure] condition {index}/{total}", flush=True)
        if not run_one(condition, args, args.out, purpose="measure"):
            failures.append(condition)

    if failures:
        print(
            f"[summary] {len(failures)}/{total} condition(s) failed after retries; "
            "the rest completed. Re-run the same command with --skip-existing to fill gaps:",
            flush=True,
        )
        for condition in failures:
            print(
                f"  - {condition.workload_id} {condition.action} "
                f"b={condition.batch_size} steps={condition.num_steps} repeat={condition.repeat}",
                flush=True,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
