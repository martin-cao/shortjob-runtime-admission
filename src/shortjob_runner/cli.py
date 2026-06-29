from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

from shortjob_runner.env import gpu_info, machine_info
from shortjob_runner.result import run_condition
from shortjob_runner.types import ACTIONS, WORKLOADS, Condition


def expand_choice(values: list[str], all_values: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        expanded.extend(all_values if value == "all" else [value])
    return list(dict.fromkeys(expanded))


def build_conditions(args: argparse.Namespace) -> list[Condition]:
    conditions = [
        Condition(workload_id, action, batch_size, num_steps, repeat)
        for workload_id, action, batch_size, num_steps, repeat in itertools.product(
            expand_choice(args.workload, WORKLOADS),
            expand_choice(args.action, ACTIONS),
            args.batch_size,
            args.num_steps,
            range(args.repeat_start, args.repeat_start + args.repeats),
        )
    ]
    if args.randomized_order:
        random.Random(args.seed).shuffle(conditions)
    return conditions


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal shortjob torch.compile / CUDA Graphs admission runner",
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
            "runs; committed n=3 artifacts are unaffected and CI uses per-n Student-t."
        ),
    )
    parser.add_argument("--repeat-start", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", help="Device such as cuda or cuda:0; CPU is not recommended for the main experiment path")
    parser.add_argument("--out", type=Path, default=Path("data/raw/shortjob_pilot.jsonl"))
    parser.add_argument("--no-randomized-order", action="store_false", dest="randomized_order")
    parser.add_argument("--time-accounting-mode", choices=("cold_total", "warm_cached_total"), default="cold_total")
    parser.add_argument("--cache-state", default="cold_cache")
    parser.add_argument("--power-limit-watts", type=float, default=None)
    parser.set_defaults(randomized_order=True)
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
    if args.first_k_steps < 0:
        raise ValueError("--first-k-steps must be >= 0")
    selected_actions = expand_choice(args.action, ACTIONS)
    only_cpu = selected_actions == ["cpu_eager"]
    if args.device.startswith("cuda") and not torch.cuda.is_available() and not only_cpu:
        raise RuntimeError("CUDA is not available; this runner is intended for single-GPU experiments")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    validate_args(args)

    device = torch.device(args.device)
    machine = machine_info()
    gpu = (
        gpu_info(device)
        if device.type != "cuda" or torch.cuda.is_available()
        else {"name": None, "available": False, "requested_device": str(device)}
    )
    for condition in build_conditions(args):
        row = run_condition(
            condition=condition,
            device=device,
            seed=args.seed,
            machine=machine,
            gpu=gpu,
            randomized_order=args.randomized_order,
            num_repeats_planned=args.repeats,
            time_accounting_mode=args.time_accounting_mode,
            cache_state=args.cache_state,
            first_k_steps=args.first_k_steps,
            power_limit_watts=args.power_limit_watts,
        )
        write_jsonl(args.out, row)

    return 0
