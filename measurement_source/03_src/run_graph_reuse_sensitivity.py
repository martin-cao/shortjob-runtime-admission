#!/usr/bin/env python3
"""Run bounded repeated-shape CUDA Graph reuse sensitivity experiments.

This runner measures a single worker process that sees multiple jobs with the
same workload, batch size, step budget, and shape. Graph actions pay capture
once, then replay the captured graph for several same-shape jobs. The result is
not a production graph cache; it is a bounded sensitivity for whether graph
capture cost changes role when the shape repeats inside a worker.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

from shortjob_runner.actions import action_eligibility, step_callable
from shortjob_runner.cli import expand_choice
from shortjob_runner.env import gpu_info, machine_info, runtime_info, utc_now
from shortjob_runner.timing import synchronize, timed, timed_loop, timed_loop_with_input_copy
from shortjob_runner.types import WORKLOADS
from shortjob_runner.workloads import make_workload


DEFAULT_WORKLOADS = (
    "cv_online_infer",
    "llm_decode_proxy",
    "dynamic_shape_infer_small",
    "short_train_small",
)
DEFAULT_ACTIONS = ("eager", "best_eager", "graphs_only", "graphs_input_copy")


def short_error(exc: BaseException) -> str:
    lines = traceback.format_exception_only(type(exc), exc)
    return "".join(lines).strip().replace("\n", " | ")[:260]


def write_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def existing_keys(path: Path) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add(
                (
                    row.get("workload_id"),
                    row.get("action"),
                    row.get("batch_size"),
                    row.get("num_steps"),
                    row.get("reuse_jobs"),
                    row.get("repeat"),
                    row.get("device_label"),
                )
            )
    return keys


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("workload_id"),
        row.get("action"),
        row.get("batch_size"),
        row.get("num_steps"),
        row.get("reuse_jobs"),
        row.get("repeat"),
        row.get("device_label"),
    )


def base_row(
    *,
    workload_id: str,
    action: str,
    batch_size: int,
    num_steps: int,
    reuse_jobs: int,
    repeat: int,
    device: torch.device,
    device_label: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": "graph_reuse_raw_v1",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "device_label": device_label,
        "machine": machine_info(),
        "gpu": gpu_info(device),
        "runtime": runtime_info(),
        "workload_id": workload_id,
        "action": action,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "reuse_jobs": reuse_jobs,
        "repeat": repeat,
        "seed": seed,
        "requested_device": str(device),
        "time_accounting_mode": "graph_reuse_total",
        "cache_state": "warm_graph_cache",
        "status": "completed",
        "action_feasible": True,
        "failure_reason": None,
        "capture_overhead_s": None,
        "graph_warmup_runtime_s": None,
        "graph_instantiate_runtime_s": None,
        "input_copy_overhead_s": 0.0,
        "input_copy_bytes": 0,
        "first_job_total_runtime_s": None,
        "median_job_runtime_s": None,
        "amortized_runtime_per_job_s": None,
        "total_group_runtime_s": None,
        "job_runtime_s": [],
    }


def median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0


def run_eager_reuse(
    workload_id: str,
    action: str,
    batch_size: int,
    num_steps: int,
    reuse_jobs: int,
    repeat: int,
    device: torch.device,
    device_label: str,
    seed: int,
) -> dict[str, Any]:
    row = base_row(
        workload_id=workload_id,
        action=action,
        batch_size=batch_size,
        num_steps=num_steps,
        reuse_jobs=reuse_jobs,
        repeat=repeat,
        device=device,
        device_label=device_label,
        seed=seed,
    )
    try:
        torch.manual_seed(seed + repeat)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + repeat)
            torch.cuda.init()
        workload = make_workload(workload_id, batch_size=batch_size, device=device)
        feasible, reason = action_eligibility(workload, action, device)
        if not feasible:
            raise RuntimeError(reason or "action_not_eligible")
        step_fn = step_callable(workload, action, device)
        job_times = [timed_loop(device, step_fn, num_steps) for _ in range(reuse_jobs)]
        total_s = sum(job_times)
        row.update(
            {
                "job_runtime_s": [round(value, 6) for value in job_times],
                "first_job_total_runtime_s": round(job_times[0], 6),
                "median_job_runtime_s": round(median(job_times) or 0.0, 6),
                "amortized_runtime_per_job_s": round(total_s / reuse_jobs, 6),
                "total_group_runtime_s": round(total_s, 6),
            }
        )
    except Exception as exc:
        row.update({"status": "infeasible", "action_feasible": False, "failure_reason": short_error(exc)})
    row["finished_at_utc"] = utc_now()
    return row


def capture_graph(
    workload_id: str,
    action: str,
    batch_size: int,
    device: torch.device,
    seed: int,
    repeat: int,
) -> tuple[torch.cuda.CUDAGraph, Callable[[], int], float, float, float]:
    torch.manual_seed(seed + repeat)
    torch.cuda.manual_seed_all(seed + repeat)
    torch.cuda.init()
    workload = make_workload(workload_id, batch_size=batch_size, device=device)
    feasible, reason = action_eligibility(workload, action, device)
    if not feasible:
        raise RuntimeError(reason or "graph_capture_not_feasible")
    if action == "graphs_input_copy":
        copy_feasible, copy_reason = workload.input_copy_feasible()
        if not copy_feasible:
            raise RuntimeError(copy_reason or "input_copy_not_supported")

    step_fn = step_callable(workload, action, device)
    warmup_s = 0.0
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            warmup_s += timed(device, step_fn)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    synchronize(device)

    graph = torch.cuda.CUDAGraph()

    def do_capture() -> None:
        with torch.cuda.graph(graph):
            step_fn()

    capture_s = timed(device, do_capture)
    capture_overhead_s = warmup_s + capture_s

    def copy_inputs() -> int:
        return workload.refresh_request_inputs() if action == "graphs_input_copy" else 0

    return graph, copy_inputs, capture_overhead_s, warmup_s, capture_s


def run_graph_reuse(
    workload_id: str,
    action: str,
    batch_size: int,
    num_steps: int,
    reuse_jobs: int,
    repeat: int,
    device: torch.device,
    device_label: str,
    seed: int,
) -> dict[str, Any]:
    row = base_row(
        workload_id=workload_id,
        action=action,
        batch_size=batch_size,
        num_steps=num_steps,
        reuse_jobs=reuse_jobs,
        repeat=repeat,
        device=device,
        device_label=device_label,
        seed=seed,
    )
    try:
        if device.type != "cuda":
            raise RuntimeError("CUDA Graph reuse requires CUDA")
        graph, copy_inputs, capture_overhead_s, warmup_s, instantiate_s = capture_graph(
            workload_id, action, batch_size, device, seed, repeat
        )
        job_times: list[float] = []
        input_copy_s = 0.0
        copied_bytes = 0
        for _ in range(reuse_jobs):
            if action == "graphs_input_copy":
                job_s, copy_s, bytes_this_job = timed_loop_with_input_copy(
                    device, copy_inputs, graph.replay, num_steps
                )
                input_copy_s += copy_s
                copied_bytes += bytes_this_job
            else:
                job_s = timed_loop(device, graph.replay, num_steps)
            job_times.append(job_s)
        total_group_s = capture_overhead_s + sum(job_times)
        row.update(
            {
                "capture_overhead_s": round(capture_overhead_s, 6),
                "graph_warmup_runtime_s": round(warmup_s, 6),
                "graph_instantiate_runtime_s": round(instantiate_s, 6),
                "input_copy_overhead_s": round(input_copy_s, 6),
                "input_copy_bytes": copied_bytes,
                "job_runtime_s": [round(value, 6) for value in job_times],
                "first_job_total_runtime_s": round(capture_overhead_s + job_times[0], 6),
                "median_job_runtime_s": round(median(job_times) or 0.0, 6),
                "amortized_runtime_per_job_s": round(total_group_s / reuse_jobs, 6),
                "total_group_runtime_s": round(total_group_s, 6),
            }
        )
    except Exception as exc:
        row.update({"status": "infeasible", "action_feasible": False, "failure_reason": short_error(exc)})
    row["finished_at_utc"] = utc_now()
    return row


def run_condition(
    workload_id: str,
    action: str,
    batch_size: int,
    num_steps: int,
    reuse_jobs: int,
    repeat: int,
    device: torch.device,
    device_label: str,
    seed: int,
) -> dict[str, Any]:
    if action in {"graphs_only", "graphs_input_copy"}:
        return run_graph_reuse(workload_id, action, batch_size, num_steps, reuse_jobs, repeat, device, device_label, seed)
    return run_eager_reuse(workload_id, action, batch_size, num_steps, reuse_jobs, repeat, device, device_label, seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded repeated-shape CUDA Graph reuse sensitivity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workload", nargs="+", choices=("all", *WORKLOADS), default=list(DEFAULT_WORKLOADS))
    parser.add_argument("--action", nargs="+", choices=DEFAULT_ACTIONS, default=list(DEFAULT_ACTIONS))
    parser.add_argument("--batch-size", nargs="+", type=int, default=[4, 16])
    parser.add_argument("--num-steps", nargs="+", type=int, default=[50, 100, 500])
    parser.add_argument("--reuse-jobs", nargs="+", type=int, default=[8])
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help=(
            "Measurement repeats per condition. Default 5 for new runs; committed "
            "n=3 graph-reuse artifacts are unaffected and CI uses per-n Student-t."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-label", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-randomized-order", action="store_false", dest="randomized_order")
    parser.set_defaults(randomized_order=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if any(value < 1 for value in args.batch_size):
        raise ValueError("--batch-size values must be >= 1")
    if any(value < 1 for value in args.num_steps):
        raise ValueError("--num-steps values must be >= 1")
    if any(value < 1 for value in args.reuse_jobs):
        raise ValueError("--reuse-jobs values must be >= 1")
    selected_actions = expand_choice(args.action, DEFAULT_ACTIONS)
    if any(action.startswith("graphs") for action in selected_actions):
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; graph reuse sensitivity requires a GPU")


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if args.device_label is None:
        args.device_label = gpu_info(device).get("name") or str(device)
    workloads = expand_choice(args.workload, WORKLOADS)
    actions = expand_choice(args.action, DEFAULT_ACTIONS)
    conditions = list(
        itertools.product(
            workloads,
            actions,
            args.batch_size,
            args.num_steps,
            args.reuse_jobs,
            range(1, args.repeats + 1),
        )
    )
    if args.randomized_order:
        random.Random(args.seed).shuffle(conditions)
    seen = existing_keys(args.out) if args.skip_existing else set()
    for workload_id, action, batch_size, num_steps, reuse_jobs, repeat in conditions:
        candidate = {
            "workload_id": workload_id,
            "action": action,
            "batch_size": batch_size,
            "num_steps": num_steps,
            "reuse_jobs": reuse_jobs,
            "repeat": repeat,
            "device_label": args.device_label,
        }
        if row_key(candidate) in seen:
            continue
        print(
            f"[graph-reuse] {args.device_label} {workload_id} {action} "
            f"b={batch_size} steps={num_steps} reuse_jobs={reuse_jobs} repeat={repeat}",
            flush=True,
        )
        row = run_condition(
            workload_id,
            action,
            batch_size,
            num_steps,
            reuse_jobs,
            repeat,
            device,
            args.device_label,
            args.seed,
        )
        write_row(args.out, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
