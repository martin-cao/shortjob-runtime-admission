#!/usr/bin/env python3
"""Probe CUDA Graph capture behavior without applying admission precheck."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

from shortjob_runner.env import gpu_info, machine_info, utc_now
from shortjob_runner.io import write_jsonl
from shortjob_runner.types import WORKLOADS
from shortjob_runner.workloads import make_workload
from shortjob_runner.actions import compile_mode_for_action


def short_error(exc: BaseException) -> str:
    lines = traceback.format_exception_only(type(exc), exc)
    return "".join(lines).strip().replace("\n", " | ")


def probe_one(workload_id: str, action: str, batch_size: int, replay_steps: int, device: torch.device) -> dict[str, Any]:
    row: dict[str, Any] = {
        "started_at_utc": utc_now(),
        "workload_id": workload_id,
        "action": action,
        "batch_size": batch_size,
        "replay_steps": replay_steps,
        "machine": machine_info(),
        "gpu": gpu_info(device),
        "status": "unknown",
        "precheck_feasible": None,
        "precheck_reason": None,
        "capture_failed": False,
        "replay_failed": False,
        "failure_stage": None,
        "failure_reason": None,
    }
    try:
        workload = make_workload(workload_id, batch_size=batch_size, device=device)
        feasible, reason = workload.graph_capture_feasible()
        row["precheck_feasible"] = feasible
        row["precheck_reason"] = reason

        if action in {"compile_plus_graphs", "compile_reduce_overhead_plus_graphs"}:
            workload.enable_compile(compile_mode_for_action(action))

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                workload.step()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                workload.step()
        except Exception as exc:
            row.update(
                {
                    "status": "capture_failed",
                    "capture_failed": True,
                    "failure_stage": "capture",
                    "failure_reason": short_error(exc),
                    "finished_at_utc": utc_now(),
                }
            )
            return row

        try:
            for _ in range(replay_steps):
                if action == "graphs_input_copy":
                    workload.refresh_request_inputs()
                graph.replay()
            torch.cuda.synchronize(device)
        except Exception as exc:
            row.update(
                {
                    "status": "replay_failed",
                    "replay_failed": True,
                    "failure_stage": "replay",
                    "failure_reason": short_error(exc),
                    "finished_at_utc": utc_now(),
                }
            )
            return row

        row.update({"status": "completed", "finished_at_utc": utc_now()})
        return row
    except Exception as exc:
        row.update(
            {
                "status": "setup_failed",
                "failure_stage": "setup",
                "failure_reason": short_error(exc),
                "finished_at_utc": utc_now(),
            }
        )
        return row


def expand_workloads(values: list[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        expanded.extend(WORKLOADS if value == "all" else [value])
    return list(dict.fromkeys(expanded))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe real CUDA Graph capture/replay behavior for shortjob workloads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workload", nargs="+", choices=("all", *WORKLOADS), default=["all"])
    parser.add_argument(
        "--action",
        nargs="+",
        choices=("graphs_only", "graphs_input_copy", "compile_plus_graphs", "compile_reduce_overhead_plus_graphs"),
        default=["graphs_only"],
    )
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--replay-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("data/raw/cuda_graph_capture_probe.jsonl"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    rows = [
        probe_one(workload_id, action, batch_size, args.replay_steps, device)
        for workload_id in expand_workloads(args.workload)
        for action in args.action
        for batch_size in args.batch_size
    ]
    write_jsonl(args.out, rows)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
