from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from shortjob_runner.actions import execution_device_for_action, is_graph_action, run_cuda_graph, run_eager_like
from shortjob_runner.env import machine_info, nvidia_smi_snapshot, runtime_info, utc_now
from shortjob_runner.workloads import make_workload


def command_available(name: str) -> bool:
    return shutil.which(name) is not None


def base_runner_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "src/run_shortjob.py",
        "--workload",
        args.workload,
        "--action",
        args.action,
        "--batch-size",
        str(args.batch_size),
        "--num-steps",
        str(args.num_steps),
        "--repeats",
        "1",
        "--device",
        args.device,
        "--out",
        str(args.runner_out),
        "--no-randomized-order",
    ]


def build_external_profile_commands(args: argparse.Namespace) -> dict[str, list[str] | None]:
    runner = base_runner_command(args)
    output_dir = args.out_dir
    return {
        "nsys": [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--force-overwrite=true",
            f"--output={output_dir / (args.profile_id + '_nsys')}",
            *runner,
        ]
        if command_available("nsys")
        else None,
        "ncu": [
            "ncu",
            "--set",
            "launch",
            "--target-processes",
            "all",
            "--export",
            str(output_dir / (args.profile_id + "_ncu")),
            *runner,
        ]
        if command_available("ncu")
        else None,
        "perf": [
            "perf",
            "record",
            "-g",
            "-o",
            str(output_dir / (args.profile_id + "_perf.data")),
            *runner,
        ]
        if command_available("perf")
        else None,
    }


def run_pytorch_profile(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    execution_device = execution_device_for_action(args.action, device)
    torch.manual_seed(args.seed)
    if execution_device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    workload = make_workload(args.workload, args.batch_size, execution_device)
    activities = [ProfilerActivity.CPU]
    if execution_device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function(f"shortjob:{args.workload}:{args.action}"):
            if is_graph_action(args.action):
                stats = run_cuda_graph(workload, args.action, args.num_steps, execution_device)
            else:
                stats = run_eager_like(workload, args.action, args.num_steps, execution_device, first_k_steps=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chrome_trace = args.out_dir / f"{args.profile_id}_torch_trace.json"
    prof.export_chrome_trace(str(chrome_trace))
    table = prof.key_averages().table(sort_by="self_cuda_time_total" if execution_device.type == "cuda" else "self_cpu_time_total", row_limit=args.row_limit)
    table_path = args.out_dir / f"{args.profile_id}_torch_ops.txt"
    table_path.write_text(table, encoding="utf-8")
    events = prof.key_averages()
    return {
        "profile_id": args.profile_id,
        "tool": "torch.profiler",
        "created_at_utc": utc_now(),
        "workload_id": args.workload,
        "action": args.action,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "device": str(device),
        "execution_device": str(execution_device),
        "machine": machine_info(),
        "runtime": runtime_info(),
        "env_snapshot": nvidia_smi_snapshot(execution_device.index if execution_device.type == "cuda" else None),
        "runner_stats": stats,
        "chrome_trace": str(chrome_trace),
        "operator_table": str(table_path),
        "operator_count": len(events),
        "total_self_cpu_time_us": round(sum(float(event.self_cpu_time_total) for event in events), 3),
        "total_self_cuda_time_us": round(
            sum(float(getattr(event, "self_cuda_time_total", 0.0)) for event in events),
            3,
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profiler wrapper for short-job root-cause evidence")
    parser.add_argument("--workload", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile-id", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results/profiles"))
    parser.add_argument("--runner-out", type=Path, default=Path("/tmp/shortjob_profile_runner.jsonl"))
    parser.add_argument("--row-limit", type=int, default=40)
    parser.add_argument("--emit-commands-only", action="store_true")
    parser.add_argument("--run-external", choices=("none", "nsys", "ncu", "perf"), default="none")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.profile_id is None:
        args.profile_id = f"{args.workload}_{args.action}_b{args.batch_size}_s{args.num_steps}"
    commands = build_external_profile_commands(args)
    if args.emit_commands_only:
        print(json.dumps({key: value for key, value in commands.items()}, ensure_ascii=False, indent=2))
        return 0
    if args.run_external != "none":
        command = commands.get(args.run_external)
        if command is None:
            raise RuntimeError(f"{args.run_external} is not available on PATH")
        subprocess.run(command, check=True)
        return 0
    summary = run_pytorch_profile(args)
    write_json(args.out_dir / f"{args.profile_id}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
