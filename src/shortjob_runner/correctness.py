from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import torch

from shortjob_runner.actions import (
    GRAPH_WARMUP_STEPS,
    execution_device_for_action,
    is_graph_action,
    run_cuda_graph,
    run_eager_like,
    step_callable,
)
from shortjob_runner.env import gpu_info, runtime_info, utc_now
from shortjob_runner.types import ACTIONS, WORKLOADS, RunnerStageError
from shortjob_runner.workloads import StepWorkload, make_workload


def expand_choice(values: list[str], all_values: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        expanded.extend(all_values if value == "all" else [value])
    return list(dict.fromkeys(expanded))


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def make_seeded_workload(workload_id: str, batch_size: int, device: torch.device, seed: int) -> StepWorkload:
    set_seed(seed, device)
    return make_workload(workload_id, batch_size, device)


def run_action(
    workload: StepWorkload,
    action: str,
    num_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    if is_graph_action(action):
        return run_cuda_graph(workload, action, num_steps, device)
    return run_eager_like(
        workload=workload,
        action=action,
        num_steps=num_steps,
        device=device,
        first_k_steps=0,
    )


def eager_reference_state(
    workload: StepWorkload,
    action: str,
    num_steps: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Drive ``workload`` with plain eager over the SAME logical trajectory the
    ``action`` candidate follows, then return its correctness state.

    A correctness failure should mean the action's *computation* diverges from
    eager for identical inputs and identical accumulated state — not that the
    harness fed the two paths different work. Two trajectory details must match:

    * Step count. CUDA-graph actions execute ``GRAPH_WARMUP_STEPS`` warm-up
      steps plus one capture step before the measured replays. For stateful
      (training) workloads those extra steps mutate parameters in place, so the
      eager reference must take the same number of steps to be comparable.
    * Input-refresh schedule. ``graphs_input_copy`` refreshes request tensors
      before each replay; the reference refreshes from an identically-seeded
      generator (see ``StepWorkload.seed_refresh``) so both observe the same
      input sequence and the final outputs are legitimately comparable.

    The step itself is always plain eager regardless of ``action`` — that is the
    reference semantics that make amp/tf32/compile/graph variants comparable to
    a single fp32 eager baseline.
    """
    step = step_callable(workload, "eager", device)
    if is_graph_action(action):
        for _ in range(GRAPH_WARMUP_STEPS + 1):  # warm-up + capture, on initial inputs
            step()
        for _ in range(num_steps):
            if action == "graphs_input_copy":
                workload.refresh_request_inputs()
            step()
    else:
        for _ in range(num_steps):
            step()
    return workload.correctness_state()


def concise_failure_reason(exc: BaseException) -> str:
    cause = exc.cause if isinstance(exc, RunnerStageError) else exc
    first_line = str(cause).splitlines()[0] if str(cause).splitlines() else cause.__class__.__name__
    if isinstance(exc, RunnerStageError):
        return f"{exc.stage}: {cause.__class__.__name__}: {first_line[:220]}"
    return f"{cause.__class__.__name__}: {first_line[:240]}"


def tensor_errors(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    ref = reference.detach().to(dtype=torch.float64, device="cpu")
    cand = candidate.detach().to(dtype=torch.float64, device="cpu")
    diff = (cand - ref).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = ref.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    return max_abs, max_rel


def compare_states(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    missing_from_candidate = sorted(set(reference) - set(candidate))
    extra_in_candidate = sorted(set(candidate) - set(reference))
    if missing_from_candidate or extra_in_candidate:
        return {
            "passed": False,
            "mismatch_key": None,
            "worst_error_key": None,
            "max_abs_error": None,
            "max_rel_error": None,
            "failure_reason": (
                f"state key mismatch; missing_from_candidate={missing_from_candidate}; "
                f"extra_in_candidate={extra_in_candidate}"
            ),
        }

    worst_abs = 0.0
    worst_rel = 0.0
    worst_key: str | None = None
    for key in sorted(reference):
        ref = reference[key]
        cand = candidate[key]
        if ref.shape != cand.shape:
            return {
                "passed": False,
                "mismatch_key": key,
                "worst_error_key": None,
                "max_abs_error": None,
                "max_rel_error": None,
                "failure_reason": f"shape mismatch for {key}: reference={tuple(ref.shape)} candidate={tuple(cand.shape)}",
            }
        max_abs, max_rel = tensor_errors(ref, cand)
        if max_abs > worst_abs:
            worst_abs = max_abs
            worst_rel = max_rel
            worst_key = key
        try:
            torch.testing.assert_close(cand, ref, rtol=rtol, atol=atol)
        except AssertionError as exc:
            return {
                "passed": False,
                "mismatch_key": key,
                "worst_error_key": key,
                "max_abs_error": round(max_abs, 12),
                "max_rel_error": round(max_rel, 12),
                "failure_reason": str(exc).splitlines()[0][:240],
            }

    return {
        "passed": True,
        "mismatch_key": None,
        "worst_error_key": worst_key,
        "max_abs_error": round(worst_abs, 12),
        "max_rel_error": round(worst_rel, 12),
        "failure_reason": None,
    }


def check_one(
    workload_id: str,
    action: str,
    batch_size: int,
    num_steps: int,
    device: torch.device,
    seed: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "checked_at_utc": utc_now(),
        "workload_id": workload_id,
        "reference_action": "eager",
        "action": action,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "seed": seed,
        "device": str(device),
        "execution_device": str(execution_device_for_action(action, device)),
        "gpu": gpu_info(device),
        "runtime": runtime_info(),
        "rtol": rtol,
        "atol": atol,
        "status": None,
        # failure_stage cleanly separates the four outcomes downstream code must
        # not conflate (reviewer item 5):
        #   None        -> passed (completed + numerically correct)
        #   "precheck"  -> rejected before execution (infeasible)
        #   "execution" -> ran but raised (runtime execution failure)
        #   "correctness" -> ran to completion but state mismatched eager (wrong output)
        "failure_stage": None,
        "correctness_failed": None,
        "action_feasible": None,
        "compared_state_keys": None,
        "num_compared_state_keys": None,
        "max_abs_error": None,
        "max_rel_error": None,
        "mismatch_key": None,
        "worst_error_key": None,
        "passed": None,
        "failure_reason": None,
        "reference_stats": None,
        "action_stats": None,
        "elapsed_s": None,
    }

    started = time.perf_counter()
    execution_device = execution_device_for_action(action, device)
    row["execution_device"] = str(execution_device)
    if is_graph_action(action) and execution_device.type != "cuda":
        row.update(
            {
                "status": "skipped",
                "failure_stage": "precheck",
                "correctness_failed": False,
                "action_feasible": False,
                "failure_reason": "CUDA Graph actions require a CUDA device",
                "elapsed_s": round(time.perf_counter() - started, 6),
            }
        )
        return row

    try:
        # Informational eager baseline timing (plain num_steps eager run).
        stats_workload = make_seeded_workload(workload_id, batch_size, execution_device, seed)
        row["reference_stats"] = run_action(stats_workload, "eager", num_steps, execution_device)

        # Correctness reference: eager driven over the SAME trajectory as the
        # action candidate (matched step count + input-refresh schedule).
        reference = make_seeded_workload(workload_id, batch_size, execution_device, seed)
        reference.seed_refresh(seed, execution_device)
        reference_state = eager_reference_state(reference, action, num_steps, execution_device)
        row["compared_state_keys"] = sorted(reference_state)
        row["num_compared_state_keys"] = len(reference_state)

        candidate = make_seeded_workload(workload_id, batch_size, execution_device, seed)
        candidate.seed_refresh(seed, execution_device)
        action_stats = run_action(candidate, action, num_steps, execution_device)
        candidate_state = candidate.correctness_state()
        row["action_stats"] = action_stats

        comparison = compare_states(reference_state, candidate_state, rtol=rtol, atol=atol)
        row.update(comparison)
        row["action_feasible"] = True
        if comparison["passed"]:
            row["status"] = "passed"
            row["failure_stage"] = None
            row["correctness_failed"] = False
        else:
            # Ran to completion but produced a wrong/incoherent state vs eager.
            # This is a CORRECTNESS failure, not runtime infeasibility.
            row["status"] = "failed"
            row["failure_stage"] = "correctness"
            row["correctness_failed"] = True
    except Exception as exc:
        failed_stage = exc.stage if isinstance(exc, RunnerStageError) else "execution"
        row["status"] = "infeasible" if failed_stage == "precheck" else "failed"
        row["failure_stage"] = "precheck" if failed_stage == "precheck" else "execution"
        row["correctness_failed"] = False
        row["action_feasible"] = False
        row["failure_reason"] = concise_failure_reason(exc)
        row["traceback_tail"] = traceback.format_exc(limit=4)
    finally:
        row["elapsed_s"] = round(time.perf_counter() - started, 6)

    return row


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shortjob runtime-action correctness gate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workload", nargs="+", choices=("all", *WORKLOADS), default=["all"])
    parser.add_argument("--action", nargs="+", choices=("all", *ACTIONS), default=["all"])
    parser.add_argument("--batch-size", nargs="+", type=int, default=[4])
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if any(value < 1 for value in args.batch_size):
        raise ValueError("--batch-size values must be >= 1")
    if args.num_steps < 1:
        raise ValueError("--num-steps must be >= 1")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("--rtol and --atol must be non-negative")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    validate_args(args)
    device = torch.device(args.device)
    workloads = expand_choice(args.workload, WORKLOADS)
    actions = expand_choice(args.action, ACTIONS)

    rows: list[dict[str, Any]] = []
    for workload_id in workloads:
        for action in actions:
            for batch_size in args.batch_size:
                row = check_one(
                    workload_id=workload_id,
                    action=action,
                    batch_size=batch_size,
                    num_steps=args.num_steps,
                    device=device,
                    seed=args.seed,
                    rtol=args.rtol,
                    atol=args.atol,
                )
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False, allow_nan=False))
                if args.out is not None:
                    write_jsonl(args.out, row)

    failed = [row for row in rows if row["status"] == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
