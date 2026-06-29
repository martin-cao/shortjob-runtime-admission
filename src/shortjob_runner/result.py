from __future__ import annotations

import time
import traceback
from typing import Any

import torch

from shortjob_runner.actions import execution_device_for_action, is_graph_action, run_cuda_graph, run_eager_like
from shortjob_runner.env import (
    nvidia_smi_snapshot,
    peak_gpu_memory_mb,
    power_limit_lock_record,
    reset_peak_gpu_memory,
    runtime_info,
    utc_now,
)
from shortjob_runner.schema import apply_total_time, base_result
from shortjob_runner.timing import synchronize
from shortjob_runner.types import Condition, RunnerStageError
from shortjob_runner.workloads import StepWorkload, make_workload


def is_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message or "cuda oom" in message


def concise_failure_reason(exc: BaseException) -> str:
    cause = exc.cause if isinstance(exc, RunnerStageError) else exc
    first_line = str(cause).splitlines()[0] if str(cause).splitlines() else cause.__class__.__name__
    if isinstance(exc, RunnerStageError):
        return f"{exc.stage}: {cause.__class__.__name__}: {first_line[:220]}"
    return f"{cause.__class__.__name__}: {first_line[:240]}"


def mark_failure(result: dict[str, Any], exc: Exception, workload: StepWorkload, action: str) -> None:
    cause = exc.cause if isinstance(exc, RunnerStageError) else exc
    failed_stage = exc.stage if isinstance(exc, RunnerStageError) else "execution"
    result["status"] = "infeasible" if failed_stage == "precheck" else "failed"
    result["failure_reason"] = concise_failure_reason(exc)
    result["oom"] = is_oom(cause)
    result["action_feasible"] = False
    result["compile_failed"] = failed_stage == "compile"
    result["precheck_failed"] = failed_stage == "precheck"
    result["capture_failed"] = failed_stage == "capture"
    result["traceback_tail"] = traceback.format_exc(limit=4)
    if workload.spec.shape_stability == "unstable" and action in {"graphs_only", "compile_plus_graphs"}:
        result["graph_break_detected"] = True


def run_condition(
    condition: Condition,
    device: torch.device,
    seed: int,
    machine: dict[str, Any],
    gpu: dict[str, Any],
    randomized_order: bool,
    num_repeats_planned: int,
    time_accounting_mode: str,
    cache_state: str,
    first_k_steps: int,
    power_limit_watts: float | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed + condition.repeat)
    requested_device = device
    execution_device = execution_device_for_action(condition.action, requested_device)
    if execution_device.type == "cuda":
        torch.cuda.manual_seed_all(seed + condition.repeat)

    cuda_context_init_s: float | None = None
    if execution_device.type == "cuda":
        context_start = time.perf_counter()
        torch.cuda.init()
        synchronize(execution_device)
        cuda_context_init_s = time.perf_counter() - context_start

    model_init_start = time.perf_counter()
    workload = make_workload(condition.workload_id, condition.batch_size, execution_device)
    model_init_runtime_s = time.perf_counter() - model_init_start
    result = base_result(
        condition=condition,
        workload=workload,
        machine=machine,
        gpu=gpu,
        runtime=runtime_info(),
        started_at_utc=utc_now(),
        condition_order_seed=seed,
        randomized_order=randomized_order,
        num_repeats_planned=num_repeats_planned,
        time_accounting_mode=time_accounting_mode,
        cache_state=cache_state,
    )
    result["requested_device"] = str(requested_device)
    result["execution_device"] = str(execution_device)
    result["model_init_runtime_s"] = round(model_init_runtime_s, 6)
    result["cuda_context_init_s"] = round(cuda_context_init_s, 6) if cuda_context_init_s is not None else None
    result["power_limit_lock"] = power_limit_lock_record(power_limit_watts)
    result["env_snapshot_before"] = nvidia_smi_snapshot(
        execution_device.index if execution_device.type == "cuda" else None
    )

    reset_peak_gpu_memory(execution_device)
    synchronize(execution_device)
    total_start = time.perf_counter()
    try:
        if is_graph_action(condition.action):
            stats = run_cuda_graph(workload, condition.action, condition.num_steps, execution_device)
        else:
            stats = run_eager_like(workload, condition.action, condition.num_steps, execution_device, first_k_steps)
        synchronize(execution_device)
        result.update(stats)
        apply_total_time(result, time.perf_counter() - total_start, time_accounting_mode)
        if result.get("measured_runtime_s") not in (None, 0):
            result["throughput_per_s"] = round(condition.num_steps / float(result["measured_runtime_s"]), 6)
        result["action_feasible"] = True
    except Exception as exc:
        synchronize(execution_device)
        apply_total_time(result, time.perf_counter() - total_start, time_accounting_mode)
        mark_failure(result, exc, workload, condition.action)
    finally:
        result["peak_gpu_memory_mb"] = peak_gpu_memory_mb(execution_device)
        result["env_snapshot_after"] = nvidia_smi_snapshot(
            execution_device.index if execution_device.type == "cuda" else None
        )
        result["finished_at_utc"] = utc_now()
    return result
