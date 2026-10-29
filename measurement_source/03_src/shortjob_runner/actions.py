from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterator

import torch

from shortjob_runner.timing import (
    avg_ms_from_total,
    mean_ms,
    median_ms,
    quantile_ms,
    timed,
    timed_loop,
    timed_loop_with_device_events,
    timed_loop_with_input_copy,
    timed_steps,
    variance_ms,
)
from shortjob_runner.types import RunnerStageError
from shortjob_runner.workloads import StepWorkload

# Number of warm-up step executions run before a CUDA graph is captured. The
# correctness gate mirrors this exactly when building the eager reference for a
# graph action so that stateful (training) workloads, whose steps mutate
# parameters in place, are compared after an identical number of update steps.
GRAPH_WARMUP_STEPS = 3


def run_stage(stage: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:
        raise RunnerStageError(stage, exc) from exc


def execution_device_for_action(action: str, requested_device: torch.device) -> torch.device:
    if action == "cpu_eager":
        return torch.device("cpu")
    return requested_device


def compile_mode_for_action(action: str) -> str | None:
    if action in {"compile_reduce_overhead", "compile_reduce_overhead_plus_graphs"}:
        return "reduce-overhead"
    if action == "compile_max_autotune":
        return "max-autotune"
    return None


def is_compile_action(action: str) -> bool:
    return action in {
        "compile_only",
        "compile_reduce_overhead",
        "compile_max_autotune",
        "compile_plus_graphs",
        "compile_reduce_overhead_plus_graphs",
    }


def is_graph_action(action: str) -> bool:
    return action in {
        "graphs_only",
        "graphs_input_copy",
        "compile_plus_graphs",
        "compile_reduce_overhead_plus_graphs",
    }


def micro_batch_factor(action: str) -> int | None:
    if action == "micro_batch_2":
        return 2
    if action == "micro_batch_4":
        return 4
    return None


def precision_context(action: str, device: torch.device) -> Iterator[None]:
    if action == "amp_fp16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if action == "amp_bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@contextlib.contextmanager
def tf32_context(action: str) -> Iterator[None]:
    if action != "tf32_eager" or not hasattr(torch.backends, "cuda"):
        yield
        return
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def step_callable(workload: StepWorkload, action: str, device: torch.device) -> Callable[[], None]:
    factor = micro_batch_factor(action)

    def call_step() -> None:
        with tf32_context(action):
            with precision_context(action, device):
                if factor is None:
                    workload.step()
                else:
                    workload.step_micro_batch(factor)

    if action in {"eager_inference_mode", "best_eager", "cpu_eager"} and not workload.spec.supports_training:
        def call_inference() -> None:
            with torch.inference_mode():
                call_step()

        return call_inference
    return call_step


def action_eligibility(workload: StepWorkload, action: str, device: torch.device) -> tuple[bool, str | None]:
    if action == "cpu_eager":
        return workload.spec.supports_cpu, None if workload.spec.supports_cpu else "cpu_baseline_not_supported"
    if action in {"amp_fp16", "amp_bf16"} and (device.type != "cuda" or not workload.spec.supports_mixed_precision):
        return False, "mixed_precision_not_supported_for_workload_or_device"
    if action in {"eager_inference_mode", "best_eager"} and workload.spec.supports_training:
        return False, "inference_mode_not_valid_for_training_workload"
    factor = micro_batch_factor(action)
    if factor is not None and not workload.spec.supports_micro_batching:
        return False, "micro_batching_not_supported_for_workload"
    if is_graph_action(action):
        return workload.graph_capture_feasible()
    return True, None


def run_eager_like(
    workload: StepWorkload,
    action: str,
    num_steps: int,
    device: torch.device,
    first_k_steps: int,
) -> dict[str, Any]:
    compile_overhead_s: float | None = None
    compile_call_overhead_s: float | None = None
    completed_steps = 0
    measured_runtime_s = 0.0
    first_k_durations_s: list[float] = []
    step_durations_s: list[float] = []
    device_event_runtime_s: float | None = None
    feasible, reason = action_eligibility(workload, action, device)
    if not feasible:
        raise RunnerStageError("precheck", RuntimeError(reason or "action_not_eligible"))

    step_fn = step_callable(workload, action, device)

    if is_compile_action(action):
        compile_call_overhead_s = run_stage(
            "compile",
            lambda: timed(device, lambda: workload.enable_compile(compile_mode_for_action(action))),
        )
        if num_steps > 0:
            first_step_s = run_stage("compile", lambda: timed(device, step_fn))
            compile_overhead_s = first_step_s
            measured_runtime_s += first_step_s
            completed_steps += 1
            step_durations_s.append(first_step_s)
    elif first_k_steps > 0 and num_steps > 0:
        probe_steps = min(first_k_steps, num_steps)
        first_k_durations_s = run_stage("execution", lambda: timed_steps(device, step_fn, probe_steps))
        measured_runtime_s += sum(first_k_durations_s)
        completed_steps += len(first_k_durations_s)
        step_durations_s.extend(first_k_durations_s)

    remaining_steps = max(0, num_steps - completed_steps)
    if remaining_steps:
        if action == "best_eager":
            remainder_s, device_event_runtime_s = run_stage(
                "execution",
                lambda: timed_loop_with_device_events(device, step_fn, remaining_steps),
            )
        else:
            remainder_s = run_stage("execution", lambda: timed_loop(device, step_fn, remaining_steps))
        measured_runtime_s += remainder_s
        completed_steps += remaining_steps

    return {
        "action_variant": action,
        "compile_mode": compile_mode_for_action(action),
        "micro_batch_factor": micro_batch_factor(action),
        "precision_mode": "fp16" if action == "amp_fp16" else "bf16" if action == "amp_bf16" else None,
        "tf32_enabled": action == "tf32_eager",
        "warmup_runtime_s": 0.0,
        "measured_runtime_s": measured_runtime_s,
        "device_event_runtime_s": device_event_runtime_s,
        "compile_call_overhead_s": compile_call_overhead_s,
        "compile_overhead_s": compile_overhead_s,
        "graph_capture_overhead_s": 0.0,
        "graph_warmup_runtime_s": None,
        "graph_instantiate_runtime_s": None,
        "input_copy_overhead_s": 0.0,
        "input_copy_bytes": 0,
        "avg_step_time_ms": avg_ms_from_total(measured_runtime_s, completed_steps),
        "median_step_time_ms": median_ms(step_durations_s),
        "p50_step_time_ms": quantile_ms(step_durations_s, 0.50),
        "p90_step_time_ms": quantile_ms(step_durations_s, 0.90),
        "p95_step_time_ms": quantile_ms(step_durations_s, 0.95),
        "p99_step_time_ms": quantile_ms(step_durations_s, 0.99),
        "first_k_steps": len(first_k_durations_s) if action == "eager" else None,
        "first_k_eager_step_time_ms": mean_ms(first_k_durations_s) if action == "eager" else None,
        "first_k_step_time_variance_ms": variance_ms(first_k_durations_s) if action == "eager" else None,
        "num_successful_steps": completed_steps,
    }


def run_cuda_graph(
    workload: StepWorkload,
    action: str,
    num_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    if device.type != "cuda":
        raise RunnerStageError("capture", RuntimeError("CUDA Graphs require a CUDA device"))

    feasible, reason = action_eligibility(workload, action, device)
    if not feasible:
        raise RunnerStageError("precheck", RuntimeError(reason or "graph_capture_not_feasible"))
    if action == "graphs_input_copy":
        copy_feasible, copy_reason = workload.input_copy_feasible()
        if not copy_feasible:
            raise RunnerStageError("precheck", RuntimeError(copy_reason or "input_copy_not_supported"))

    compile_overhead_s: float | None = None
    compile_call_overhead_s: float | None = None
    if is_compile_action(action):
        compile_call_overhead_s = run_stage(
            "compile",
            lambda: timed(device, lambda: workload.enable_compile(compile_mode_for_action(action))),
        )
    step_fn = step_callable(workload, action, device)

    graph = torch.cuda.CUDAGraph()
    warmup_runtime_s = 0.0
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for idx in range(GRAPH_WARMUP_STEPS):
            stage = "compile" if is_compile_action(action) and idx == 0 else "capture"
            duration = run_stage(stage, lambda: timed(device, step_fn))
            warmup_runtime_s += duration
            if is_compile_action(action) and idx == 0:
                compile_overhead_s = duration
    torch.cuda.current_stream().wait_stream(warmup_stream)

    def capture() -> None:
        with torch.cuda.graph(graph):
            step_fn()

    capture_runtime_s = run_stage("capture", lambda: timed(device, capture))
    if action == "graphs_input_copy":
        replay_runtime_s, input_copy_s, copied_bytes = run_stage(
            "execution",
            lambda: timed_loop_with_input_copy(device, workload.refresh_request_inputs, graph.replay, num_steps),
        )
    else:
        replay_runtime_s, device_event_runtime_s = run_stage(
            "execution",
            lambda: timed_loop_with_device_events(device, graph.replay, num_steps),
        )
        input_copy_s = 0.0
        copied_bytes = 0

    return {
        "action_variant": action,
        "compile_mode": compile_mode_for_action(action),
        "micro_batch_factor": None,
        "precision_mode": None,
        "tf32_enabled": False,
        "warmup_runtime_s": warmup_runtime_s,
        "measured_runtime_s": replay_runtime_s,
        "device_event_runtime_s": device_event_runtime_s if action != "graphs_input_copy" else None,
        "compile_call_overhead_s": compile_call_overhead_s,
        "compile_overhead_s": compile_overhead_s,
        "graph_capture_overhead_s": warmup_runtime_s + capture_runtime_s,
        "graph_warmup_runtime_s": warmup_runtime_s,
        "graph_instantiate_runtime_s": capture_runtime_s,
        "input_copy_overhead_s": input_copy_s,
        "input_copy_bytes": copied_bytes,
        "avg_step_time_ms": avg_ms_from_total(replay_runtime_s, num_steps),
        "median_step_time_ms": None,
        "p50_step_time_ms": None,
        "p90_step_time_ms": None,
        "p95_step_time_ms": None,
        "p99_step_time_ms": None,
        "num_successful_steps": num_steps,
    }
