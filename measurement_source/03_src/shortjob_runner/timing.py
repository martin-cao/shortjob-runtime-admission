from __future__ import annotations

import time
from typing import Callable

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn: Callable[[], None]) -> float:
    synchronize(device)
    start = time.perf_counter()
    fn()
    synchronize(device)
    return time.perf_counter() - start


def timed_loop(device: torch.device, fn: Callable[[], None], steps: int) -> float:
    def loop() -> None:
        for _ in range(steps):
            fn()

    return timed(device, loop)


def timed_steps(device: torch.device, fn: Callable[[], None], steps: int) -> list[float]:
    durations: list[float] = []
    for _ in range(steps):
        durations.append(timed(device, fn))
    return durations


def timed_loop_with_input_copy(
    device: torch.device,
    copy_fn: Callable[[], int],
    fn: Callable[[], None],
    steps: int,
) -> tuple[float, float, int]:
    execution_s = 0.0
    copy_s = 0.0
    copied_bytes = 0
    for _ in range(steps):
        copy_start = time.perf_counter()
        copied_bytes += copy_fn()
        synchronize(device)
        copy_s += time.perf_counter() - copy_start
        execution_s += timed(device, fn)
    return execution_s, copy_s, copied_bytes


def timed_loop_with_device_events(
    device: torch.device,
    fn: Callable[[], None],
    steps: int,
) -> tuple[float, float | None]:
    if device.type != "cuda":
        return timed_loop(device, fn, steps), None
    synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    for _ in range(steps):
        fn()
    end_event.record()
    synchronize(device)
    wall_s = time.perf_counter() - wall_start
    device_s = start_event.elapsed_time(end_event) / 1000.0
    return wall_s, round(device_s, 9)


def mean_ms(values_s: list[float]) -> float | None:
    if not values_s:
        return None
    return round((sum(values_s) / len(values_s)) * 1000.0, 6)


def variance_ms(values_s: list[float]) -> float | None:
    if len(values_s) < 2:
        return 0.0 if values_s else None
    values_ms = [value * 1000.0 for value in values_s]
    mean_value = sum(values_ms) / len(values_ms)
    variance = sum((value - mean_value) ** 2 for value in values_ms) / len(values_ms)
    return round(variance, 6)


def avg_ms_from_total(total_s: float, steps: int) -> float | None:
    if steps <= 0:
        return None
    return round((total_s / steps) * 1000.0, 6)


def median_ms(values_s: list[float]) -> float | None:
    if not values_s:
        return None
    values = sorted(values_s)
    mid = len(values) // 2
    median_s = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0
    return round(median_s * 1000.0, 6)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 6)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    value = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(float(value), 6)


def quantile_ms(values_s: list[float], q: float) -> float | None:
    value = quantile(values_s, q)
    return round(value * 1000.0, 6) if value is not None else None
