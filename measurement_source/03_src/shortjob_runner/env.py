from __future__ import annotations

import datetime as dt
import os
import platform
import shutil
import socket
import subprocess
from typing import Any

import torch


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def machine_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def gpu_info(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"name": None, "available": False}
    props = torch.cuda.get_device_properties(device)
    return {
        "name": torch.cuda.get_device_name(device),
        "available": True,
        "index": device.index if device.index is not None else torch.cuda.current_device(),
        "capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "total_memory_mb": round(props.total_memory / 1024 / 1024, 3),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def runtime_info() -> dict[str, Any]:
    return {
        "compile_mode": "default",
        "torchinductor_max_autotune": os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE"),
        "torchinductor_max_autotune_gemm": os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"),
        "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def nvidia_smi_snapshot(device_index: int | None = None) -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi_not_found"}
    query = (
        "timestamp,name,index,driver_version,power.draw,power.limit,"
        "temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,memory.used"
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    except Exception as exc:
        return {"available": False, "reason": f"{exc.__class__.__name__}: {str(exc)[:160]}"}
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(query.split(",")):
            continue
        row = dict(zip(query.split(","), parts, strict=True))
        rows.append(row)
    if device_index is not None:
        rows = [row for row in rows if row.get("index") == str(device_index)]
    return {"available": True, "gpus": rows}


def power_limit_lock_record(requested_watts: float | None) -> dict[str, Any]:
    if requested_watts is None:
        return {"requested": False, "attempted": False}
    if shutil.which("nvidia-smi") is None:
        return {"requested": True, "attempted": False, "reason": "nvidia-smi_not_found"}
    command = ["nvidia-smi", "-pl", str(requested_watts)]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except Exception as exc:
        return {
            "requested": True,
            "attempted": True,
            "succeeded": False,
            "reason": f"{exc.__class__.__name__}: {str(exc)[:160]}",
        }
    return {
        "requested": True,
        "attempted": True,
        "succeeded": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[:400],
        "stderr": completed.stderr.strip()[:400],
    }


def peak_gpu_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 3)


def reset_peak_gpu_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
