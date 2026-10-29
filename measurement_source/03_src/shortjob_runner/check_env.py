from __future__ import annotations

import json
import platform
import sys
from typing import Any

import torch

EXPECTED_GPU_TORCH = "2.12.0+cu126"
EXPECTED_GPU_CUDA = "12.6"


def is_linux_x86_64() -> bool:
    return platform.system() == "Linux" and platform.machine() == "x86_64"


def collect_environment() -> dict[str, Any]:
    row: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        row.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "capability": f"{props.major}.{props.minor}",
                "total_memory_mb": round(props.total_memory / 1024 / 1024, 3),
            }
        )
    return row


def validate_environment(row: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if not row["python"].startswith("3.11."):
        problems.append(f"expected Python 3.11.x, got {row['python']}")
    if is_linux_x86_64():
        if row["torch"] != EXPECTED_GPU_TORCH:
            problems.append(f"expected torch {EXPECTED_GPU_TORCH}, got {row['torch']}")
        if row["torch_cuda"] != EXPECTED_GPU_CUDA:
            problems.append(f"expected torch CUDA {EXPECTED_GPU_CUDA}, got {row['torch_cuda']}")
        if not row["cuda_available"]:
            problems.append("expected CUDA to be available on the Linux x86_64 GPU machines")
    return problems


def main() -> int:
    row = collect_environment()
    problems = validate_environment(row)
    payload = {"ok": not problems, "environment": row, "problems": problems}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
