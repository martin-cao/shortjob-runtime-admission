#!/usr/bin/env python3
"""Entry point for the short-lived GPU job runner."""

from __future__ import annotations

import os


def _import_torch_pinned() -> None:
    """Load torch with the process pinned to a single CPU core, then restore.

    PyTorch 2.12's ``c10::ApproximateClockToUnixTimeConverter`` runs in the
    ``libc10_cuda`` static initializer (i.e. at ``import torch``) and asserts the
    CPU timestamp counter is monotonic across two back-to-back reads. On
    multi-CCD CPUs (Ryzen 7945HX, EPYC 9654, ...) those two reads can land on
    cores whose TSCs are not perfectly synchronized, aborting the process with
    ``fast_1 >= fast_0 ... getCount is non-monotonic``. With ~8000 isolated
    subprocess launches per sweep, even a tiny per-launch probability is enough
    to kill a multi-hour run.

    Pinning to one core for the duration of the import makes both reads share a
    single TSC (monotonic), then we restore the original affinity so torch.compile
    workers and the measured workload are not throttled.
    """
    restore: set[int] | None = None
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        try:
            current = os.sched_getaffinity(0)
            if current:
                os.sched_setaffinity(0, {min(current)})
                restore = current
        except OSError:
            restore = None
    try:
        import torch  # noqa: F401  # triggers libc10_cuda load under the pinned core

        # Touch the driver so any clock init tied to first CUDA use also happens
        # while pinned; harmless if CUDA is unavailable.
        try:
            torch.cuda.is_available()
        except Exception:
            pass
    finally:
        if restore is not None:
            try:
                os.sched_setaffinity(0, restore)
            except OSError:
                pass


_import_torch_pinned()

from shortjob_runner.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
