#!/usr/bin/env python3
"""Replay action-summary service times through a minimal worker-local queue."""

from __future__ import annotations

from shortjob_runner.queue_replay import main


if __name__ == "__main__":
    raise SystemExit(main())
