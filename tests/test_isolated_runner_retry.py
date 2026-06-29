"""CPU unit tests for the isolated-runner subprocess retry/continue logic.

A single per-condition subprocess can die at import from the transient PyTorch
2.12 TSC assert. These tests verify that one crash is retried and that a
persistent failure is recorded and skipped rather than aborting the whole sweep.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _load_isolated():
    spec = importlib.util.spec_from_file_location(
        "run_shortjob_isolated", REPO / "src" / "run_shortjob_isolated.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ISO = _load_isolated()
Condition = ISO.Condition


def _args(tmp: Path) -> argparse.Namespace:
    return argparse.Namespace(
        first_k_steps=5,
        seed=42,
        device="cuda",
        out=tmp / "out.jsonl",
        time_accounting_mode="cold_total",
        cache_state="cold_cache",
        cache_dir_root=tmp / "cache",
        power_limit_watts=None,
    )


class _FakeResult:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


class RunOneRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cond = Condition("gnn_irregular", "compile_only", 1, 10, 3)
        self.tmp = Path(self.id())  # not touched on disk; only used to build paths
        self._orig_run = ISO.subprocess.run
        self._orig_retries = ISO.SUBPROCESS_RETRIES
        self._orig_backoff = ISO.SUBPROCESS_RETRY_BACKOFF_S
        ISO.SUBPROCESS_RETRIES = 3
        ISO.SUBPROCESS_RETRY_BACKOFF_S = 0.0
        self.calls = 0

    def tearDown(self) -> None:
        ISO.subprocess.run = self._orig_run
        ISO.SUBPROCESS_RETRIES = self._orig_retries
        ISO.SUBPROCESS_RETRY_BACKOFF_S = self._orig_backoff

    def _patch(self, returncodes):
        seq = list(returncodes)

        def fake_run(command, env=None):  # noqa: ANN001
            self.calls += 1
            return _FakeResult(seq.pop(0) if seq else 0)

        ISO.subprocess.run = fake_run

    def test_success_first_try(self) -> None:
        self._patch([0])
        ok = ISO.run_one(self.cond, _args(Path("/tmp/x")), Path("/tmp/x/out.jsonl"), purpose="measure")
        self.assertTrue(ok)
        self.assertEqual(self.calls, 1)

    def test_recovers_after_transient_crash(self) -> None:
        # SIGABRT shows up as a negative return code; crash twice, then succeed.
        self._patch([-6, -6, 0])
        ok = ISO.run_one(self.cond, _args(Path("/tmp/x")), Path("/tmp/x/out.jsonl"), purpose="measure")
        self.assertTrue(ok)
        self.assertEqual(self.calls, 3)

    def test_persistent_failure_returns_false(self) -> None:
        self._patch([-6, -6, -6, -6])  # 1 + 3 retries all fail
        ok = ISO.run_one(self.cond, _args(Path("/tmp/x")), Path("/tmp/x/out.jsonl"), purpose="measure")
        self.assertFalse(ok)
        self.assertEqual(self.calls, 4)


if __name__ == "__main__":
    unittest.main()
