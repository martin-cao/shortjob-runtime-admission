"""CPU unit tests for the runtime-action correctness gate.

These exercise the like-for-like comparison logic that the gate relies on,
without requiring a GPU: the CUDA-graph execution path is covered on device,
but the *reference* trajectory, the seeded input-refresh determinism, and the
micro-batch reconstruction are all pure-tensor logic that runs on CPU.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shortjob_runner.actions import GRAPH_WARMUP_STEPS  # noqa: E402
from shortjob_runner.correctness import check_one, eager_reference_state, make_seeded_workload  # noqa: E402
from shortjob_runner.workloads import make_workload  # noqa: E402

CPU = torch.device("cpu")


class SeededRefreshTests(unittest.TestCase):
    """Two independently-built workloads must refresh to identical inputs once
    their refresh generators are seeded the same way (the basis of the
    graphs_input_copy correctness check)."""

    def test_refresh_sequence_is_reproducible(self) -> None:
        a = make_seeded_workload("synthetic_kernel_chain", 4, CPU, seed=42)
        b = make_seeded_workload("synthetic_kernel_chain", 4, CPU, seed=42)
        a.seed_refresh(42, CPU)
        b.seed_refresh(42, CPU)
        for _ in range(5):
            a.refresh_request_inputs()
            b.refresh_request_inputs()
            self.assertTrue(torch.equal(a.x, b.x))

    def test_unseeded_refresh_matches_global_randn(self) -> None:
        """Timing path leaves the generator unset; refresh must still draw from
        the global RNG exactly as before (no behavioural change off the gate)."""
        wl = make_seeded_workload("synthetic_kernel_chain", 4, CPU, seed=7)
        torch.manual_seed(123)
        expected = torch.randn_like(wl.x)
        torch.manual_seed(123)
        wl.refresh_request_inputs()
        self.assertTrue(torch.equal(wl.x, expected))


class MicroBatchTests(unittest.TestCase):
    def test_fixed_shape_micro_batch_reconstructs_full_batch(self) -> None:
        reference = make_seeded_workload("fixed_shape_infer_small", 8, CPU, seed=0)
        reference.step()
        candidate = make_seeded_workload("fixed_shape_infer_small", 8, CPU, seed=0)
        candidate.step_micro_batch(4)
        self.assertEqual(candidate.last_output.shape, reference.last_output.shape)
        torch.testing.assert_close(candidate.last_output, reference.last_output)


class ReferenceTrajectoryTests(unittest.TestCase):
    def test_graph_reference_matches_warmup_adjusted_eager(self) -> None:
        """For a stateful (training) workload, the graph reference must take the
        warm-up + capture + replay step count, not just num_steps."""
        num_steps = 3
        ref = make_seeded_workload("short_train_small", 4, CPU, seed=42)
        ref.seed_refresh(42, CPU)
        state = eager_reference_state(ref, "graphs_only", num_steps, CPU)

        manual = make_seeded_workload("short_train_small", 4, CPU, seed=42)
        for _ in range(GRAPH_WARMUP_STEPS + 1 + num_steps):
            manual.step()
        torch.testing.assert_close(state["loss"], manual.last_loss.detach())

    def test_graph_reference_differs_from_plain_num_steps(self) -> None:
        """Sanity: the fix actually changes the training reference (otherwise the
        old num_steps-only reference would have been fine)."""
        num_steps = 3
        ref = make_seeded_workload("short_train_small", 4, CPU, seed=42)
        graph_state = eager_reference_state(ref, "graphs_only", num_steps, CPU)
        plain = make_seeded_workload("short_train_small", 4, CPU, seed=42)
        plain_state = eager_reference_state(plain, "eager", num_steps, CPU)
        self.assertFalse(torch.allclose(graph_state["loss"], plain_state["loss"]))


class CheckOneCpuTests(unittest.TestCase):
    """End-to-end gate rows for the CPU-eligible actions."""

    def _row(self, workload_id: str, action: str):
        return check_one(
            workload_id=workload_id,
            action=action,
            batch_size=4,
            num_steps=3,
            device=CPU,
            seed=42,
            rtol=1e-4,
            atol=1e-5,
        )

    def test_eager_passes(self) -> None:
        self.assertEqual(self._row("synthetic_kernel_chain", "eager")["status"], "passed")

    def test_micro_batch_passes_after_concat_fix(self) -> None:
        row = self._row("fixed_shape_infer_small", "micro_batch_2")
        self.assertEqual(row["status"], "passed", row.get("failure_reason"))


if __name__ == "__main__":
    unittest.main()
