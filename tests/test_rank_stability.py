from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "summarize_rank_stability.py"
SPEC = importlib.util.spec_from_file_location("summarize_rank_stability", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
rank_stability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rank_stability)


COMMON_FIELDS = {
    "workload_id": "synthetic_kernel_chain",
    "batch_size": 16,
    "num_steps": 500,
    "input_mode": "synthetic",
    "time_accounting_mode": "cold_total",
    "cache_state": "cold_cache",
    "execution_device": "cuda",
}


class RankStabilityTests(unittest.TestCase):
    def _make_rows(self, *, with_is_best_action: bool) -> list[dict]:
        eager = {
            **COMMON_FIELDS,
            "action": "eager",
            "action_feasible": True,
            "median_total_runtime_s": 0.100,
            "ci95_total_runtime_low_s": 0.090,
            "ci95_total_runtime_high_s": 0.110,
        }
        graphs = {
            **COMMON_FIELDS,
            "action": "graphs_only",
            "action_feasible": True,
            "median_total_runtime_s": 0.095,
            "ci95_total_runtime_low_s": 0.092,
            "ci95_total_runtime_high_s": 0.098,
        }
        compile_ = {
            **COMMON_FIELDS,
            "action": "compile_only",
            "action_feasible": False,
            "median_total_runtime_s": None,
        }
        if with_is_best_action:
            eager["is_best_action"] = False
            graphs["is_best_action"] = True
            compile_["is_best_action"] = False
        return [eager, graphs, compile_]

    def _assert_condition(self, per_condition: list, device: dict) -> None:
        self.assertEqual(len(per_condition), 1)
        self.assertEqual(per_condition[0]["oracle_action"], "graphs_only")
        self.assertEqual(per_condition[0]["runner_up_action"], "eager")
        self.assertEqual(per_condition[0]["marginal_rank"], True)
        self.assertEqual(per_condition[0]["ci95_overlap"], True)
        self.assertEqual(device["graph_oracle_count"], 1)
        self.assertEqual(device["graph_oracle_marginal_count"], 1)

    def test_with_is_best_action_flag(self) -> None:
        """When is_best_action is pre-computed, the flag drives oracle selection."""
        rows = self._make_rows(with_is_best_action=True)
        per_condition = rank_stability.summarize_conditions(
            rows, device_label="RTX 4060", marginal_threshold=0.06
        )
        device = rank_stability.summarize_device("RTX 4060", per_condition)
        self._assert_condition(per_condition, device)

    def test_fallback_speed_oracle_without_is_best_action(self) -> None:
        """Without is_best_action, falls back to speed-based oracle."""
        rows = self._make_rows(with_is_best_action=False)
        per_condition = rank_stability.summarize_conditions(
            rows, device_label="RTX 4060", marginal_threshold=0.06
        )
        device = rank_stability.summarize_device("RTX 4060", per_condition)
        self._assert_condition(per_condition, device)

    def test_blocked_action_excluded_from_runner_up(self) -> None:
        """A correctness-blocked action must not appear as runner-up even if faster."""
        rows = self._make_rows(with_is_best_action=True)
        # Simulate a scenario where eager is the correctness-constrained oracle
        # (is_best_action=True) and graphs_only is fast but blocked.
        for row in rows:
            if row["action"] == "eager":
                row["is_best_action"] = True
                row["median_total_runtime_s"] = 0.100
            elif row["action"] == "graphs_only":
                row["is_best_action"] = False
                # graphs_only is fast but fails the gate; it must NOT be runner-up
        blocked = frozenset([("synthetic_kernel_chain", "graphs_only")])
        per_condition = rank_stability.summarize_conditions(
            rows, device_label="RTX 4060", marginal_threshold=0.06,
            blocked_actions=blocked,
        )
        self.assertEqual(len(per_condition), 1)
        self.assertEqual(per_condition[0]["oracle_action"], "eager")
        # runner_up must not be graphs_only (it is blocked)
        self.assertNotEqual(per_condition[0].get("runner_up_action"), "graphs_only")


if __name__ == "__main__":
    unittest.main()
