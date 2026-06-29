from __future__ import annotations

import unittest

from shortjob_runner.policies import evaluate_policies, policy_set
from shortjob_runner.summarize import GROUP_FIELDS, attach_oracle_and_regret


def _summary_row(workload: str, steps: int, action: str, runtime: float, *, stability: str = "stable", startup: float = 0.0) -> dict:
    row = {field: None for field in GROUP_FIELDS}
    row.update(
        {
            "workload_id": workload,
            "shape_stability": stability,
            "batch_size": 4,
            "num_steps": steps,
            "input_mode": "synthetic",
            "time_accounting_mode": "cold_total",
            "cache_state": "cold_cache",
            "execution_device": "cuda",
            "action": action,
            "action_feasible": True,
            "median_total_runtime_s": runtime,
            "median_compile_overhead_s": startup if action != "eager" else None,
        }
    )
    return row


def _build_summary() -> list[dict]:
    rows: list[dict] = []
    # Two workloads, two job lengths each, three actions.
    # Long jobs: graphs amortizes (fast); short jobs: eager wins.
    for workload in ("infer_a", "train_b"):
        # short job: eager fastest
        rows.append(_summary_row(workload, 10, "eager", 0.10))
        rows.append(_summary_row(workload, 10, "graphs_only", 0.20, startup=0.15))
        rows.append(_summary_row(workload, 10, "compile_only", 2.00, startup=1.90))
        # long job: graphs fastest
        rows.append(_summary_row(workload, 500, "eager", 1.00))
        rows.append(_summary_row(workload, 500, "graphs_only", 0.60, startup=0.15))
        rows.append(_summary_row(workload, 500, "compile_only", 3.00, startup=1.90))
    return attach_oracle_and_regret(rows)


class PolicyAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = _build_summary()

    def test_new_policies_registered(self) -> None:
        names = set(policy_set(100, self.summary))
        for expected in (
            "workload_agnostic_logreg_policy",
            "workload_agnostic_tree_policy",
            "amort_ablation_startup_only",
            "amort_ablation_step_only",
            "amort_ablation_startup_step",
            "amort_ablation_startup_step_risk",
        ):
            self.assertIn(expected, names)

    def test_startup_step_risk_matches_amortization(self) -> None:
        rows = evaluate_policies(self.summary, threshold=100)
        amort = {
            (r["workload_id"], r["num_steps"]): r["chosen_action"]
            for r in rows
            if r["policy"] == "amortization_policy"
        }
        ablation = {
            (r["workload_id"], r["num_steps"]): r["chosen_action"]
            for r in rows
            if r["policy"] == "amort_ablation_startup_step_risk"
        }
        self.assertEqual(amort, ablation)

    def test_startup_only_always_eager(self) -> None:
        rows = evaluate_policies(self.summary, threshold=100)
        chosen = {
            r["chosen_action"]
            for r in rows
            if r["policy"] == "amort_ablation_startup_only"
        }
        # Startup-only cannot see per-step benefit, so it never admits an
        # action with non-zero startup; eager (startup 0) always wins.
        self.assertEqual(chosen, {"eager"})

    def test_workload_agnostic_policies_emit_feasible_or_eager(self) -> None:
        rows = evaluate_policies(self.summary, threshold=100)
        for policy in ("workload_agnostic_logreg_policy", "workload_agnostic_tree_policy"):
            for r in rows:
                if r["policy"] != policy:
                    continue
                # Either eager or a concrete action that was feasible (not failed).
                self.assertFalse(r["policy_failed"], f"{policy} produced an infeasible choice")


if __name__ == "__main__":
    unittest.main()
