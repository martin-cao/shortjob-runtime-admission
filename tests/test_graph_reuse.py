from __future__ import annotations

import unittest

from shortjob_runner.graph_reuse import condition_key, summarize_conditions, summarize_device


class GraphReuseSummaryTests(unittest.TestCase):
    def test_oracle_uses_amortized_runtime_per_job(self) -> None:
        rows = [
            {
                "device_label": "RTX 4060",
                "workload_id": "llm_decode_proxy",
                "batch_size": 16,
                "num_steps": 100,
                "reuse_jobs": 8,
                "action": "eager",
                "status": "completed",
                "action_feasible": True,
                "amortized_runtime_per_job_s": 0.120,
                "first_job_total_runtime_s": 0.120,
            },
            {
                "device_label": "RTX 4060",
                "workload_id": "llm_decode_proxy",
                "batch_size": 16,
                "num_steps": 100,
                "reuse_jobs": 8,
                "action": "graphs_only",
                "status": "completed",
                "action_feasible": True,
                "amortized_runtime_per_job_s": 0.070,
                "first_job_total_runtime_s": 0.180,
            },
        ]

        summary = summarize_conditions(rows, metric="amortized_runtime_per_job_s")

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["oracle_action"], "graphs_only")
        self.assertAlmostEqual(summary[0]["graphs_only_speedup_vs_eager"], 0.120 / 0.070)
        self.assertEqual(summary[0]["graphs_only_first_job_slower_than_eager"], True)

    def test_infeasible_graph_rows_do_not_become_oracle(self) -> None:
        rows = [
            {
                "device_label": "V100",
                "workload_id": "dynamic_shape_infer_small",
                "batch_size": 16,
                "num_steps": 100,
                "reuse_jobs": 8,
                "action": "eager",
                "status": "completed",
                "action_feasible": True,
                "amortized_runtime_per_job_s": 0.050,
            },
            {
                "device_label": "V100",
                "workload_id": "dynamic_shape_infer_small",
                "batch_size": 16,
                "num_steps": 100,
                "reuse_jobs": 8,
                "action": "graphs_only",
                "status": "infeasible",
                "action_feasible": False,
                "amortized_runtime_per_job_s": None,
            },
        ]

        summary = summarize_conditions(rows, metric="amortized_runtime_per_job_s")

        self.assertEqual(summary[0]["oracle_action"], "eager")
        self.assertEqual(summary[0]["graphs_only_feasible"], False)

    def test_repeats_are_aggregated_before_selecting_oracle(self) -> None:
        rows = [
            {
                "device_label": "RTX 4060",
                "workload_id": "cv_online_infer",
                "batch_size": 4,
                "num_steps": 50,
                "reuse_jobs": 8,
                "action": "eager",
                "repeat": 1,
                "status": "completed",
                "action_feasible": True,
                "amortized_runtime_per_job_s": 0.10,
                "first_job_total_runtime_s": 0.10,
            },
            {
                "device_label": "RTX 4060",
                "workload_id": "cv_online_infer",
                "batch_size": 4,
                "num_steps": 50,
                "reuse_jobs": 8,
                "action": "eager",
                "repeat": 2,
                "status": "completed",
                "action_feasible": True,
                "amortized_runtime_per_job_s": 0.30,
                "first_job_total_runtime_s": 0.30,
            },
            {
                "device_label": "RTX 4060",
                "workload_id": "cv_online_infer",
                "batch_size": 4,
                "num_steps": 50,
                "reuse_jobs": 8,
                "action": "graphs_only",
                "repeat": 1,
                "status": "completed",
                "action_feasible": True,
                "amortized_runtime_per_job_s": 0.08,
                "first_job_total_runtime_s": 0.18,
            },
            {
                "device_label": "RTX 4060",
                "workload_id": "cv_online_infer",
                "batch_size": 4,
                "num_steps": 50,
                "reuse_jobs": 8,
                "action": "graphs_only",
                "repeat": 2,
                "status": "completed",
                "action_feasible": True,
                "amortized_runtime_per_job_s": 0.20,
                "first_job_total_runtime_s": 0.40,
            },
        ]

        summary = summarize_conditions(rows, metric="amortized_runtime_per_job_s")

        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["eager_runtime_s"], 0.20)
        self.assertAlmostEqual(summary[0]["graphs_only_runtime_s"], 0.14)
        self.assertEqual(summary[0]["oracle_action"], "graphs_only")
        self.assertAlmostEqual(summary[0]["graphs_only_speedup_vs_eager"], 0.20 / 0.14)

    def test_device_summary_counts_graph_wins_and_infeasible_cases(self) -> None:
        rows = [
            {
                "device_label": "RTX 4060",
                "workload_id": "a",
                "batch_size": 4,
                "num_steps": 50,
                "reuse_jobs": 8,
                "action": "eager",
                "status": "completed",
                "action_feasible": True,
                "amortized_runtime_per_job_s": 0.100,
            },
            {
                "device_label": "RTX 4060",
                "workload_id": "a",
                "batch_size": 4,
                "num_steps": 50,
                "reuse_jobs": 8,
                "action": "graphs_only",
                "status": "completed",
                "action_feasible": True,
                "amortized_runtime_per_job_s": 0.080,
            },
            {
                "device_label": "RTX 4060",
                "workload_id": "b",
                "batch_size": 4,
                "num_steps": 50,
                "reuse_jobs": 8,
                "action": "eager",
                "status": "completed",
                "action_feasible": True,
                "amortized_runtime_per_job_s": 0.100,
            },
            {
                "device_label": "RTX 4060",
                "workload_id": "b",
                "batch_size": 4,
                "num_steps": 50,
                "reuse_jobs": 8,
                "action": "graphs_only",
                "status": "infeasible",
                "action_feasible": False,
                "amortized_runtime_per_job_s": None,
            },
        ]

        per_condition = summarize_conditions(rows, metric="amortized_runtime_per_job_s")
        summary = summarize_device("RTX 4060", per_condition)

        self.assertEqual(summary["n_conditions"], 2)
        self.assertEqual(summary["graph_oracle_count"], 1)
        self.assertEqual(summary["graphs_only_infeasible_count"], 1)
        self.assertEqual(summary["graphs_only_win_vs_eager_count"], 1)

    def test_condition_key_separates_reuse_jobs(self) -> None:
        base = {
            "device_label": "RTX 4060",
            "workload_id": "llm_decode_proxy",
            "batch_size": 16,
            "num_steps": 100,
        }
        first = {**base, "reuse_jobs": 4}
        second = {**base, "reuse_jobs": 8}

        self.assertNotEqual(condition_key(first), condition_key(second))

    def test_correctness_blocked_graph_actions_are_not_positive_evidence(self) -> None:
        rows = [
            {
                "device_label": "RTX 4060",
                "workload_id": "short_train_small",
                "batch_size": 16,
                "num_steps": 100,
                "reuse_jobs": 8,
                "action": "eager",
                "status": "completed",
                "action_feasible": True,
                "amortized_runtime_per_job_s": 0.100,
                "first_job_total_runtime_s": 0.100,
            },
            {
                "device_label": "RTX 4060",
                "workload_id": "short_train_small",
                "batch_size": 16,
                "num_steps": 100,
                "reuse_jobs": 8,
                "action": "graphs_only",
                "status": "completed",
                "action_feasible": True,
                "amortized_runtime_per_job_s": 0.020,
                "first_job_total_runtime_s": 0.030,
            },
            {
                "device_label": "RTX 4060",
                "workload_id": "short_train_small",
                "batch_size": 16,
                "num_steps": 100,
                "reuse_jobs": 8,
                "action": "graphs_input_copy",
                "status": "completed",
                "action_feasible": True,
                "amortized_runtime_per_job_s": 0.010,
                "first_job_total_runtime_s": 0.020,
            },
        ]
        blocked_by_device = {
            "RTX 4060": frozenset(
                {
                    ("short_train_small", "graphs_only"),
                    ("short_train_small", "graphs_input_copy"),
                }
            )
        }

        per_condition = summarize_conditions(
            rows,
            metric="amortized_runtime_per_job_s",
            blocked_actions_by_device=blocked_by_device,
        )
        device = summarize_device("RTX 4060", per_condition)

        self.assertEqual(per_condition[0]["oracle_action"], "eager")
        self.assertEqual(per_condition[0]["graphs_only_feasible"], False)
        self.assertEqual(per_condition[0]["graphs_only_correctness_pass"], False)
        self.assertIsNone(per_condition[0]["graphs_only_speedup_vs_eager"])
        self.assertEqual(per_condition[0]["graphs_only_win_vs_eager"], False)
        self.assertEqual(device["graph_oracle_count"], 0)
        self.assertEqual(device["graphs_only_win_vs_eager_count"], 0)
        self.assertEqual(device["graphs_only_correctness_blocked_count"], 1)
        self.assertIsNone(device["median_graphs_only_speedup_vs_eager"])


if __name__ == "__main__":
    unittest.main()
