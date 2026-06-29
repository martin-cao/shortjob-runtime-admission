from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "summarize_queue_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("summarize_queue_sensitivity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
queue_sensitivity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue_sensitivity)


class QueueSensitivityTests(unittest.TestCase):
    def test_summarize_rows_filters_load_level_and_averages_seeds(self) -> None:
        rows = [
            {
                "load_level": "heavy",
                "job_mix_id": "mix_a",
                "policy": "always_eager",
                "mean_completion_time_s": 1.0,
                "p95_completion_time_s": 2.0,
                "throughput_jobs_per_s": 3.0,
                "wasted_optimization_overhead_s": 0.0,
                "wrong_admit_count": 0,
                "missed_opportunity_count": 4,
            },
            {
                "load_level": "heavy",
                "job_mix_id": "mix_a",
                "policy": "always_eager",
                "mean_completion_time_s": 3.0,
                "p95_completion_time_s": 4.0,
                "throughput_jobs_per_s": 5.0,
                "wasted_optimization_overhead_s": 0.0,
                "wrong_admit_count": 2,
                "missed_opportunity_count": 6,
            },
            {
                "load_level": "light",
                "job_mix_id": "mix_a",
                "policy": "always_eager",
                "mean_completion_time_s": 100.0,
                "p95_completion_time_s": 100.0,
                "throughput_jobs_per_s": 100.0,
                "wasted_optimization_overhead_s": 100.0,
                "wrong_admit_count": 100,
                "missed_opportunity_count": 100,
            },
        ]

        summary = queue_sensitivity.summarize_rows("queue.jsonl", rows, load_level="heavy")

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["source_file"], "queue.jsonl")
        self.assertEqual(summary[0]["job_mix_id"], "mix_a")
        self.assertEqual(summary[0]["heavy_mean_completion_time_s"], 2.0)
        self.assertEqual(summary[0]["heavy_p95_completion_time_s"], 3.0)
        self.assertEqual(summary[0]["heavy_wrong_admit_count"], 1.0)
        self.assertEqual(summary[0]["heavy_missed_opportunity_count"], 5.0)


if __name__ == "__main__":
    unittest.main()
