from __future__ import annotations

import unittest

from shortjob_runner.queue_replay import parse_num_steps_weights, sample_trace


def action_row(workload_id: str, num_steps: int) -> dict[str, object]:
    return {
        "workload_id": workload_id,
        "batch_size": 4,
        "num_steps": num_steps,
        "input_mode": "synthetic",
        "time_accounting_mode": "cold_total",
        "cache_state": "cold_cache",
        "execution_device": "cuda",
        "action": "eager",
        "oracle_action": "eager",
        "median_total_runtime_s": 0.1,
    }


class QueueReplayTests(unittest.TestCase):
    def test_num_steps_weight_can_force_short_step_mix(self) -> None:
        short_key = ("w", 4, 10, "synthetic", "cold_total", "cold_cache", "cuda")
        long_key = ("w", 4, 500, "synthetic", "cold_total", "cold_cache", "cuda")
        action_index = {
            short_key: {"eager": action_row("w", 10)},
            long_key: {"eager": action_row("w", 500)},
        }

        _, trace = sample_trace(
            [short_key, long_key],
            action_index,
            num_jobs=20,
            seed=1,
            target_utilization=0.35,
            num_steps_weights={10: 1.0, 500: 0.0},
        )

        self.assertEqual({job["condition"] for job in trace}, {short_key})

    def test_parse_num_steps_weights(self) -> None:
        self.assertEqual(parse_num_steps_weights(["10=6", "500=0.5"]), {10: 6.0, 500: 0.5})


if __name__ == "__main__":
    unittest.main()
