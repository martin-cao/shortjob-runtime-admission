"""Regression tests for the submission-hardening pass (reviewer items 1-7)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shortjob_runner.policies import (
    LeaveOneWorkloadOutLearnedPolicy,
    evaluate_policies,
    evaluate_policy_on_group,
    policy_set,
)
from shortjob_runner.summarize import (
    GROUP_FIELDS,
    attach_oracle_and_regret,
    group_key,
    load_correctness_blocklist,
)
from shortjob_runner import queue_replay as qr
from shortjob_runner import sharding


REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "src" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = _load_script("summarize_policy_bootstrap")
rank_stability = _load_script("summarize_rank_stability")


def _summary_row(workload, steps, action, runtime, *, oracle, feasible=True, stability="stable"):
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
            "action_feasible": feasible,
            "median_total_runtime_s": runtime if feasible else None,
            "oracle_action": oracle,
        }
    )
    return row


# --------------------------------------------------------------------------- #
# Item 1: workload-agnostic baseline must not leak held-out measured info.
# --------------------------------------------------------------------------- #
class LearnedPolicyLeakageTests(unittest.TestCase):
    def _training_summary(self) -> list[dict]:
        # Several training workloads: stable + long steps -> graphs is oracle,
        # stable + short steps -> eager. Enough samples (>= min_samples_leaf on
        # each side) for the shallow tree to learn the job-length split.
        rows = []
        for wl in ("train_a", "train_b", "train_c", "train_d", "train_e", "train_f"):
            rows.append(_summary_row(wl, 10, "eager", 0.10, oracle="eager"))
            rows.append(_summary_row(wl, 500, "eager", 1.00, oracle="graphs_only"))
        return rows

    def test_policy_does_not_receive_action_rows(self) -> None:
        # Structural guarantee: the policy cannot inspect held-out measured rows.
        self.assertFalse(LeaveOneWorkloadOutLearnedPolicy.needs_action_rows)

    def test_choice_independent_of_held_out_feasibility(self) -> None:
        summary = self._training_summary()
        policy = LeaveOneWorkloadOutLearnedPolicy(summary, model_kind="tree")
        # Held-out condition: stable + long steps -> model should predict graphs.
        held_out = _summary_row("held_out", 500, "eager", 1.0, oracle="eager")
        # Two measurement realities for the held-out condition: graphs feasible
        # vs graphs infeasible. The decision must be identical.
        feasible_rows = {
            "eager": _summary_row("held_out", 500, "eager", 1.0, oracle="eager"),
            "graphs_only": _summary_row("held_out", 500, "graphs_only", 0.6, oracle="eager"),
        }
        infeasible_rows = {
            "eager": _summary_row("held_out", 500, "eager", 1.0, oracle="eager"),
            "graphs_only": _summary_row(
                "held_out", 500, "graphs_only", 0.6, oracle="eager", feasible=False
            ),
        }
        group = group_key(held_out)
        res_feasible = evaluate_policy_on_group(group, feasible_rows, "wa", policy)
        res_infeasible = evaluate_policy_on_group(group, infeasible_rows, "wa", policy)
        self.assertEqual(res_feasible["chosen_action"], "graphs_only")
        # Identical decision regardless of held-out measured feasibility -> no leak.
        self.assertEqual(
            res_feasible["chosen_action"], res_infeasible["chosen_action"]
        )

    def test_infeasible_prediction_is_honest_failure(self) -> None:
        summary = self._training_summary()
        policy = LeaveOneWorkloadOutLearnedPolicy(summary, model_kind="tree")
        held_out = _summary_row("held_out", 500, "eager", 1.0, oracle="eager")
        infeasible_rows = {
            "eager": _summary_row("held_out", 500, "eager", 1.0, oracle="eager"),
            "graphs_only": _summary_row(
                "held_out", 500, "graphs_only", 0.6, oracle="eager", feasible=False
            ),
        }
        res = evaluate_policy_on_group(group_key(held_out), infeasible_rows, "wa", policy)
        # Predicted graphs is infeasible -> recorded as a real failure, not eager.
        self.assertEqual(res["chosen_action"], "graphs_only")
        self.assertTrue(res["policy_failed"])


# --------------------------------------------------------------------------- #
# Item 2: paired condition bootstrap.
# --------------------------------------------------------------------------- #
class BootstrapTests(unittest.TestCase):
    def _rows(self) -> list[dict]:
        rows = []
        for policy in ("p_a", "p_b"):
            for i in range(8):
                r = {field: None for field in GROUP_FIELDS}
                r.update(
                    {
                        "workload_id": f"w{i}",
                        "batch_size": 4,
                        "num_steps": 100,
                        "input_mode": "synthetic",
                        "time_accounting_mode": "cold_total",
                        "cache_state": "cold_cache",
                        "execution_device": "cuda",
                        "policy": policy,
                        "normalized_regret_vs_oracle": 0.01 * i,
                        "wrong_enable": (i % 2 == 0),
                        "missed_opportunity": (i % 3 == 0),
                        "policy_failed": False,
                    }
                )
                rows.append(r)
        return rows

    def test_indices_deterministic(self) -> None:
        a = bootstrap.build_bootstrap_indices(8, 50, seed_material="0:dev")
        b = bootstrap.build_bootstrap_indices(8, 50, seed_material="0:dev")
        self.assertTrue(np.array_equal(a, b))
        c = bootstrap.build_bootstrap_indices(8, 50, seed_material="0:other")
        self.assertFalse(np.array_equal(a, c))

    def test_resampling_is_paired(self) -> None:
        rows, indices = bootstrap.device_bootstrap_rows(
            "dev", self._rows(), n_samples=64, seed=0
        )
        conditions, by_policy = bootstrap.collect_device(self._rows())
        # Recompute each policy's wrong_enable_rate CI with the SAME indices and
        # confirm it matches the reported CI -> every policy used these indices.
        for policy in ("p_a", "p_b"):
            arrays = bootstrap.metric_arrays(by_policy[policy], conditions)
            draws = arrays["wrong"][indices].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            reported = next(
                r for r in rows
                if r["policy"] == policy and r["metric"] == "wrong_enable_rate"
            )
            self.assertAlmostEqual(reported["ci95_low"], round(float(lo), 6), places=6)
            self.assertAlmostEqual(reported["ci95_high"], round(float(hi), 6), places=6)

    def test_duplicate_rows_rejected(self) -> None:
        rows = self._rows()
        rows.append(rows[0])  # duplicate (policy, condition)
        with self.assertRaises(bootstrap.DuplicatePolicyConditionRow):
            bootstrap.collect_device(rows)

    def test_inconsistent_condition_sets_rejected(self) -> None:
        rows = self._rows()
        rows = [r for r in rows if not (r["policy"] == "p_b" and r["workload_id"] == "w0")]
        with self.assertRaises(bootstrap.InconsistentConditionSets):
            bootstrap.collect_device(rows)


# --------------------------------------------------------------------------- #
# Item 3: queue replay metrics.
# --------------------------------------------------------------------------- #
class QueueMetricsTests(unittest.TestCase):
    def _summary(self) -> list[dict]:
        rows = []
        for steps in (10, 500):
            rows.append(_summary_row("w", steps, "eager", 0.1, oracle="eager"))
            rows.append(_summary_row("w", steps, "compile_only", 1.0, oracle="eager"))
        return rows

    def _policies(self) -> list[dict]:
        rows = []
        for steps in (10, 500):
            base = {field: None for field in GROUP_FIELDS}
            base.update(
                {
                    "workload_id": "w",
                    "shape_stability": "stable",
                    "batch_size": 4,
                    "num_steps": steps,
                    "input_mode": "synthetic",
                    "time_accounting_mode": "cold_total",
                    "cache_state": "cold_cache",
                    "execution_device": "cuda",
                    "oracle_action": "eager",
                }
            )
            rows.append({**base, "policy": "oracle_best_action", "chosen_action": "eager"})
            rows.append({**base, "policy": "slow_policy", "chosen_action": "compile_only"})
        return rows

    def _run(self, burst_probability=0.0, burst_size=1):
        return qr.replay_queue(
            summary_rows=self._summary(),
            policy_rows=self._policies(),
            seeds=[1],
            load_levels={"heavy": 0.9},
            num_jobs=60,
            job_mix_id="t",
            trace_id_prefix="t",
            burst_probability=burst_probability,
            burst_size=burst_size,
        )

    def test_effective_gpu_busy_time_is_null(self) -> None:
        for row in self._run():
            self.assertIsNone(row["effective_gpu_busy_time_s"])

    def test_effective_offered_load_definition_and_overload(self) -> None:
        rows = {r["policy"]: r for r in self._run()}
        for r in rows.values():
            expected = round(r["arrival_rate"] * r["mean_service_time_s"], 6)
            self.assertAlmostEqual(r["effective_offered_load"], expected, places=6)
        # oracle (all eager, 0.1) stays near target; slow (all compile, 1.0) overloads.
        self.assertLess(rows["oracle_best_action"]["effective_offered_load"], 1.0)
        self.assertGreater(rows["slow_policy"]["effective_offered_load"], 1.0)

    def test_matched_arrivals_across_policies(self) -> None:
        action_index = qr.index_action_rows(self._summary())
        policy_index = qr.index_policy_rows(self._policies())
        conditions = qr.eligible_conditions(action_index)
        _, trace = qr.sample_trace(conditions, action_index, 40, 1, 0.9)
        jobs_a = qr.build_policy_jobs(trace, "oracle_best_action", policy_index, action_index)
        jobs_b = qr.build_policy_jobs(trace, "slow_policy", policy_index, action_index)
        self.assertEqual(
            [j["arrival_time_s"] for j in jobs_a],
            [j["arrival_time_s"] for j in jobs_b],
        )

    def test_bursty_arrivals_recorded(self) -> None:
        rows = self._run(burst_probability=0.5, burst_size=3)
        for r in rows:
            self.assertEqual(r["burst_probability"], 0.5)
            self.assertEqual(r["burst_size"], 3)
            self.assertIsNotNone(r["effective_offered_load"])


# --------------------------------------------------------------------------- #
# Item 4: per-device correctness gates.
# --------------------------------------------------------------------------- #
class DeviceGateTests(unittest.TestCase):
    def _write_gate(self, tmp: Path, name: str, pairs) -> Path:
        path = tmp / name
        with path.open("w") as handle:
            for workload, action in pairs:
                handle.write(
                    json.dumps(
                        {
                            "workload_id": workload,
                            "action": action,
                            "status": "failed",
                            "action_feasible": True,
                            "failure_stage": "correctness",
                        }
                    )
                    + "\n"
                )
        return path

    def test_each_device_gets_its_own_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gate_a = self._write_gate(tmp, "a.jsonl", [("w1", "graphs_only")])
            gate_b = self._write_gate(tmp, "b.jsonl", [("w2", "compile_only")])
            gates = rank_stability.resolve_device_gates(
                ["DevA", "DevB"],
                [f"DevA={gate_a}", f"DevB={gate_b}"],
                allow_unconstrained=False,
            )
            self.assertEqual(gates["DevA"], frozenset({("w1", "graphs_only")}))
            self.assertEqual(gates["DevB"], frozenset({("w2", "compile_only")}))
            self.assertNotEqual(gates["DevA"], gates["DevB"])

    def test_missing_gate_without_unconstrained_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gate_a = self._write_gate(Path(tmpdir), "a.jsonl", [("w1", "graphs_only")])
            with self.assertRaises(SystemExit):
                rank_stability.resolve_device_gates(
                    ["DevA", "DevB"], [f"DevA={gate_a}"], allow_unconstrained=False
                )

    def test_explicit_unconstrained_allowed(self) -> None:
        gates = rank_stability.resolve_device_gates(
            ["DevA", "DevB"], None, allow_unconstrained=True
        )
        self.assertEqual(gates["DevA"], frozenset())
        self.assertEqual(gates["DevB"], frozenset())

    def test_no_gate_no_unconstrained_errors(self) -> None:
        with self.assertRaises(SystemExit):
            rank_stability.resolve_device_gates(["DevA"], None, allow_unconstrained=False)


# --------------------------------------------------------------------------- #
# Item 5: correctness vs runtime-infeasibility separation in the blocklist.
# --------------------------------------------------------------------------- #
class CorrectnessBlocklistTests(unittest.TestCase):
    def _load(self, rows) -> frozenset:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gate.jsonl"
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            return load_correctness_blocklist(path)

    def test_blocks_correctness_failures_only(self) -> None:
        rows = [
            {"workload_id": "w", "action": "graphs_only", "status": "failed",
             "action_feasible": True, "failure_stage": "correctness"},
            {"workload_id": "w", "action": "compile_only", "status": "failed",
             "action_feasible": False, "failure_stage": "execution"},
            {"workload_id": "w", "action": "eager", "status": "passed",
             "action_feasible": True, "failure_stage": None},
        ]
        blocked = self._load(rows)
        self.assertIn(("w", "graphs_only"), blocked)
        self.assertNotIn(("w", "compile_only"), blocked)  # execution failure
        self.assertNotIn(("w", "eager"), blocked)

    def test_legacy_gate_uses_feasibility_to_distinguish(self) -> None:
        # Legacy rows have no failure_stage; correctness failure = ran but failed.
        rows = [
            {"workload_id": "w", "action": "graphs_only", "status": "failed",
             "action_feasible": True},
            {"workload_id": "w", "action": "compile_only", "status": "failed",
             "action_feasible": False},
        ]
        blocked = self._load(rows)
        self.assertIn(("w", "graphs_only"), blocked)
        self.assertNotIn(("w", "compile_only"), blocked)


# --------------------------------------------------------------------------- #
# Item 7: deterministic condition sharding.
# --------------------------------------------------------------------------- #
class ShardingTests(unittest.TestCase):
    def _conditions(self):
        from shortjob_runner.types import Condition

        return [
            Condition("w", a, b, s, r)
            for a in ("eager", "graphs_only", "compile_only")
            for b in (1, 4, 16)
            for s in (10, 100, 500)
            for r in (1, 2, 3, 4, 5)
        ]

    def test_shards_disjoint_and_cover_full_matrix(self) -> None:
        conditions = self._conditions()
        shard_count = 4
        seen = []
        for index in range(shard_count):
            shard = sharding.select_shard(list(conditions), shard_count, index)
            seen.append(shard)
        flat = [c for shard in seen for c in shard]
        # union == full, no overlap
        self.assertEqual(len(flat), len(conditions))
        self.assertEqual(
            {tuple(map(str, (c.workload_id, c.action, c.batch_size, c.num_steps, c.repeat))) for c in flat},
            {tuple(map(str, (c.workload_id, c.action, c.batch_size, c.num_steps, c.repeat))) for c in conditions},
        )

    def test_shard_assignment_deterministic(self) -> None:
        conditions = self._conditions()
        a = sharding.select_shard(list(conditions), 3, 1)
        b = sharding.select_shard(list(conditions), 3, 1)
        self.assertEqual([c.repeat for c in a], [c.repeat for c in b])
        self.assertEqual(
            [(c.workload_id, c.action, c.batch_size, c.num_steps, c.repeat) for c in a],
            [(c.workload_id, c.action, c.batch_size, c.num_steps, c.repeat) for c in b],
        )


if __name__ == "__main__":
    unittest.main()
