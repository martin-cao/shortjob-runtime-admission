from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

from shortjob_runner.io import read_jsonl
from shortjob_runner.policies import ACTION_ORDER, post_startup_step_time_s, startup_cost_s
from shortjob_runner.summarize import GROUP_FIELDS, group_key

F = TypeVar("F", bound=Callable[..., Any])


class AdmissionAnalyzer:
    """Estimate worker-local runtime optimization admission from calibration rows."""

    def __init__(self, calibration_rows: list[dict[str, Any]]) -> None:
        self.calibration_rows = calibration_rows
        self.stats = self._build_stats(calibration_rows)

    @classmethod
    def from_summary(
        cls,
        path: str | Path,
        *,
        exclude_job: dict[str, Any] | None = None,
    ) -> "AdmissionAnalyzer":
        rows = read_jsonl(Path(path))
        if exclude_job is not None:
            rows = [row for row in rows if group_key(row) != group_key(exclude_job)]
        if not rows:
            raise ValueError("no calibration rows available")
        return cls(rows)

    def analyze_job(
        self,
        *,
        workload_id: str,
        batch_size: int,
        num_steps: int,
        task_family: str | None = None,
        workload_class: str | None = None,
        shape_stability: str | None = None,
        input_mode: str | None = None,
        time_accounting_mode: str | None = None,
        cache_state: str | None = None,
        first_k_eager_step_time_ms: float | None = None,
        first_k_steps: int | None = None,
    ) -> dict[str, Any]:
        defaults = infer_defaults(self.calibration_rows, workload_id)
        job = {
            "workload_id": workload_id,
            "task_family": task_family or defaults.get("task_family"),
            "workload_class": workload_class or defaults.get("workload_class"),
            "shape_stability": shape_stability or defaults.get("shape_stability"),
            "batch_size": batch_size,
            "num_steps": num_steps,
            "input_mode": input_mode or defaults.get("input_mode", "synthetic"),
            "time_accounting_mode": time_accounting_mode or defaults.get("time_accounting_mode", "cold_total"),
            "cache_state": cache_state or defaults.get("cache_state", "cold_cache"),
            "first_k_eager_step_time_ms": first_k_eager_step_time_ms,
            "first_k_steps": first_k_steps,
        }
        missing = [field for field in ("task_family", "workload_class", "shape_stability") if job.get(field) is None]
        if missing:
            raise ValueError(f"missing job metadata: {', '.join(missing)}")
        return self.analyze(job)

    def choose_action(self, **job_kwargs: Any) -> str:
        return str(self.analyze_job(**job_kwargs)["selected_action"])

    def analyze(self, job: dict[str, Any]) -> dict[str, Any]:
        estimates = [self._estimate_action(job, action) for action in ACTION_ORDER]
        feasible = [item for item in estimates if item["admissible"]]
        selected = min(feasible, key=lambda item: (item["estimated_total_runtime_s"], ACTION_ORDER.index(item["action"])))
        return {
            "selected_action": selected["action"],
            "selected_estimated_total_runtime_s": selected["estimated_total_runtime_s"],
            "decision_basis": "worker_local_offline_calibration",
            "policy_information_budget": (
                "static_metadata,num_steps,batch_size,shape_stability,cheap_feasibility_rules,"
                "offline_calibration_summary"
            ),
            "job": {field: job.get(field) for field in GROUP_FIELDS},
            "action_estimates": estimates,
        }

    def _estimate_action(self, job: dict[str, Any], action: str) -> dict[str, Any]:
        exact_key = self._stats_key(job)
        fallback_key = self._fallback_key(job)
        stats = self.stats.get((exact_key, action)) or self.stats.get((fallback_key, action))
        calibration_match = "none"
        blocked_reason = None
        if action in {"graphs_only", "compile_plus_graphs"} and job.get("shape_stability") == "unstable":
            blocked_reason = "unstable_shape_not_graph_capturable"
        if not stats:
            return {
                "action": action,
                "admissible": False,
                "blocked_reason": blocked_reason or "missing_calibration",
                "estimated_total_runtime_s": None,
                "startup_cost_s": None,
                "step_time_s": None,
                "risk_penalty_s": None,
                "calibration_match": "none",
            }
        calibration_match = stats["match_level"]
        if not stats["action_feasible"]:
            blocked_reason = blocked_reason or "no_feasible_calibration_rows"
        step_time_s, step_time_source = self._step_time_s(job, action, stats, exact_key, fallback_key)
        estimate = (
            stats["startup_cost_s"]
            + int(job["num_steps"]) * step_time_s
            + stats["risk_penalty_s"]
        )
        return {
            "action": action,
            "admissible": blocked_reason is None,
            "blocked_reason": blocked_reason,
            "estimated_total_runtime_s": round(float(estimate), 6),
            "startup_cost_s": round(float(stats["startup_cost_s"]), 6),
            "step_time_s": round(float(step_time_s), 9),
            "step_time_source": step_time_source,
            "risk_penalty_s": round(float(stats["risk_penalty_s"]), 6),
            "calibration_match": calibration_match,
            "n_calibration_rows": stats["n_rows"],
            "n_feasible_rows": stats["n_feasible_rows"],
            "infeasible_rate": round(float(stats["infeasible_rate"]), 6),
        }

    def _step_time_s(
        self,
        job: dict[str, Any],
        action: str,
        stats: dict[str, Any],
        exact_key: tuple[Any, ...],
        fallback_key: tuple[Any, ...],
    ) -> tuple[float, str]:
        observed_ms = job.get("first_k_eager_step_time_ms")
        if observed_ms is None:
            return float(stats["step_time_s"]), "offline_calibration"
        observed_eager_step_s = float(observed_ms) / 1000.0
        if observed_eager_step_s <= 0:
            return float(stats["step_time_s"]), "offline_calibration"
        if action == "eager":
            return observed_eager_step_s, "first_k_eager_observation"

        eager_stats = self.stats.get((exact_key, "eager")) or self.stats.get((fallback_key, "eager"))
        if not eager_stats or not eager_stats["action_feasible"] or eager_stats["step_time_s"] <= 0:
            return float(stats["step_time_s"]), "offline_calibration"
        scale = observed_eager_step_s / float(eager_stats["step_time_s"])
        return float(stats["step_time_s"]) * scale, "first_k_scaled_from_calibration"

    def _build_stats(self, rows: list[dict[str, Any]]) -> dict[tuple[tuple[Any, ...], str], dict[str, Any]]:
        buckets: dict[tuple[tuple[Any, ...], str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("median_total_runtime_s") is None:
                continue
            buckets[(self._stats_key(row), row["action"])].append(row)
            buckets[(self._fallback_key(row), row["action"])].append(row)

        stats: dict[tuple[tuple[Any, ...], str], dict[str, Any]] = {}
        for (key, action), action_rows in buckets.items():
            feasible_rows = [row for row in action_rows if row.get("action_feasible")]
            startup_costs = [startup_cost_s(row) for row in feasible_rows]
            step_times = [post_startup_step_time_s(row) for row in feasible_rows]
            infeasible_rate = 1.0 - (len(feasible_rows) / len(action_rows)) if action_rows else 1.0
            max_runtime = max((row.get("median_total_runtime_s") or 0.0) for row in action_rows)
            stats[(key, action)] = {
                "action_feasible": bool(feasible_rows),
                "startup_cost_s": statistics.median(startup_costs) if startup_costs else 0.0,
                "step_time_s": statistics.median(step_times) if step_times else float("inf"),
                "risk_penalty_s": infeasible_rate * max_runtime,
                "infeasible_rate": infeasible_rate,
                "n_rows": len(action_rows),
                "n_feasible_rows": len(feasible_rows),
                "match_level": "fallback" if key[-1] == "*" else "exact",
            }
        return stats

    @staticmethod
    def _stats_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row.get("workload_id"), row.get("shape_stability"), row.get("batch_size"))

    @staticmethod
    def _fallback_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row.get("workload_id"), row.get("shape_stability"), "*")


def with_admission_decision(
    analyzer: AdmissionAnalyzer,
    **job_kwargs: Any,
) -> Callable[[F], F]:
    """Inject an admission decision into a user function.

    The wrapped function receives an ``admission_decision`` keyword argument.
    This decorator intentionally does not perform generic CUDA Graph capture;
    it is a low-risk integration point for applications that decide how to map
    actions onto their own execution path.
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            decision = analyzer.analyze_job(**job_kwargs)
            kwargs.setdefault("admission_decision", decision)
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def infer_defaults(summary_rows: list[dict[str, Any]], workload_id: str) -> dict[str, Any]:
    matches = [row for row in summary_rows if row.get("workload_id") == workload_id]
    if not matches:
        return {}
    defaults: dict[str, Any] = {}
    for field in ("task_family", "workload_class", "shape_stability", "input_mode", "time_accounting_mode", "cache_state"):
        values = [row.get(field) for row in matches if row.get(field) is not None]
        if values:
            defaults[field] = statistics.mode(values)
    return defaults


def build_job(args: argparse.Namespace, summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    defaults = infer_defaults(summary_rows, args.workload_id)
    job = {
        "workload_id": args.workload_id,
        "task_family": args.task_family or defaults.get("task_family"),
        "workload_class": args.workload_class or defaults.get("workload_class"),
        "shape_stability": args.shape_stability or defaults.get("shape_stability"),
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "input_mode": args.input_mode or defaults.get("input_mode", "synthetic"),
        "time_accounting_mode": args.time_accounting_mode or defaults.get("time_accounting_mode", "cold_total"),
        "cache_state": args.cache_state or defaults.get("cache_state", "cold_cache"),
        "first_k_eager_step_time_ms": args.first_k_eager_step_time_ms,
        "first_k_steps": args.first_k_steps,
    }
    missing = [field for field in ("task_family", "workload_class", "shape_stability") if job.get(field) is None]
    if missing:
        raise ValueError(f"missing job metadata: {', '.join(missing)}")
    return job


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a calibration summary to make a worker-local optimization admission decision for one short-lived GPU job",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summary", type=Path, default=Path("results/tables/shortjob_action_summary.jsonl"))
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--task-family")
    parser.add_argument("--workload-class")
    parser.add_argument("--shape-stability", choices=("stable", "mostly_stable", "unstable"))
    parser.add_argument("--input-mode")
    parser.add_argument("--time-accounting-mode")
    parser.add_argument("--cache-state")
    parser.add_argument("--first-k-eager-step-time-ms", type=float)
    parser.add_argument("--first-k-steps", type=int)
    parser.add_argument(
        "--include-current-condition",
        action="store_true",
        help="By default, exclude calibration rows from the same condition to avoid peeking at the evaluation condition",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    summary_rows = read_jsonl(args.summary)
    job = build_job(args, summary_rows)
    analyzer = AdmissionAnalyzer(summary_rows) if args.include_current_condition else AdmissionAnalyzer.from_summary(
        args.summary,
        exclude_job=job,
    )
    decision = analyzer.analyze(job)
    print(json.dumps(decision, ensure_ascii=False, allow_nan=False, indent=2 if args.pretty else None))
    return 0
