from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from shortjob_runner.io import read_jsonl, write_jsonl
from shortjob_runner.summarize import group_key, load_correctness_blocklist


LOAD_LEVELS = {
    "light": 0.35,
    "medium": 0.65,
    "heavy": 0.9,
}
EAGER_FAMILY_ACTIONS = {"eager", "best_eager", "eager_inference_mode"}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * (len(ordered) - 1)))
    return round(ordered[index], 6)


def mean_or_none(values: list[float]) -> float | None:
    return round(float(statistics.fmean(values)), 6) if values else None


def median_or_none(values: list[float]) -> float | None:
    return round(float(statistics.median(values)), 6) if values else None


def condition_id(row: dict[str, Any]) -> tuple[Any, ...]:
    return group_key(row)


def action_overhead_s(action_row: dict[str, Any] | None) -> float:
    if not action_row or action_row.get("action") == "eager":
        return 0.0
    overheads = [
        action_row.get("median_compile_call_overhead_s"),
        action_row.get("median_compile_overhead_s"),
        action_row.get("median_graph_capture_overhead_s"),
        action_row.get("median_input_copy_overhead_s"),
        action_row.get("median_warmup_runtime_s"),
    ]
    numeric = [float(item) for item in overheads if item is not None]
    return round(max(numeric), 6) if numeric else 0.0


def index_action_rows(summary_rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, dict[str, Any]]]:
    indexed: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        indexed[condition_id(row)][row["action"]] = row
    return indexed


def index_policy_rows(policy_rows: list[dict[str, Any]]) -> dict[str, dict[tuple[Any, ...], dict[str, Any]]]:
    indexed: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
    for row in policy_rows:
        indexed[row["policy"]][condition_id(row)] = row
    return indexed


def valid_action_row(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("action_feasible") and row.get("median_total_runtime_s") is not None)


def eager_baseline_row(
    key: tuple[Any, ...],
    policy_row: dict[str, Any],
    action_index: dict[tuple[Any, ...], dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    actions = action_index[key]
    policy_baseline = policy_row.get("eager_baseline_action")
    if policy_baseline and valid_action_row(actions.get(str(policy_baseline))):
        return actions[str(policy_baseline)]
    for action in ("best_eager", "eager"):
        row = actions.get(action)
        if valid_action_row(row):
            return row
    return actions["eager"]


def eligible_conditions(action_index: dict[tuple[Any, ...], dict[str, dict[str, Any]]]) -> list[tuple[Any, ...]]:
    conditions = []
    for key, actions in action_index.items():
        oracle = next((row for row in actions.values() if row.get("action") == row.get("oracle_action")), None)
        if oracle and oracle.get("median_total_runtime_s") is not None:
            conditions.append(key)
    return sorted(conditions)


def sample_trace(
    conditions: list[tuple[Any, ...]],
    action_index: dict[tuple[Any, ...], dict[str, dict[str, Any]]],
    num_jobs: int,
    seed: int,
    target_utilization: float,
    burst_probability: float = 0.0,
    burst_size: int = 1,
    workload_weights: dict[str, float] | None = None,
    num_steps_weights: dict[int, float] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    rng = random.Random(seed)
    oracle_service_times = []
    for key in conditions:
        reference_row = next(iter(action_index[key].values()))
        oracle_action = reference_row["oracle_action"]
        oracle_service_times.append(float(action_index[key][oracle_action]["median_total_runtime_s"]))
    mean_oracle_service = statistics.fmean(oracle_service_times)
    arrival_rate = target_utilization / mean_oracle_service if mean_oracle_service > 0 else 1.0

    now = 0.0
    trace: list[dict[str, Any]] = []
    weighted_conditions = conditions
    weights: list[float] | None = None
    if workload_weights or num_steps_weights:
        weights = []
        for key in weighted_conditions:
            row = next(iter(action_index[key].values()))
            weight = 1.0
            if workload_weights:
                weight *= float(workload_weights.get(row.get("workload_id"), 1.0))
            if num_steps_weights and row.get("num_steps") is not None:
                weight *= float(num_steps_weights.get(int(row["num_steps"]), 1.0))
            weights.append(weight)
    job_index = 0
    while job_index < num_jobs:
        if job_index > 0:
            now += rng.expovariate(arrival_rate)
        key = rng.choices(weighted_conditions, weights=weights, k=1)[0] if weights else rng.choice(conditions)
        trace.append({"job_index": job_index, "arrival_time_s": round(now, 6), "condition": key})
        job_index += 1
        if burst_probability > 0 and rng.random() < burst_probability:
            for _ in range(max(0, burst_size - 1)):
                if job_index >= num_jobs:
                    break
                key = rng.choices(weighted_conditions, weights=weights, k=1)[0] if weights else rng.choice(conditions)
                trace.append({"job_index": job_index, "arrival_time_s": round(now, 6), "condition": key})
                job_index += 1
    return round(arrival_rate, 6), trace


def replay_single_server(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    last_finish = 0.0
    completion_times: list[float] = []
    wait_times: list[float] = []
    service_times: list[float] = []

    for job in jobs:
        arrival = float(job["arrival_time_s"])
        service = float(job["service_time_s"])
        start = max(arrival, last_finish)
        finish = start + service
        last_finish = finish
        completion_times.append(finish - arrival)
        wait_times.append(start - arrival)
        service_times.append(service)

    makespan = max(last_finish - float(jobs[0]["arrival_time_s"]), 0.0) if jobs else 0.0
    return {
        "mean_completion_time_s": mean_or_none(completion_times),
        "median_completion_time_s": median_or_none(completion_times),
        "p95_completion_time_s": percentile(completion_times, 0.95),
        "p99_completion_time_s": percentile(completion_times, 0.99),
        "mean_wait_time_s": mean_or_none(wait_times),
        "p95_wait_time_s": percentile(wait_times, 0.95),
        "throughput_jobs_per_s": round(len(jobs) / makespan, 6) if makespan > 0 else None,
        "worker_busy_time_s": round(sum(service_times), 6),
        "per_job_completion_times": completion_times,
    }


def build_policy_jobs(
    trace: list[dict[str, Any]],
    policy_name: str,
    policy_index: dict[str, dict[tuple[Any, ...], dict[str, Any]]],
    action_index: dict[tuple[Any, ...], dict[str, dict[str, Any]]],
    *,
    blocked_actions: frozenset[tuple[str, str]] = frozenset(),
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for trace_job in trace:
        key = trace_job["condition"]
        policy_row = policy_index[policy_name][key]
        chosen_action = policy_row.get("chosen_action")
        chosen_action_row = action_index[key].get(chosen_action) if chosen_action else None
        baseline_row = eager_baseline_row(key, policy_row, action_index)
        baseline_action = str(baseline_row["action"])
        baseline_service_time_s = float(baseline_row["median_total_runtime_s"])

        # Safety net: if the policy chose a correctness-blocked action, fall back
        # to the per-condition eager-family baseline.
        # This ensures no policy can appear faster than the correctness-constrained oracle by
        # using a fast-but-incorrect action's service time.
        workload_id = policy_row.get("workload_id", "")
        correctness_failure = bool(
            chosen_action
            and chosen_action != "eager"
            and blocked_actions
            and (workload_id, chosen_action) in blocked_actions
        )
        if correctness_failure or policy_row.get("policy_failed") or not chosen_action_row:
            chosen_action = baseline_action
            chosen_action_row = baseline_row

        oracle_action = chosen_action_row.get("oracle_action")
        overhead_s = action_overhead_s(chosen_action_row)
        service_time_s = float(chosen_action_row["median_total_runtime_s"])
        wrong_admit = bool(policy_row.get("wrong_enable")) or correctness_failure
        jobs.append(
            {
                **trace_job,
                "policy": policy_name,
                "chosen_action": chosen_action,
                "admitted_action": policy_row.get("admitted_action", chosen_action),
                "oracle_action": oracle_action,
                "eager_baseline_action": baseline_action,
                "eager_baseline_service_time_s": baseline_service_time_s,
                "service_time_s": service_time_s,
                "optimization_overhead_s": overhead_s,
                "wasted_optimization_overhead_s": overhead_s
                if (
                    (chosen_action not in EAGER_FAMILY_ACTIONS and service_time_s > baseline_service_time_s)
                    or correctness_failure
                )
                else 0.0,
                "wrong_admit": wrong_admit,
                "missed_opportunity": bool(policy_row.get("missed_opportunity")),
                "correctness_failure": correctness_failure,
            }
        )
    return jobs


def replay_queue(
    summary_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    seeds: list[int],
    load_levels: dict[str, float],
    num_jobs: int,
    job_mix_id: str,
    trace_id_prefix: str,
    burst_probability: float = 0.0,
    burst_size: int = 1,
    workload_weights: dict[str, float] | None = None,
    num_steps_weights: dict[int, float] | None = None,
    blocked_actions: frozenset[tuple[str, str]] = frozenset(),
) -> list[dict[str, Any]]:
    action_index = index_action_rows(summary_rows)
    policy_index = index_policy_rows(policy_rows)
    conditions = eligible_conditions(action_index)
    policies = sorted(policy_index)
    outputs: list[dict[str, Any]] = []

    for load_name, target_utilization in load_levels.items():
        for seed in seeds:
            arrival_rate, trace = sample_trace(
                conditions,
                action_index,
                num_jobs,
                seed,
                target_utilization,
                burst_probability=burst_probability,
                burst_size=burst_size,
                workload_weights=workload_weights,
                num_steps_weights=num_steps_weights,
            )
            oracle_jobs = build_policy_jobs(
                trace, "oracle_best_action", policy_index, action_index,
                blocked_actions=blocked_actions,
            )
            oracle_replay = replay_single_server(oracle_jobs)
            oracle_completion = oracle_replay["per_job_completion_times"]

            for policy_name in policies:
                policy_jobs = build_policy_jobs(
                    trace, policy_name, policy_index, action_index,
                    blocked_actions=blocked_actions,
                )
                replay = replay_single_server(policy_jobs)
                completion = replay.pop("per_job_completion_times")
                oracle_gaps = [current - oracle for current, oracle in zip(completion, oracle_completion, strict=True)]
                trace_id = f"{trace_id_prefix}_{load_name}_seed{seed}"

                # Per-policy effective load. The shared arrival rate is fixed from
                # the oracle mean service time, so a slow policy's offered load
                # (arrival_rate x its own mean service time) can exceed 1.0; that
                # overload — not a steady-state tail — is exactly why always_compile
                # blows up. Report it explicitly instead of leaving it implicit.
                policy_service_times = [float(job["service_time_s"]) for job in policy_jobs]
                mean_service_time_s = (
                    round(statistics.fmean(policy_service_times), 6) if policy_service_times else None
                )
                effective_offered_load = (
                    round(arrival_rate * mean_service_time_s, 6)
                    if mean_service_time_s is not None
                    else None
                )
                unstable_policy = bool(
                    effective_offered_load is not None and effective_offered_load >= 1.0
                )
                outputs.append(
                    {
                        "queue_run_id": f"{trace_id}_{policy_name}",
                        "trace_id": trace_id,
                        "arrival_process": "poisson_replay_from_measured_service_times",
                        "burst_probability": burst_probability,
                        "burst_size": burst_size,
                        "arrival_rate": arrival_rate,
                        "arrival_rate_lambda": arrival_rate,
                        "load_level": load_name,
                        "target_utilization": target_utilization,
                        "job_mix_id": job_mix_id,
                        "policy": policy_name,
                        "queue_seed": seed,
                        "num_jobs": num_jobs,
                        **{key: value for key, value in replay.items() if key != "per_job_completion_times"},
                        # effective_gpu_busy_time_s stays null on purpose: replay
                        # only knows worker wall-clock service time, not measured
                        # GPU activity. Reporting service time here would be a fake
                        # utilization number. worker_busy_time_s carries the
                        # wall-clock occupancy; effective_offered_load carries the
                        # offered load. (reviewer item 3)
                        "effective_gpu_busy_time_s": None,
                        "mean_service_time_s": mean_service_time_s,
                        "effective_offered_load": effective_offered_load,
                        "policy_unstable": unstable_policy,
                        "stability_status": "unstable_rho_ge_1" if unstable_policy else "stable_rho_lt_1",
                        "total_optimization_overhead_s": round(
                            sum(job["optimization_overhead_s"] for job in policy_jobs), 6
                        ),
                        "wasted_optimization_overhead_s": round(
                            sum(job["wasted_optimization_overhead_s"] for job in policy_jobs), 6
                        ),
                        "wrong_admit_count": sum(1 for job in policy_jobs if job["wrong_admit"]),
                        "missed_opportunity_count": sum(1 for job in policy_jobs if job["missed_opportunity"]),
                        "oracle_gap_mean": mean_or_none(oracle_gaps),
                        "oracle_gap_p95": percentile(oracle_gaps, 0.95),
                    }
                )
    return outputs


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay measured short-job results through a local worker queue")
    parser.add_argument("--summary", type=Path, default=Path("results/tables/shortjob_action_summary.jsonl"))
    parser.add_argument("--policy-eval", type=Path, default=Path("results/tables/shortjob_policy_eval.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/tables/shortjob_queue_replay.jsonl"))
    parser.add_argument("--num-jobs", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--job-mix-id", default="uniform_pilot_conditions")
    parser.add_argument("--trace-id-prefix", default="synthetic_poisson")
    parser.add_argument("--burst-probability", type=float, default=0.0)
    parser.add_argument("--burst-size", type=int, default=1)
    parser.add_argument(
        "--workload-weight",
        action="append",
        default=[],
        help="Optional workload_id=weight entry for job-mix sensitivity; can be repeated.",
    )
    parser.add_argument(
        "--num-steps-weight",
        action="append",
        default=[],
        help="Optional num_steps=weight entry for job-length mix sensitivity; can be repeated.",
    )
    parser.add_argument(
        "--correctness-gate",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "This device's correctness gate JSONL from check_correctness.py. "
            "Correctness-failed actions are replaced with eager service time in "
            "replay. Required unless --allow-unconstrained is set."
        ),
    )
    parser.add_argument(
        "--allow-unconstrained",
        action="store_true",
        help="Explicitly run unconstrained replay with no correctness gate.",
    )
    return parser.parse_args(argv)


def parse_workload_weights(entries: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for entry in entries:
        name, sep, value = entry.partition("=")
        if not sep:
            raise ValueError(f"invalid --workload-weight entry: {entry}")
        weights[name] = float(value)
    return weights


def parse_num_steps_weights(entries: list[str]) -> dict[int, float]:
    weights: dict[int, float] = {}
    for entry in entries:
        name, sep, value = entry.partition("=")
        if not sep:
            raise ValueError(f"invalid --num-steps-weight entry: {entry}")
        weights[int(name)] = float(value)
    return weights


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.correctness_gate is None and not args.allow_unconstrained:
        raise SystemExit(
            "error: --correctness-gate PATH is required (this device's own gate) "
            "or pass --allow-unconstrained to run unconstrained replay explicitly."
        )
    blocked: frozenset[tuple[str, str]] = frozenset()
    if args.correctness_gate is not None:
        blocked = load_correctness_blocklist(args.correctness_gate)
    rows = replay_queue(
        summary_rows=read_jsonl(args.summary),
        policy_rows=read_jsonl(args.policy_eval),
        seeds=args.seeds,
        load_levels=LOAD_LEVELS,
        num_jobs=args.num_jobs,
        job_mix_id=args.job_mix_id,
        trace_id_prefix=args.trace_id_prefix,
        burst_probability=args.burst_probability,
        burst_size=args.burst_size,
        workload_weights=parse_workload_weights(args.workload_weight),
        num_steps_weights=parse_num_steps_weights(args.num_steps_weight),
        blocked_actions=blocked,
    )
    write_jsonl(args.out, rows)
    return 0
