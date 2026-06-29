#!/usr/bin/env python3
"""Evaluate admission policies under declared-budget error.

This is a derived analysis: measured service times stay fixed, while the
policy-facing job budget is perturbed before action selection. The selected
action is then scored against the original measured condition.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from shortjob_runner.io import read_jsonl, write_jsonl
from shortjob_runner.policies import (
    ACTION_ORDER,
    AmortizationPolicy,
    StructuredAdmissionPolicy,
    always,
    oracle_best,
    p90,
    shape_only,
)
from shortjob_runner.summarize import GROUP_FIELDS, group_key


PolicyFn = Callable[..., str | None]


def perturbed_steps(num_steps: int, factor: float) -> int:
    return max(1, int(round(num_steps * factor)))


def group_rows(summary_rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        grouped[group_key(row)][row["action"]] = row
    return grouped


def job_length_only_from_estimated(threshold: int) -> PolicyFn:
    def policy(row: dict[str, Any]) -> str:
        return "compile_only" if int(row["num_steps"]) >= threshold else "eager"

    return policy


def static_threshold_from_estimated(threshold: int) -> PolicyFn:
    def policy(row: dict[str, Any]) -> str:
        stability = row.get("shape_stability")
        if stability == "unstable":
            return "eager"
        if int(row["num_steps"]) < threshold:
            return "eager"
        return "graphs_only" if stability == "stable" else "compile_only"

    return policy


def policy_set_for_condition(
    calibration_rows: list[dict[str, Any]],
    threshold: int,
) -> dict[str, PolicyFn]:
    return {
        "always_eager": always("eager"),
        "always_compile": always("compile_only"),
        "always_graphs": always("graphs_only"),
        "always_compile_plus_graphs": always("compile_plus_graphs"),
        "oracle_best_action": oracle_best,
        "job_length_only": job_length_only_from_estimated(threshold),
        "shape_only": shape_only,
        "static_threshold": static_threshold_from_estimated(threshold),
        "amortization_policy": AmortizationPolicy(calibration_rows),
        "static_gate_only": StructuredAdmissionPolicy(
            calibration_rows,
            threshold=threshold,
            use_cost_model=False,
            use_action_eligibility=False,
            use_first_k_proxy=False,
            risk_margin_scale=0.0,
            require_eager_margin=False,
        ),
        "static_gate_plus_eligibility": StructuredAdmissionPolicy(
            calibration_rows,
            threshold=threshold,
            use_cost_model=False,
            use_action_eligibility=True,
            use_first_k_proxy=False,
            risk_margin_scale=0.0,
            require_eager_margin=False,
        ),
        "first_k_probe_policy": StructuredAdmissionPolicy(
            calibration_rows,
            threshold=threshold,
            use_cost_model=True,
            use_action_eligibility=True,
            use_first_k_proxy=True,
            risk_margin_scale=0.0,
            require_eager_margin=False,
        ),
        "risk_aware_policy": StructuredAdmissionPolicy(
            calibration_rows,
            threshold=threshold,
            use_cost_model=True,
            use_action_eligibility=True,
            use_first_k_proxy=True,
            risk_margin_scale=1.0,
            require_eager_margin=True,
        ),
    }


def evaluate_policy(
    true_group: tuple[Any, ...],
    actual_action_rows: dict[str, dict[str, Any]],
    policy_reference_row: dict[str, Any],
    policy_name: str,
    policy: PolicyFn,
    perturbation_factor: float,
) -> dict[str, Any]:
    if getattr(policy, "needs_action_rows", False):
        chosen_action = policy(policy_reference_row, actual_action_rows)
    else:
        chosen_action = policy(policy_reference_row)

    chosen_row = actual_action_rows.get(chosen_action) if chosen_action else None
    reference_row = next(iter(actual_action_rows.values()))
    oracle_action = reference_row.get("oracle_action")

    result = {
        **dict(zip(GROUP_FIELDS, true_group, strict=True)),
        "budget_perturbation_factor": perturbation_factor,
        "budget_perturbation_pct": round((perturbation_factor - 1.0) * 100.0, 3),
        "true_num_steps": int(reference_row["num_steps"]),
        "estimated_num_steps": int(policy_reference_row["num_steps"]),
        "policy": policy_name,
        "chosen_action": chosen_action,
        "oracle_action": oracle_action,
        "policy_failed": chosen_row is None or not chosen_row.get("action_feasible"),
        "total_runtime_s": None,
        "regret_vs_oracle_s": None,
        "normalized_regret_vs_oracle": None,
        "wrong_enable": None,
        "missed_opportunity": None,
        "policy_information_budget": getattr(
            policy,
            "information_budget",
            "leave_one_condition_out_offline_calibration"
            if policy_name == "amortization_policy"
            else "static_or_oracle_baseline",
        ),
    }

    if chosen_row and chosen_row.get("action_feasible"):
        result.update(
            {
                "total_runtime_s": chosen_row.get("median_total_runtime_s"),
                "regret_vs_oracle_s": chosen_row.get("regret_vs_oracle_s"),
                "normalized_regret_vs_oracle": chosen_row.get("normalized_regret_vs_oracle"),
                "wrong_enable": chosen_row.get("wrong_enable"),
                "missed_opportunity": chosen_row.get("missed_opportunity"),
            }
        )
    return result


def evaluate_budget_uncertainty(
    summary_rows: list[dict[str, Any]],
    factors: list[float],
    threshold: int,
) -> list[dict[str, Any]]:
    grouped = group_rows(summary_rows)
    outputs: list[dict[str, Any]] = []

    for true_group, action_rows in grouped.items():
        current_key = true_group
        calibration_rows = [row for row in summary_rows if group_key(row) != current_key]
        policies = policy_set_for_condition(calibration_rows, threshold)
        reference_row = next(iter(action_rows.values()))

        for factor in factors:
            policy_reference_row = dict(reference_row)
            policy_reference_row["num_steps"] = perturbed_steps(int(reference_row["num_steps"]), factor)
            for policy_name, policy in policies.items():
                outputs.append(
                    evaluate_policy(
                        true_group=current_key,
                        actual_action_rows=action_rows,
                        policy_reference_row=policy_reference_row,
                        policy_name=policy_name,
                        policy=policy,
                        perturbation_factor=factor,
                    )
                )
    return outputs


def aggregate_budget_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(float(row["budget_perturbation_factor"]), row["policy"])].append(row)

    aggregates: list[dict[str, Any]] = []
    for (factor, policy), policy_rows in sorted(by_key.items()):
        regrets = [
            float(row["normalized_regret_vs_oracle"])
            for row in policy_rows
            if row.get("normalized_regret_vs_oracle") is not None
        ]
        absolute_regrets = [
            float(row["regret_vs_oracle_s"])
            for row in policy_rows
            if row.get("regret_vs_oracle_s") is not None
        ]
        failed = [row for row in policy_rows if row.get("policy_failed")]
        wrong_enable = [row for row in policy_rows if row.get("wrong_enable")]
        missed = [row for row in policy_rows if row.get("missed_opportunity")]
        aggregates.append(
            {
                "budget_perturbation_factor": factor,
                "budget_perturbation_pct": round((factor - 1.0) * 100.0, 3),
                "policy": policy,
                "n_conditions": len(policy_rows),
                "policy_failure_rate": round(len(failed) / len(policy_rows), 6) if policy_rows else None,
                "wrong_enable_rate": round(len(wrong_enable) / len(policy_rows), 6) if policy_rows else None,
                "missed_opportunity_rate": round(len(missed) / len(policy_rows), 6) if policy_rows else None,
                "median_normalized_regret": round(float(statistics.median(regrets)), 6) if regrets else None,
                "p90_normalized_regret": p90(regrets),
                "max_normalized_regret": round(max(regrets), 6) if regrets else None,
                "max_regret_vs_oracle_s": round(max(absolute_regrets), 6) if absolute_regrets else None,
            }
        )
    return aggregates


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate policy sensitivity to declared-budget error")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--aggregate-out", type=Path, required=True)
    parser.add_argument(
        "--factors",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.25, 1.5],
        help="Multipliers applied to policy-facing num_steps",
    )
    parser.add_argument("--threshold", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = evaluate_budget_uncertainty(read_jsonl(args.summary), args.factors, threshold=args.threshold)
    write_jsonl(args.out, rows)
    write_jsonl(args.aggregate_out, aggregate_budget_results(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
