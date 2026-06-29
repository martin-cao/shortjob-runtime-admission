from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from shortjob_runner.io import read_jsonl, write_jsonl
from shortjob_runner.learned_baselines import (
    LABEL_TO_ACTION,
    build_model,
    collapse_oracle_label,
    static_feature_vector,
)
from shortjob_runner.summarize import GROUP_FIELDS, group_key, load_correctness_blocklist
from shortjob_runner.types import ACTIONS


PolicyFn = Callable[..., str | None]

ACTION_ORDER = (
    "eager",
    "best_eager",
    "eager_inference_mode",
    "cpu_eager",
    "graphs_only",
    "graphs_input_copy",
    "compile_reduce_overhead",
    "compile_only",
    "compile_reduce_overhead_plus_graphs",
    "compile_plus_graphs",
    "amp_bf16",
    "amp_fp16",
    "tf32_eager",
    "micro_batch_2",
    "micro_batch_4",
    "compile_max_autotune",
)
ACTION_RANK = {action: rank for rank, action in enumerate(ACTION_ORDER)}
GRAPH_ACTIONS = {"graphs_only", "graphs_input_copy", "compile_plus_graphs", "compile_reduce_overhead_plus_graphs"}
COMPILE_ACTIONS = {"compile_only", "compile_reduce_overhead", "compile_max_autotune", "compile_plus_graphs", "compile_reduce_overhead_plus_graphs"}
EAGER_FAMILY_ACTIONS = {"eager", "best_eager", "eager_inference_mode"}


class AmortizationPolicy:
    def __init__(
        self,
        summary_rows: Iterable[dict[str, Any]],
        *,
        blocked_actions: frozenset[tuple[str, str]] = frozenset(),
        use_startup: bool = True,
        use_step: bool = True,
        use_risk: bool = True,
    ) -> None:
        self.blocked_actions = blocked_actions
        # Component-ablation switches. Defaults reproduce the full
        # startup + per-step amortization + risk-penalty estimate, so the
        # canonical ``amortization_policy`` is numerically unchanged.
        self.use_startup = use_startup
        self.use_step = use_step
        self.use_risk = use_risk
        self.summary_rows = list(summary_rows)
        self.stats = self._build_stats(self.summary_rows)

    def __call__(self, row: dict[str, Any]) -> str:
        key = self._stats_key(row)
        workload_id = row.get("workload_id", "")
        candidates: list[tuple[float, str]] = []
        for action in ACTION_ORDER:
            if (workload_id, action) in self.blocked_actions:
                continue
            stats = self.stats.get((key, action)) or self.stats.get((self._fallback_key(row), action))
            if not stats or not stats["action_feasible"]:
                continue
            if action in GRAPH_ACTIONS and row.get("shape_stability") == "unstable":
                continue
            estimate = (
                (stats["startup_cost_s"] if self.use_startup else 0.0)
                + (int(row["num_steps"]) * stats["step_time_s"] if self.use_step else 0.0)
                + (stats["risk_penalty_s"] if self.use_risk else 0.0)
            )
            candidates.append((estimate, action))
        if not candidates:
            return "eager"
        return min(candidates, key=lambda item: (item[0], ACTION_RANK.get(item[1], len(ACTION_RANK))))[1]

    def _build_stats(self, rows: list[dict[str, Any]]) -> dict[tuple[tuple[Any, ...], str], dict[str, Any]]:
        buckets: dict[tuple[tuple[Any, ...], str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("median_total_runtime_s") is None:
                continue
            # Exclude blocked (workload_id, action) pairs from calibration data
            if (row.get("workload_id", ""), row.get("action", "")) in self.blocked_actions:
                continue
            buckets[(self._stats_key(row), row["action"])].append(row)
            buckets[(self._fallback_key(row), row["action"])].append(row)

        stats: dict[tuple[tuple[Any, ...], str], dict[str, Any]] = {}
        for key, action_rows in buckets.items():
            feasible_rows = [row for row in action_rows if row.get("action_feasible")]
            startup_costs = [startup_cost_s(row) for row in feasible_rows]
            step_times = [post_startup_step_time_s(row) for row in feasible_rows]
            infeasible_rate = 1.0 - (len(feasible_rows) / len(action_rows)) if action_rows else 1.0
            stats[key] = {
                "action_feasible": bool(feasible_rows),
                "startup_cost_s": statistics.median(startup_costs) if startup_costs else 0.0,
                "step_time_s": statistics.median(step_times) if step_times else float("inf"),
                "risk_penalty_s": infeasible_rate * max((row.get("median_total_runtime_s") or 0.0) for row in action_rows),
            }
        return stats

    @staticmethod
    def _stats_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row.get("workload_id"), row.get("shape_stability"), row.get("batch_size"))

    @staticmethod
    def _fallback_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row.get("workload_id"), row.get("shape_stability"), "*")


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * (len(ordered) - 1)))
    return ordered[index]


class StructuredAdmissionPolicy:
    """Layered admission rules for controller ablation.

    The policy is intentionally rule based.  Its purpose is not to be a general
    tuning system, but to expose which low-cost information layers reduce
    wrong admission: static gates, action eligibility, first-k eager proxy, and
    risk margins.
    """

    needs_action_rows = True

    def __init__(
        self,
        summary_rows: Iterable[dict[str, Any]],
        *,
        threshold: int,
        use_cost_model: bool,
        use_action_eligibility: bool,
        use_first_k_proxy: bool,
        risk_margin_scale: float,
        require_eager_margin: bool,
        blocked_actions: frozenset[tuple[str, str]] = frozenset(),
    ) -> None:
        self.summary_rows = list(summary_rows)
        self.threshold = threshold
        self.use_cost_model = use_cost_model
        self.use_action_eligibility = use_action_eligibility
        self.use_first_k_proxy = use_first_k_proxy
        self.risk_margin_scale = risk_margin_scale
        self.require_eager_margin = require_eager_margin
        self.blocked_actions = blocked_actions
        self.stats = self._build_stats(self.summary_rows)
        self.information_budget = self._information_budget()

    def __call__(self, row: dict[str, Any], action_rows: dict[str, dict[str, Any]]) -> str:
        if not self.use_cost_model:
            action = self._static_gate_action(row)
            if self.use_action_eligibility and not self._is_admissible(row, action):
                return "eager"
            return action

        estimates = [(self._estimate_action(row, action, action_rows), action) for action in ACTION_ORDER]
        candidates = [(estimate, action) for estimate, action in estimates if estimate is not None]
        if not candidates:
            return "eager"
        eager_estimate = next((estimate for estimate, action in candidates if action == "eager"), None)
        best_estimate, best_action = min(candidates, key=lambda item: (item[0], ACTION_RANK.get(item[1], len(ACTION_RANK))))
        if (
            self.require_eager_margin
            and best_action != "eager"
            and eager_estimate is not None
            and best_estimate >= eager_estimate * 0.98
        ):
            return "eager"
        return best_action

    def _information_budget(self) -> str:
        parts = ["static_metadata", "num_steps", "batch_size", "shape_stability"]
        if self.use_action_eligibility:
            parts.append("cheap_action_eligibility")
        if self.use_first_k_proxy:
            parts.append("first_k_eager_timing_or_legacy_proxy")
        if self.use_cost_model:
            parts.append("leave_one_condition_out_offline_calibration")
        if self.risk_margin_scale > 0:
            parts.append("calibration_iqr_risk_margin")
        return ",".join(parts)

    def _static_gate_action(self, row: dict[str, Any]) -> str:
        stability = row.get("shape_stability")
        steps = int(row["num_steps"])
        if stability == "unstable" or steps < self.threshold:
            return "eager"
        if stability in {"stable", "mostly_stable"}:
            return "graphs_only"
        return "eager"

    def _is_admissible(self, row: dict[str, Any], action: str) -> bool:
        if action == "eager":
            return True
        workload_id = row.get("workload_id", "")
        if (workload_id, action) in self.blocked_actions:
            return False
        if action in GRAPH_ACTIONS and row.get("shape_stability") == "unstable":
            return False
        stats = self._stats_for(row, action)
        return bool(stats and stats["action_feasible"] and stats["infeasible_rate"] == 0)

    def _estimate_action(
        self,
        row: dict[str, Any],
        action: str,
        action_rows: dict[str, dict[str, Any]],
    ) -> float | None:
        if self.use_action_eligibility and not self._is_admissible(row, action):
            return None
        if not self.use_action_eligibility:
            workload_id = row.get("workload_id", "")
            if (workload_id, action) in self.blocked_actions:
                return None
        if action in COMPILE_ACTIONS | GRAPH_ACTIONS and int(row["num_steps"]) < min(self.threshold, 50):
            return None
        stats = self._stats_for(row, action)
        if not stats or not stats["action_feasible"]:
            return None
        step_time_s = float(stats["step_time_s"])
        if self.use_first_k_proxy:
            step_time_s = self._first_k_scaled_step_time_s(row, action, action_rows, stats)
        estimate = (
            float(stats["startup_cost_s"])
            + int(row["num_steps"]) * step_time_s
            + float(stats["risk_penalty_s"])
            + self.risk_margin_scale * float(stats["total_iqr_s"])
        )
        return estimate

    def _first_k_scaled_step_time_s(
        self,
        row: dict[str, Any],
        action: str,
        action_rows: dict[str, dict[str, Any]],
        stats: dict[str, Any],
    ) -> float:
        eager_current = action_rows.get("eager")
        observed_ms = None
        if eager_current:
            observed_ms = eager_current.get("median_first_k_eager_step_time_ms")
            if observed_ms is None:
                observed_ms = eager_current.get("median_avg_step_time_ms")
        if observed_ms is None:
            return float(stats["step_time_s"])
        observed_eager_step_s = float(observed_ms) / 1000.0
        if action == "eager":
            return observed_eager_step_s
        eager_stats = self._stats_for(row, "eager")
        if not eager_stats or eager_stats["step_time_s"] <= 0:
            return float(stats["step_time_s"])
        scale = observed_eager_step_s / float(eager_stats["step_time_s"])
        return float(stats["step_time_s"]) * scale

    def _stats_for(self, row: dict[str, Any], action: str) -> dict[str, Any] | None:
        return self.stats.get((self._stats_key(row), action)) or self.stats.get((self._fallback_key(row), action))

    def _build_stats(self, rows: list[dict[str, Any]]) -> dict[tuple[tuple[Any, ...], str], dict[str, Any]]:
        buckets: dict[tuple[tuple[Any, ...], str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("median_total_runtime_s") is None:
                continue
            # Exclude blocked (workload_id, action) pairs from calibration data
            if (row.get("workload_id", ""), row.get("action", "")) in self.blocked_actions:
                continue
            buckets[(self._stats_key(row), row["action"])].append(row)
            buckets[(self._fallback_key(row), row["action"])].append(row)

        stats: dict[tuple[tuple[Any, ...], str], dict[str, Any]] = {}
        for key, action_rows in buckets.items():
            feasible_rows = [row for row in action_rows if row.get("action_feasible")]
            startup_costs = [startup_cost_s(row) for row in feasible_rows]
            step_times = [post_startup_step_time_s(row) for row in feasible_rows]
            total_times = [float(row["median_total_runtime_s"]) for row in feasible_rows]
            infeasible_rate = 1.0 - (len(feasible_rows) / len(action_rows)) if action_rows else 1.0
            q25 = quantile(total_times, 0.25)
            q75 = quantile(total_times, 0.75)
            total_iqr_s = (q75 - q25) if q25 is not None and q75 is not None else 0.0
            stats[key] = {
                "action_feasible": bool(feasible_rows),
                "startup_cost_s": statistics.median(startup_costs) if startup_costs else 0.0,
                "step_time_s": statistics.median(step_times) if step_times else float("inf"),
                "risk_penalty_s": infeasible_rate * max((row.get("median_total_runtime_s") or 0.0) for row in action_rows),
                "total_iqr_s": total_iqr_s,
                "infeasible_rate": infeasible_rate,
            }
        return stats

    @staticmethod
    def _stats_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row.get("workload_id"), row.get("shape_stability"), row.get("batch_size"))

    @staticmethod
    def _fallback_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row.get("workload_id"), row.get("shape_stability"), "*")


class LeaveOneConditionOutAmortizationPolicy:
    def __init__(
        self,
        summary_rows: list[dict[str, Any]],
        *,
        blocked_actions: frozenset[tuple[str, str]] = frozenset(),
        use_startup: bool = True,
        use_step: bool = True,
        use_risk: bool = True,
        information_budget: str | None = None,
    ) -> None:
        self.summary_rows = summary_rows
        self.blocked_actions = blocked_actions
        self.amort_kwargs = {
            "use_startup": use_startup,
            "use_step": use_step,
            "use_risk": use_risk,
        }
        # Only set when an ablation variant supplies a label; leaving it unset for
        # the canonical amortization_policy preserves its existing budget string.
        if information_budget is not None:
            self.information_budget = information_budget
        self.cache: dict[tuple[Any, ...], AmortizationPolicy] = {}

    def __call__(self, row: dict[str, Any]) -> str:
        current_key = group_key(row)
        if current_key not in self.cache:
            calibration_rows = [item for item in self.summary_rows if group_key(item) != current_key]
            self.cache[current_key] = AmortizationPolicy(
                calibration_rows, blocked_actions=self.blocked_actions, **self.amort_kwargs
            )
        return self.cache[current_key](row)


class LeaveOneConditionOutStructuredPolicy:
    def __init__(
        self,
        summary_rows: list[dict[str, Any]],
        *,
        blocked_actions: frozenset[tuple[str, str]] = frozenset(),
        **kwargs: Any,
    ) -> None:
        self.summary_rows = summary_rows
        self.blocked_actions = blocked_actions
        self.kwargs = kwargs
        self.cache: dict[tuple[Any, ...], StructuredAdmissionPolicy] = {}
        self.needs_action_rows = True
        self.information_budget = "leave_one_condition_out_structured_admission"

    def __call__(self, row: dict[str, Any], action_rows: dict[str, dict[str, Any]]) -> str:
        current_key = group_key(row)
        if current_key not in self.cache:
            calibration_rows = [item for item in self.summary_rows if group_key(item) != current_key]
            self.cache[current_key] = StructuredAdmissionPolicy(
                calibration_rows, blocked_actions=self.blocked_actions, **self.kwargs
            )
        return self.cache[current_key](row, action_rows)


class LeaveOneWorkloadFamilyOutStructuredPolicy:
    def __init__(
        self,
        summary_rows: list[dict[str, Any]],
        *,
        blocked_actions: frozenset[tuple[str, str]] = frozenset(),
        **kwargs: Any,
    ) -> None:
        self.summary_rows = summary_rows
        self.blocked_actions = blocked_actions
        self.kwargs = kwargs
        self.cache: dict[str, StructuredAdmissionPolicy] = {}
        self.needs_action_rows = True
        self.information_budget = "leave_one_workload_family_out_structured_admission"

    def __call__(self, row: dict[str, Any], action_rows: dict[str, dict[str, Any]]) -> str:
        workload_id = str(row.get("workload_id"))
        if workload_id not in self.cache:
            calibration_rows = [item for item in self.summary_rows if item.get("workload_id") != workload_id]
            self.cache[workload_id] = StructuredAdmissionPolicy(
                calibration_rows, blocked_actions=self.blocked_actions, **self.kwargs
            )
        return self.cache[workload_id](row, action_rows)


class LeaveOneWorkloadOutLearnedPolicy:
    """Workload-agnostic learned baseline, evaluated leave-one-workload-out.

    The model sees only static features (job length, batch size, shape
    stability) — never the workload id — and is trained on every *other*
    workload's conditions, then asked to recommend an action for the held-out
    workload.  This is the honest test of cross-family generalization that the
    structured ``workload_family_holdout_policy`` could not provide, because the
    structured estimator keys its calibration on the workload id and therefore
    has nothing to fall back on once that workload is removed.

    Leakage discipline (reviewer item 1): the decision uses *only*
    admission-time information about the target condition:

    - workload-agnostic static features (job length, batch size, shape stability);
    - the device correctness gate (``blocked_actions``), which is known a priori
      from a separate correctness probe — this is *correctness eligibility*, not
      measured runtime feasibility.

    It deliberately does **not** look at the held-out condition's measured
    runtimes, oracle action, or ``action_feasible``.  Consequently this policy
    does not receive ``action_rows`` (``needs_action_rows = False``).  If the
    predicted action turns out to be infeasible at evaluation time, that surfaces
    as a real ``policy_failed`` in ``evaluate_policy_on_group`` rather than a
    silent fall back to eager — so the reported failure/regret are honest.
    """

    needs_action_rows = False

    def __init__(
        self,
        summary_rows: list[dict[str, Any]],
        *,
        model_kind: str,
        blocked_actions: frozenset[tuple[str, str]] = frozenset(),
    ) -> None:
        self.summary_rows = list(summary_rows)
        self.model_kind = model_kind
        self.blocked_actions = blocked_actions
        self.cache: dict[str, Any] = {}
        self.information_budget = (
            f"workload_agnostic_static_features_{model_kind}_leave_one_workload_out"
        )

    def _train_for(self, holdout_workload: str) -> Any:
        seen: set[tuple[Any, ...]] = set()
        feats: list[list[float]] = []
        labels: list[str] = []
        for item in self.summary_rows:
            if item.get("workload_id") == holdout_workload:
                continue
            key = group_key(item)
            if key in seen:
                continue
            seen.add(key)
            feats.append(static_feature_vector(item))
            labels.append(collapse_oracle_label(item.get("oracle_action")))
        model = build_model(self.model_kind)
        if feats:
            model.fit(np.array(feats, dtype=float), labels)
        return model

    def __call__(self, row: dict[str, Any]) -> str:
        workload_id = str(row.get("workload_id"))
        if workload_id not in self.cache:
            self.cache[workload_id] = self._train_for(workload_id)
        label = self.cache[workload_id].predict_one(static_feature_vector(row))
        action = LABEL_TO_ACTION.get(label, "eager")
        if action == "eager":
            return "eager"
        # Admission-time correctness eligibility (a priori device gate) and a
        # static shape gate are allowed; measured runtime feasibility is NOT.
        if (workload_id, action) in self.blocked_actions:
            return "eager"
        if action in GRAPH_ACTIONS and row.get("shape_stability") == "unstable":
            return "eager"
        return action


def startup_cost_s(row: dict[str, Any]) -> float:
    if row.get("action") == "eager":
        return 0.0
    values = [
        row.get("median_compile_call_overhead_s"),
        row.get("median_compile_overhead_s"),
        row.get("median_graph_capture_overhead_s"),
        row.get("median_input_copy_overhead_s"),
        row.get("median_warmup_runtime_s"),
    ]
    numeric = [float(value) for value in values if value is not None]
    return max(numeric) if numeric else 0.0


def post_startup_step_time_s(row: dict[str, Any]) -> float:
    total = float(row["median_total_runtime_s"])
    startup = startup_cost_s(row)
    steps = max(int(row["num_steps"]), 1)
    return max(total - startup, 0.0) / steps


def always(action: str) -> PolicyFn:
    return lambda _: action


def oracle_best(row: dict[str, Any]) -> str | None:
    return row.get("oracle_action")


def job_length_only(threshold: int) -> PolicyFn:
    def policy(row: dict[str, Any]) -> str:
        return "compile_only" if int(row["num_steps"]) >= threshold else "eager"

    return policy


def shape_only(row: dict[str, Any]) -> str:
    stability = row.get("shape_stability")
    if stability == "stable":
        return "graphs_only"
    if stability == "mostly_stable":
        return "compile_only"
    return "eager"


def static_threshold(threshold: int) -> PolicyFn:
    def policy(row: dict[str, Any]) -> str:
        stability = row.get("shape_stability")
        if stability == "unstable":
            return "eager"
        if int(row["num_steps"]) < threshold:
            return "eager"
        return "graphs_only" if stability == "stable" else "compile_only"

    return policy


def policy_set(
    threshold: int,
    summary_rows: list[dict[str, Any]] | None = None,
    *,
    blocked_actions: frozenset[tuple[str, str]] = frozenset(),
) -> dict[str, PolicyFn]:
    rows = summary_rows or []
    policies: dict[str, PolicyFn] = {
        "always_eager": always("eager"),
        "always_best_eager": always("best_eager"),
        "always_cpu": always("cpu_eager"),
        "always_compile": always("compile_only"),
        "always_compile_reduce_overhead": always("compile_reduce_overhead"),
        "always_graphs": always("graphs_only"),
        "always_graphs_input_copy": always("graphs_input_copy"),
        "always_compile_plus_graphs": always("compile_plus_graphs"),
        "oracle_best_action": oracle_best,
        "job_length_only": job_length_only(threshold),
        "shape_only": shape_only,
        "static_threshold": static_threshold(threshold),
    }
    if rows:
        policies["amortization_policy"] = LeaveOneConditionOutAmortizationPolicy(
            rows, blocked_actions=blocked_actions
        )
        policies["static_gate_only"] = LeaveOneConditionOutStructuredPolicy(
            rows,
            blocked_actions=blocked_actions,
            threshold=threshold,
            use_cost_model=False,
            use_action_eligibility=False,
            use_first_k_proxy=False,
            risk_margin_scale=0.0,
            require_eager_margin=False,
        )
        policies["static_gate_plus_eligibility"] = LeaveOneConditionOutStructuredPolicy(
            rows,
            blocked_actions=blocked_actions,
            threshold=threshold,
            use_cost_model=False,
            use_action_eligibility=True,
            use_first_k_proxy=False,
            risk_margin_scale=0.0,
            require_eager_margin=False,
        )
        policies["first_k_probe_policy"] = LeaveOneConditionOutStructuredPolicy(
            rows,
            blocked_actions=blocked_actions,
            threshold=threshold,
            use_cost_model=True,
            use_action_eligibility=True,
            use_first_k_proxy=True,
            risk_margin_scale=0.0,
            require_eager_margin=False,
        )
        policies["risk_aware_policy"] = LeaveOneConditionOutStructuredPolicy(
            rows,
            blocked_actions=blocked_actions,
            threshold=threshold,
            use_cost_model=True,
            use_action_eligibility=True,
            use_first_k_proxy=True,
            risk_margin_scale=1.0,
            require_eager_margin=True,
        )
        policies["workload_family_holdout_policy"] = LeaveOneWorkloadFamilyOutStructuredPolicy(
            rows,
            blocked_actions=blocked_actions,
            threshold=threshold,
            use_cost_model=True,
            use_action_eligibility=True,
            use_first_k_proxy=True,
            risk_margin_scale=1.0,
            require_eager_margin=True,
        )
        # Workload-agnostic learned baselines (static features only),
        # evaluated leave-one-workload-out. These directly test whether an
        # estimator can generalize to an unseen workload family.
        policies["workload_agnostic_logreg_policy"] = LeaveOneWorkloadOutLearnedPolicy(
            rows, model_kind="logreg", blocked_actions=blocked_actions
        )
        policies["workload_agnostic_tree_policy"] = LeaveOneWorkloadOutLearnedPolicy(
            rows, model_kind="tree", blocked_actions=blocked_actions
        )
        # Cost-model component ablations: which terms of the amortization
        # estimate actually matter. startup+step+risk reproduces
        # amortization_policy and is included for a complete ablation row.
        policies["amort_ablation_startup_only"] = LeaveOneConditionOutAmortizationPolicy(
            rows,
            blocked_actions=blocked_actions,
            use_startup=True,
            use_step=False,
            use_risk=False,
            information_budget="amort_ablation_startup_only",
        )
        policies["amort_ablation_step_only"] = LeaveOneConditionOutAmortizationPolicy(
            rows,
            blocked_actions=blocked_actions,
            use_startup=False,
            use_step=True,
            use_risk=False,
            information_budget="amort_ablation_step_only",
        )
        policies["amort_ablation_startup_step"] = LeaveOneConditionOutAmortizationPolicy(
            rows,
            blocked_actions=blocked_actions,
            use_startup=True,
            use_step=True,
            use_risk=False,
            information_budget="amort_ablation_startup_step",
        )
        policies["amort_ablation_startup_step_risk"] = LeaveOneConditionOutAmortizationPolicy(
            rows,
            blocked_actions=blocked_actions,
            use_startup=True,
            use_step=True,
            use_risk=True,
            information_budget="amort_ablation_startup_step_risk",
        )
    return policies


def quantile_floor(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * (len(ordered) - 1)))
    return round(ordered[index], 6)


def p90(values: list[float]) -> float | None:
    return quantile_floor(values, 0.9)


def group_rows(summary_rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        grouped[group_key(row)][row["action"]] = row
    return grouped


def valid_action_row(
    row: dict[str, Any] | None,
    *,
    blocked_actions: frozenset[tuple[str, str]] = frozenset(),
) -> bool:
    if row is None:
        return False
    workload_id = row.get("workload_id", "")
    action = row.get("action", "")
    return bool(
        row.get("action_feasible")
        and row.get("median_total_runtime_s") is not None
        and (workload_id, action) not in blocked_actions
    )


def eager_baseline_action(
    action_rows: dict[str, dict[str, Any]],
    *,
    blocked_actions: frozenset[tuple[str, str]] = frozenset(),
) -> str:
    for action in ("best_eager", "eager"):
        if valid_action_row(action_rows.get(action), blocked_actions=blocked_actions):
            return action
    return "eager"


def evaluate_policy_on_group(
    group: tuple[Any, ...],
    action_rows: dict[str, dict[str, Any]],
    policy_name: str,
    policy: PolicyFn,
    *,
    blocked_actions: frozenset[tuple[str, str]] = frozenset(),
) -> dict[str, Any]:
    reference_row = next(iter(action_rows.values()))
    if getattr(policy, "needs_action_rows", False):
        admitted_action = policy(reference_row, action_rows)
    else:
        admitted_action = policy(reference_row)
    chosen_action = admitted_action
    oracle_action = reference_row.get("oracle_action")
    workload_id = reference_row.get("workload_id", "")
    base_action = str(reference_row.get("eager_baseline_action") or eager_baseline_action(
        action_rows, blocked_actions=blocked_actions
    ))
    base_row = action_rows.get(base_action) or action_rows.get("eager")
    base_time = base_row.get("median_total_runtime_s") if base_row else None

    # Safety net: if the policy chose a correctness-blocked action, fall back to eager.
    # This can only trigger for policies that don't check blocked_actions themselves
    # (e.g. always_graphs on short_train_small).
    policy_correctness_failure = bool(
        chosen_action
        and chosen_action != "eager"
        and blocked_actions
        and (workload_id, chosen_action) in blocked_actions
    )
    if policy_correctness_failure:
        chosen_action = base_action

    chosen_row = action_rows.get(chosen_action) if chosen_action else None
    policy_failed = chosen_row is None or not chosen_row.get("action_feasible") or policy_correctness_failure

    result = {
        **dict(zip(GROUP_FIELDS, group, strict=True)),
        "policy": policy_name,
        "admitted_action": admitted_action,
        "chosen_action": chosen_action,
        "oracle_action": oracle_action,
        "eager_baseline_action": base_action,
        "eager_baseline_total_runtime_s": base_time,
        "fallback_action_used": chosen_action != admitted_action,
        "fallback_reason": "correctness_blocked" if policy_correctness_failure else None,
        "policy_failed": policy_failed,
        "policy_correctness_failure": policy_correctness_failure,
        "total_runtime_s": None,
        "regret_vs_oracle_s": None,
        "normalized_regret_vs_oracle": None,
        "decision_excess_vs_eager_baseline_s": None,
        "normalized_excess_vs_eager_baseline": None,
        "missed_benefit_vs_eager_baseline_s": None,
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
        chosen_time = chosen_row.get("median_total_runtime_s")
        non_eager_oracle = bool(oracle_action and oracle_action not in EAGER_FAMILY_ACTIONS)
        missed_benefit = (
            round(float(base_time) - float(reference_row["oracle_total_runtime_s"]), 6)
            if non_eager_oracle
            and base_time is not None
            and reference_row.get("oracle_total_runtime_s") is not None
            else None
        )
        result.update(
            {
                "total_runtime_s": chosen_time,
                "regret_vs_oracle_s": chosen_row.get("regret_vs_oracle_s"),
                "normalized_regret_vs_oracle": chosen_row.get("normalized_regret_vs_oracle"),
                "decision_excess_vs_eager_baseline_s": (
                    round(float(chosen_time) - float(base_time), 6)
                    if chosen_time is not None and base_time is not None
                    else None
                ),
                "normalized_excess_vs_eager_baseline": (
                    round((float(chosen_time) - float(base_time)) / float(base_time), 6)
                    if chosen_time is not None and base_time not in (None, 0)
                    else None
                ),
                "missed_benefit_vs_eager_baseline_s": missed_benefit,
                "wrong_enable": policy_correctness_failure or bool(
                    admitted_action not in EAGER_FAMILY_ACTIONS
                    and chosen_time is not None
                    and base_time is not None
                    and float(chosen_time) > float(base_time)
                ),
                "missed_opportunity": bool(
                    admitted_action in EAGER_FAMILY_ACTIONS
                    and non_eager_oracle
                    and chosen_time is not None
                    and reference_row.get("oracle_total_runtime_s") is not None
                    and float(chosen_time) > float(reference_row["oracle_total_runtime_s"])
                ),
            }
        )
    return result


def evaluate_policies(
    summary_rows: list[dict[str, Any]],
    threshold: int,
    *,
    blocked_actions: frozenset[tuple[str, str]] = frozenset(),
) -> list[dict[str, Any]]:
    policies = policy_set(threshold, summary_rows, blocked_actions=blocked_actions)
    outputs: list[dict[str, Any]] = []
    for group, action_rows in group_rows(summary_rows).items():
        for policy_name, policy in policies.items():
            outputs.append(
                evaluate_policy_on_group(
                    group, action_rows, policy_name, policy, blocked_actions=blocked_actions
                )
            )
    return outputs


def aggregate_policy_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[row["policy"]].append(row)

    aggregates: list[dict[str, Any]] = []
    for policy, policy_rows in sorted(by_policy.items()):
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
        failed = [row for row in policy_rows if row.get("policy_failed") or row.get("policy_correctness_failure")]
        wrong_enable = [row for row in policy_rows if row.get("wrong_enable")]
        missed = [row for row in policy_rows if row.get("missed_opportunity")]
        aggregates.append(
            {
                "policy": policy,
                "n_conditions": len(policy_rows),
                "policy_failure_count": len(failed),
                "wrong_enable_count": len(wrong_enable),
                "missed_opportunity_count": len(missed),
                "policy_failure_rate": round(len(failed) / len(policy_rows), 6) if policy_rows else None,
                "wrong_enable_rate": round(len(wrong_enable) / len(policy_rows), 6) if policy_rows else None,
                "missed_opportunity_rate": round(len(missed) / len(policy_rows), 6) if policy_rows else None,
                "mean_normalized_regret": round(float(statistics.fmean(regrets)), 6) if regrets else None,
                "median_normalized_regret": round(float(statistics.median(regrets)), 6) if regrets else None,
                "p90_normalized_regret": p90(regrets),
                "p95_normalized_regret": quantile_floor(regrets, 0.95),
                "max_normalized_regret": round(max(regrets), 6) if regrets else None,
                "total_regret_vs_oracle_s": round(sum(absolute_regrets), 6) if absolute_regrets else None,
                "mean_regret_vs_oracle_s": round(float(statistics.fmean(absolute_regrets)), 6) if absolute_regrets else None,
                "max_regret_vs_oracle_s": round(max(absolute_regrets), 6) if absolute_regrets else None,
            }
        )
    return aggregates


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate shortjob always-* and simple rule baselines")
    parser.add_argument("--summary", type=Path, default=Path("results/tables/shortjob_action_summary.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/tables/shortjob_policy_eval.jsonl"))
    parser.add_argument("--aggregate-out", type=Path, default=Path("results/tables/shortjob_policy_aggregate.jsonl"))
    parser.add_argument("--threshold", type=int, default=100, help="num_steps threshold for job_length/static threshold policies")
    parser.add_argument(
        "--correctness-gate",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "This device's correctness gate JSONL from check_correctness.py. "
            "Correctness failures are excluded from policy candidate sets. Required "
            "unless --allow-unconstrained is set."
        ),
    )
    parser.add_argument(
        "--allow-unconstrained",
        action="store_true",
        help="Explicitly run unconstrained policy selection with no correctness gate.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.correctness_gate is None and not args.allow_unconstrained:
        raise SystemExit(
            "error: --correctness-gate PATH is required (this device's own gate) "
            "or pass --allow-unconstrained to run unconstrained selection explicitly."
        )
    blocked: frozenset[tuple[str, str]] = frozenset()
    if args.correctness_gate is not None:
        blocked = load_correctness_blocklist(args.correctness_gate)
    per_condition = evaluate_policies(read_jsonl(args.summary), threshold=args.threshold, blocked_actions=blocked)
    write_jsonl(args.out, per_condition)
    write_jsonl(args.aggregate_out, aggregate_policy_results(per_condition))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
