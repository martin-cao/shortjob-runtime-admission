#!/usr/bin/env python3
"""Build paper-facing figures from the 2026-06-15 current-protocol shortjob data."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TABLE_ROOT = ROOT / "results" / "tables"
OUT_FIG = ROOT / "results" / "figures" / "paper_official_20260627"
OUT_TAB = ROOT / "results" / "tables" / "paper_official_20260627"

# Per-device bar colors for the multi-device grouped figures.
DEVICE_BAR_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]

DEVICES = {
    "RTX 4060": {
        "action": TABLE_ROOT / "official_4060_20260625" / "cold_core" / "action_summary.jsonl",
        "policy_eval": TABLE_ROOT / "official_4060_20260625" / "cold_core" / "policy_eval.jsonl",
        "policy": TABLE_ROOT / "official_4060_20260625" / "cold_core" / "policy_aggregate.jsonl",
        "queue": TABLE_ROOT / "official_4060_20260625" / "cold_core" / "queue_replay.jsonl",
    },
    "V100": {
        "action": TABLE_ROOT / "official_v100_20260625" / "cold_core" / "action_summary.jsonl",
        "policy_eval": TABLE_ROOT / "official_v100_20260625" / "cold_core" / "policy_eval.jsonl",
        "policy": TABLE_ROOT / "official_v100_20260625" / "cold_core" / "policy_aggregate.jsonl",
        "queue": TABLE_ROOT / "official_v100_20260625" / "cold_core" / "queue_replay.jsonl",
    },
    "A100": {
        "action": TABLE_ROOT / "official_a100_20260626" / "cold_core" / "action_summary.jsonl",
        "policy_eval": TABLE_ROOT / "official_a100_20260626" / "cold_core" / "policy_eval.jsonl",
        "policy": TABLE_ROOT / "official_a100_20260626" / "cold_core" / "policy_aggregate.jsonl",
        "queue": TABLE_ROOT / "official_a100_20260626" / "cold_core" / "queue_replay.jsonl",
    },
    "H100": {
        "action": TABLE_ROOT / "official_h100_20260627" / "cold_core" / "action_summary.jsonl",
        "policy_eval": TABLE_ROOT / "official_h100_20260627" / "cold_core" / "policy_eval.jsonl",
        "policy": TABLE_ROOT / "official_h100_20260627" / "cold_core" / "policy_aggregate.jsonl",
        "queue": TABLE_ROOT / "official_h100_20260627" / "cold_core" / "queue_replay.jsonl",
    },
}

DEVICE_CONTEXT_SOURCES = {
    "RTX 4060": ROOT / "data" / "raw" / "official_4060_20260625" / "cold_core.jsonl",
    "V100": ROOT / "data" / "raw" / "official_v100_20260625" / "cold_core.jsonl",
    "A100": ROOT / "data" / "raw" / "official_a100_20260626" / "cold_core.jsonl",
    "H100": ROOT / "data" / "raw" / "official_h100_20260627" / "cold_core.jsonl",
}

WORKLOAD_LABELS = {
    "cv_online_infer": "CV infer",
    "decode_toy": "Decode toy",
    "dlrm_recommendation": "DLRM proxy",
    "dynamic_shape_infer_small": "Dynamic infer",
    "fixed_shape_infer_small": "Fixed infer",
    "gnn_irregular": "Irregular GNN",
    "llm_decode_proxy": "LLM decode",
    "prefill_toy": "Prefill toy",
    "rl_policy_infer": "RL policy",
    "short_text_transformer_dynamic": "Dyn. text tr.",
    "short_text_transformer_padded": "Padded text tr.",
    "short_train_small": "Short train",
    "synthetic_kernel_chain": "Kernel chain",
}

POLICY_LABELS = {
    "oracle_best_action": "Oracle",
    "amortization_policy": "Amortization",
    "risk_aware_policy": "Risk-aware",
    "first_k_probe_policy": "First-k",
    "static_gate_plus_eligibility": "Eligibility gate",
    "workload_family_holdout_policy": "Family holdout",
    "always_best_eager": "Best eager",
    "always_graphs": "Always graphs",
    "static_threshold": "Static threshold",
    "job_length_only": "Length only",
    "shape_only": "Shape only",
    "always_compile": "Always compile",
}

POLICY_ORDER = [
    "oracle_best_action",
    "amortization_policy",
    "risk_aware_policy",
    "static_gate_plus_eligibility",
    "always_best_eager",
    "always_graphs",
    "static_threshold",
    "job_length_only",
    "shape_only",
    "always_compile",
]

BOUNDARY_WORKLOADS = [
    ("short_train_small", 16, "Short training"),
    ("llm_decode_proxy", 16, "LLM decode proxy"),
    ("cv_online_infer", 16, "Online CV inference"),
    ("dynamic_shape_infer_small", 16, "Dynamic-shape inference"),
]

WORKLOAD_METADATA = {
    "cv_online_infer": ("realistic proxy", "inference", "fixed", "yes", "yes", "convolution / normalization / activation", "online CV inference"),
    "decode_toy": ("synthetic toy", "inference", "fixed", "yes", "yes", "small matrix / sampling-like decode loop", "token decode micro-work"),
    "dlrm_recommendation": ("realistic proxy", "inference", "fixed", "yes", "yes", "embedding / MLP / feature interaction", "recommendation inference"),
    "dynamic_shape_infer_small": ("synthetic stress", "inference", "dynamic", "no", "yes", "shape-changing tensor ops", "dynamic request shapes"),
    "fixed_shape_infer_small": ("mechanism workload", "inference", "fixed", "yes", "yes", "MLP / activation", "small fixed-shape inference"),
    "gnn_irregular": ("realistic proxy", "inference", "irregular", "no", "yes", "scatter / gather / sparse-style aggregation", "irregular graph inference"),
    "llm_decode_proxy": ("realistic proxy", "inference", "fixed", "yes", "yes", "attention-like decode / projection", "LLM decode proxy"),
    "prefill_toy": ("synthetic toy", "inference", "fixed", "yes", "yes", "attention-like prefill loop", "LLM prefill micro-work"),
    "rl_policy_infer": ("realistic proxy", "inference", "fixed", "yes", "yes", "MLP policy network", "online RL policy inference"),
    "short_text_transformer_dynamic": ("realistic proxy", "inference", "dynamic", "no", "yes", "transformer block / dynamic sequence length", "variable-length text inference"),
    "short_text_transformer_padded": ("realistic proxy", "inference", "padded fixed", "partial", "no for one graph capture case", "transformer block / padded sequence", "padded text inference"),
    "short_train_small": ("mechanism workload", "training", "fixed", "no positive graph evidence", "graph fails correctness", "forward / backward / optimizer", "short training job"),
    "synthetic_kernel_chain": ("synthetic stress", "inference", "fixed", "yes", "yes", "pointwise / reduction kernel chain", "kernel-launch dominated chain"),
}

POLICY_DEFINITION_ROWS = [
    {
        "policy": "empirical oracle",
        "code_identifier": "oracle_best_action",
        "inputs": "all measured action repeat medians; evaluation only",
        "calibration": "none",
        "eligibility": "correctness gate plus runtime feasibility",
        "decision": "minimum median total runtime with deterministic action tie order",
        "fallback": "not a deployable policy",
    },
    {
        "policy": "calibrated admission",
        "code_identifier": "amortization_policy",
        "inputs": "device, workload id, batch size, step budget, shape stability",
        "calibration": "leave-one-condition-out per-device known-family summaries",
        "eligibility": "correctness blocklist and graph shape gate",
        "decision": "minimize startup plus N steps times step-time plus infeasibility risk",
        "fallback": "best_eager when legal, otherwise eager if no candidate",
    },
    {
        "policy": "risk-aware admission",
        "code_identifier": "risk_aware_policy",
        "inputs": "calibrated inputs plus first-k eager timing proxy",
        "calibration": "leave-one-condition-out per-device known-family summaries",
        "eligibility": "correctness blocklist, graph shape gate, zero-infeasible calibration bucket",
        "decision": "calibrated estimate plus IQR margin; abstain if best non-eager is within 2 percent of eager",
        "fallback": "best_eager when legal, otherwise eager",
    },
    {
        "policy": "first-k policy",
        "code_identifier": "first_k_probe_policy",
        "inputs": "static metadata plus first-k eager timing proxy",
        "calibration": "leave-one-condition-out per-device known-family summaries",
        "eligibility": "same as risk-aware, without IQR/eager margin",
        "decision": "scale calibrated step estimates by observed first-k eager timing",
        "fallback": "best_eager when legal, otherwise eager",
    },
    {
        "policy": "eligibility-gated policy",
        "code_identifier": "static_gate_plus_eligibility",
        "inputs": "step budget and shape stability",
        "calibration": "leave-one-condition-out only for action feasibility",
        "eligibility": "correctness blocklist, graph shape gate, zero-infeasible calibration bucket",
        "decision": "static step threshold chooses graphs for stable shapes, otherwise eager",
        "fallback": "eager-family baseline on ineligible actions",
    },
    {
        "policy": "family-holdout policy",
        "code_identifier": "workload_family_holdout_policy",
        "inputs": "static metadata and first-k proxy, no target workload-family calibration rows",
        "calibration": "leave-one-workload-family-out per device",
        "eligibility": "same structured eligibility gate",
        "decision": "risk-aware structured estimate after removing target workload family",
        "fallback": "best_eager when legal, otherwise eager",
    },
    {
        "policy": "graph-only baseline",
        "code_identifier": "always_graphs",
        "inputs": "none beyond assigned job",
        "calibration": "none",
        "eligibility": "none at decision time",
        "decision": "always choose graphs_only",
        "fallback": "optimistic zero-cost eager-family fallback if invalid",
    },
    {
        "policy": "always-best-eager baseline",
        "code_identifier": "always_best_eager",
        "inputs": "none beyond assigned job",
        "calibration": "none",
        "eligibility": "best_eager legality only observed at evaluation",
        "decision": "always choose best_eager",
        "fallback": "literal eager when best_eager is illegal",
    },
    {
        "policy": "static-threshold baseline",
        "code_identifier": "static_threshold",
        "inputs": "step budget and shape stability",
        "calibration": "fixed threshold, not tuned on test labels",
        "eligibility": "none beyond unstable-shape eager guard",
        "decision": "graphs for stable long jobs, compile for mostly stable long jobs, eager otherwise",
        "fallback": "optimistic zero-cost eager-family fallback if invalid",
    },
    {
        "policy": "job-length-only baseline",
        "code_identifier": "job_length_only",
        "inputs": "step budget only",
        "calibration": "fixed threshold, not tuned on test labels",
        "eligibility": "none",
        "decision": "compile_only above threshold, eager below threshold",
        "fallback": "optimistic zero-cost eager-family fallback if invalid",
    },
    {
        "policy": "shape-only baseline",
        "code_identifier": "shape_only",
        "inputs": "shape stability only",
        "calibration": "none",
        "eligibility": "none",
        "decision": "graphs for stable, compile for mostly stable, eager for unstable",
        "fallback": "optimistic zero-cost eager-family fallback if invalid",
    },
    {
        "policy": "always-compile baseline",
        "code_identifier": "always_compile",
        "inputs": "none beyond assigned job",
        "calibration": "none",
        "eligibility": "none",
        "decision": "always choose compile_only",
        "fallback": "not normally triggered because compile_only is feasible in this suite",
    },
]

ACTION_COLORS = {
    "eager_family": "#4E79A7",
    "graphs_family": "#59A14F",
    "compile_family": "#E15759",
    "infeasible": "#BAB0AC",
    "other": "#9C755F",
}

ACTION_LABELS = {
    "eager": "Eager",
    "best_eager": "Best eager",
    "graphs_only": "Graphs",
    "graphs_input_copy": "Graphs + copy",
    "compile_only": "Compile",
    "compile_reduce_overhead": "Compile reduce",
    "compile_plus_graphs": "Compile + graphs",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.7,
            "lines.markersize": 4,
        }
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="\n", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_or_none(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return round(float(sum(numeric) / len(numeric)), 6) if numeric else None


def quantile_or_none(values: Iterable[Any], q: float) -> float | None:
    numeric = sorted(float(value) for value in values if value is not None)
    if not numeric:
        return None
    index = min(len(numeric) - 1, int(q * (len(numeric) - 1)))
    return round(numeric[index], 6)


def condition_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("workload_id"),
        row.get("batch_size"),
        row.get("num_steps"),
        row.get("input_mode"),
        row.get("time_accounting_mode"),
        row.get("cache_state"),
        row.get("execution_device"),
    )


def collapse_action(action: str) -> str:
    if action in {"eager", "best_eager", "eager_inference_mode"}:
        return "eager_family"
    if action in {"graphs_only", "graphs_input_copy"}:
        return "graphs_family"
    if "compile" in action:
        return "compile_family"
    return "other"


def oracle_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one row per condition: the correctness-constrained oracle.

    Uses the pre-computed ``is_best_action`` flag from action_summary.jsonl,
    which reflects the correctness gate applied in summarize.py.  This keeps
    the figure and table oracle counts consistent with the policy-evaluation
    oracle rather than re-deriving from speed alone.
    """
    return [row for row in rows if row.get("is_best_action") is True]


def save_figure(fig: plt.Figure, name: str) -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG / f"{name}.pdf")
    fig.savefig(OUT_FIG / f"{name}.png")
    plt.close(fig)


def plot_oracle_action_mix(device_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    workloads = sorted(WORKLOAD_LABELS)
    n_dev = len(device_rows)
    fig, axes = plt.subplots(1, n_dev, figsize=(2.45 * n_dev, 4.35), sharey=True)
    axes = np.atleast_1d(axes)
    summary_rows: list[dict[str, Any]] = []

    for idx, (ax, (device, rows)) in enumerate(zip(axes, device_rows.items())):
        oracle = oracle_rows(rows)
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in oracle:
            counts[row["workload_id"]][collapse_action(row["action"])] += 1

        y = np.arange(len(workloads))
        left = np.zeros(len(workloads))
        for family, label in [
            ("eager_family", "Eager family"),
            ("graphs_family", "CUDA Graphs"),
            ("compile_family", "Compile family"),
            ("other", "Other"),
        ]:
            vals = np.array([counts[workload][family] / max(1, sum(counts[workload].values())) for workload in workloads])
            ax.barh(
                y,
                vals,
                left=left,
                height=0.58,
                color=ACTION_COLORS[family],
                edgecolor="white",
                linewidth=0.5,
                label=label,
            )
            left += vals
        ax.set_title(device)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Oracle share")
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "50%", "100%"])
        # Suppress tick labels at shared panel seams so the right edge of one
        # panel ("100%") does not collide with the left edge of the next ("0").
        seam_labels = ax.get_xticklabels()
        if idx != 0:
            seam_labels[0].set_visible(False)
        if idx != n_dev - 1:
            seam_labels[-1].set_visible(False)
        ax.invert_yaxis()
        for workload in workloads:
            total = sum(counts[workload].values())
            for family in ("eager_family", "graphs_family", "compile_family", "other"):
                summary_rows.append(
                    {
                        "device": device,
                        "workload_id": workload,
                        "action_family": family,
                        "count": counts[workload][family],
                        "total_conditions": total,
                        "share": round(counts[workload][family] / total, 6) if total else 0.0,
                    }
                )

    axes[0].set_yticks(np.arange(len(workloads)))
    axes[0].set_yticklabels([WORKLOAD_LABELS[w] for w in workloads])
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.16, wspace=0.12)
    save_figure(fig, "fig_oracle_action_mix")
    return summary_rows


def select_action(row_map: dict[str, dict[str, Any]], action: str) -> dict[str, Any] | None:
    return row_map.get(action)


def plot_selected_boundaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["workload_id"], int(row["batch_size"]), int(row["num_steps"]))][row["action"]] = row

    actions = ["eager", "best_eager", "graphs_only", "compile_only"]
    colors = {
        "eager": "#4E79A7",
        "best_eager": "#76B7B2",
        "graphs_only": "#59A14F",
        "compile_only": "#E15759",
    }
    markers = {"eager": "o", "best_eager": "s", "graphs_only": "^", "compile_only": "D"}
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.75), sharex=True)
    summary_rows: list[dict[str, Any]] = []

    for ax, (workload, batch, title) in zip(axes.flat, BOUNDARY_WORKLOADS):
        step_values = sorted({key[2] for key in grouped if key[0] == workload and key[1] == batch})
        for action in actions:
            xs, ys = [], []
            for steps in step_values:
                row = select_action(grouped[(workload, batch, steps)], action)
                if row and row.get("action_feasible") and row.get("median_total_runtime_s") is not None:
                    xs.append(steps)
                    ys.append(float(row["median_total_runtime_s"]))
                    summary_rows.append(
                        {
                            "workload_id": workload,
                            "batch_size": batch,
                            "num_steps": steps,
                            "action": action,
                            "median_total_runtime_s": row["median_total_runtime_s"],
                            "oracle_action": row.get("oracle_action"),
                            "normalized_regret_vs_oracle": row.get("normalized_regret_vs_oracle"),
                        }
                    )
            if xs:
                ax.plot(xs, ys, marker=markers[action], color=colors[action], label=ACTION_LABELS[action])
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Job length (steps)")
        ax.set_ylabel("Total time (s)")
        ax.set_xticks([10, 50, 100, 500])
        ax.set_xticklabels(["10", "50", "100", "500"])
        ax.grid(True, which="both", axis="y")
        if workload in {"dynamic_shape_infer_small"}:
            ax.text(
                0.03,
                0.92,
                "graphs infeasible",
                transform=ax.transAxes,
                fontsize=7.5,
                color="#5F5F5F",
                va="top",
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.20, hspace=0.40, wspace=0.28)
    save_figure(fig, "fig_selected_runtime_boundaries")
    return summary_rows


def plot_selected_boundaries_normalized(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Companion to plot_selected_boundaries with runtime normalized to eager.

    Reviewers noted the absolute-time panels have mismatched y-axis magnitudes,
    so cross-workload slope comparison is misleading.  This variant divides each
    action's total runtime by the eager total runtime at the same job length and
    shares a single log y-axis across panels, with a y=1 (break-even vs eager)
    reference line.  Values below 1 are genuine speedups; above 1 are wrong
    admissions.
    """
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["workload_id"], int(row["batch_size"]), int(row["num_steps"]))][row["action"]] = row

    actions = ["best_eager", "graphs_only", "compile_only"]
    colors = {"best_eager": "#76B7B2", "graphs_only": "#59A14F", "compile_only": "#E15759"}
    markers = {"best_eager": "s", "graphs_only": "^", "compile_only": "D"}
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.75), sharex=True, sharey=True)
    summary_rows: list[dict[str, Any]] = []

    for ax, (workload, batch, title) in zip(axes.flat, BOUNDARY_WORKLOADS):
        step_values = sorted({key[2] for key in grouped if key[0] == workload and key[1] == batch})
        for action in actions:
            xs, ys = [], []
            for steps in step_values:
                cell = grouped[(workload, batch, steps)]
                eager = select_action(cell, "eager")
                row = select_action(cell, action)
                if (
                    eager
                    and eager.get("median_total_runtime_s")
                    and row
                    and row.get("action_feasible")
                    and row.get("median_total_runtime_s") is not None
                ):
                    ratio = float(row["median_total_runtime_s"]) / float(eager["median_total_runtime_s"])
                    xs.append(steps)
                    ys.append(ratio)
                    summary_rows.append(
                        {
                            "workload_id": workload,
                            "batch_size": batch,
                            "num_steps": steps,
                            "action": action,
                            "runtime_ratio_vs_eager": round(ratio, 6),
                        }
                    )
            if xs:
                ax.plot(xs, ys, marker=markers[action], color=colors[action], label=ACTION_LABELS[action])
        ax.axhline(1.0, color="#9A9A9A", linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Job length (steps)")
        ax.set_ylabel("Runtime / eager")
        ax.set_xticks([10, 50, 100, 500])
        ax.set_xticklabels(["10", "50", "100", "500"])
        ax.grid(True, which="both", axis="y")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.20, hspace=0.40, wspace=0.28)
    save_figure(fig, "fig_selected_runtime_boundaries_normalized")
    return summary_rows


def plot_policy_and_queue(device_policy: dict[str, list[dict[str, Any]]], device_queue: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.85))
    x = np.arange(len(POLICY_ORDER))
    n_dev = len(device_policy)
    width = 0.8 / n_dev
    summary_rows: list[dict[str, Any]] = []

    for idx, (device, rows) in enumerate(device_policy.items()):
        by_policy = {row["policy"]: row for row in rows}
        vals = []
        for policy in POLICY_ORDER:
            val = by_policy[policy].get("p90_normalized_regret")
            vals.append(float(val) if val is not None else np.nan)
            summary_rows.append(
                {
                    "device": device,
                    "policy": policy,
                    "policy_failure_rate": by_policy[policy].get("policy_failure_rate"),
                    "wrong_enable_rate": by_policy[policy].get("wrong_enable_rate"),
                    "missed_opportunity_rate": by_policy[policy].get("missed_opportunity_rate"),
                    "p90_normalized_regret": by_policy[policy].get("p90_normalized_regret"),
                    "max_regret_vs_oracle_s": by_policy[policy].get("max_regret_vs_oracle_s"),
                }
            )
        offset = (idx - (n_dev - 1) / 2) * width
        axes[0].bar(x + offset, np.maximum(vals, 1e-4), width=width, label=device, color=DEVICE_BAR_COLORS[idx])

    axes[0].set_yscale("log")
    axes[0].set_ylabel("P90 normalized regret")
    axes[0].set_title("Single-job regret")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([POLICY_LABELS[p] for p in POLICY_ORDER], rotation=40, ha="right")
    axes[0].legend(frameon=False)

    queue_policies = [
        "oracle_best_action",
        "amortization_policy",
        "static_gate_plus_eligibility",
        "always_best_eager",
        "always_graphs",
        "static_threshold",
        "always_compile",
    ]
    x2 = np.arange(len(queue_policies))
    for idx, (device, rows) in enumerate(device_queue.items()):
        vals = []
        for policy in queue_policies:
            heavy = [row for row in rows if row["load_level"] == "heavy" and row["policy"] == policy]
            p95 = sum(float(row["p95_completion_time_s"]) for row in heavy) / len(heavy)
            waste = sum(float(row["wasted_optimization_overhead_s"]) for row in heavy) / len(heavy)
            rho = mean_or_none(row.get("effective_offered_load") for row in heavy)
            mean_service = mean_or_none(row.get("mean_service_time_s") for row in heavy)
            arrival_rate = mean_or_none(row.get("arrival_rate_lambda", row.get("arrival_rate")) for row in heavy)
            vals.append(p95)
            summary_rows.append(
                {
                    "device": device,
                    "policy": policy,
                    "load_level": "heavy",
                    "target_utilization": 0.9,
                    "mean_arrival_rate_lambda": arrival_rate,
                    "mean_service_time_s": mean_service,
                    "mean_effective_offered_load": rho,
                    "unstable_seed_count": sum(1 for row in heavy if row.get("policy_unstable")),
                    "mean_p95_completion_time_s": round(p95, 6),
                    "mean_wasted_optimization_overhead_s": round(waste, 6),
                }
            )
        offset = (idx - (n_dev - 1) / 2) * width
        axes[1].bar(x2 + offset, np.maximum(vals, 1e-3), width=width, label=device, color=DEVICE_BAR_COLORS[idx])

    axes[1].set_yscale("log")
    axes[1].set_ylabel("Heavy-load P95 completion (s)")
    axes[1].set_title("Worker-queue consequence")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([POLICY_LABELS[p] for p in queue_policies], rotation=40, ha="right")
    axes[1].legend(frameon=False)
    fig.subplots_adjust(bottom=0.32, wspace=0.35)
    save_figure(fig, "fig_policy_regret_and_queue")
    return summary_rows


def write_core_tables(device_rows: dict[str, list[dict[str, Any]]]) -> None:
    rows_out: list[dict[str, Any]] = []
    for device, rows in device_rows.items():
        oracle = oracle_rows(rows)
        counts = Counter(collapse_action(row["action"]) for row in oracle)
        concrete_counts = Counter(row["action"] for row in oracle)
        compile_actions = [row for row in rows if "compile" in row["action"]]
        graph_rows = [row for row in rows if row["action"] in {"graphs_only", "graphs_input_copy"}]
        conditions = {
            condition_key(row)
            for row in rows
            if row.get("action") == "eager" and row.get("median_total_runtime_s") is not None
        }
        rows_out.append(
            {
                "device": device,
                "conditions": len(oracle),
                "conditions_with_best_eager_baseline": sum(
                    1
                    for row in rows
                    if row.get("action") == "eager"
                    and row.get("eager_baseline_action") == "best_eager"
                    and condition_key(row) in conditions
                ),
                "oracle_literal_eager": concrete_counts["eager"],
                "oracle_best_eager": concrete_counts["best_eager"],
                "oracle_eager_family": counts["eager_family"],
                "oracle_graphs_family": counts["graphs_family"],
                "oracle_compile_family": counts["compile_family"],
                "compile_action_rows": len(compile_actions),
                "compile_best_rows": sum(1 for row in compile_actions if row.get("is_best_action")),
                "graphs_best_rows": sum(1 for row in graph_rows if row.get("is_best_action")),
                "graphs_infeasible_rows": sum(1 for row in graph_rows if not row.get("action_feasible")),
            }
        )
    write_csv(OUT_TAB / "core_oracle_counts.csv", rows_out)


def write_device_runtime_context_table() -> None:
    rows_out: list[dict[str, Any]] = []
    for device, path in DEVICE_CONTEXT_SOURCES.items():
        first_row = read_jsonl(path)[0]
        machine = first_row["machine"]
        gpu = first_row["gpu"]
        runtime = first_row["runtime"]
        env_gpu = (first_row.get("env_snapshot_before") or {}).get("gpus", [{}])[0]
        rows_out.append(
            {
                "device": device,
                "gpu_name": gpu.get("name"),
                "compute_capability": gpu.get("capability"),
                "sm_count": gpu.get("multi_processor_count"),
                "gpu_memory_mb": round(float(gpu.get("total_memory_mb", 0.0)), 1),
                "host": machine.get("hostname"),
                "platform": machine.get("platform"),
                "cpu_count": machine.get("cpu_count"),
                "driver_version": env_gpu.get("driver_version"),
                "python": machine.get("python"),
                "torch": gpu.get("torch_version"),
                "torch_cuda": gpu.get("cuda_version"),
                "torch_float32_matmul_precision": runtime.get("torch_float32_matmul_precision"),
                "cuda_matmul_allow_tf32": runtime.get("cuda_matmul_allow_tf32"),
                "cudnn_allow_tf32": runtime.get("cudnn_allow_tf32"),
            }
        )
    write_csv(OUT_TAB / "device_runtime_context_table.csv", rows_out)


def write_policy_definition_table() -> None:
    write_csv(OUT_TAB / "policy_definition_table.csv", POLICY_DEFINITION_ROWS)


def write_workload_metadata_table(device_rows: dict[str, list[dict[str, Any]]]) -> None:
    rows_out: list[dict[str, Any]] = []
    representative_rows = next(iter(device_rows.values()))
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in representative_rows:
        by_workload[row["workload_id"]].append(row)
    for workload in sorted(by_workload):
        kind, task, shape, graph_eligible, correctness, operators, analogue = WORKLOAD_METADATA[workload]
        batches = sorted({int(row["batch_size"]) for row in by_workload[workload]})
        steps = sorted({int(row["num_steps"]) for row in by_workload[workload]})
        rows_out.append(
            {
                "workload_id": workload,
                "paper_label": WORKLOAD_LABELS[workload],
                "workload_kind": kind,
                "task": task,
                "shape_regime": shape,
                "graph_eligible": graph_eligible,
                "correctness_validated": correctness,
                "main_operators": operators,
                "intended_analogue": analogue,
                "batch_sizes": " ".join(str(item) for item in batches),
                "step_budgets": " ".join(str(item) for item in steps),
            }
        )
    write_csv(OUT_TAB / "workload_metadata_table.csv", rows_out)


def write_policy_error_severity(device_policy_eval: dict[str, list[dict[str, Any]]]) -> None:
    rows_out: list[dict[str, Any]] = []
    selected = [
        "amortization_policy",
        "risk_aware_policy",
        "first_k_probe_policy",
        "static_gate_plus_eligibility",
        "workload_family_holdout_policy",
        "always_graphs",
        "static_threshold",
        "job_length_only",
        "shape_only",
        "always_compile",
    ]
    for device, rows in device_policy_eval.items():
        by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_policy[str(row["policy"])].append(row)
        for policy in selected:
            policy_rows = by_policy.get(policy, [])
            if not policy_rows:
                continue
            wrong = [row for row in policy_rows if row.get("wrong_enable")]
            missed = [row for row in policy_rows if row.get("missed_opportunity")]
            wrong_excess = [
                max(0.0, float(row["decision_excess_vs_eager_baseline_s"]))
                for row in wrong
                if row.get("decision_excess_vs_eager_baseline_s") is not None
            ]
            wrong_norm = [
                max(0.0, float(row["normalized_excess_vs_eager_baseline"]))
                for row in wrong
                if row.get("normalized_excess_vs_eager_baseline") is not None
            ]
            missed_benefit = [
                max(0.0, float(row["missed_benefit_vs_eager_baseline_s"]))
                for row in missed
                if row.get("missed_benefit_vs_eager_baseline_s") is not None
            ]
            normalized_regrets = [
                float(row["normalized_regret_vs_oracle"])
                for row in policy_rows
                if row.get("normalized_regret_vs_oracle") is not None
            ]
            rows_out.append(
                {
                    "device": device,
                    "policy": policy,
                    "conditions": len(policy_rows),
                    "wrong_admit_count": len(wrong),
                    "wrong_admit_rate": round(len(wrong) / len(policy_rows), 6),
                    "wrong_admit_median_excess_vs_base_s": quantile_or_none(wrong_excess, 0.5),
                    "wrong_admit_p90_excess_vs_base_s": quantile_or_none(wrong_excess, 0.9),
                    "wrong_admit_median_norm_excess_vs_base": quantile_or_none(wrong_norm, 0.5),
                    "wrong_admit_p90_norm_excess_vs_base": quantile_or_none(wrong_norm, 0.9),
                    "wrong_admit_total_excess_worker_time_s": round(sum(wrong_excess), 6),
                    "wrong_admit_marginal_count_lt5pct": sum(1 for value in wrong_norm if value < 0.05),
                    "wrong_admit_decisive_count_ge5pct": sum(1 for value in wrong_norm if value >= 0.05),
                    "missed_opportunity_count": len(missed),
                    "missed_opportunity_rate": round(len(missed) / len(policy_rows), 6),
                    "missed_median_benefit_vs_base_s": quantile_or_none(missed_benefit, 0.5),
                    "missed_p90_benefit_vs_base_s": quantile_or_none(missed_benefit, 0.9),
                    "missed_total_benefit_vs_base_s": round(sum(missed_benefit), 6),
                    "mean_normalized_regret": mean_or_none(normalized_regrets),
                    "p95_normalized_regret": quantile_or_none(normalized_regrets, 0.95),
                    "policy_failure_count": sum(
                        1
                        for row in policy_rows
                        if row.get("policy_failed") or row.get("policy_correctness_failure")
                    ),
                }
            )
    write_csv(OUT_TAB / "policy_error_severity_summary.csv", rows_out)


def write_queue_load_summary(device_queue: dict[str, list[dict[str, Any]]]) -> None:
    rows_out: list[dict[str, Any]] = []
    for device, rows in device_queue.items():
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["load_level"]), str(row["policy"]))].append(row)
        for (load_level, policy), group in sorted(grouped.items()):
            rows_out.append(
                {
                    "device": device,
                    "load_level": load_level,
                    "policy": policy,
                    "target_utilization": mean_or_none(row.get("target_utilization") for row in group),
                    "mean_arrival_rate_lambda": mean_or_none(row.get("arrival_rate_lambda", row.get("arrival_rate")) for row in group),
                    "mean_service_time_s": mean_or_none(row.get("mean_service_time_s") for row in group),
                    "mean_effective_offered_load": mean_or_none(row.get("effective_offered_load") for row in group),
                    "unstable_seed_count": sum(1 for row in group if row.get("policy_unstable")),
                    "num_seeds": len(group),
                    "mean_p95_completion_time_s": mean_or_none(row.get("p95_completion_time_s") for row in group),
                    "mean_p95_wait_time_s": mean_or_none(row.get("p95_wait_time_s") for row in group),
                    "mean_throughput_jobs_per_s": mean_or_none(row.get("throughput_jobs_per_s") for row in group),
                    "mean_wasted_optimization_overhead_s": mean_or_none(
                        row.get("wasted_optimization_overhead_s") for row in group
                    ),
                    "mean_wrong_admit_count": mean_or_none(row.get("wrong_admit_count") for row in group),
                    "mean_missed_opportunity_count": mean_or_none(
                        row.get("missed_opportunity_count") for row in group
                    ),
                }
            )
    write_csv(OUT_TAB / "queue_replay_load_summary.csv", rows_out)


HARDER_SPLIT_POLICIES = [
    "amortization_policy",
    "risk_aware_policy",
    "first_k_probe_policy",
    "workload_family_holdout_policy",
    "always_eager",
    "always_graphs",
    "static_gate_plus_eligibility",
]

HARDER_SPLIT_FIELDS = [
    "n_conditions",
    "policy_failure_rate",
    "wrong_enable_rate",
    "missed_opportunity_rate",
    "median_normalized_regret",
    "p90_normalized_regret",
    "max_normalized_regret",
    "max_regret_vs_oracle_s",
]


def write_policy_harder_split(device_policy: dict[str, list[dict[str, Any]]]) -> None:
    rows_out: list[dict[str, Any]] = []
    for device, rows in device_policy.items():
        by_policy = {row["policy"]: row for row in rows}
        for policy in HARDER_SPLIT_POLICIES:
            if policy not in by_policy:
                continue
            source = by_policy[policy]
            rows_out.append(
                {
                    "device": device,
                    "policy": policy,
                    **{field: source.get(field) for field in HARDER_SPLIT_FIELDS},
                }
            )
    write_csv(OUT_TAB / "policy_harder_split_summary.csv", rows_out)


def main() -> int:
    setup_style()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)

    device_action = {device: read_jsonl(paths["action"]) for device, paths in DEVICES.items()}
    device_policy_eval = {device: read_jsonl(paths["policy_eval"]) for device, paths in DEVICES.items()}
    device_policy = {device: read_jsonl(paths["policy"]) for device, paths in DEVICES.items()}
    device_queue = {device: read_jsonl(paths["queue"]) for device, paths in DEVICES.items()}

    write_policy_definition_table()
    write_workload_metadata_table(device_action)
    write_device_runtime_context_table()
    write_core_tables(device_action)
    write_csv(OUT_TAB / "oracle_action_mix_by_workload.csv", plot_oracle_action_mix(device_action))
    write_csv(OUT_TAB / "selected_runtime_boundaries.csv", plot_selected_boundaries(device_action["RTX 4060"]))
    write_csv(
        OUT_TAB / "selected_runtime_boundaries_normalized.csv",
        plot_selected_boundaries_normalized(device_action["RTX 4060"]),
    )
    write_csv(OUT_TAB / "policy_regret_and_queue_summary.csv", plot_policy_and_queue(device_policy, device_queue))
    write_policy_error_severity(device_policy_eval)
    write_queue_load_summary(device_queue)
    write_policy_harder_split(device_policy)
    print(f"Wrote figures to {OUT_FIG}")
    print(f"Wrote tables to {OUT_TAB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
