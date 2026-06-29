#!/usr/bin/env python3
"""Generate pilot v1 figures for the RTX 4060 short-job experiment."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ACTION_LABELS = {
    "eager": "Eager",
    "eager_inference_mode": "Inference mode",
    "best_eager": "Best eager",
    "cpu_eager": "CPU eager",
    "compile_only": "Compile",
    "compile_reduce_overhead": "Compile reduce-overhead",
    "compile_max_autotune": "Compile max-autotune",
    "graphs_only": "Graphs",
    "graphs_input_copy": "Graphs input-copy",
    "compile_plus_graphs": "Compile+Graphs",
    "compile_reduce_overhead_plus_graphs": "Compile reduce-overhead+graphs",
    "amp_fp16": "AMP FP16",
    "amp_bf16": "AMP BF16",
    "tf32_eager": "TF32 eager",
    "micro_batch_2": "Micro-batch 2",
    "micro_batch_4": "Micro-batch 4",
}

ACTION_COLORS = {
    "eager": "#56B4E9",
    "eager_inference_mode": "#8CCBEA",
    "best_eager": "#0072B2",
    "cpu_eager": "#999999",
    "compile_only": "#D55E00",
    "compile_reduce_overhead": "#E69F00",
    "compile_max_autotune": "#A65628",
    "graphs_only": "#009E73",
    "graphs_input_copy": "#44AA99",
    "compile_plus_graphs": "#CC79A7",
    "compile_reduce_overhead_plus_graphs": "#AA4499",
    "amp_fp16": "#F0E442",
    "amp_bf16": "#B2DF8A",
    "tf32_eager": "#6A3D9A",
    "micro_batch_2": "#80B1D3",
    "micro_batch_4": "#FDB462",
    "failed": "#BDBDBD",
}

WORKLOAD_LABELS = {
    "dynamic_shape_infer_small": "Dynamic-shape inference",
    "fixed_shape_infer_small": "Fixed-shape inference",
    "short_train_small": "Short training",
    "prefill_toy": "Toy prefill",
    "decode_toy": "Toy decode",
    "cv_online_infer": "CV online inference",
    "short_text_transformer_padded": "Padded text Transformer",
    "short_text_transformer_dynamic": "Dynamic text Transformer",
    "llm_decode_proxy": "LLM decode proxy",
    "dlrm_recommendation": "DLRM-like recommendation",
    "gnn_irregular": "Irregular GNN",
    "rl_policy_infer": "RL policy inference",
    "synthetic_kernel_chain": "Synthetic kernel chain",
}

POLICY_LABELS = {
    "always_eager": "Always eager",
    "always_best_eager": "Always best eager",
    "always_cpu": "Always CPU",
    "always_compile": "Always compile",
    "always_compile_reduce_overhead": "Always compile reduce-overhead",
    "always_graphs": "Always graphs",
    "always_graphs_input_copy": "Always graphs input-copy",
    "always_compile_plus_graphs": "Always compile+graphs",
    "job_length_only": "Job length only",
    "shape_only": "Shape only",
    "static_threshold": "Static threshold",
    "amortization_policy": "Amortization policy",
    "static_gate_only": "Static gate",
    "static_gate_plus_eligibility": "Static gate+elig.",
    "first_k_probe_policy": "First-k probe",
    "risk_aware_policy": "Risk-aware",
    "workload_family_holdout_policy": "Workload holdout",
    "oracle_best_action": "Oracle",
}

ACTION_ORDER = [
    "eager",
    "eager_inference_mode",
    "best_eager",
    "cpu_eager",
    "compile_only",
    "compile_reduce_overhead",
    "compile_max_autotune",
    "graphs_only",
    "graphs_input_copy",
    "compile_plus_graphs",
    "compile_reduce_overhead_plus_graphs",
    "amp_fp16",
    "amp_bf16",
    "tf32_eager",
    "micro_batch_2",
    "micro_batch_4",
]

WORKLOAD_ORDER = [
    "fixed_shape_infer_small",
    "short_train_small",
    "dynamic_shape_infer_small",
    "prefill_toy",
    "decode_toy",
    "cv_online_infer",
    "short_text_transformer_padded",
    "short_text_transformer_dynamic",
    "llm_decode_proxy",
    "dlrm_recommendation",
    "gnn_irregular",
    "rl_policy_infer",
    "synthetic_kernel_chain",
]


def label_action(action: str) -> str:
    return ACTION_LABELS.get(action, action.replace("_", " ").title())


def label_workload(workload: str) -> str:
    return WORKLOAD_LABELS.get(workload, workload.replace("_", " ").title())


def label_policy(policy: str) -> str:
    return POLICY_LABELS.get(policy, policy.replace("_", " ").title())


def ordered_subset(values: set[str], preferred: list[str]) -> list[str]:
    return [item for item in preferred if item in values] + sorted(values - set(preferred))


def action_palette(actions: list[str]) -> dict[str, str]:
    fallback = sns.color_palette("tab20", max(len(actions), 1)).as_hex()
    return {action: ACTION_COLORS.get(action, fallback[index % len(fallback)]) for index, action in enumerate(actions)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_oracle_action_map(summary: pd.DataFrame, out_dir: Path) -> None:
    oracle_actions = {str(action) for action in summary["oracle_action"].dropna().unique()}
    action_order = ordered_subset(oracle_actions, ACTION_ORDER)
    action_to_idx = {action: index for index, action in enumerate(action_order)}
    colors = action_palette(action_order)
    missing_idx = len(action_order)
    cmap = ListedColormap([colors[action] for action in action_order] + [ACTION_COLORS["failed"]])

    workloads = ordered_subset({str(item) for item in summary["workload_id"].dropna().unique()}, WORKLOAD_ORDER)
    batches = sorted(summary["batch_size"].unique())
    steps = sorted(summary["num_steps"].unique())

    fig, axes = plt.subplots(
        nrows=len(workloads),
        ncols=len(batches),
        figsize=(max(7.2, 2.0 * len(batches)), max(2.4, 1.7 * len(workloads))),
        sharex=True,
        sharey=False,
        constrained_layout=True,
        squeeze=False,
    )

    for row_idx, workload in enumerate(workloads):
        for col_idx, batch_size in enumerate(batches):
            ax = axes[row_idx][col_idx]
            subset = summary[
                (summary["workload_id"] == workload)
                & (summary["batch_size"] == batch_size)
                & (summary["action"] == summary["oracle_action"])
            ].sort_values("num_steps")
            oracle_by_step = {row["num_steps"]: row["oracle_action"] for _, row in subset.iterrows()}
            values = [action_to_idx.get(oracle_by_step.get(step), missing_idx) for step in steps]
            ax.imshow([values], aspect="auto", cmap=cmap, vmin=0, vmax=missing_idx)
            ax.set_xticks(range(len(steps)), labels=[str(step) for step in steps])
            ax.set_yticks([])
            ax.set_title(f"batch={batch_size}")
            if col_idx == 0:
                ax.set_ylabel(textwrap.fill(label_workload(workload), width=18), rotation=0, labelpad=58, va="center")
            for idx, step in enumerate(steps):
                action = oracle_by_step.get(step)
                label = label_action(str(action)).replace("+", "\n+") if action else "n/a"
                ax.text(idx, 0, label, ha="center", va="center", fontsize=6)
            ax.set_xlabel("Job length (steps)" if row_idx == len(workloads) - 1 else "")

    handles = [Patch(facecolor=colors[action], label=label_action(action)) for action in action_order]
    fig.legend(handles=handles, loc="upper center", ncol=min(4, len(handles)), frameon=False, bbox_to_anchor=(0.52, 1.05))
    fig.suptitle("Oracle runtime-optimization action changes across short-job conditions", y=1.10)
    save_figure(fig, out_dir, "fig_pilot_4060_oracle_action_map")


def plot_infeasibility_rate(summary: pd.DataFrame, out_dir: Path) -> None:
    workloads = ordered_subset({str(item) for item in summary["workload_id"].dropna().unique()}, WORKLOAD_ORDER)
    actions = ordered_subset({str(item) for item in summary["action"].dropna().unique()}, ACTION_ORDER)
    pivot = summary.pivot_table(
        index="workload_id",
        columns="action",
        values="infeasible_precheck_rate",
        aggfunc="mean",
    ).reindex(index=workloads, columns=actions)
    nonzero = pivot.fillna(0.0)
    pivot = pivot.loc[nonzero.sum(axis=1) > 0, nonzero.sum(axis=0) > 0]
    if pivot.empty:
        pivot = pd.DataFrame([[0.0]], index=["No precheck infeasibility"], columns=["All actions"])
    else:
        pivot.index = [textwrap.fill(label_workload(str(item)), width=22) for item in pivot.index]
        pivot.columns = [textwrap.fill(label_action(str(item)), width=18) for item in pivot.columns]
    annotations = pivot.map(lambda value: "" if pd.isna(value) or float(value) == 0.0 else f"{float(value):.2f}")

    fig_width = max(6.5, 1.35 * len(pivot.columns) + 2.0)
    fig_height = max(2.6, 0.55 * len(pivot.index) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    sns.heatmap(
        pivot,
        ax=ax,
        annot=annotations,
        fmt="",
        cmap=sns.light_palette("#D55E00", as_cmap=True),
        vmin=0,
        vmax=1,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Precheck infeasibility rate"},
    )
    ax.set_xlabel("Execution action")
    ax.set_ylabel("")
    ax.set_title("Precheck infeasibility is part of the admission boundary")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    save_figure(fig, out_dir, "fig_pilot_4060_precheck_infeasibility")


def plot_policy_regret(aggregate: pd.DataFrame, out_dir: Path) -> None:
    policies = [
        "always_cpu",
        "always_best_eager",
        "always_compile",
        "always_compile_reduce_overhead",
        "always_compile_plus_graphs",
        "job_length_only",
        "shape_only",
        "static_threshold",
        "amortization_policy",
        "static_gate_only",
        "static_gate_plus_eligibility",
        "first_k_probe_policy",
        "risk_aware_policy",
        "workload_family_holdout_policy",
        "always_graphs_input_copy",
        "always_graphs",
        "always_eager",
        "oracle_best_action",
    ]
    data = aggregate.set_index("policy").loc[[policy for policy in policies if policy in set(aggregate["policy"])]].reset_index()
    data["label"] = [label_policy(str(item)) for item in data["policy"]]
    data["p90_normalized_regret"] = pd.to_numeric(data["p90_normalized_regret"], errors="coerce")
    data["median_normalized_regret"] = pd.to_numeric(data["median_normalized_regret"], errors="coerce")

    fig, (ax, rate_ax) = plt.subplots(
        ncols=2,
        figsize=(8.4, 4.2),
        width_ratios=(2.5, 1.15),
        constrained_layout=True,
    )
    y_positions = range(len(data))
    ax.barh(
        y_positions,
        data["p90_normalized_regret"],
        color="#0072B2",
        alpha=0.82,
        label="p90 normalized regret",
    )
    ax.scatter(
        data["median_normalized_regret"],
        list(y_positions),
        color="#E69F00",
        zorder=3,
        label="median normalized regret",
    )
    ax.set_yticks(list(y_positions), labels=data["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Normalized regret vs. oracle")
    ax.set_title("Regret distribution")
    ax.legend(loc="lower right", frameon=True)
    finite_p90 = data["p90_normalized_regret"].replace([float("inf"), float("-inf")], pd.NA).dropna()
    ax.set_xlim(0, max(float(finite_p90.max()) * 1.08, 1.0) if not finite_p90.empty else 1.0)

    rate_data = data[
        ["policy_failure_rate", "wrong_enable_rate", "missed_opportunity_rate"]
    ].rename(
        columns={
            "policy_failure_rate": "failure",
            "wrong_enable_rate": "wrong-admit",
            "missed_opportunity_rate": "missed-opportunity",
        }
    )
    sns.heatmap(
        rate_data,
        ax=rate_ax,
        annot=True,
        fmt=".2f",
        cmap=sns.light_palette("#D55E00", as_cmap=True),
        vmin=0,
        vmax=1,
        linewidths=0.8,
        linecolor="white",
        cbar=False,
    )
    rate_ax.set_yticks([])
    rate_ax.set_xticklabels(rate_ax.get_xticklabels(), rotation=35, ha="right")
    rate_ax.set_xlabel("")
    rate_ax.set_ylabel("")
    rate_ax.set_title("Decision errors")
    fig.suptitle("Naive always-enable policies create large tail regret")
    save_figure(fig, out_dir, "fig_pilot_4060_policy_regret")


def plot_runtime_boundary(summary: pd.DataFrame, out_dir: Path) -> None:
    workloads = ordered_subset({str(item) for item in summary["workload_id"].dropna().unique()}, WORKLOAD_ORDER)
    actions = ordered_subset({str(item) for item in summary["action"].dropna().unique()}, ACTION_ORDER)
    colors = action_palette(actions)
    batches = sorted(summary["batch_size"].unique())
    steps = sorted(summary["num_steps"].unique())

    fig, axes = plt.subplots(
        nrows=len(workloads),
        ncols=len(batches),
        figsize=(max(8.0, 2.2 * len(batches)), max(3.0, 2.0 * len(workloads))),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )

    for row_idx, workload in enumerate(workloads):
        for col_idx, batch_size in enumerate(batches):
            ax = axes[row_idx][col_idx]
            subset = summary[
                (summary["workload_id"] == workload)
                & (summary["batch_size"] == batch_size)
                & (summary["action_feasible"])
            ]
            for action, action_rows in subset.groupby("action"):
                action_rows = action_rows.sort_values("num_steps")
                ax.plot(
                    action_rows["num_steps"],
                    action_rows["median_total_runtime_s"],
                    marker="o",
                    linewidth=1.6,
                    markersize=4,
                    color=colors.get(action, ACTION_COLORS["failed"]),
                    label=label_action(str(action)),
                )
            ax.set_xscale("log")
            ax.set_xticks(steps, labels=[str(step) for step in steps])
            ax.set_title(f"{label_workload(workload)}, batch={batch_size}")
            if col_idx == 0:
                ax.set_ylabel("Median total runtime (s)")
            if row_idx == len(workloads) - 1:
                ax.set_xlabel("Job length (steps)")
            ax.grid(True, which="major", linewidth=0.5, alpha=0.4)

    handles = [Patch(facecolor=colors[action], label=label_action(action)) for action in actions]
    fig.legend(handles=handles, loc="upper center", ncol=min(4, len(handles)), frameon=False, bbox_to_anchor=(0.52, 1.035))
    fig.suptitle("Cold-total runtime boundary across execution actions", y=1.06)
    save_figure(fig, out_dir, "fig_pilot_4060_runtime_boundary")


def plot_queue_replay(queue: pd.DataFrame, out_dir: Path) -> None:
    policies = [
        "always_cpu",
        "always_best_eager",
        "always_compile",
        "always_compile_reduce_overhead",
        "always_compile_plus_graphs",
        "job_length_only",
        "shape_only",
        "static_threshold",
        "amortization_policy",
        "static_gate_only",
        "static_gate_plus_eligibility",
        "first_k_probe_policy",
        "risk_aware_policy",
        "workload_family_holdout_policy",
        "always_graphs_input_copy",
        "always_graphs",
        "always_eager",
        "oracle_best_action",
    ]
    load_order = ["light", "medium", "heavy"]
    data = queue[queue["policy"].isin(policies)].copy()
    data["policy_label"] = data["policy"].map(lambda item: label_policy(str(item)))

    aggregate = (
        data.groupby(["load_level", "policy", "policy_label"], as_index=False)
        .agg(
            p95_completion_time_s=("p95_completion_time_s", "mean"),
            wasted_optimization_overhead_s=("wasted_optimization_overhead_s", "mean"),
        )
        .sort_values(["load_level", "policy"])
    )
    aggregate["load_level"] = pd.Categorical(aggregate["load_level"], categories=load_order, ordered=True)
    aggregate = aggregate.sort_values(["load_level", "p95_completion_time_s"])

    fig, (latency_ax, waste_ax) = plt.subplots(
        ncols=2,
        figsize=(9.2, 4.6),
        width_ratios=(1.35, 1.0),
        constrained_layout=True,
    )
    sns.lineplot(
        data=aggregate,
        x="load_level",
        y="p95_completion_time_s",
        hue="policy_label",
        marker="o",
        linewidth=1.6,
        ax=latency_ax,
    )
    latency_ax.set_xlabel("Replay load level")
    latency_ax.set_ylabel("Mean P95 completion time (s)")
    latency_ax.set_yscale("log")
    latency_ax.set_title("Tail latency under worker-queue replay")
    latency_ax.legend(title="", loc="upper left", fontsize=7, frameon=True)

    heavy = aggregate[aggregate["load_level"] == "heavy"].copy()
    heavy = heavy.sort_values("wasted_optimization_overhead_s", ascending=True)
    waste_ax.barh(
        heavy["policy_label"],
        heavy["wasted_optimization_overhead_s"],
        color="#D55E00",
        alpha=0.82,
    )
    waste_ax.set_xlabel("Wasted optimization overhead (s)")
    waste_ax.set_ylabel("")
    waste_ax.set_title("Heavy-load overhead waste")
    fig.suptitle("Local queue replay turns admission mistakes into worker-level cost")
    save_figure(fig, out_dir, "fig_pilot_4060_queue_replay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RTX 4060 pilot v1 figures")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/tables/pilot_4060_v1_action_summary.jsonl"),
    )
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path("results/tables/pilot_4060_v1_policy_aggregate.jsonl"),
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("results/tables/pilot_4060_v1_queue_replay.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures/pilot_4060_v1"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_style()
    summary = pd.DataFrame(read_jsonl(args.summary))
    aggregate = pd.DataFrame(read_jsonl(args.aggregate))
    queue = pd.DataFrame(read_jsonl(args.queue)) if args.queue.exists() else pd.DataFrame()
    plot_oracle_action_map(summary, args.out_dir)
    plot_infeasibility_rate(summary, args.out_dir)
    plot_policy_regret(aggregate, args.out_dir)
    plot_runtime_boundary(summary, args.out_dir)
    if not queue.empty:
        plot_queue_replay(queue, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
