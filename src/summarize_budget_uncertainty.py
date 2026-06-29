#!/usr/bin/env python3
"""Summarize budget-uncertainty admission results."""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from shortjob_runner.io import read_jsonl, write_jsonl


KEY_POLICIES = (
    "always_eager",
    "always_graphs",
    "static_threshold",
    "amortization_policy",
    "first_k_probe_policy",
    "risk_aware_policy",
)


def mean(values: list[float]) -> float | None:
    return round(float(statistics.fmean(values)), 6) if values else None


def parse_factor(row: dict[str, Any]) -> float:
    match = re.search(r"budget_factor(\d+)", str(row.get("trace_id") or row.get("queue_run_id") or ""))
    if not match:
        raise ValueError(f"cannot parse budget factor from queue row: {row}")
    return int(match.group(1)) / 100.0


def aggregate_queue_rows(queue_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in queue_rows:
        by_key[(parse_factor(row), row["load_level"], row["policy"])].append(row)

    outputs: list[dict[str, Any]] = []
    for (factor, load_level, policy), rows in sorted(by_key.items()):
        outputs.append(
            {
                "budget_perturbation_factor": factor,
                "budget_perturbation_pct": round((factor - 1.0) * 100.0, 3),
                "load_level": load_level,
                "policy": policy,
                "n_replays": len(rows),
                "mean_p95_completion_time_s": mean(
                    [float(row["p95_completion_time_s"]) for row in rows if row.get("p95_completion_time_s") is not None]
                ),
                "mean_p99_completion_time_s": mean(
                    [float(row["p99_completion_time_s"]) for row in rows if row.get("p99_completion_time_s") is not None]
                ),
                "mean_wasted_optimization_overhead_s": mean(
                    [
                        float(row["wasted_optimization_overhead_s"])
                        for row in rows
                        if row.get("wasted_optimization_overhead_s") is not None
                    ]
                ),
                "mean_wrong_admit_count": mean(
                    [float(row["wrong_admit_count"]) for row in rows if row.get("wrong_admit_count") is not None]
                ),
                "mean_missed_opportunity_count": mean(
                    [
                        float(row["missed_opportunity_count"])
                        for row in rows
                        if row.get("missed_opportunity_count") is not None
                    ]
                ),
                "mean_oracle_gap_p95": mean(
                    [float(row["oracle_gap_p95"]) for row in rows if row.get("oracle_gap_p95") is not None]
                ),
            }
        )
    return outputs


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def table_for_policy_aggregate(rows: list[dict[str, Any]]) -> str:
    filtered = [row for row in rows if row["policy"] in KEY_POLICIES]
    lines = [
        "| Budget error | Policy | Wrong | Missed | P90 norm regret | Max regret s |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in filtered:
        lines.append(
            "| {err:+.0f}% | `{policy}` | {wrong} | {missed} | {p90} | {max_s} |".format(
                err=float(row["budget_perturbation_pct"]),
                policy=row["policy"],
                wrong=format_float(row["wrong_enable_rate"]),
                missed=format_float(row["missed_opportunity_rate"]),
                p90=format_float(row["p90_normalized_regret"]),
                max_s=format_float(row["max_regret_vs_oracle_s"]),
            )
        )
    return "\n".join(lines)


def table_for_heavy_queue(rows: list[dict[str, Any]]) -> str:
    filtered = [
        row for row in rows if row["policy"] in KEY_POLICIES and row["load_level"] == "heavy"
    ]
    lines = [
        "| Budget error | Policy | P95 completion s | Waste s | Wrong count | Missed count |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in filtered:
        lines.append(
            "| {err:+.0f}% | `{policy}` | {p95} | {waste} | {wrong} | {missed} |".format(
                err=float(row["budget_perturbation_pct"]),
                policy=row["policy"],
                p95=format_float(row["mean_p95_completion_time_s"]),
                waste=format_float(row["mean_wasted_optimization_overhead_s"]),
                wrong=format_float(row["mean_wrong_admit_count"], digits=1),
                missed=format_float(row["mean_missed_opportunity_count"], digits=1),
            )
        )
    return "\n".join(lines)


def write_markdown(path: Path, aggregate_rows: list[dict[str, Any]], queue_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Budget Uncertainty Sensitivity Summary",
                "",
                "This is a derived analysis over the RTX 4060 combined-v3 first-k summary. "
                "Measured service times are unchanged; only the policy-facing declared budget is perturbed.",
                "",
                "## Policy-Level Sensitivity",
                "",
                table_for_policy_aggregate(aggregate_rows),
                "",
                "## Heavy-Load Queue Replay",
                "",
                table_for_heavy_queue(queue_rows),
                "",
                "## Interpretation",
                "",
                "- `amortization_policy` is stable under +/-25% and +50% budget error in this pilot.",
                "- Under -50% budget error, `amortization_policy` becomes more conservative: wrong-admit drops, missed opportunities increase.",
                "- `first_k_probe_policy` and `risk_aware_policy` remain more graph-aggressive when the budget is accurate or overestimated, so the current risk margin does not yet justify a robust-controller claim.",
                "- This supports a conservative admission / abstention framing rather than a deployable two-stage online controller claim.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize budget uncertainty sensitivity outputs")
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--queue", type=Path, nargs="+", required=True)
    parser.add_argument("--queue-aggregate-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    aggregate_rows = read_jsonl(args.aggregate)
    queue_rows: list[dict[str, Any]] = []
    for path in args.queue:
        queue_rows.extend(read_jsonl(path))
    queue_aggregate = aggregate_queue_rows(queue_rows)
    write_jsonl(args.queue_aggregate_out, queue_aggregate)
    write_markdown(args.markdown_out, aggregate_rows, queue_aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
