#!/usr/bin/env python3
"""Generate paper-facing action-validity reason artifacts.

This is a derived-data script: it reads the saved correctness gates and writes
CSV/Markdown summaries. It does not rerun GPU workloads or alter performance
measurements.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "tables" / "paper_official_20260627"

GATES = {
    "RTX 4060": ROOT / "data" / "raw" / "official_4060_20260625" / "correctness_gate.jsonl",
    "V100": ROOT / "data" / "raw" / "official_v100_20260625" / "correctness_gate.jsonl",
    "A100": ROOT / "data" / "raw" / "official_a100_20260626" / "correctness_gate.jsonl",
    "H100": ROOT / "data" / "raw" / "official_h100_20260627" / "correctness_gate.jsonl",
}

ACTION_ORDER = [
    "eager",
    "best_eager",
    "graphs_only",
    "graphs_input_copy",
    "compile_only",
    "compile_reduce_overhead",
    "compile_plus_graphs",
    "compile_reduce_overhead_plus_graphs",
]

EVALUATION_TREATMENT = {
    "eager": "eager baseline",
    "best_eager": "eager baseline when legal",
    "graphs_only": "eligible candidate when valid",
    "graphs_input_copy": "eligible candidate when valid",
    "compile_only": "performance stress baseline",
    "compile_reduce_overhead": "performance stress baseline",
    "compile_plus_graphs": "diagnostic candidate",
    "compile_reduce_overhead_plus_graphs": "excluded from positive evidence",
}

FAILURE_MECHANISM_ROWS = [
    {
        "failure_class": "reduce_overhead_plus_explicit_graph_capture_failure",
        "cells": 10,
        "actions": "compile_reduce_overhead_plus_graphs",
        "workloads": "all graph-precheck-feasible workloads",
        "stage": "capture/execution",
        "likely_mechanism": (
            "explicit torch.cuda.CUDAGraph capture wraps a torch.compile(mode='reduce-overhead') "
            "callable; PyTorch reduce-overhead itself uses CUDA graphs to reduce Python overhead, "
            "so the observed errors are a nested/manual-plus-framework capture anti-pattern"
        ),
        "divergence_class": "feasibility rejection, not numerical divergence",
        "paper_treatment": "avoid-target / negative-control stress case; excluded from positive graph evidence",
    },
    {
        "failure_class": "padded_text_explicit_graph_capture_failure",
        "cells": 2,
        "actions": "graphs_only; graphs_input_copy",
        "workloads": "short_text_transformer_padded",
        "stage": "capture/execution",
        "likely_mechanism": (
            "manual graph capture around the padded Transformer path hits stream-capture-invalidated "
            "runtime errors despite fixed tensor shapes"
        ),
        "divergence_class": "capture feasibility rejection, not completed-output divergence",
        "paper_treatment": "graph eligibility caveat for this workload family",
    },
    {
        "failure_class": "short_train_loss_mismatch",
        "cells": 3,
        "actions": "graphs_only; graphs_input_copy; compile_plus_graphs",
        "workloads": "short_train_small",
        "stage": "correctness",
        "likely_mechanism": (
            "stateful training graph replay compares final loss and parameters after matched warm-up, "
            "capture, and replay trajectory; only the loss scalar trips the strict gate"
        ),
        "divergence_class": "strict-tolerance state rejection; possible tolerance artifact, not proved semantic failure",
        "paper_treatment": "exclude short-train graph paths from positive graph evidence",
    },
]

MAIN_LIMITATION = {
    "eager": "none",
    "best_eager": "training-ineligible",
    "graphs_only": "two failed workloads",
    "graphs_input_copy": "two failed workloads",
    "compile_only": "high startup cost",
    "compile_reduce_overhead": "high startup cost",
    "compile_plus_graphs": "one failed workload",
    "compile_reduce_overhead_plus_graphs": "fails capture-feasible workloads",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gate_status(row: dict[str, Any]) -> str:
    if row.get("status") == "passed":
        return "pass"
    if row.get("failure_stage") == "precheck" or row.get("status") in {"infeasible", "skipped"}:
        return "infeasible"
    return "fail"


def parse_exception(row: dict[str, Any]) -> tuple[str, str]:
    reason = str(row.get("failure_reason") or "")
    parts = reason.split(": ", 2)
    if len(parts) == 3 and parts[0] in {"precheck", "compile", "capture", "execution"}:
        return parts[1], parts[2]
    if len(parts) >= 2:
        return parts[0], ": ".join(parts[1:])
    return "", reason


def failure_category(row: dict[str, Any]) -> str:
    action = str(row["action"])
    workload = str(row["workload_id"])
    stage = row.get("failure_stage")
    reason = str(row.get("failure_reason") or "")

    if gate_status(row) == "pass":
        return "passed"
    if stage == "precheck":
        if "inference_mode_not_valid_for_training_workload" in reason:
            return "precheck_training_inference_mode"
        if "unstable_shape_not_graph_capturable" in reason:
            return "precheck_unstable_shape"
        if "workload_declares_cuda_graph_ineligible" in reason:
            return "precheck_declared_graph_ineligible"
        return "precheck_other"
    if stage == "correctness" and workload == "short_train_small" and row.get("mismatch_key") == "loss":
        return "short_train_loss_mismatch"
    if stage == "execution" and action == "compile_reduce_overhead_plus_graphs":
        return "reduce_overhead_plus_explicit_graph_capture_failure"
    if stage == "execution" and workload == "short_text_transformer_padded":
        return "padded_text_explicit_graph_capture_failure"
    if stage == "execution":
        return "runtime_execution_failure"
    return "other_failure"


def evaluation_treatment(row: dict[str, Any]) -> str:
    return EVALUATION_TREATMENT[str(row["action"])]


def concise(value: Any, limit: int = 180) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def reason_row(device: str, row: dict[str, Any]) -> dict[str, Any]:
    status = gate_status(row)
    exc_type, exc_msg = parse_exception(row)
    mismatch_key = str(row.get("mismatch_key") or "")
    return {
        "device": device,
        "workload": row["workload_id"],
        "action": row["action"],
        "status": status,
        "failure_stage": row.get("failure_stage") or "",
        "reference_action": row.get("reference_action") or "",
        "batch_size": row.get("batch_size"),
        "num_steps": row.get("num_steps"),
        "seed": row.get("seed"),
        "precheck_reason": concise(row.get("failure_reason")) if status == "infeasible" else "",
        "runtime_failure": str(row.get("failure_stage") == "execution").lower(),
        "exception_type": exc_type if status != "pass" else "",
        "exception_message_short": concise(exc_msg),
        "correctness_metric": (
            "torch.testing.assert_close over workload correctness_state"
            if row.get("failure_stage") == "correctness"
            else ""
        ),
        "atol": row.get("atol"),
        "rtol": row.get("rtol"),
        "max_abs_error": row.get("max_abs_error"),
        "max_rel_error": row.get("max_rel_error"),
        "output_mismatch": str(mismatch_key == "output").lower(),
        "state_mismatch": str(bool(mismatch_key and mismatch_key != "output")).lower(),
        "mismatch_key": mismatch_key,
        "compared_state_keys": " ".join(row.get("compared_state_keys") or []),
        "failure_category": failure_category(row),
        "evaluation_treatment": evaluation_treatment(row),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_action_validity_audit(all_rows: dict[str, list[dict[str, Any]]]) -> None:
    rows_out: list[dict[str, Any]] = []
    for device, rows in all_rows.items():
        by_action: dict[str, Counter[str]] = defaultdict(Counter)
        feasible_false: Counter[str] = Counter()
        for row in rows:
            action = str(row["action"])
            by_action[action][gate_status(row)] += 1
            if row.get("action_feasible") is False:
                feasible_false[action] += 1
        for action in ACTION_ORDER:
            rows_out.append(
                {
                    "dataset": device,
                    "action": action,
                    "passed_workloads": by_action[action]["pass"],
                    "infeasible_workloads": by_action[action]["infeasible"],
                    "failed_workloads": by_action[action]["fail"],
                    "action_feasible_false": feasible_false[action],
                    "main_limitation": MAIN_LIMITATION[action],
                    "evaluation_treatment": EVALUATION_TREATMENT[action],
                }
            )
    write_csv(OUT_DIR / "action_validity_audit.csv", rows_out)


def write_failure_mechanisms(reason_rows: list[dict[str, Any]]) -> None:
    representative_failures = [
        row for row in reason_rows if row["device"] == "RTX 4060" and row["status"] == "fail"
    ]
    counts = Counter(row["failure_category"] for row in representative_failures)
    rows_out = []
    for row in FAILURE_MECHANISM_ROWS:
        observed = counts[row["failure_class"]]
        rows_out.append({**row, "observed_cells": observed, "count_matches_observed": observed == row["cells"]})
    write_csv(OUT_DIR / "action_failure_mechanisms.csv", rows_out)


def markdown_summary(all_rows: dict[str, list[dict[str, Any]]], reason_rows: list[dict[str, Any]]) -> str:
    representative = next(iter(all_rows.values()))
    status_counts = Counter(gate_status(row) for row in representative)
    fail_counts = Counter(row["action"] for row in representative if gate_status(row) == "fail")
    fail_categories = Counter(
        row["failure_category"] for row in reason_rows if row["device"] == "RTX 4060" and row["status"] == "fail"
    )
    infeasible_categories = Counter(
        row["failure_category"]
        for row in reason_rows
        if row["device"] == "RTX 4060" and row["status"] == "infeasible"
    )

    failed_cells = [
        row for row in reason_rows if row["device"] == "RTX 4060" and row["status"] == "fail"
    ]

    lines = [
        "# Action Validity and Correctness Gate",
        "",
        "This artifact is generated from the saved four-device `correctness_gate.jsonl` files by `src/summarize_action_validity_reasons.py`. It does not rerun GPU workloads or change performance numbers.",
        "",
        "## Checker Definition",
        "",
        "- Reference action: literal `eager`, not `a_base`.",
        "- Official gate condition: one workload-action row at `batch_size=4`, `num_steps=3`, `seed=42` per device.",
        "- Inputs and initial model state are reproducible because the checker seeds model/input construction before building both the eager reference and candidate workload.",
        "- For `graphs_input_copy`, refreshed request inputs use an explicitly seeded generator so the eager reference and graph candidate see the same refreshed-input sequence.",
        "- Graph-action references run the same logical trajectory as the candidate: three graph warm-up steps, one capture-equivalent step, then the requested replay steps.",
        "- Inference workloads compare `output` from `correctness_state()`.",
        "- The training workload compares `loss` and model `param:*` tensors from `correctness_state()`; gradients and optimizer state are not separately saved by the current checker.",
        "- Pass uses `torch.testing.assert_close(candidate, eager_reference, rtol=1e-4, atol=1e-5)` for every compared state tensor.",
        "",
        "## Outcome Definitions",
        "",
        "- `pass`: the action executes and all compared output/state tensors match the eager reference within tolerance.",
        "- `infeasible`: precheck rejects the action before execution, for example because inference mode is illegal for training or the workload declares graph capture ineligible.",
        "- `fail`: execution/capture raises after admission, or execution completes but the output/state equivalence check fails.",
        "- Table I counts workload-action validation cells. The current official gate has one tested condition per workload-action cell, so no multi-condition vote is used.",
        "",
        "## Four-Device Summary",
        "",
        f"- Each device has {status_counts['pass']} pass, {status_counts['infeasible']} infeasible, and {status_counts['fail']} fail cells out of 104.",
        "- The pass/infeasible/fail pattern is identical on RTX 4060, V100, A100, and H100.",
        "",
        "## Fail Distribution",
        "",
    ]
    for action in ACTION_ORDER:
        if fail_counts[action]:
            lines.append(f"- `{action}`: {fail_counts[action]} fail cells.")
    lines.extend(
        [
            "",
            "Failure categories on the representative device:",
        ]
    )
    for category, count in sorted(fail_categories.items()):
        lines.append(f"- `{category}`: {count} cells.")
    lines.extend(
        [
            "",
            "The 15 fail cells are concentrated in three observed categories rather than 15 unrelated defects. `compile_reduce_overhead_plus_graphs` is an explicit manual capture wrapped around a `torch.compile(mode='reduce-overhead')` callable; PyTorch documents `reduce-overhead` as a CUDA-graphs-based mode for reducing Python overhead, so this row is treated as a nested/manual-plus-framework capture anti-pattern rather than generic CUDA Graph fragility. The short-train loss mismatch is a strict-tolerance state rejection; the current gate does not prove semantic training divergence.",
            "",
            "## Failure Mechanism Classes",
            "",
            "| Class | Cells | Actions | Mechanism / cause | Divergence class | Treatment |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for row in FAILURE_MECHANISM_ROWS:
        lines.append(
            f"| `{row['failure_class']}` | {row['cells']} | `{row['actions']}` | "
            f"{row['likely_mechanism']} | {row['divergence_class']} | {row['paper_treatment']} |"
        )
    lines.extend(
        [
            "",
            "## Failed Workload-Action Cells",
            "",
            "| Workload | Action | Stage | Category | Reason |",
            "|---|---|---|---|---|",
        ]
    )
    for row in failed_cells:
        reason = row["exception_message_short"] or row["mismatch_key"]
        lines.append(
            f"| `{row['workload']}` | `{row['action']}` | `{row['failure_stage']}` | `{row['failure_category']}` | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Infeasible Categories",
            "",
        ]
    )
    for category, count in sorted(infeasible_categories.items()):
        lines.append(f"- `{category}`: {count} cells.")
    lines.extend(
        [
            "",
            "## Evaluation Treatment",
            "",
            "- `eager` and legal `best_eager` are eager-family baselines.",
            "- `graphs_only` and `graphs_input_copy` are eligible candidates only for workload families that pass the gate and runtime feasibility checks.",
            "- `compile_only` and `compile_reduce_overhead` pass correctness but are retained as performance stress baselines because their startup cost is not amortized in the cold-total matrix.",
            "- `compile_plus_graphs` is a diagnostic combined action.",
            "- `compile_reduce_overhead_plus_graphs` is excluded from positive evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    all_rows = {device: read_jsonl(path) for device, path in GATES.items()}
    reason_rows = [reason_row(device, row) for device, rows in all_rows.items() for row in rows]
    write_csv(OUT_DIR / "action_validity_reasons.csv", reason_rows)
    write_action_validity_audit(all_rows)
    write_failure_mechanisms(reason_rows)
    (OUT_DIR / "ACTION_VALIDITY.md").write_text(markdown_summary(all_rows, reason_rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
