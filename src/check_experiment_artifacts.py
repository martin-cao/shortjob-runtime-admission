#!/usr/bin/env python3
"""Report whether the paper-facing experiment packages have expected artifacts.

This script is intentionally read-only. It does not run GPU workloads; it only
checks that raw data, completeness reports, summaries, policy outputs, queue
replays, and figures exist in the expected repository locations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ArtifactPackage:
    package_id: str
    device: str
    experiment: str
    expected_raw_rows: int | None
    expected_summary_rows: int | None
    raw: Path
    completeness: Path | None = None
    prime_raw: Path | None = None
    summary: Path | None = None
    policy_eval: Path | None = None
    policy_aggregate: Path | None = None
    queue_replay: Path | None = None
    figure_dir: Path | None = None
    extra_required: tuple[Path, ...] = ()


PAPER_TABLES = ROOT / "results/tables/paper_official_20260627"
PAPER_FIGURES = ROOT / "results/figures/paper_official_20260627"

# (package_dir, device label) for the four paper-facing devices, in paper order.
DEVICE_PACKAGES = [
    ("official_4060_20260625", "RTX 4060 Laptop"),
    ("official_v100_20260625", "V100"),
    ("official_a100_20260626", "A100-SXM4-80GB"),
    ("official_h100_20260627", "H100 80GB HBM3"),
]


def _cold_package(pkg: str, device: str) -> ArtifactPackage:
    base = ROOT / "results/tables" / pkg / "cold_core"
    return ArtifactPackage(
        package_id=f"{pkg}/cold_core",
        device=device,
        experiment="cold_core",
        expected_raw_rows=8320,
        expected_summary_rows=1664,
        raw=ROOT / "data/raw" / pkg / "cold_core.jsonl",
        completeness=base / "completeness_report.json",
        summary=base / "action_summary.jsonl",
        policy_eval=base / "policy_eval.jsonl",
        policy_aggregate=base / "policy_aggregate.jsonl",
        queue_replay=base / "queue_replay.jsonl",
    )


def _warm_package(pkg: str, device: str) -> ArtifactPackage:
    base = ROOT / "results/tables" / pkg / "warm_reuse_core"
    return ArtifactPackage(
        package_id=f"{pkg}/warm_reuse_core",
        device=device,
        experiment="warm_compile_cache",
        expected_raw_rows=360,
        expected_summary_rows=120,
        raw=ROOT / "data/raw" / pkg / "warm_reuse_core.jsonl",
        prime_raw=ROOT / "data/raw" / pkg / "warm_reuse_core_prime.jsonl",
        completeness=base / "completeness_report.json",
        summary=base / "action_summary.jsonl",
        policy_eval=base / "policy_eval.jsonl",
        policy_aggregate=base / "policy_aggregate.jsonl",
    )


def _graph_reuse_package(pkg: str, device: str) -> ArtifactPackage:
    return ArtifactPackage(
        package_id=f"{pkg}/graph_reuse_core",
        device=device,
        experiment="graph_reuse",
        expected_raw_rows=288,
        expected_summary_rows=None,
        raw=ROOT / "data/raw" / pkg / "graph_reuse_core.jsonl",
        # The distilled graph-reuse summary is device-pooled in the paper package
        # (graph_reuse_sensitivity_device_summary.csv); there is no per-device
        # distilled table, so only the raw run is validated here.
    )


PACKAGES = (
    [_cold_package(pkg, dev) for pkg, dev in DEVICE_PACKAGES]
    + [_warm_package(pkg, dev) for pkg, dev in DEVICE_PACKAGES]
    + [_graph_reuse_package(pkg, dev) for pkg, dev in DEVICE_PACKAGES]
    + [
        ArtifactPackage(
            package_id="paper_official_20260627/distilled",
            device="all four",
            experiment="paper_distilled",
            expected_raw_rows=None,
            expected_summary_rows=None,
            # Anchor on the oracle-count table the LaTeX boundary numbers cite.
            raw=PAPER_TABLES / "core_oracle_counts.csv",
            summary=PAPER_TABLES / "action_validity_audit.csv",
            figure_dir=PAPER_FIGURES,
            extra_required=(
                PAPER_TABLES / "rank_stability_device_summary.csv",
                PAPER_TABLES / "action_validity_reasons.csv",
                PAPER_TABLES / "ACTION_VALIDITY.md",
                PAPER_TABLES / "warm_compile_cache_sensitivity_summary.csv",
                PAPER_TABLES / "graph_reuse_sensitivity_device_summary.csv",
                PAPER_TABLES / "policy_regret_and_queue_summary.csv",
                PAPER_TABLES / "policy_harder_split_summary.csv",
                PAPER_TABLES / "queue_replay_job_mix_sensitivity_heavy_summary.csv",
                # Figures actually included by the LaTeX submission source.
                PAPER_FIGURES / "fig_oracle_action_mix.pdf",
                PAPER_FIGURES / "fig_selected_runtime_boundaries.pdf",
            ),
        )
    ]
)


def count_lines(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def path_status(path: Path | None) -> str:
    if path is None:
        return "n/a"
    return "ok" if path.exists() else "missing"


def figure_count(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return sum(1 for item in path.iterdir() if item.is_file())


def load_completeness(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_package(package: ArtifactPackage) -> dict[str, Any]:
    raw_rows = count_lines(package.raw)
    summary_rows = count_lines(package.summary)
    prime_rows = count_lines(package.prime_raw)
    completeness = load_completeness(package.completeness)
    missing_core = [
        name
        for name, path in (
            ("raw", package.raw),
            ("completeness", package.completeness),
            ("summary", package.summary),
            ("policy_eval", package.policy_eval),
            ("policy_aggregate", package.policy_aggregate),
        )
        if path is not None and not path.exists()
    ]
    if package.experiment == "cold_core" and package.queue_replay is not None and not package.queue_replay.exists():
        missing_core.append("queue_replay")
    for extra in package.extra_required:
        if not extra.exists():
            missing_core.append(rel(extra) or str(extra))

    return {
        "package_id": package.package_id,
        "device": package.device,
        "experiment": package.experiment,
        "raw": rel(package.raw),
        "raw_status": path_status(package.raw),
        "raw_rows": raw_rows,
        "expected_raw_rows": package.expected_raw_rows,
        "summary": rel(package.summary),
        "summary_status": path_status(package.summary),
        "summary_rows": summary_rows,
        "expected_summary_rows": package.expected_summary_rows,
        "completeness": rel(package.completeness),
        "completeness_status": path_status(package.completeness),
        "missing_count": completeness.get("missing_count"),
        "duplicate_condition_count": completeness.get("duplicate_condition_count"),
        "unexpected_condition_count": completeness.get("unexpected_condition_count"),
        "runtime_failure_count": completeness.get("runtime_failure_count"),
        "prime_raw": rel(package.prime_raw),
        "prime_raw_status": path_status(package.prime_raw),
        "prime_rows": prime_rows,
        "policy_eval_status": path_status(package.policy_eval),
        "policy_aggregate_status": path_status(package.policy_aggregate),
        "queue_replay_status": path_status(package.queue_replay),
        "figure_dir": rel(package.figure_dir),
        "figure_dir_status": path_status(package.figure_dir),
        "figure_file_count": figure_count(package.figure_dir),
        "missing_core_artifacts": missing_core,
    }


def status_label(row: dict[str, Any]) -> str:
    if row["missing_core_artifacts"]:
        return "missing"
    if (
        row["expected_raw_rows"] is not None
        and row["raw_rows"] is not None
        and row["raw_rows"] != row["expected_raw_rows"]
    ):
        return "row-mismatch"
    if (
        row["expected_summary_rows"] is not None
        and row["summary_rows"] is not None
        and row["summary_rows"] != row["expected_summary_rows"]
    ):
        return "row-mismatch"
    if row["missing_count"] not in (None, 0):
        return "incomplete"
    if row["duplicate_condition_count"] not in (None, 0):
        return "duplicates"
    if row["unexpected_condition_count"] not in (None, 0):
        return "unexpected"
    return "ok"


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| Package | Device | Experiment | Status | Raw rows | Summary rows | Runtime failures | Missing core |")
    print("|---|---|---|---|---:|---:|---:|---|")
    for row in rows:
        print(
            "| {package_id} | {device} | {experiment} | {status} | {raw_rows} | "
            "{summary_rows} | {runtime_failure_count} | {missing_core} |".format(
                package_id=row["package_id"],
                device=row["device"],
                experiment=row["experiment"],
                status=row["status"],
                raw_rows=row["raw_rows"] if row["raw_rows"] is not None else "n/a",
                summary_rows=row["summary_rows"] if row["summary_rows"] is not None else "n/a",
                runtime_failure_count=(
                    row["runtime_failure_count"] if row["runtime_failure_count"] is not None else "n/a"
                ),
                missing_core=", ".join(row["missing_core_artifacts"]) or "-",
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--out", type=Path, default=None, help="Optional output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for package in PACKAGES:
        row = summarize_package(package)
        row["status"] = status_label(row)
        rows.append(row)

    if args.format == "json":
        output = json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    else:
        from io import StringIO
        import contextlib

        buffer = StringIO()
        with contextlib.redirect_stdout(buffer):
            print_markdown(rows)
        output = buffer.getvalue()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
