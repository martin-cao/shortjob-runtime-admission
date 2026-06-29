#!/usr/bin/env python3
"""Condition-level bootstrap confidence intervals for policy metrics.

The three-repeat Student-t intervals quantify per-condition measurement noise.
This script answers a different question raised in review: how stable are the
*policy-level* aggregates (normalized regret, wrong-admit rate, missed-opportunity
rate) under resampling of the condition set itself?

It is a no-GPU derived analysis: it consumes the per-condition ``policy_eval``
rows already produced by evaluate_baselines / shortjob_runner.policies and
resamples conditions with replacement (paired across policies, so every policy
sees the same resampled conditions in each draw).  Pure numpy; no new deps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from shortjob_runner.io import read_jsonl
from shortjob_runner.summarize import GROUP_FIELDS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "tables" / "paper_official_20260627" / "policy_bootstrap_ci.csv"


def condition_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in GROUP_FIELDS)


def p90(values: np.ndarray) -> float:
    """P90 matching the index convention used in policy aggregation."""
    if values.size == 0:
        return float("nan")
    ordered = np.sort(values)
    index = min(ordered.size - 1, int(0.9 * (ordered.size - 1)))
    return float(ordered[index])


def mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.parent.name, path
    label, path = value.split("=", 1)
    return label, Path(path)


class InconsistentConditionSets(ValueError):
    """Raised when policies on one device do not cover identical condition sets."""


class DuplicatePolicyConditionRow(ValueError):
    """Raised when a (policy, condition) pair appears more than once."""


def collect_device(
    rows: list[dict[str, Any]],
) -> tuple[list[tuple[Any, ...]], dict[str, dict[tuple[Any, ...], dict[str, Any]]]]:
    """Index rows by policy and condition, rejecting duplicates and inconsistency.

    - A repeated (policy, condition) pair is a hard error (no silent overwrite).
    - Every policy must cover exactly the same condition set; a missing or extra
      condition for any policy is a hard error (no silent zero-fill).
    """
    by_policy: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
    conditions: list[tuple[Any, ...]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        policy = row["policy"]
        key = condition_key(row)
        if key in by_policy[policy]:
            raise DuplicatePolicyConditionRow(
                f"duplicate row for policy={policy!r} condition={key!r}"
            )
        by_policy[policy][key] = row
        if key not in seen:
            seen.add(key)
            conditions.append(key)

    condition_set = set(conditions)
    for policy, rows_by_cond in by_policy.items():
        policy_conditions = set(rows_by_cond)
        if policy_conditions != condition_set:
            missing = sorted(map(str, condition_set - policy_conditions))[:5]
            extra = sorted(map(str, policy_conditions - condition_set))[:5]
            raise InconsistentConditionSets(
                f"policy={policy!r} condition set differs from device set; "
                f"missing={missing} extra={extra}"
            )
    return sorted(conditions), by_policy


def metric_arrays(
    policy_rows: dict[tuple[Any, ...], dict[str, Any]],
    conditions: list[tuple[Any, ...]],
) -> dict[str, np.ndarray]:
    """Per-condition metric vectors aligned to ``conditions``.

    ``regret_mask`` marks conditions with a defined regret (a feasible choice);
    a failed/None regret is excluded from regret statistics via the mask rather
    than being silently coerced to zero.  Boolean rates and the failure rate are
    defined over all conditions, matching ``aggregate_policy_results`` so the
    bootstrap brackets exactly those point estimates.
    """
    regret: list[float] = []
    regret_mask: list[bool] = []
    wrong: list[float] = []
    miss: list[float] = []
    failed: list[float] = []
    for key in conditions:
        row = policy_rows[key]  # guaranteed present by collect_device
        nr = row.get("normalized_regret_vs_oracle")
        if nr is None:
            regret.append(np.nan)
            regret_mask.append(False)
        else:
            regret.append(float(nr))
            regret_mask.append(True)
        wrong.append(1.0 if row.get("wrong_enable") else 0.0)
        miss.append(1.0 if row.get("missed_opportunity") else 0.0)
        failed.append(1.0 if row.get("policy_failed") else 0.0)
    return {
        "regret": np.array(regret),
        "regret_mask": np.array(regret_mask, dtype=bool),
        "wrong": np.array(wrong),
        "miss": np.array(miss),
        "failed": np.array(failed),
    }


def build_bootstrap_indices(n: int, n_samples: int, *, seed_material: str) -> np.ndarray:
    """Deterministic (n_samples x n) resample index matrix.

    Built ONCE per device and reused for every policy so that, within each
    bootstrap draw, all policies are evaluated on exactly the same resampled
    conditions (paired bootstrap).  hashlib seeding avoids PYTHONHASHSEED
    randomization.
    """
    digest = hashlib.sha256(seed_material.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return rng.integers(0, n, size=(n_samples, n))


def bootstrap_ci(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, tuple[float, float, float]]:
    """Percentile CIs for one policy using a shared resample-index matrix."""
    regret = arrays["regret"]
    regret_mask = arrays["regret_mask"]
    observed_regret = regret[regret_mask]
    point = {
        "mean_normalized_regret": mean(observed_regret),
        "p90_normalized_regret": p90(observed_regret),
        "wrong_enable_rate": mean(arrays["wrong"]),
        "missed_opportunity_rate": mean(arrays["miss"]),
        "policy_failure_rate": mean(arrays["failed"]),
    }
    n_samples = indices.shape[0]
    draws = {key: np.empty(n_samples) for key in point}
    # Vectorized rate metrics (paired: same indices for every policy/metric).
    draws["wrong_enable_rate"][:] = arrays["wrong"][indices].mean(axis=1)
    draws["missed_opportunity_rate"][:] = arrays["miss"][indices].mean(axis=1)
    draws["policy_failure_rate"][:] = arrays["failed"][indices].mean(axis=1)
    # Regret metrics need per-draw masking of undefined regrets.
    for b in range(n_samples):
        idx = indices[b]
        r = regret[idx]
        m = regret_mask[idx]
        r_valid = r[m]
        draws["mean_normalized_regret"][b] = mean(r_valid)
        draws["p90_normalized_regret"][b] = p90(r_valid)
    out: dict[str, tuple[float, float, float]] = {}
    for key, sample in draws.items():
        valid = sample[~np.isnan(sample)]
        if valid.size == 0:
            out[key] = (point[key], float("nan"), float("nan"))
            continue
        lo, hi = np.percentile(valid, [2.5, 97.5])
        out[key] = (point[key], float(lo), float(hi))
    return out


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy-eval",
        nargs="+",
        required=True,
        help="Inputs as DEVICE_LABEL=policy_eval.jsonl.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def device_bootstrap_rows(
    device_label: str,
    rows: list[dict[str, Any]],
    *,
    n_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Bootstrap every policy on one device with a single shared index matrix."""
    conditions, by_policy = collect_device(rows)
    indices = build_bootstrap_indices(
        len(conditions), n_samples, seed_material=f"{seed}:{device_label}"
    )
    out_rows: list[dict[str, Any]] = []
    for policy in sorted(by_policy):
        arrays = metric_arrays(by_policy[policy], conditions)
        cis = bootstrap_ci(arrays, indices)
        for metric, (point, lo, hi) in cis.items():
            out_rows.append(
                {
                    "device_label": device_label,
                    "policy": policy,
                    "metric": metric,
                    "point_estimate": round(point, 6) if point == point else None,
                    "ci95_low": round(lo, 6) if lo == lo else None,
                    "ci95_high": round(hi, 6) if hi == hi else None,
                    "n_conditions": len(conditions),
                    "bootstrap_samples": n_samples,
                }
            )
    return out_rows, indices


def main() -> int:
    args = parse_args()
    out_rows: list[dict[str, Any]] = []
    for item in args.policy_eval:
        device_label, path = parse_input(item)
        rows, _indices = device_bootstrap_rows(
            device_label,
            read_jsonl(path),
            n_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        out_rows.extend(rows)
    write_csv(args.out, out_rows)
    print(
        json.dumps(
            {
                "devices": sorted({r["device_label"] for r in out_rows}),
                "policies": len({r["policy"] for r in out_rows}),
                "rows": len(out_rows),
                "out": str(args.out),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
