"""Deterministic condition sharding for safe multi-GPU / multi-machine reruns.

Kept torch-free so it can be unit-tested and reasoned about without importing the
runtime stack.  Sharding only partitions the condition list; it never introduces
concurrency on a single GPU, DDP, NCCL, or multiprocessing — the caller is
responsible for running at most one process per GPU.
"""

from __future__ import annotations

import hashlib

from shortjob_runner.types import Condition


def condition_identity(condition: Condition) -> str:
    """Stable identity string for a condition (order-, device-, process-independent)."""
    return "|".join(
        str(part)
        for part in (
            condition.workload_id,
            condition.action,
            condition.batch_size,
            condition.num_steps,
            condition.repeat,
        )
    )


def shard_of(condition: Condition, shard_count: int) -> int:
    """Deterministic, reproducible shard id for a condition.

    Hashes the stable identity with hashlib (not builtin hash(), which is
    PYTHONHASHSEED-randomized) so the same condition lands in the same shard on
    every machine.
    """
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    digest = hashlib.sha256(condition_identity(condition).encode()).hexdigest()
    return int(digest, 16) % shard_count


def select_shard(conditions: list[Condition], shard_count: int, shard_index: int) -> list[Condition]:
    """Return the subset of conditions assigned to ``shard_index``.

    Over all indices the shards are disjoint and their union is the full input,
    so merged shard outputs pass the normal completeness checker.
    """
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    if shard_count == 1:
        return conditions
    return [c for c in conditions if shard_of(c, shard_count) == shard_index]
