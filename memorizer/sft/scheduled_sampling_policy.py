"""Dependency-free policy helpers for sequence-level Scheduled Sampling."""

from __future__ import annotations

import hashlib
from typing import Sequence, Tuple


ContextKey = Tuple[str, int]


def visible_selected_history(
    selected: Sequence[int],
    *,
    timestep: int,
    buffer_size: int,
    max_history: int,
) -> list[int]:
    """Return selected timesteps represented in the bounded prompt at ``timestep``."""

    t = int(timestep)
    buffer_size = int(buffer_size)
    max_history = int(max_history)
    past = [int(value) for value in selected if int(value) < t]
    buffer_start = max(0, t - buffer_size)
    in_buffer = [value for value in past if buffer_start <= value < t]
    long_term = [value for value in past if value < buffer_start]
    if len(long_term) > max_history:
        long_term = long_term[-max_history:] if max_history > 0 else []
    return long_term + in_buffer


def memory_conditioned_supervision(
    timestep: int,
    *,
    segments: Sequence[tuple[int, int]],
    anchors: Sequence[int],
    selected: Sequence[int],
) -> tuple[int, float]:
    """Return the target/mask for the exact autoregressive memory shown at ``timestep``."""

    t = int(timestep)
    normalized = [(int(start), int(end)) for start, end in segments]
    anchor_values = [int(anchor) for anchor in anchors]
    selected_values = [int(value) for value in selected]

    active: tuple[int, int, int] | None = None
    for (start, end), anchor in zip(normalized, anchor_values):
        if start <= t <= end:
            active = (start, end, anchor)

    if active is None:
        return 0, 1.0

    start, _end, anchor = active
    covered = any(start <= value < t for value in selected_values)
    if covered:
        return 0, 1.0
    if t < anchor:
        return 0, 0.0
    return 1, 1.0


def scheduled_sampling_probability(
    progress: float, *, warmup_ratio: float, max_probability: float
) -> float:
    """Linear ramp from 0 after ``warmup_ratio`` to ``max_probability`` at progress 1."""
    progress = min(1.0, max(0.0, float(progress)))
    if max_probability == 0.0 or progress <= warmup_ratio:
        return 0.0
    ramp = (progress - warmup_ratio) / (1.0 - warmup_ratio)
    return min(max_probability, max_probability * ramp)


def use_model_feedback(
    probability: float, *, seed: int, epoch: int, key: ContextKey, timestep: int
) -> bool:
    """A deterministic Bernoulli draw, invariant to rank count and episode sharding."""
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    payload = f"{seed}|{epoch}|{key[0]}|{key[1]}|{timestep}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value / float(1 << 64) < probability
