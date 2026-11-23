"""Validation and conversion of raw, full-rate memory intervals."""

from __future__ import annotations

from typing import Any, Iterable


def get_mem_boundary_from_episode(episode: dict[str, Any]) -> Any:
    """Read the canonical top-level boundary, with ManiSkill ``info`` fallback."""

    if "mem_boundary" in episode:
        return episode["mem_boundary"]
    info = episode.get("info")
    if isinstance(info, dict):
        return info.get("mem_boundary")
    return None


def mem_boundary_to_pairs(
    boundary: Iterable[Any] | None, *, episode_length: int | None = None
) -> list[list[int]]:
    """Return inclusive ``[start, end]`` pairs from an arbitrarily long boundary."""

    if boundary is None:
        return []
    values = [int(value) for value in boundary if int(value) >= 0]
    pairs: list[list[int]] = []
    for start, end in zip(values[0::2], values[1::2]):
        pairs.append([start, end])
    return pairs


def keyframe_record(
    episode: dict[str, Any], *, episode_index: int, episode_length: int
) -> dict[str, Any]:
    """Build one LeRobot keyframe row, preserving optional rich v3 metadata."""

    pairs = mem_boundary_to_pairs(
        get_mem_boundary_from_episode(episode), episode_length=episode_length
    )
    row: dict[str, Any] = {
        "episode_index": int(episode_index),
        "mem_boundary_pairs": pairs,
    }
    for name in ("keyframe_metadata", "phase_intervals"):
        value = episode.get(name)
        if value is None and isinstance(episode.get("info"), dict):
            value = episode["info"].get(name)
        if value is not None:
            row[name] = value
    return row
