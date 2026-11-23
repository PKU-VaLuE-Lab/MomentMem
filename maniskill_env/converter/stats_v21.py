"""Episode statistics compatible with LeRobot dataset v2.1."""

from __future__ import annotations

from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _stats(array: np.ndarray, *, axes=0, keepdims: bool = False) -> dict[str, Any]:
    values = np.asarray(array)
    finite = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(finite, axis=axes, keepdims=keepdims),
        "max": np.max(finite, axis=axes, keepdims=keepdims),
        "mean": np.mean(finite, axis=axes, keepdims=keepdims),
        "std": np.std(finite, axis=axes, keepdims=keepdims),
        "count": np.asarray([values.shape[0]], dtype=np.int64),
    }


def _video_stats(dataset, n_frames: int) -> dict[str, Any]:
    sample_count = min(100, int(n_frames))
    indices = np.linspace(0, n_frames - 1, sample_count, dtype=np.int64)
    nhwc = np.stack([np.asarray(dataset[int(index)]) for index in indices])
    nchw = np.transpose(nhwc.astype(np.float64) / 255.0, (0, 3, 1, 2))
    result = _stats(nchw, axes=(0, 2, 3), keepdims=True)
    # LeRobot stores channel statistics as Cx1x1, not 1xCx1x1.
    for key in ("min", "max", "mean", "std"):
        result[key] = np.squeeze(result[key], axis=0)
    result["count"] = np.asarray([n_frames], dtype=np.int64)
    return result


def build_episode_stats(
    *,
    actions: np.ndarray,
    extra_actions: dict[str, np.ndarray],
    states: np.ndarray,
    extra_states: dict[str, np.ndarray],
    rgb_by_video_key: dict[str, Any],
    episode_index: int,
    frame_index: np.ndarray,
    timestamp: np.ndarray,
    index: np.ndarray,
    task_index: np.ndarray,
    policy_action_mask: np.ndarray,
) -> dict[str, Any]:
    n = len(actions)
    policy_action_mask = np.asarray(policy_action_mask, dtype=bool).reshape(-1)
    result: dict[str, Any] = {
        # Normalization for every action representation is based only on rows
        # that can actually be sampled for VLA training.
        "action": _stats(actions[policy_action_mask]),
        "observation.state": _stats(states),
    }
    for name, values in extra_actions.items():
        result[name] = _stats(values[policy_action_mask])
    for name, values in extra_states.items():
        result[name] = _stats(values)
    for key, dataset in rgb_by_video_key.items():
        result[key] = _video_stats(dataset, n)
    scalar_columns = {
        "episode_index": np.full(n, episode_index, dtype=np.float64),
        "frame_index": frame_index,
        "timestamp": timestamp,
        "index": index,
        "task_index": task_index,
    }
    for name, values in scalar_columns.items():
        result[name] = _stats(np.asarray(values).reshape(-1, 1), keepdims=False)
    result["policy_action_mask"] = _stats(
        policy_action_mask.astype(np.float32).reshape(-1, 1), keepdims=False
    )
    return _jsonable(result)
