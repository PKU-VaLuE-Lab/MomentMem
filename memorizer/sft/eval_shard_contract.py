"""Dependency-free provenance and completeness checks for AR evaluation shards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import hashlib
import os
from pathlib import Path


EVALUATION_PROTOCOL = "autoregressive"
LABEL_GENERATION_PROTOCOL = "autoregressive_label_generation"


def path_id(value: str | Path | None) -> str | None:
    """Identify a local input without writing its path into exported metadata."""
    if value is None:
        return None
    normalized = str(Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve())
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def episode_id(leaf: str | Path, episode: int) -> str:
    return f"{path_id(leaf)}|{int(episode)}"


def build_merge_identity(args: Any, processor_path: str, thresholds: Sequence[float]) -> dict:
    """Return every evaluation setting that must agree before shards can merge."""
    return {
        "model_name_or_path": path_id(str(args.model_name_or_path)),
        "processor_name_or_path": path_id(str(processor_path)),
        "dataset_root": path_id(args.dataset_root),
        "dataset_spec": path_id(args.dataset_spec),
        "train_task_spec": path_id(args.train_task_spec),
        "split": str(args.split),
        "split_ratio": float(args.split_ratio),
        "split_seed": int(args.split_seed),
        "seen_task_eval_episodes": int(args.seen_task_eval_episodes),
        "max_eval_episodes": args.max_eval_episodes,
        "video_fps": int(args.video_fps),
        "sample_fps": int(args.sample_fps),
        "buffer_size": int(args.buffer_size),
        "max_history": int(args.max_history),
        "camera_mode": str(args.camera_mode),
        "image_size": None if args.image_size is None else int(args.image_size),
        "max_length": int(args.max_length),
        "add_memory_instruction": bool(args.add_memory_instruction),
        "thresholds": [f"{float(value):.2f}" for value in thresholds],
    }


def build_label_merge_identity(args: Any, processor_path: str) -> dict:
    """Return the deployment settings that must match across label shards."""

    return {
        "model_name_or_path": path_id(str(args.model_name_or_path)),
        "processor_name_or_path": path_id(str(processor_path)),
        "dataset_root": path_id(str(args.dataset_root)),
        "dataset_spec": None,
        "train_task_spec": None,
        "split": "all",
        "max_eval_episodes": None,
        "video_fps": int(args.video_fps),
        "sample_fps": int(args.sample_fps),
        "buffer_size": int(args.buffer_size),
        "max_history": int(args.max_history),
        "camera_mode": str(args.camera_mode),
        "image_size": None if args.image_size is None else int(args.image_size),
        "max_length": int(args.max_length),
        "add_memory_instruction": bool(args.add_memory_instruction),
        "thresholds": [f"{float(args.threshold):.2f}"],
    }


def _mapping(raw: Mapping, field: str, source: str) -> Mapping:
    value = raw.get(field)
    return value


def _shard_integer(raw: Mapping, field: str, source: str) -> int:
    value = raw.get(field)
    return value


def build_label_raw_output(
    *,
    merge_identity: Mapping,
    episode_shard_index: int,
    episode_shard_count: int,
    threshold: float,
    predictions: Mapping,
    task_by_episode: Mapping,
) -> dict:
    """Build the prediction-only raw format without privileged annotations."""

    threshold_key = f"{float(threshold):.2f}"
    return {
        "protocol": LABEL_GENERATION_PROTOCOL,
        "merge_identity": dict(merge_identity),
        "episode_shard_index": int(episode_shard_index),
        "episode_shard_count": int(episode_shard_count),
        "preds_by_ep_by_threshold": {threshold_key: dict(predictions)},
        "task_by_ep": dict(task_by_episode),
    }


def merge_raw_shards(shards: Sequence[tuple[str, Mapping]]) -> dict:
    """Validate a complete shard set, then return its merged raw dictionaries."""
    if not shards:
        raise ValueError("no raw prediction shards were provided")

    records = []
    expected_count = None
    expected_identity = None
    expected_thresholds = None
    expected_protocol = None
    indices: set[int] = set()
    all_episodes: set[str] = set()

    for source, raw in shards:
        protocol = raw.get("protocol")

        identity = _mapping(raw, "merge_identity", source)
        count = _shard_integer(raw, "episode_shard_count", source)
        index = _shard_integer(raw, "episode_shard_index", source)

        predictions = _mapping(raw, "preds_by_ep_by_threshold", source)
        task_by_episode = _mapping(raw, "task_by_ep", source)
        thresholds = set(predictions)
        if protocol == LABEL_GENERATION_PROTOCOL:
            windows = {}
            episode_keys = set(task_by_episode)
        else:
            windows = _mapping(raw, "windows_by_ep", source)
            episode_keys = set(windows)

        if expected_count is None:
            expected_count = count
            expected_identity = dict(identity)
            expected_thresholds = thresholds
            expected_protocol = protocol
        else:
            if count != expected_count:
                raise ValueError(
                    f"{source}: shard count {count} differs from expected {expected_count}"
                )
            if dict(identity) != expected_identity:
                raise ValueError(f"{source}: merge_identity differs from the other shards")
            if protocol != expected_protocol:
                raise ValueError(f"{source}: shard protocol differs from the other shards")
            if thresholds != expected_thresholds:
                raise ValueError(f"{source}: threshold set differs from the other shards")
        overlap = all_episodes.intersection(episode_keys)
        if overlap:
            raise ValueError(f"{source}: episode overlap across shards: {sorted(overlap)[:3]}")

        indices.add(index)
        all_episodes.update(episode_keys)
        records.append((predictions, windows, task_by_episode))

    missing = set(range(expected_count)) - indices
    if missing:
        raise ValueError(f"missing shard indices: {sorted(missing)}")

    merged_predictions = {threshold: {} for threshold in expected_thresholds}
    merged_windows = {}
    merged_tasks = {}
    for predictions, windows, task_by_episode in records:
        merged_windows.update(windows)
        merged_tasks.update(task_by_episode)
        for threshold, by_episode in predictions.items():
            merged_predictions[threshold].update(by_episode)

    return {
        "protocol": expected_protocol,
        "predictions": merged_predictions,
        "windows": merged_windows,
        "task_by_episode": merged_tasks,
        "merge_identity": expected_identity,
        "shard_count": expected_count,
    }
