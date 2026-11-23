#!/usr/bin/env python3
"""Export autoregressive VLM predictions as VLA keyframe labels."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


MEMORIZER = Path(__file__).resolve().parents[2]
REPO_ROOT = MEMORIZER.parent
UTILS = REPO_ROOT / "utils"
for path in (MEMORIZER, UTILS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_spec import resolve_dataset_root  # noqa: E402
from sft.eval_shard_contract import merge_raw_shards, path_id  # noqa: E402


EXCLUDED_METADATA = {
    "keyframes.json",
    "segments.json",
    "vlm_keyframe_generation.json",
}


def load_raw_shards(paths: Sequence[Path]) -> dict:
    shards = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            shards.append((str(path), json.load(handle)))
    return merge_raw_shards(shards)


def load_episode_lengths(leaf: Path) -> dict[int, int]:
    lengths = {}
    with (leaf / "meta" / "episodes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                lengths[int(record["episode_index"])] = int(record["length"])
    return lengths


def _parse_episode_key(raw_key: str) -> tuple[str, int]:
    leaf, separator, episode = raw_key.rpartition("|")
    return leaf, int(episode)


def build_export_plan(
    dataset_root: Path,
    merged: Mapping[str, Any],
    threshold: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    identity = merged["merge_identity"]
    if identity.get("dataset_root") != path_id(dataset_root):
        raise ValueError("raw prediction dataset identity does not match --dataset-root")
    video_fps = int(identity["video_fps"])
    sample_fps = int(identity["sample_fps"])
    stride = video_fps // sample_fps
    threshold_key = f"{float(threshold):.2f}"
    if threshold_key not in merged["predictions"]:
        available = sorted(merged["predictions"], key=float)
        raise ValueError(
            f"threshold {threshold_key} is unavailable; choose one of {available}"
        )
    predictions = merged["predictions"][threshold_key]
    predictions_by_episode = {
        _parse_episode_key(raw_key): values for raw_key, values in predictions.items()
    }

    resolved_leaves = list(resolve_dataset_root(dataset_root))
    lengths_by_leaf = {
        path_id(resolved.leaf_dir): load_episode_lengths(Path(resolved.leaf_dir))
        for resolved in resolved_leaves
    }
    expected_episodes = {
        (leaf, episode_index)
        for leaf, lengths in lengths_by_leaf.items()
        for episode_index in lengths
    }
    actual_episodes = set(predictions_by_episode)
    if actual_episodes != expected_episodes:
        missing = sorted(str(key) for key in expected_episodes - actual_episodes)
        extra = sorted(str(key) for key in actual_episodes - expected_episodes)
        raise ValueError(
            f"episode coverage mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )

    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    task_summary: dict[str, dict[str, int]] = {}
    for resolved in resolved_leaves:
        leaf = path_id(resolved.leaf_dir)
        rows = []
        positive_count = 0
        for episode_index in sorted(lengths_by_leaf[leaf]):
            episode_key = (leaf, episode_index)
            frame_predictions = predictions_by_episode[episode_key]
            positive_frames = sorted(
                int(frame_sub) * stride
                for frame_sub, decision in frame_predictions.items()
                if int(decision) == 1
            )
            positive_count += len(positive_frames)
            rows.append(
                {
                    "episode_index": episode_index,
                    "mem_boundary_pairs": [[frame, frame] for frame in positive_frames],
                }
            )
        rows_by_task[resolved.task_name] = rows
        task_summary[resolved.task_name] = {
            "episodes": len(rows),
            "positive_frames": positive_count,
        }

    provenance = {
        "schema_version": 1,
        "threshold": float(threshold_key),
        "video_fps": video_fps,
        "sample_fps": sample_fps,
        "stride": stride,
        "shard_count": int(merged["shard_count"]),
        "merge_identity": identity,
        "tasks": task_summary,
    }
    return provenance, rows_by_task


def relative_symlink(source: Path, link: Path) -> None:
    target = os.path.relpath(source, start=link.parent)
    link.symlink_to(target, target_is_directory=source.is_dir())


def _materialize(
    dataset_root: Path,
    output_root: Path,
    provenance: Mapping[str, Any],
    rows_by_task: Mapping[str, list[dict[str, Any]]],
) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        output_lerobot = temporary / "lerobot_format"
        output_lerobot.mkdir()
        leaves = {
            entry.task_name: Path(entry.leaf_dir)
            for entry in resolve_dataset_root(dataset_root)
        }
        for task_name, rows in rows_by_task.items():
            source_leaf = leaves[task_name]
            output_leaf = output_lerobot / task_name
            output_meta = output_leaf / "meta"
            output_meta.mkdir(parents=True)
            for source_entry in source_leaf.iterdir():
                if source_entry.name != "meta":
                    relative_symlink(source_entry, output_leaf / source_entry.name)
            for source_entry in (source_leaf / "meta").iterdir():
                if source_entry.name not in EXCLUDED_METADATA:
                    relative_symlink(source_entry, output_meta / source_entry.name)
            (output_meta / "keyframes.json").write_text(
                json.dumps({"episodes": rows}, indent=2, ensure_ascii=False) + "\n"
            )
            (output_meta / "vlm_keyframe_generation.json").write_text(
                json.dumps(provenance, indent=2, ensure_ascii=False) + "\n"
            )
        temporary.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def export_dataset(
    *,
    dataset_root: Path,
    raw_prediction_paths: Sequence[Path],
    output_root: Path,
    threshold: float,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).expanduser().absolute()
    output_root = output_root.parent.resolve() / output_root.name
    if os.path.lexists(output_root):
        raise FileExistsError(f"output root already exists: {output_root}")
    raw_paths = [Path(path).resolve() for path in raw_prediction_paths]
    merged = load_raw_shards(raw_paths)
    provenance, rows_by_task = build_export_plan(dataset_root, merged, threshold)
    _materialize(dataset_root, output_root, provenance, rows_by_task)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--raw-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_dataset(
        dataset_root=args.dataset_root,
        raw_prediction_paths=args.raw_predictions,
        output_root=args.output_root,
        threshold=args.threshold,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
