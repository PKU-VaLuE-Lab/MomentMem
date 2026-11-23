#!/usr/bin/env python3
"""Merge disjoint eval_keyframe_ar raw-prediction shards and recompute exact metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))
MEMORIZER = os.path.join(REPO, "memorizer")
UTILS = os.path.join(REPO, "utils")
if MEMORIZER not in sys.path:
    sys.path.insert(0, MEMORIZER)
if UTILS not in sys.path:
    sys.path.insert(0, UTILS)

from keyframe_metrics import compute_keyframe_metrics  # noqa: E402
from sft.eval_shard_contract import merge_raw_shards, path_id  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True, help="Metrics output; must end in .json.")
    ap.add_argument("--expected-episodes", type=int, default=None)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--eval-spec", default=None)
    args = ap.parse_args()
    if not args.output.lower().endswith(".json"):
        ap.error("--output must be a .json filename")

    shards = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        shards.append((path, raw))

    merged = merge_raw_shards(shards)
    predictions = merged["predictions"]
    windows = merged["windows"]
    task_by_episode = merged["task_by_episode"]
    merge_identity = merged["merge_identity"]
    model_path = merge_identity.get("model_name_or_path")
    eval_spec = merge_identity.get("dataset_spec") or merge_identity.get("dataset_root")
    if args.model_path is not None and path_id(args.model_path) != model_path:
        raise ValueError(
            f"--model-path {args.model_path!r} does not match shard merge_identity {model_path!r}"
        )
    if args.eval_spec is not None and args.eval_spec != eval_spec:
        raise ValueError(
            f"--eval-spec {args.eval_spec!r} does not match shard merge_identity {eval_spec!r}"
        )

    if args.expected_episodes is not None and len(windows) != args.expected_episodes:
        raise ValueError(f"expected {args.expected_episodes} episodes, got {len(windows)}")
    parsed_windows = {
        episode: [tuple(pair) for pair in episode_windows]
        for episode, episode_windows in windows.items()
    }
    episodes_by_task: dict[str, list[str]] = {}
    for episode, task_name in task_by_episode.items():
        episodes_by_task.setdefault(task_name, []).append(episode)

    by_threshold = {}
    by_task_threshold: dict[str, dict[str, dict]] = {
        task_name: {} for task_name in sorted(episodes_by_task)
    }
    for threshold, by_episode in sorted(predictions.items(), key=lambda item: float(item[0])):
        parsed = {
            episode: {int(frame): int(value) for frame, value in frame_predictions.items()}
            for episode, frame_predictions in by_episode.items()
        }
        by_threshold[threshold] = compute_keyframe_metrics(parsed, parsed_windows)
        for task_name, episodes in sorted(episodes_by_task.items()):
            task_preds = {episode: parsed[episode] for episode in episodes}
            task_windows = {episode: parsed_windows[episode] for episode in episodes}
            by_task_threshold[task_name][threshold] = compute_keyframe_metrics(
                task_preds, task_windows
            )

    best_segment_threshold = max(
        by_threshold, key=lambda threshold: by_threshold[threshold]["overall_score"]
    )
    overall = {
        "trajectory_count": len(windows),
        "best_segment_threshold": best_segment_threshold,
        "best_segment": by_threshold[best_segment_threshold],
        "by_threshold": by_threshold,
    }
    by_task = {}
    for task_name, task_metrics in by_task_threshold.items():
        task_best = max(
            task_metrics, key=lambda threshold: task_metrics[threshold]["overall_score"]
        )
        by_task[task_name] = {
            "trajectory_count": len(episodes_by_task[task_name]),
            "best_segment_threshold": task_best,
            "best_segment": task_metrics[task_best],
            "by_threshold": task_metrics,
        }
    trajectory_counts = {
        "overall": len(windows),
        "by_task": {
            task_name: len(episodes) for task_name, episodes in sorted(episodes_by_task.items())
        },
    }
    result = {
        "protocol": "autoregressive_full100_sharded",
        "model_path": model_path,
        "eval_spec": args.eval_spec or eval_spec,
        "merge_identity": merge_identity,
        "n_episodes": len(windows),
        "n_shards": merged["shard_count"],
        "overall": overall,
        "by_task": by_task,
        "trajectory_counts": trajectory_counts,
        "by_threshold": by_threshold,
        "raw_prediction_shards": args.inputs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
