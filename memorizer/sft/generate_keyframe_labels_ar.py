#!/usr/bin/env python3
"""Generate deployment-style VLM keyframe labels without ground truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


MEMORIZER = Path(__file__).resolve().parents[1]
REPO_ROOT = MEMORIZER.parent
for path in (MEMORIZER, REPO_ROOT / "utils"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--processor-name-or-path", default=None)
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="LeRobot dataset root; every episode in episodes.jsonl is labeled.",
    )
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--sample-fps", type=int, default=5)
    parser.add_argument("--buffer-size", type=int, default=10)
    parser.add_argument("--max-history", type=int, default=48)
    parser.add_argument(
        "--camera-mode", choices=["overview", "overview+wrist"], default="overview+wrist"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--native-image-resolution",
        action="store_true",
        help="Keep native frame dimensions and use Qwen dynamic resolution when no checkpoint "
        "sidecar overrides the image contract.",
    )
    parser.add_argument("--add-memory-instruction", action="store_true")
    parser.add_argument("--prompt-config", default=None)
    parser.add_argument("--cache-vision", action="store_true")
    parser.add_argument("--vision-cache-cap", type=int, default=20000)
    parser.add_argument("--episode-shard-count", type=int, default=1)
    parser.add_argument("--episode-shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.image_size <= 0:
        parser.error("--image-size must be positive")
    if args.native_image_resolution:
        args.image_size = None
    if not str(args.output).lower().endswith(".json"):
        parser.error("--output must be a .json filename")
    if not math.isfinite(args.threshold) or not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be finite and in [0, 1]")
    if args.max_history < 0:
        parser.error("--max-history must be >= 0")
    if args.episode_shard_count < 1:
        parser.error("--episode-shard-count must be >= 1")
    if not 0 <= args.episode_shard_index < args.episode_shard_count:
        parser.error("--episode-shard-index must be in [0, episode-shard-count)")
    return args


def generate_labels(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from dataset import MomentMemDataset
    from dataset_spec import resolve_dataset_root
    from sft.eval_keyframe_ar import (
        _rollout_threshold,
        _rollout_threshold_cached,
    )
    from sft.eval_shard_contract import (
        episode_id,
        build_label_merge_identity,
        build_label_raw_output,
    )
    from sft.vision_cache import VisionCache

    contexts = []
    task_by_key = {}
    for resolved in resolve_dataset_root(args.dataset_root):
        leaf = str(Path(resolved.leaf_dir).resolve())
        dataset = MomentMemDataset(
            dataset_root=leaf,
            mode="all",
            video_fps=args.video_fps,
            sample_fps=args.sample_fps,
            transform=None,
            include_t0_in_index=True,
            image_size=args.image_size,
            buffer_size=args.buffer_size,
            max_history=args.max_history,
            add_memory_instruction=args.add_memory_instruction,
            camera_mode=args.camera_mode,
            build_supervision=False,
            load_annotations=False,
        )
        for context in dataset.episodes:
            key = (leaf, context.episode_index)
            contexts.append((key, dataset, context))
            task_by_key[key] = resolved.task_name
        print(f"[label-ar] {leaf}: {len(dataset.episodes)} episodes queued")

    if not contexts:
        raise RuntimeError(f"no episodes found under {args.dataset_root}")
    shard_count = int(args.episode_shard_count)
    shard_index = int(args.episode_shard_index)
    all_contexts = contexts
    contexts = [
        item
        for ordinal, item in enumerate(all_contexts)
        if ordinal % shard_count == shard_index
    ]
    print(
        f"[label-ar] episode shard {shard_index}/{shard_count}: "
        f"{len(contexts)}/{len(all_contexts)} episodes"
    )

    processor_path = args.processor_name_or_path or args.model_name_or_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval().cuda()

    cache = VisionCache(processor, cap=args.vision_cache_cap) if args.cache_vision else None
    with torch.no_grad():
        if cache is None:
            predictions = _rollout_threshold(
                model, processor, contexts, args.threshold, args, yes_id, no_id
            )
        else:
            predictions = _rollout_threshold_cached(
                model, cache, contexts, args.threshold, args, yes_id, no_id
            )

    raw = build_label_raw_output(
        merge_identity=build_label_merge_identity(args, processor_path),
        episode_shard_index=shard_index,
        episode_shard_count=shard_count,
        threshold=args.threshold,
        predictions={
            episode_id(key[0], key[1]): frames for key, frames in predictions.items()
        },
        task_by_episode={
            episode_id(key[0], key[1]): task_by_key[key] for key, _, _ in contexts
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(raw, ensure_ascii=False) + "\n")
    print(f"[label-ar] wrote {args.output} ({len(contexts)} episodes)")
    return raw


def main() -> None:
    args = parse_args()
    args.dataset_root = str(Path(args.dataset_root).expanduser().resolve())
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    from sft.eval_utils import _apply_prompt_config

    _apply_prompt_config(args)
    generate_labels(args)


if __name__ == "__main__":
    main()
