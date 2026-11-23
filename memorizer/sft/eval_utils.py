"""Shared input and argument helpers for autoregressive keyframe evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT, _REPO_ROOT / "utils"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dataset_spec import resolve_dataset_root, resolve_spec  # noqa: E402
from prompts import (  # noqa: E402
    PROMPT_LAYOUT_VERSION,
    build_user_parts,
    camera_mode_from_config,
    image_size_from_config,
    render_user_content,
    system_prompt,
)
from sft.train_qwen3_vl_sft import _to_rgb_pil  # noqa: E402


def discover_eval_roots(args) -> List[Tuple[str, str]]:
    """Resolve ``(leaf_dir, task_name)`` pairs from a spec or dataset root."""
    if getattr(args, "dataset_spec", None):
        return [(r.leaf_dir, r.task_name) for r in resolve_spec(args.dataset_spec)]
    if args.dataset_root:
        return [(r.leaf_dir, r.task_name) for r in resolve_dataset_root(args.dataset_root)]
    raise ValueError("provide either --dataset-spec or --dataset-root")


def _apply_prompt_config(args: argparse.Namespace) -> None:
    """Apply the training-time sidecar so the eval prompt structure cannot drift from training."""
    path = args.prompt_config
    if not path:
        candidate = os.path.join(args.model_name_or_path, "keyframe_prompt_config.json")
        path = candidate if os.path.exists(candidate) else None
    if not path:
        return
    if not os.path.exists(path):
        raise SystemExit(f"prompt-config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    layout = cfg.get("prompt_layout")
    if layout is None:
        raise SystemExit(f"prompt-config {path}: missing required prompt_layout")
    if int(layout) != int(PROMPT_LAYOUT_VERSION):
        raise SystemExit(
            f"prompt-config {path}: prompt_layout={layout} but this code builds layout "
            f"{PROMPT_LAYOUT_VERSION}. Evaluate this checkpoint with the matching code revision."
        )

    def _override(key: str, attr: str, cast) -> None:
        if key not in cfg:
            return
        value = cast(cfg[key])
        if value != cast(getattr(args, attr)):
            print(f"[eval-ar] prompt-config {path}: {key}={value} "
                  f"(overrides CLI {getattr(args, attr)})")
        setattr(args, attr, value)

    _override("add_memory_instruction", "add_memory_instruction", bool)
    _override("buffer_size", "buffer_size", int)
    _override("max_history", "max_history", int)
    _override("video_fps", "video_fps", int)
    _override("sample_fps", "sample_fps", int)
    try:
        value = image_size_from_config(cfg)
    except ValueError as exc:
        raise SystemExit(f"prompt-config {path}: {exc}") from exc
    if value != args.image_size:
        print(
            f"[eval-ar] prompt-config {path}: image_size={value} "
            f"(overrides CLI {args.image_size})"
        )
    args.image_size = value
    _override("max_length", "max_length", int)
    try:
        camera_mode = camera_mode_from_config(cfg)
    except ValueError as exc:
        raise SystemExit(f"prompt-config {path}: {exc}") from exc
    if camera_mode is not None:
        if camera_mode != args.camera_mode:
            print(f"[eval-ar] prompt-config {path}: camera_mode={camera_mode} "
                  f"(overrides CLI {args.camera_mode})")
        args.camera_mode = camera_mode


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Autoregressive keyframe evaluation")
    ap.add_argument("--model-name-or-path", type=str, required=True)
    ap.add_argument("--processor-name-or-path", type=str, default=None,
                    help="Optional tokenizer/processor source when a model-only checkpoint omits it.")
    ap.add_argument("--dataset-root", type=str, default=None,
                    help="Evaluate every task under <dataset-root>/lerobot_format.")
    ap.add_argument("--dataset-spec", type=str, default=None,
                    help="JSON/YAML file (or inline JSON) mapping dataset roots to task names.")
    ap.add_argument(
        "--train-task-spec",
        type=str,
        default=None,
        help="Optional training dataset spec used for mixed seen/unseen evaluation. Tasks present "
        "in this spec use the deterministic validation split; tasks absent from it use all "
        "episodes. Requires --dataset-spec to describe the complete evaluation task set.",
    )
    ap.add_argument(
        "--seen-task-eval-episodes",
        type=int,
        default=20,
        help="Maximum validation-split trajectories per task listed in --train-task-spec.",
    )
    ap.add_argument("--skip-bad-leaves", action="store_true")
    ap.add_argument("--split", type=str, default="val", choices=["val", "train"])
    ap.add_argument("--split-ratio", type=float, default=0.8)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--video-fps", type=int, default=20)
    ap.add_argument("--sample-fps", type=int, default=5)
    ap.add_argument("--buffer-size", type=int, default=10,
                    help="Recent sampled timesteps before the current frame; must match deployment config.")
    ap.add_argument(
        "--camera-mode", choices=["overview", "overview+wrist"], default="overview+wrist",
        help="Camera layout used by the checkpoint; a prompt sidecar overrides this value.",
    )
    ap.add_argument("--max-history", type=int, default=48,
                    help="Keep only the most recent N long-term keyframes, matching deployment.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--max-eval-episodes", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=224,
                    help="Square decoded-frame size; must match training.")
    ap.add_argument(
        "--native-image-resolution",
        action="store_true",
        help="Keep native frame dimensions and use Qwen dynamic resolution when no checkpoint "
        "sidecar overrides the image contract.",
    )
    ap.add_argument("--add-memory-instruction", action="store_true",
                    help="Append task_memory from meta/tasks.jsonl; must match training.")
    ap.add_argument("--prompt-config", default=None,
                    help="Training keyframe_prompt_config.json; defaults to the model directory.")
    ap.add_argument("--thresholds", type=str, default="0.5",
                    help="Comma-separated thresholds; each causes a separate full rollout.")
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional metrics output; when provided, the filename must end in .json.",
    )
    ap.add_argument("--cache-vision", action="store_true")
    ap.add_argument("--vision-cache-cap", type=int, default=20000)
    ap.add_argument("--eval-all-episodes", action="store_true")
    ap.add_argument("--episode-shard-count", type=int, default=1,
                    help="Split the selected episodes into this many deterministic disjoint shards.")
    ap.add_argument("--episode-shard-index", type=int, default=0,
                    help="Zero-based shard to evaluate; episode ordinal modulo shard count is used.")
    args = ap.parse_args()
    if args.image_size <= 0:
        ap.error("--image-size must be positive")
    if args.native_image_resolution:
        args.image_size = None
    if args.eval_all_episodes:
        args.split = "all"
    _apply_prompt_config(args)
    if args.train_task_spec and not args.dataset_spec:
        raise ValueError("--train-task-spec requires --dataset-spec for the full evaluation set")
    if args.output is not None and not args.output.lower().endswith(".json"):
        ap.error("--output must be a .json filename")
    return args


def build_infer_inputs(processor, items: List[Dict[str, Any]], max_length: int = 16384):
    """Build the system+user multimodal batch ending at the assistant generation prompt."""
    texts: List[str] = []
    images_all: List[List[Any]] = []
    for ex in items:
        num_cameras = int(ex.get("num_cameras", len(ex["current_image"])))
        flat_history = [_to_rgb_pil(v) for views in ex["history_images"] for v in views]
        flat_buffer = [_to_rgb_pil(v) for views in ex.get("buffer_images", []) for v in views]
        current_views = [_to_rgb_pil(v) for v in ex["current_image"]]
        images = flat_history + flat_buffer + current_views
        history_count = len(ex["history_images"])
        buffer_flags = [bool(x) for x in ex.get("buffer_is_keyframe", [])]
        image_content = [{"type": "image"} for _ in range(
            (history_count + len(buffer_flags) + 1) * num_cameras
        )]
        parts = build_user_parts(
            ex["instruction"], history_count, num_cameras,
            buffer_keyframe_flags=buffer_flags,
            buffer_frame_indices=[int(x) for x in ex.get("buffer_frame_indices_sub", [])],
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt(num_cameras)}]},
            {"role": "user", "content": render_user_content(parts, image_content)},
        ]
        texts.append(processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))
        images_all.append(images)
    # NEVER truncate: the Yes/No answer is read from the last token's logits, so truncation
    # would silently score the wrong position. max_length is a hard guard instead.
    enc = processor(text=texts, images=images_all, return_tensors="pt", padding=True)
    seq_len = int(enc["input_ids"].shape[1])
    if seq_len > max_length:
        raise ValueError(
            f"VLM prompt is {seq_len} tokens > --max-length={max_length}. Truncating would "
            "corrupt the Yes/No answer position, so this is a hard error: raise --max-length, "
            "or lower --max-history / --buffer-size / --image-size so the prompt fits."
        )
    return enc
