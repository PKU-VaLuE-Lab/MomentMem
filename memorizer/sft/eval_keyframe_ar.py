#!/usr/bin/env python
"""Evaluate a keyframe VLM with the deployment-style autoregressive protocol.

Usage:
  python sft/eval_keyframe_ar.py \
      --model-name-or-path {memorizer_path} \
      --dataset-root {data_path} --batch-size 8 --thresholds 0.5
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import torch

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_CUR_DIR)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
_REPO_ROOT = os.path.dirname(_PROJ_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_UTILS_ROOT = os.path.join(_REPO_ROOT, "utils")
if _UTILS_ROOT not in sys.path:
    sys.path.insert(0, _UTILS_ROOT)

# Single source of truth for the 6 keyframe metrics, used in memorizer evaluation.
from keyframe_metrics import compute_keyframe_metrics  # noqa: E402

from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

from dataset import MomentMemDataset  # noqa: E402
from sft.eval_shard_contract import build_merge_identity  # noqa: E402
from sft.eval_shard_contract import episode_id, path_id
from sft.eval_utils import build_infer_inputs, parse_args, discover_eval_roots  # noqa: E402
from sft.vision_cache import VisionCache, build_cached_item  # noqa: E402


@torch.no_grad()
def _p_yes_batch(model, processor, items: List[Dict[str, Any]], yes_id: int, no_id: int, max_length: int) -> List[float]:
    """Constrained P(Yes) for a batch of already-built autoregressive items."""
    batch = build_infer_inputs(processor, items, max_length=max_length).to("cuda")
    out = model(**batch)
    logits = out.logits  # (B, T, V)
    last_idx = batch["attention_mask"].sum(dim=1) - 1
    ans = logits[torch.arange(logits.size(0)), last_idx]  # (B, V)
    ly = ans[:, yes_id].float()
    ln = ans[:, no_id].float()
    p_yes = torch.softmax(torch.stack([ln, ly], dim=1), dim=1)[:, 1]
    return p_yes.tolist()


def _build_item(ds: MomentMemDataset, ctx, t_sub: int, selected_sub: List[int], buffer_size: int, max_history: int | None = None) -> Dict[str, Any]:
    """Build one inference item at frame t_sub with history = the MODEL's own past picks."""
    buffer_sub = list(range(max(0, t_sub - buffer_size), t_sub)) if buffer_size > 0 else []
    buffer_set = set(buffer_sub)
    selected_set = set(selected_sub)
    history_sub = [k for k in selected_sub if k < t_sub and k not in buffer_set]
    if max_history is not None and len(history_sub) > max_history:
        history_sub = history_sub[-max_history:] if max_history > 0 else []
    current_views = ds._load_views_by_sub_idx(ctx.video_paths, t_sub)
    history_views = [ds._load_views_by_sub_idx(ctx.video_paths, k) for k in history_sub]
    buffer_views = [ds._load_views_by_sub_idx(ctx.video_paths, k) for k in buffer_sub]
    return {
        "instruction": ctx.instruction,
        "history_images": history_views,
        "buffer_images": buffer_views,
        "buffer_is_keyframe": [k in selected_set for k in buffer_sub],
        "buffer_frame_indices_sub": buffer_sub,
        "current_image": current_views,
        "num_cameras": len(ds._camera_keys),
    }


def _rollout_threshold(model, processor, contexts: List[Tuple[Any, MomentMemDataset, Any]], thr: float, args, yes_id: int, no_id: int) -> Dict[Any, Dict[int, int]]:
    """One free-running rollout over ALL episodes at a fixed decision threshold."""
    selected: Dict[Any, List[int]] = {key: [] for key, _, _ in contexts}
    preds: Dict[Any, Dict[int, int]] = {key: {} for key, _, _ in contexts}
    max_len = max(ctx.sub_len for _, _, ctx in contexts)
    for t_sub in range(max_len):
        active = [(key, ds, ctx) for key, ds, ctx in contexts if t_sub < ctx.sub_len]
        if not active:
            break
        items = [_build_item(ds, ctx, t_sub, selected[key], args.buffer_size,
                             max_history=getattr(args, "max_history", None))
                 for key, ds, ctx in active]
        probs: List[float] = []
        for i in range(0, len(items), args.batch_size):
            probs.extend(_p_yes_batch(model, processor, items[i:i + args.batch_size], yes_id, no_id, args.max_length))
        for (key, ds, ctx), p in zip(active, probs):
            d = 1 if p > thr else 0  # strict `>`, matching the online memorizer
            preds[key][t_sub] = d
            if d:
                selected[key].append(t_sub)
    return preds


def _p_yes_batch_cached(model, vc: VisionCache, cached_items: List[Dict[str, Any]], yes_id: int, no_id: int, max_length: int) -> List[float]:
    """P(Yes) using the vision cache. build_batch is byte-identical to build_infer_inputs, so this
    returns the SAME logits as _p_yes_batch — only the redundant image preprocessing is skipped."""
    batch = {k: v.to("cuda") for k, v in vc.build_batch(cached_items, max_length=max_length).items()}
    out = model(**batch)
    logits = out.logits
    last_idx = batch["attention_mask"].sum(dim=1) - 1
    ans = logits[torch.arange(logits.size(0)), last_idx]
    ly = ans[:, yes_id].float()
    ln = ans[:, no_id].float()
    return torch.softmax(torch.stack([ln, ly], dim=1), dim=1)[:, 1].tolist()


def _rollout_threshold_cached(model, vc: VisionCache, contexts: List[Tuple[Any, MomentMemDataset, Any]], thr: float, args, yes_id: int, no_id: int) -> Dict[Any, Dict[int, int]]:
    """Same free-running rollout as _rollout_threshold, but per-frame pixel_values are cached
    (VisionCache) instead of re-preprocessed every timestep. Results are identical."""
    selected: Dict[Any, List[int]] = {key: [] for key, _, _ in contexts}
    preds: Dict[Any, Dict[int, int]] = {key: {} for key, _, _ in contexts}
    max_len = max(ctx.sub_len for _, _, ctx in contexts)
    for t_sub in range(max_len):
        active = [(key, ds, ctx) for key, ds, ctx in contexts if t_sub < ctx.sub_len]
        if not active:
            break
        items = [build_cached_item(ctx, ds, t_sub, selected[key], args.buffer_size, vc,
                                   max_history=getattr(args, "max_history", None))
                 for key, ds, ctx in active]
        probs: List[float] = []
        for i in range(0, len(items), args.batch_size):
            probs.extend(_p_yes_batch_cached(model, vc, items[i:i + args.batch_size], yes_id, no_id, args.max_length))
        for (key, ds, ctx), p in zip(active, probs):
            d = 1 if p > thr else 0
            preds[key][t_sub] = d
            if d:
                selected[key].append(t_sub)
    return preds


@torch.no_grad()
def evaluate_ar(args) -> Dict[str, Any]:
    processor_path = args.processor_name_or_path or args.model_name_or_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.eval().cuda()

    # Gather every episode across all leaf datasets; invalid episodes are dropped by MomentMemDataset.
    contexts: List[Tuple[Any, MomentMemDataset, Any]] = []
    windows_by_ep: Dict[Any, List[Tuple[int, int]]] = {}
    task_by_leaf: Dict[str, str] = {}
    skipped_leaves: List[str] = []

    trained_tasks = None
    if args.train_task_spec:
        from dataset_spec import resolve_spec

        trained_tasks = {entry.task_name for entry in resolve_spec(args.train_task_spec)}
        print(
            f"[eval-ar] mixed seen/unseen policy: seen={sorted(trained_tasks)} -> "
            f"val(max={args.seen_task_eval_episodes}); unseen -> all episodes"
        )

    for root, task_name in discover_eval_roots(args):
        # Use full leaf paths while merging entries with the same task name.
        leaf = root
        task_by_leaf[leaf] = task_name
        task_is_seen = trained_tasks is None or task_name in trained_tasks
        if trained_tasks is None:
            task_mode = args.split
            task_max_eval_episodes = args.max_eval_episodes
        elif task_is_seen:
            task_mode = "val"
            task_max_eval_episodes = args.seen_task_eval_episodes
        else:
            task_mode = "all"
            task_max_eval_episodes = None
        try:
            ds = MomentMemDataset(dataset_root=root, mode=task_mode, split_ratio=args.split_ratio,
                            split_seed=args.split_seed, video_fps=args.video_fps,
                            sample_fps=args.sample_fps, transform=None, include_t0_in_index=True,
                            image_size=args.image_size,
                            max_eval_episodes=task_max_eval_episodes,
                            buffer_size=args.buffer_size,
                            max_history=args.max_history,
                            add_memory_instruction=args.add_memory_instruction,
                            camera_mode=args.camera_mode,
                            build_supervision=False)
        except Exception as e:  # noqa: BLE001
            if not args.skip_bad_leaves:
                raise RuntimeError(
                    f"failed to load leaf {leaf} ({e}); pass --skip-bad-leaves to skip it instead."
                ) from e
            skipped_leaves.append(leaf)
            print(f"[eval-ar] skip {leaf}: {e}")
            continue
        for ctx in ds.episodes:
            key = (leaf, ctx.episode_index)
            contexts.append((key, ds, ctx))
            windows_by_ep[key] = list(ctx.segments_sub)
        seen_label = "seen" if task_is_seen else "unseen"
        print(
            f"[eval-ar] {leaf}: {len(ds.episodes)} episodes queued "
            f"(task={task_name}, {seen_label}, split={task_mode})"
        )

    if skipped_leaves:
        print(f"[eval-ar] WARNING: skipped {len(skipped_leaves)} leaf dataset(s): {skipped_leaves}")
    if not contexts:
        raise RuntimeError(
            f"no episodes were found under {args.dataset_spec or args.dataset_root} "
            f"(skipped leaves: {skipped_leaves or 'none'}). Nothing to score."
        )

    # Deterministic, disjoint episode sharding for multi-GPU evaluation.  Each process loads one
    # model replica and evaluates one ordinal-modulo shard; the raw predictions are merged after
    # all shards finish.  Dataset discovery and episode ordering are deterministic for a fixed
    # dataset spec and split seed.
    shard_count = int(getattr(args, "episode_shard_count", 1))
    shard_index = int(getattr(args, "episode_shard_index", 0))
    if shard_count > 1:
        all_contexts = contexts
        contexts = [item for ordinal, item in enumerate(all_contexts)
                    if ordinal % shard_count == shard_index]
        selected_keys = {key for key, _, _ in contexts}
        windows_by_ep = {key: windows_by_ep[key] for key in selected_keys}
        print(f"[eval-ar] episode shard {shard_index}/{shard_count}: "
              f"{len(contexts)}/{len(all_contexts)} episodes")

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    if len(thresholds) > 1:
        print(f"[eval-ar] NOTE: {len(thresholds)} thresholds -> {len(thresholds)} separate "
              f"free-running rollouts (the memory diverges per threshold, so it cannot be swept "
              f"post-hoc). This multiplies the VLM cost by {len(thresholds)}x.")
    merge_identity = build_merge_identity(args, processor_path, thresholds)

    # Roll out memory per threshold while sharing threshold-independent vision features.
    vc = VisionCache(processor, cap=args.vision_cache_cap) if args.cache_vision else None
    if vc is not None:
        print(f"[eval-ar] vision cache ON (cap={args.vision_cache_cap}): per-frame pixel_values "
              f"cached across steps & thresholds")
    preds_by_thr: Dict[float, Dict[Any, Dict[int, int]]] = {}
    for thr in thresholds:
        if vc is not None:
            preds_by_thr[thr] = _rollout_threshold_cached(model, vc, contexts, thr, args, yes_id, no_id)
            print(f"[eval-ar] rollout @ thr={thr:.2f} done (cache hits={vc.hits} misses={vc.misses})")
        else:
            preds_by_thr[thr] = _rollout_threshold(model, processor, contexts, thr, args, yes_id, no_id)
            print(f"[eval-ar] rollout @ thr={thr:.2f} done")

    def score_subset(keys: List[Any]) -> Dict[str, Any]:
        """Score precomputed per-threshold autoregressive rollouts."""
        wins_sub = {k: windows_by_ep.get(k, []) for k in keys}
        by_thr: Dict[str, Dict[str, Any]] = {}
        best_seg = None
        for thr in thresholds:
            preds = {k: preds_by_thr[thr][k] for k in keys}
            m = compute_keyframe_metrics(preds, wins_sub)
            by_thr[f"{thr:.2f}"] = m
            if best_seg is None or m["overall_score"] > best_seg[2]:
                best_seg = (thr, m, m["overall_score"])
        return {"best_segment_threshold": best_seg[0], "best_segment": best_seg[1],
                "by_threshold": by_thr,
                "trajectory_count": len(keys)}

    tasks: Dict[str, List[Any]] = {}
    for key, _, _ in contexts:
        tasks.setdefault(task_by_leaf.get(key[0], key[0]), []).append(key)

    all_keys = [key for key, _, _ in contexts]
    overall = score_subset(all_keys)
    by_task = {t: score_subset(ks) for t, ks in sorted(tasks.items())}
    trajectory_counts = {
        "overall": len(all_keys),
        "by_task": {task_name: len(keys) for task_name, keys in sorted(tasks.items())},
    }

    def _seg_line(name, r):
        m = r["best_segment"]
        segment_overall = m["segment"]["overall"]
        print(f"{name:28s} | thr {r['best_segment_threshold']:.2f} "
              f"| precision {m['precision']:.4f} recall {m['recall']:.4f} f1 {m['f1']:.4f} "
              f"| segment_hit {segment_overall['hit']:.4f} "
              f"segment_acc {segment_overall['acc']:.4f} non_key_acc {m['non_key_acc']:.4f} "
              f"overall_score {m['overall_score']:.4f} "
              f"| n_win {m['n_windows']} totYes {m['total_yes_pred']} "
              f"segments {m['segment']}")
    print("\n=== AUTOREGRESSIVE keyframe-selection metrics (best-overall_score threshold): "
          "precision/recall/f1 + segment_acc/non_key_acc/overall_score ===")
    for t in sorted(by_task):
        _seg_line(t, by_task[t])
    _seg_line("OVERALL", overall)

    out = {"protocol": "autoregressive", "camera_mode": args.camera_mode,
           "merge_identity": merge_identity,
           "evaluation_policy": {
               "train_task_spec": path_id(args.train_task_spec),
               "trained_tasks": sorted(trained_tasks) if trained_tasks is not None else None,
               "seen_tasks": "validation_split" if trained_tasks is not None else args.split,
               "seen_task_eval_episodes": (
                   args.seen_task_eval_episodes if trained_tasks is not None else args.max_eval_episodes
               ),
               "unseen_tasks": "all_episodes" if trained_tasks is not None else args.split,
           },
           "episode_shard_index": shard_index, "episode_shard_count": shard_count,
           "overall": overall, "by_task": by_task,
           "trajectory_counts": trajectory_counts,
           "by_threshold": overall["by_threshold"]}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[eval-ar] wrote {args.output}")
        raw = {"protocol": "autoregressive", "merge_identity": merge_identity,
               "episode_shard_index": shard_index, "episode_shard_count": shard_count,
               "preds_by_ep_by_threshold": {
                   f"{thr:.2f}": {episode_id(k[0], k[1]): v for k, v in preds_by_thr[thr].items()}
                   for thr in thresholds},
               "windows_by_ep": {episode_id(k[0], k[1]): v for k, v in windows_by_ep.items()},
               "task_by_ep": {
                   episode_id(k[0], k[1]): task_by_leaf.get(k[0], k[0])
                   for k in windows_by_ep
               }}
        output_path = Path(args.output)
        raw_path = output_path.with_name(f"{output_path.stem}_rawpreds.json")
        with open(raw_path, "w") as f:
            json.dump(raw, f)
        print(f"[eval-ar] wrote {raw_path}")
    return out


if __name__ == "__main__":
    evaluate_ar(parse_args())
