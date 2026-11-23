"""In-training autoregressive validation for the Qwen keyframe selector."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import TrainerCallback

_UTILS_ROOT = Path(__file__).resolve().parents[2] / "utils"
if str(_UTILS_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(_UTILS_ROOT))

from dataset_spec import task_name_from_leaf  # noqa: E402
from keyframe_metrics import compute_keyframe_metrics  # noqa: E402

from dataset import MomentMemDataset  # noqa: E402
from sft.vision_cache import VisionCache, build_cached_item  # noqa: E402


def _compute_metrics_by_task(
    preds: Dict[Any, Dict[int, int]],
    windows: Dict[Any, List[Tuple[int, int]]],
) -> Dict[str, Dict[str, Any]]:
    """Score task subsets from an already-computed autoregressive rollout."""
    keys_by_task: Dict[str, List[Any]] = {}
    for key in preds:
        root = str(key[0])
        keys_by_task.setdefault(task_name_from_leaf(root), []).append(key)

    by_task: Dict[str, Dict[str, Any]] = {}
    for task_name, keys in sorted(keys_by_task.items()):
        task_preds = {key: preds[key] for key in keys}
        task_windows = {key: windows[key] for key in keys}
        by_task[task_name] = compute_keyframe_metrics(task_preds, task_windows)
        by_task[task_name]["trajectory_count"] = len(keys)
    return by_task


def build_validation_contexts(args, roots: Sequence[str]) -> List[Tuple[Any, MomentMemDataset, Any]]:
    """Build the configured untouched episode split used for autoregressive validation."""
    contexts: List[Tuple[Any, MomentMemDataset, Any]] = []
    for root in roots:
        try:
            ds = MomentMemDataset(
                dataset_root=root,
                mode="val",
                split_ratio=args.train_split_ratio,
                split_seed=args.split_seed,
                gamma=args.gamma,
                video_fps=args.video_fps,
                sample_fps=args.sample_fps,
                transform=None,
                include_t0_in_index=args.include_t0_in_index,
                image_size=args.image_size,
                buffer_size=args.buffer_size,
                max_history=args.max_history,
                add_memory_instruction=args.add_memory_instruction,
                camera_mode=args.camera_mode,
                augment=False,
                build_supervision=False,
            )
        except Exception as exc:  # noqa: BLE001
            if not args.skip_bad_leaves:
                raise RuntimeError(f"[data] val: failed to load {root}: {exc}") from exc
            print(f"[data] val: skip {root} ({exc})", flush=True)
            continue
        for ctx in ds.episodes:
            key = (root, int(ctx.episode_index))
            contexts.append((key, ds, ctx))
        print(
            f"[data] val: loaded {os.path.basename(root)} "
            f"({len(ds.episodes)} episodes, camera_mode={ds.resolved_camera_mode})",
            flush=True,
        )
    if not contexts:
        raise ValueError("No validation episodes were found.")
    return contexts


class AutoregressiveValidationCallback(TrainerCallback):
    """Evaluate at each save and store metrics without changing checkpoint names."""

    def __init__(
        self,
        *,
        processor,
        contexts: Sequence[Tuple[Any, MomentMemDataset, Any]],
        threshold: float = 0.5,
        batch_size: int = 2,
        buffer_size: int = 10,
        max_history: int = 48,
        max_length: int = 16384,
        vision_cache_cap: int = 6000,
        validation_split_ratio: float = 0.2,
    ) -> None:
        self.processor = processor
        self.contexts = list(contexts)
        self.threshold = float(threshold)
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.max_history = int(max_history)
        self.max_length = int(max_length)
        self.vision_cache_cap = int(vision_cache_cap)
        self.validation_split_ratio = round(float(validation_split_ratio), 12)
        self.yes_id = int(processor.tokenizer.encode("Yes", add_special_tokens=False)[0])
        self.no_id = int(processor.tokenizer.encode("No", add_special_tokens=False)[0])
        self._vision_cache: VisionCache | None = None

    @staticmethod
    def _dist_info() -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    @torch.no_grad()
    def _rollout(self, model, contexts) -> tuple[Dict[Any, Dict[int, int]], Dict[Any, List[Tuple[int, int]]]]:
        if self._vision_cache is None:
            self._vision_cache = VisionCache(self.processor, cap=self.vision_cache_cap)
        vc = self._vision_cache
        selected: Dict[Any, List[int]] = {key: [] for key, _, _ in contexts}
        preds: Dict[Any, Dict[int, int]] = {key: {} for key, _, _ in contexts}
        windows = {key: list(ctx.segments_sub) for key, _, ctx in contexts}
        max_len = max((ctx.sub_len for _, _, ctx in contexts), default=0)
        device = next(model.parameters()).device

        for t_sub in range(max_len):
            active = [(key, ds, ctx) for key, ds, ctx in contexts if t_sub < ctx.sub_len]
            items = [
                build_cached_item(
                    ctx,
                    ds,
                    t_sub,
                    selected[key],
                    self.buffer_size,
                    vc,
                    max_history=self.max_history,
                )
                for key, ds, ctx in active
            ]
            probabilities: List[float] = []
            for start in range(0, len(items), self.batch_size):
                batch = {
                    key: value.to(device)
                    for key, value in vc.build_batch(
                        items[start : start + self.batch_size], max_length=self.max_length
                    ).items()
                }
                outputs = model(**batch)
                last_idx = batch["attention_mask"].sum(dim=1) - 1
                logits = outputs.logits[torch.arange(outputs.logits.size(0), device=device), last_idx]
                yes_logits = logits[:, self.yes_id].float()
                no_logits = logits[:, self.no_id].float()
                probabilities.extend(
                    torch.softmax(torch.stack([no_logits, yes_logits], dim=1), dim=1)[:, 1]
                    .detach()
                    .cpu()
                    .tolist()
                )
            for (key, _ds, _ctx), probability in zip(active, probabilities):
                decision = int(float(probability) > self.threshold)
                preds[key][t_sub] = decision
                if decision:
                    selected[key].append(t_sub)
        return preds, windows

    def on_save(self, args, state, control, model=None, **kwargs):
        rank, world_size = self._dist_info()
        local_contexts = self.contexts[rank::world_size]
        was_training = bool(model.training)
        model.eval()
        print(
            f"[val-ar] step={state.global_step} rank={rank}/{world_size} "
            f"episodes={len(local_contexts)} threshold={self.threshold}",
            flush=True,
        )
        local_preds, local_windows = self._rollout(model, local_contexts)
        if was_training:
            model.train()

        gathered = [None for _ in range(world_size)]
        payload = (local_preds, local_windows)
        if world_size > 1:
            torch.distributed.all_gather_object(gathered, payload)
        else:
            gathered[0] = payload

        if rank == 0:
            preds: Dict[Any, Dict[int, int]] = {}
            windows: Dict[Any, List[Tuple[int, int]]] = {}
            for shard_preds, shard_windows in gathered:
                preds.update(shard_preds)
                windows.update(shard_windows)
            overall_metrics = compute_keyframe_metrics(preds, windows)
            overall_metrics["trajectory_count"] = len(preds)
            by_task = _compute_metrics_by_task(preds, windows)
            trajectory_counts = {
                "overall": len(preds),
                "by_task": {
                    task_name: task_metrics["trajectory_count"]
                    for task_name, task_metrics in by_task.items()
                },
            }
            # Expose overall metrics both at the top level and in the grouped view.
            metrics = dict(overall_metrics)
            metrics["overall"] = overall_metrics
            metrics["by_task"] = by_task
            metrics["trajectory_counts"] = trajectory_counts
            metrics.update(
                {
                    "step": int(state.global_step),
                    "threshold": self.threshold,
                    "split": "val",
                    "split_ratio": self.validation_split_ratio,
                    "train_split_ratio": round(
                        1.0 - self.validation_split_ratio, 12
                    ),
                    "validation_split_ratio": self.validation_split_ratio,
                    "protocol": "autoregressive",
                }
            )
            source = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            with open(os.path.join(source, "validation_metrics.json"), "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            print(
                f"[val-ar] step={state.global_step} "
                f"segment_hit={metrics['segment']['overall']['hit']:.4f} "
                f"segment_acc={metrics['segment']['overall']['acc']:.4f} "
                f"non_key_acc={metrics['non_key_acc']:.4f} "
                f"non_key_framewise={metrics['non_key_frame_acc']:.4f} -> "
                f"checkpoint-{state.global_step}/validation_metrics.json",
                flush=True,
            )
            print(
                f"[val-ar] overall segment={metrics['segment']}",
                flush=True,
            )
            for task_name, task_metrics in metrics["by_task"].items():
                print(
                    f"[val-ar]   {task_name}: "
                    f"segment_hit={task_metrics['segment']['overall']['hit']:.4f} "
                    f"segment_acc={task_metrics['segment']['overall']['acc']:.4f} "
                    f"non_key_acc={task_metrics['non_key_acc']:.4f} "
                    f"non_key_framewise={task_metrics['non_key_frame_acc']:.4f} "
                    f"segment={task_metrics['segment']} "
                    f"episodes={task_metrics['n_episodes']}",
                    flush=True,
                )
        if world_size > 1:
            torch.distributed.barrier()
        return control
