"""Sequence-level Scheduled Sampling for keyframe-selector SFT."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import TrainerCallback

from dataset import MomentMemDataset
from sft.scheduled_sampling_policy import (
    ContextKey,
    memory_conditioned_supervision,
    scheduled_sampling_probability,
    use_model_feedback,
    visible_selected_history,
)
from sft.vision_cache import VisionCache, build_cached_item


TrainingContext = Tuple[ContextKey, MomentMemDataset, Any]
FeedbackPayload = Tuple[List[int], np.ndarray, np.ndarray]


def _iter_mrm_datasets(dataset: Any):
    if isinstance(dataset, MomentMemDataset):
        yield dataset
        return
    for child in getattr(dataset, "datasets", []):
        yield from _iter_mrm_datasets(child)


def build_training_contexts(dataset: Any) -> List[TrainingContext]:
    contexts: List[TrainingContext] = []
    seen: set[ContextKey] = set()
    for ds in _iter_mrm_datasets(dataset):
        root = os.path.realpath(ds.dataset_root)
        for ctx in ds.episodes:
            key = (root, int(ctx.episode_index))
            seen.add(key)
            contexts.append((key, ds, ctx))
    return contexts


def clear_feedback_history(contexts: Sequence[TrainingContext]) -> None:
    for _key, _ds, ctx in contexts:
        ctx.feedback_selected_sub = None
        ctx.feedback_labels = None
        ctx.feedback_loss_mask = None


def apply_feedback_supervision(
    contexts: Sequence[TrainingContext], feedback: Dict[ContextKey, FeedbackPayload]
) -> None:
    for key, _ds, ctx in contexts:
        history, labels, loss_mask = feedback[key]
        ctx.feedback_selected_sub = sorted({int(t) for t in history})
        ctx.feedback_labels = np.asarray(labels, dtype=np.float32)
        ctx.feedback_loss_mask = np.asarray(loss_mask, dtype=np.float32)


class ScheduledSamplingCallback(TrainerCallback):
    """Build mixed teacher/model histories at epoch boundaries."""

    def __init__(
        self,
        *,
        processor,
        train_dataset,
        max_probability: float = 0.4,
        warmup_ratio: float = 0.0,
        threshold: float = 0.5,
        batch_size: int = 2,
        buffer_size: int = 10,
        max_history: int = 48,
        max_length: int = 16384,
        vision_cache_cap: int = 20000,
        seed: int = 42,
    ) -> None:
        self.processor = processor
        self.contexts = build_training_contexts(train_dataset)
        self.max_probability = float(max_probability)
        self.warmup_ratio = float(warmup_ratio)
        self.threshold = float(threshold)
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.max_history = int(max_history)
        self.max_length = int(max_length)
        self.vision_cache_cap = int(vision_cache_cap)
        self.seed = int(seed)
        # Validate schedule parameters eagerly.
        scheduled_sampling_probability(
            0.0, warmup_ratio=self.warmup_ratio, max_probability=self.max_probability
        )
        self.yes_id = int(processor.tokenizer.encode("Yes", add_special_tokens=False)[0])
        self.no_id = int(processor.tokenizer.encode("No", add_special_tokens=False)[0])
        self._vision_cache: VisionCache | None = None

    @staticmethod
    def _dist_info() -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    @staticmethod
    def _progress(args, state) -> float:
        """Map epoch boundaries to [0, 1], including both endpoints."""

        if state.epoch is None:
            return 0.0
        total_epochs = int(
            getattr(state, "num_train_epochs", 0)
            or getattr(args, "num_train_epochs", 0)
        )
        if total_epochs <= 1:
            # With only one rollout boundary there is no interval over which to ramp. Treat the
            # sole epoch as the final epoch so the configured maximum remains meaningful.
            return 1.0
        epoch_index = int(float(state.epoch))
        return min(1.0, max(0.0, float(epoch_index) / float(total_epochs - 1)))

    @torch.no_grad()
    def _rollout(
        self,
        model,
        contexts: Sequence[TrainingContext],
        *,
        probability: float,
        epoch: int,
    ) -> tuple[Dict[ContextKey, FeedbackPayload], Dict[str, int]]:
        if self._vision_cache is None:
            self._vision_cache = VisionCache(self.processor, cap=self.vision_cache_cap)
        vc = self._vision_cache
        selected: Dict[ContextKey, List[int]] = {key: [] for key, _ds, _ctx in contexts}
        labels: Dict[ContextKey, np.ndarray] = {
            key: np.zeros((ctx.sub_len,), dtype=np.float32) for key, _ds, ctx in contexts
        }
        loss_masks: Dict[ContextKey, np.ndarray] = {
            key: np.ones((ctx.sub_len,), dtype=np.float32) for key, _ds, ctx in contexts
        }
        anchors: Dict[ContextKey, List[int]] = {}
        for key, _ds, ctx in contexts:
            episode_anchors: List[int] = []
            for start, end in ctx.segments_sub:
                matches = [
                    t for t in range(int(start), int(end) + 1)
                    if 0 <= t < ctx.sub_len and float(ctx.labels[t]) > 0.5
                ]
                episode_anchors.append(matches[0])
            anchors[key] = episode_anchors
        stats = {
            "decisions": 0,
            "model_feedback": 0,
            "model_false_positive": 0,
            "model_false_negative": 0,
        }
        max_len = max((ctx.sub_len for _key, _ds, ctx in contexts), default=0)
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
                logits = outputs.logits[
                    torch.arange(outputs.logits.size(0), device=device), last_idx
                ]
                yes_logits = logits[:, self.yes_id].float()
                no_logits = logits[:, self.no_id].float()
                probabilities.extend(
                    torch.softmax(torch.stack([no_logits, yes_logits], dim=1), dim=1)[:, 1]
                    .detach()
                    .cpu()
                    .tolist()
                )

            for (key, _ds, ctx), predicted_probability in zip(active, probabilities):
                prediction = int(float(predicted_probability) > self.threshold)
                target, loss_mask = memory_conditioned_supervision(
                    t_sub,
                    segments=ctx.segments_sub,
                    anchors=anchors[key],
                    selected=visible_selected_history(
                        selected[key],
                        timestep=t_sub,
                        buffer_size=self.buffer_size,
                        max_history=self.max_history,
                    ),
                )
                labels[key][t_sub] = float(target)
                loss_masks[key][t_sub] = float(loss_mask)
                take_model = use_model_feedback(
                    probability,
                    seed=self.seed,
                    epoch=epoch,
                    key=key,
                    timestep=t_sub,
                )
                feedback = prediction if take_model else target
                stats["decisions"] += 1
                if take_model:
                    stats["model_feedback"] += 1
                    if loss_mask > 0.0:
                        stats["model_false_positive"] += int(prediction == 1 and target == 0)
                        stats["model_false_negative"] += int(prediction == 0 and target == 1)
                if feedback:
                    selected[key].append(t_sub)
        payload = {
            key: (selected[key], labels[key], loss_masks[key]) for key in selected
        }
        return payload, stats

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        rank, world_size = self._dist_info()
        epoch = int(state.epoch) if state.epoch is not None else 0
        probability = scheduled_sampling_probability(
            self._progress(args, state),
            warmup_ratio=self.warmup_ratio,
            max_probability=self.max_probability,
        )
        if probability <= 0.0:
            # EpochResampleCallback has already cleared this, but make the contract explicit.
            clear_feedback_history(self.contexts)
            if rank == 0:
                print(
                    f"[scheduled-sampling] epoch={epoch} teacher forcing "
                    f"(model_feedback_probability=0.0000)",
                    flush=True,
                )
            return control

        local_contexts = self.contexts[rank::world_size]
        was_training = bool(model.training)
        model.eval()
        try:
            local_feedback, local_stats = self._rollout(
                model, local_contexts, probability=probability, epoch=epoch
            )
        finally:
            if was_training:
                model.train()

        payload = (local_feedback, local_stats)
        gathered = [None for _ in range(world_size)]
        if world_size > 1:
            torch.distributed.all_gather_object(gathered, payload)
        else:
            gathered[0] = payload

        feedback: Dict[ContextKey, FeedbackPayload] = {}
        totals = {name: 0 for name in local_stats}
        for shard_feedback, shard_stats in gathered:
            feedback.update(shard_feedback)
            for name, value in shard_stats.items():
                totals[name] += int(value)
        apply_feedback_supervision(self.contexts, feedback)

        if rank == 0:
            print(
                f"[scheduled-sampling] epoch={epoch} probability={probability:.4f} "
                f"model_feedback={totals['model_feedback']}/{totals['decisions']} "
                f"feedback_fp={totals['model_false_positive']} "
                f"feedback_fn={totals['model_false_negative']}",
                flush=True,
            )
        if world_size > 1:
            torch.distributed.barrier()
        return control
