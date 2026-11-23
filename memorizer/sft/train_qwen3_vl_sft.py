import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import random

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, Sampler
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

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

from prompts import (  # noqa: E402
    PROMPT_LAYOUT_VERSION,
    build_user_parts,
    image_size_from_config,
    render_user_content,
    system_prompt,
)
from dataset import MomentMemDataset  # noqa: E402
from dataset_spec import resolve_dataset_root, resolve_spec  # noqa: E402
from sft.online_validation import (  # noqa: E402
    AutoregressiveValidationCallback,
    build_validation_contexts,
)
from sft.scheduled_sampling import ScheduledSamplingCallback  # noqa: E402


def _parse_int_csv(value: str) -> List[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {value!r}") from exc
    return parsed


def _parse_float_csv(value: str) -> List[float]:
    try:
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated floats, got {value!r}") from exc
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("MomentMem memorizer training")
    parser.add_argument("--config", type=Path, help="JSON training options; explicit CLI options take precedence.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument(
        "--processor-name-or-path",
        type=str,
        default=None,
        help="Optional processor/tokenizer source. Defaults to --model-name-or-path; useful "
             "when continuing from a weights-only Trainer checkpoint.",
    )
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="dataset root containing lerobot_format/<task> leaves. "
                        "Mutually exclusive with --dataset-spec.")
    parser.add_argument("--dataset-spec", type=str, default=None,
                        help="mix tasks across datasets: a JSON/YAML file (or inline JSON) of "
                        "{dataset_root: [task_name, ...]}. Same task_name across roots = one task. "
                        "Overrides --dataset-root when given.")
    parser.add_argument("--skip-bad-leaves", action="store_true",
                        help="skip a leaf dataset that fails to load instead of hard-failing (default: "
                        "fail fast, so we never silently train on fewer tasks than intended).")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Trainer/DeepSpeed checkpoint directory to resume from. Dataset and callback "
             "state is rebuilt from the saved Trainer step, so dynamic epoch resampling is "
             "deterministic but not bit-exact across an interrupted run.",
    )

    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)

    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--sample-fps", type=int, default=5)
    parser.add_argument(
        "--temporal-stride-augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Training-only temporal-frequency augmentation. Instead of fixed video_fps/sample_fps "
        "spacing, resample each episode timeline per epoch with independent original-frame gaps. "
        "Validation and inference remain at the fixed sample_fps cadence.",
    )
    parser.add_argument(
        "--temporal-stride-choices",
        type=_parse_int_csv,
        default=[1, 2, 3, 4],
        help="Comma-separated positive original-frame gaps used by temporal stride augmentation.",
    )
    parser.add_argument(
        "--temporal-stride-probabilities",
        type=_parse_float_csv,
        default=[0.1, 0.2, 0.3, 0.4],
        help="Comma-separated categorical weights corresponding to --temporal-stride-choices.",
    )
    parser.add_argument("--buffer-size", type=int, default=10,
                        help="Number of recent sampled timesteps before the current frame to include "
                        "in the VLM prompt. 0 disables the recent buffer.")
    parser.add_argument(
        "--camera-mode", choices=["overview", "overview+wrist"], default="overview",
        help="Images per timestep used for this checkpoint. overview ignores wrist videos; "
        "overview+wrist uses a synchronized pair in overview-then-wrist order.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-split-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--include-t0-in-index", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether frame 0 participates like a normal frame.")

    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="If >0, overrides num-train-epochs (handy for smoke tests).")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False,
                        help="Trade compute for memory; recommended for the 4B model.")
    parser.add_argument("--attn-implementation", type=str, default="sdpa",
                        choices=["sdpa", "eager", "flash_attention_2"],
                        help="Attention kernel. flash_attention_2 requires flash-attn installed.")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to a DeepSpeed config JSON (for sharded full fine-tuning).")
    parser.add_argument("--max-train-time-hours", type=float, default=0.0,
                        help="If >0, stop training after this many wall-clock hours (then save).")
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--save-only-model", action="store_true",
                        help="Only save model weights (safetensors), skip optimizer/scheduler/RNG "
                             "state (DeepSpeed global_step*). Saves much disk/IO; cannot resume.")
    parser.add_argument("--validate-on-save", action="store_true",
                        help="Run deployment-style autoregressive validation after every checkpoint save.")
    parser.add_argument("--eval-threshold", type=float, default=0.5)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument(
        "--max-history", "--eval-max-history", dest="max_history", type=int, default=48,
        help="Maximum long-term keyframes outside the recent buffer. This one value is shared "
        "by training, Scheduled Sampling, validation, and deployment. --eval-max-history is a "
        "legacy alias.",
    )
    parser.add_argument("--eval-vision-cache-cap", type=int, default=6000)

    # Sequence-level Scheduled Sampling: autoregressively roll out the current model at epoch
    # boundaries, then use a mixture of teacher and model decisions as training prompt history.
    parser.add_argument("--scheduled-sampling-max-prob", type=float, default=0.4,
                        help="Maximum probability of feeding a model decision (instead of the GT "
                        "decision) into subsequent training history. Set 0 to disable.")
    parser.add_argument("--scheduled-sampling-warmup-ratio", type=float, default=0.0,
                        help="Training-progress fraction kept as pure teacher forcing; the model "
                        "feedback probability then ramps linearly to its maximum.")
    parser.add_argument("--scheduled-sampling-threshold", type=float, default=0.5,
                        help="P(Yes) threshold used during training-set autoregressive rollout.")
    parser.add_argument("--scheduled-sampling-batch-size", type=int, default=8,
                        help="Per-rank inference batch size for the epoch-boundary rollout.")
    parser.add_argument("--scheduled-sampling-vision-cache-cap", type=int, default=20000,
                        help="Per-rank number of decoded/processed frame views cached for rollout.")

    # Focal loss balances sparse Yes labels; gamma=0 and alpha=0.5 gives cross-entropy.
    parser.add_argument("--focal-gamma", type=float, default=2.5,
                        help="Focusing parameter; higher down-weights easy (well-classified) tokens.")
    parser.add_argument("--focal-alpha", type=float, default=0.8,
                        help="Weight on the positive (Yes/keyframe) class; (1-alpha) on negatives.")
    parser.add_argument("--balance-sampler", action=argparse.BooleanOptionalAction, default=True,
                        help="Oversample keyframe (Yes) frames so each batch has positives. "
                        "Essential given the ~1%% positive rate.")
    parser.add_argument("--positive-fraction", type=float, default=0.34,
                        help="Target fraction of positive (Yes) samples per epoch (balance sampler).")
    parser.add_argument("--hard-negative-fraction", type=float, default=0.33,
                        help="Target fraction of HARD negatives (within-window 'redundant' No frames) "
                        "per epoch — teaches 'already in history -> No'. Rest are easy negatives.")

    # Consistent per-sample augmentation (same jitter + same weak crop for all images in a sample).
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply consistent color-jitter + weak random-crop to the current frame "
                        "and all history keyframes (train split only). Use --no-augment to disable.")
    parser.add_argument("--aug-brightness", type=float, default=0.3)
    parser.add_argument("--aug-contrast", type=float, default=0.3)
    parser.add_argument("--aug-saturation", type=float, default=0.3)
    parser.add_argument("--aug-hue", type=float, default=0.05)
    parser.add_argument("--aug-crop-scale", type=float, default=0.9,
                        help="Weak crop: each side kept in [scale, 1.0] then resized back. Higher = milder.")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Square-resize each decoded frame to this size. Ignored with "
                        "--native-image-resolution; eval must use the same preprocessing contract.")
    parser.add_argument(
        "--native-image-resolution",
        action="store_true",
        help="Keep each decoded frame at its native aspect ratio and let the Qwen processor "
        "apply dynamic-resolution patching instead of manually resizing it to a square.",
    )
    parser.add_argument("--max-train-episodes", type=int, default=None,
                        help="Cap TRAIN episodes per task (val unchanged) — for train-set-size sweeps.")
    parser.add_argument("--add-memory-instruction", action="store_true",
                        help="Feed the VLM INSTRUCTION_TASK + INSTRUCTION_MEMORY (the 'what to "
                        "remember' hint from meta/tasks.jsonl) instead of the task text alone.")

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_path = config_parser.parse_known_args()[0].config
    if config_path is not None:
        options = json.loads(config_path.read_text())
        known = {action.dest: action for action in parser._actions}
        if not isinstance(options, dict) or set(options) - set(known):
            parser.error("config must be an object containing known argument names")
        parser.set_defaults(**options)
        for key in options:
            known[key].required = False
    args = parser.parse_args()
    if args.image_size <= 0:
        parser.error("--image-size must be positive")
    if args.native_image_resolution:
        args.image_size = None
    return args


def _to_rgb_pil(img: Any):
    # Normalize dataset images to RGB PIL.
    if isinstance(img, torch.Tensor):
        # Expect CHW in [0,1] or [0,255]
        x = img.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0).numpy()
        x = x.clip(0, 1) if x.max() <= 1.0 else (x.clip(0, 255) / 255.0)
        from PIL import Image
        import numpy as np

        arr = (x * 255.0).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    return img.convert("RGB")


@dataclass
class QwenKeyframeCollator:
    processor: Any
    max_length: int

    def _build_messages(
        self,
        task_field: str,
        history_count: int,
        num_cameras: int,
        label: int,
        buffer_keyframe_flags: List[bool] | None = None,
        buffer_frame_indices: List[int] | None = None,
    ) -> List[Dict[str, Any]]:
        # Preserve the history, buffer, current, and camera order used by the processor.
        buffer_keyframe_flags = list(buffer_keyframe_flags or [])
        num_timesteps = history_count + len(buffer_keyframe_flags) + 1
        user_parts = build_user_parts(
            task_field,
            history_count,
            num_cameras,
            buffer_keyframe_flags=buffer_keyframe_flags,
            buffer_frame_indices=buffer_frame_indices,
        )
        assistant_text = "Yes" if label == 1 else "No"
        image_content = [{"type": "image"} for _ in range(num_timesteps * num_cameras)]
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt(num_cameras)}],
            },
            {
                "role": "user",
                "content": render_user_content(user_parts, image_content),
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts_all: List[str] = []
        texts_prompt: List[str] = []
        images_all: List[List[Any]] = []
        skip_mask: List[bool] = []
        focal_classes: List[int] = []
        action_token_ids: List[int] = []

        for ex in batch:
            # Each history item / current item is a list of camera views (overview, wrist).
            num_cameras = int(ex.get("num_cameras", len(ex["current_image"])))
            flat_history: List[Any] = []
            for views in ex["history_images"]:          # per keyframe timestep
                flat_history.extend(_to_rgb_pil(v) for v in views)
            flat_buffer: List[Any] = []
            for views in ex.get("buffer_images", []):    # per recent-buffer timestep
                flat_buffer.extend(_to_rgb_pil(v) for v in views)
            current_views = [_to_rgb_pil(v) for v in ex["current_image"]]
            images = flat_history + flat_buffer + current_views
            history_count = len(ex["history_images"])    # number of history TIMESTEPS
            label = int(ex["label"].item() > 0.5)

            msgs_all = self._build_messages(
                ex["instruction"],
                history_count,
                num_cameras,
                label,
                buffer_keyframe_flags=[bool(x) for x in ex.get("buffer_is_keyframe", [])],
                buffer_frame_indices=[int(x) for x in ex.get("buffer_frame_indices_sub", [])],
            )
            msgs_prompt = [msgs_all[0], msgs_all[1]]  # system + user

            text_all = self.processor.apply_chat_template(
                msgs_all, tokenize=False, add_generation_prompt=False
            )
            text_prompt = self.processor.apply_chat_template(
                msgs_prompt, tokenize=False, add_generation_prompt=True
            )

            texts_all.append(text_all)
            texts_prompt.append(text_prompt)
            images_all.append(images)
            skip_mask.append(bool(ex["loss_mask"].item() <= 0.0))
            focal_classes.append(label)
            action_text = "Yes" if label == 1 else "No"
            action_ids = self.processor.tokenizer.encode(action_text, add_special_tokens=False)
            if len(action_ids) != 1:
                raise ValueError(
                    f"[collator] {action_text!r} must tokenize to exactly one action token, got "
                    f"{len(action_ids)} tokens: {action_ids}. The Stage-1/Stage-2 binary policy "
                    "contract requires single-token Yes/No actions."
                )
            action_token_ids.append(int(action_ids[0]))

        batch_all = self.processor(
            text=texts_all,
            images=images_all,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch_prompt = self.processor(
            text=texts_prompt,
            images=images_all,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Start fully masked and restore exactly one action token per valid sample below. In
        # particular, the assistant suffix (for example <|im_end|> and a trailing newline) must
        # not receive focal/class weighting.
        labels = torch.full_like(batch_all["input_ids"], -100)

        # Supervise only the Yes/No completion after the prompt prefix.
        prompt_lens = batch_prompt["attention_mask"].sum(dim=1).tolist()
        all_lens = batch_all["attention_mask"].sum(dim=1).tolist()
        # Reject samples whose answer was truncated away.
        for i, (p_len, a_len, action_id) in enumerate(
            zip(prompt_lens, all_lens, action_token_ids)
        ):
            if skip_mask[i]:
                continue
            p_len, a_len = int(p_len), int(a_len)
            if a_len <= p_len:
                raise ValueError(
                    f"[collator] sample {i}: batch_all has {a_len} real tokens but the prompt alone "
                    f"is {p_len} — the Yes/No answer was truncated away (max_length={self.max_length}). "
                    f"Increase --max-length or reduce buffer_size/history/cameras."
                )
            labels[i, p_len] = action_id

        batch_all["labels"] = labels
        # Per-sample class (1=Yes/keyframe, 0=No) for focal-loss alpha weighting.
        batch_all["focal_class"] = torch.tensor(focal_classes, dtype=torch.long)
        return batch_all


class FocalTrainer(Trainer):
    """Trainer that applies focal loss to the single supervised Yes/No action token, countering
    the heavy imbalance between sparse keyframes and negative frames.
    """

    def __init__(
        self,
        *args,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        balance_sampler: bool = False,
        positive_fraction: float = 0.34,
        hard_negative_fraction: float = 0.33,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.balance_sampler = bool(balance_sampler)
        self.positive_fraction = float(positive_fraction)
        self.hard_negative_fraction = float(hard_negative_fraction)

    def _get_train_sampler(self, *args, **kwargs):
        if self.balance_sampler:
            return BalancedKeyframeSampler(
                self.train_dataset, self.positive_fraction, self.hard_negative_fraction, self.args.seed
            )
        return super()._get_train_sampler(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        focal_class = inputs.pop("focal_class", None)  # (B,) 1=Yes, 0=No; not a model arg
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)

        # Causal shift: predict token t+1 from position t.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        vocab = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab)
        flat_labels = shift_labels.view(-1)

        ce = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=-100)
        valid = flat_labels.ne(-100)
        pt = torch.exp(-ce)  # prob of the true token (1 at ignored positions, so focal->0 there)
        focal = ((1.0 - pt) ** self.focal_gamma) * ce

        if focal_class is not None:
            cls = focal_class.to(focal.device).view(-1, 1).expand(-1, shift_labels.size(1)).reshape(-1)
            alpha_pos = torch.as_tensor(self.focal_alpha, device=focal.device, dtype=focal.dtype)
            alpha_w = torch.where(cls == 1, alpha_pos, 1.0 - alpha_pos)
            focal = focal * alpha_w

        denom = valid.sum().clamp(min=1)
        loss = (focal * valid.to(focal.dtype)).sum() / denom
        return (loss, outputs) if return_outputs else loss


class BalancedKeyframeSampler(Sampler):
    """Three-way balanced sampler: positives / hard negatives / easy negatives."""

    def __init__(
        self,
        dataset: Dataset,
        positive_fraction: float = 0.34,
        hard_negative_fraction: float = 0.33,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.positive_fraction = float(positive_fraction)
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = len(dataset)

    def _gather(self):
        ds = self.dataset
        subs = ds.datasets if hasattr(ds, "datasets") else [ds]  # ConcatDataset or single
        pos: List[int] = []
        hard: List[int] = []
        easy: List[int] = []
        offset = 0
        for sub in subs:
            p, h, e = sub.supervised_groups()
            pos.extend(offset + i for i in p)
            hard.extend(offset + i for i in h)
            easy.extend(offset + i for i in e)
            offset += len(sub)
        return pos, hard, easy

    @staticmethod
    def _draw(pool, k, g):
        if k <= 0 or not pool:
            return []
        if k <= len(pool):
            return g.sample(pool, k)
        return [pool[g.randrange(len(pool))] for _ in range(k)]

    def __iter__(self):
        g = random.Random(self.seed + self.epoch * 7919)
        self.epoch += 1
        pos, hard, easy = self._gather()
        n = self.num_samples
        # If a group is empty, fold its budget into the remaining groups proportionally.
        groups = [(pos, self.positive_fraction), (hard, self.hard_negative_fraction),
                  (easy, max(0.0, 1.0 - self.positive_fraction - self.hard_negative_fraction))]
        groups = [(p, f) for (p, f) in groups if p]
        if not groups:
            idx = list(range(n))
            g.shuffle(idx)
            return iter(idx)
        tot_f = sum(f for _, f in groups) or 1.0
        idx: List[int] = []
        for pool, f in groups:
            idx += self._draw(pool, int(round(n * f / tot_f)), g)
        # pad/truncate to exactly n (rounding)
        while len(idx) < n:
            pool = groups[0][0]
            idx.append(pool[g.randrange(len(pool))])
        idx = idx[:n]
        g.shuffle(idx)
        return iter(idx)

    def __len__(self):
        return self.num_samples


def _set_epoch_recursive(ds: Any, epoch: int) -> None:
    """Propagate set_epoch through (possibly concatenated) MomentMemDataset(s)."""
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(epoch)
    elif hasattr(ds, "datasets"):  # torch ConcatDataset
        for sub in ds.datasets:
            _set_epoch_recursive(sub, epoch)


class EpochResampleCallback(TrainerCallback):
    """Resample counterfactual labels each epoch for MomentMemDataset(train)."""

    def __init__(self, train_dataset: Dataset):
        self.train_dataset = train_dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        ep = int(state.epoch) if state.epoch is not None else 0
        _set_epoch_recursive(self.train_dataset, ep)
        return control


class TimeStopCallback(TrainerCallback):
    """Stop training after a wall-clock budget (checked every step). Trainer then saves."""

    def __init__(self, max_hours: float):
        import time
        self._time = time.time
        self.max_seconds = max_hours * 3600.0
        self.start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = self._time()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.start is not None and (self._time() - self.start) >= self.max_seconds:
            control.should_training_stop = True
        return control


def build_train_dataset(args: argparse.Namespace) -> Dataset:
    """Build one training MomentMemDataset per leaf LeRobot dataset and concatenate them."""
    roots = resolve_data_roots(args)
    src_desc = f"spec {args.dataset_spec}" if args.dataset_spec else args.dataset_root
    subs: List[MomentMemDataset] = []
    for r in roots:
        try:
            subs.append(
                MomentMemDataset(
                    dataset_root=r,
                    mode="train",
                    split_ratio=args.train_split_ratio,
                    split_seed=args.split_seed,
                    gamma=args.gamma,
                    video_fps=args.video_fps,
                    sample_fps=args.sample_fps,
                    transform=None,  # keep PIL for the processor
                    include_t0_in_index=args.include_t0_in_index,
                    image_size=args.image_size,
                    max_train_episodes=args.max_train_episodes,
                    buffer_size=args.buffer_size,
                    max_history=args.max_history,
                    add_memory_instruction=args.add_memory_instruction,
                    camera_mode=args.camera_mode,
                    temporal_stride_augmentation=args.temporal_stride_augmentation,
                    temporal_stride_choices=tuple(args.temporal_stride_choices),
                    temporal_stride_probabilities=tuple(args.temporal_stride_probabilities),
                    # A decode failure must stop training; blank visual inputs silently corrupt
                    # keyframe supervision, especially with LeRobot's default AV1 videos.
                    allow_blank_frames=False,
                    # Share one augmentation across all images in each training sample.
                    augment=args.augment,
                    aug_brightness=args.aug_brightness,
                    aug_contrast=args.aug_contrast,
                    aug_saturation=args.aug_saturation,
                    aug_hue=args.aug_hue,
                    aug_crop_scale=args.aug_crop_scale,
                )
            )
            print(f"[data] train: loaded {os.path.basename(r)} "
                  f"({len(subs[-1])} frames, {len(subs[-1].episodes)} episodes, "
                  f"temporal_stride_aug={subs[-1].temporal_stride_augmentation})")
        except Exception as e:  # noqa: BLE001 - a leaf may have an empty split for this mode
            if not args.skip_bad_leaves:
                raise RuntimeError(
                    f"[data] train: failed to load leaf {os.path.basename(r)} ({e}); pass "
                    "--skip-bad-leaves to skip it."
                ) from e
            print(f"[data] train: skip {os.path.basename(r)} ({e})")
    if not subs:
        raise ValueError(f"No usable train data under {src_desc}")
    # Require one camera schema across all dataset leaves.
    cam_schemas = {os.path.basename(s.dataset_root): tuple(s._camera_keys) for s in subs}
    if len(set(cam_schemas.values())) > 1:
        raise ValueError(
            f"[data] train: leaves have inconsistent camera schemas: {cam_schemas}. "
            "All datasets in one run must share identical camera keys."
        )
    total = sum(len(s) for s in subs)
    print(f"[data] train: {len(subs)} dataset(s), {total} total frame-samples")
    return subs[0] if len(subs) == 1 else ConcatDataset(subs)


def resolve_data_roots(args: argparse.Namespace) -> List[str]:
    if args.dataset_spec:
        return [r.leaf_dir for r in resolve_spec(args.dataset_spec)]
    if args.dataset_root:
        return [r.leaf_dir for r in resolve_dataset_root(args.dataset_root)]
    raise ValueError("provide either --dataset-spec or --dataset-root")


def build_model_and_processor(args: argparse.Namespace):
    processor_source = args.processor_name_or_path or args.model_name_or_path
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)

    # Right padding preserves prompt-prefix label masking.
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    # Use the multimodal model class available in the installed Transformers version.
    model = None
    last_err = None
    for model_cls in (AutoModelForImageTextToText, AutoModelForVision2Seq):
        try:
            model = model_cls.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                attn_implementation=args.attn_implementation,
            )
            print(f"[load] loaded model with {model_cls.__name__}")
            break
        except Exception as e:  # noqa: BLE001 - logged, then re-raised if all classes fail
            print(f"[load] {model_cls.__name__} failed: {e}")
            last_err = e
    if model is None:
        raise RuntimeError("Failed to load a vision-language model.") from last_err

    if args.gradient_checkpointing:
        model.config.use_cache = False

    return model, processor


def main():
    args = parse_args()
    if args.resume_from_checkpoint:
        args.resume_from_checkpoint = os.path.abspath(args.resume_from_checkpoint)
        if not os.path.isdir(args.resume_from_checkpoint):
            raise FileNotFoundError(
                f"--resume-from-checkpoint is not a directory: {args.resume_from_checkpoint}"
            )
        trainer_state = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
        if not os.path.isfile(trainer_state):
            raise FileNotFoundError(
                f"Resume checkpoint is missing trainer_state.json: {args.resume_from_checkpoint}"
            )
    probability_sum = sum(args.temporal_stride_probabilities)
    args.temporal_stride_probabilities = [
        weight / probability_sum for weight in args.temporal_stride_probabilities
    ]
    os.makedirs(args.output_dir, exist_ok=True)

    model, processor = build_model_and_processor(args)

    train_ds = build_train_dataset(args)
    val_contexts = build_validation_contexts(args, resolve_data_roots(args)) if args.validate_on_save else []
    collator = QwenKeyframeCollator(
        processor=processor,
        max_length=args.max_length,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="no",
        save_strategy="steps",
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        deepspeed=args.deepspeed if args.deepspeed else None,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none",
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    # Store prompt settings with every checkpoint for consistent serving.
    _cams = getattr(train_ds, "_camera_keys", None)
    if _cams is None and getattr(train_ds, "datasets", None):
        _cams = getattr(train_ds.datasets[0], "_camera_keys", None)
    _camera_mode = "overview+wrist" if len(_cams) == 2 else "overview"
    prompt_cfg = {
        "add_memory_instruction": bool(args.add_memory_instruction),
        "buffer_size": int(args.buffer_size),
        "max_history": int(args.max_history),
        "camera_mode": _camera_mode,
        "use_wrist": _camera_mode == "overview+wrist",
        "camera_keys": list(_cams),
        "num_cameras": len(_cams),
        # Record the training-time retrieval stride for online inference.
        "video_fps": int(args.video_fps),
        "sample_fps": int(args.sample_fps),
        "retrieval_stride": max(1, int(args.video_fps) // max(1, int(args.sample_fps))),
        "temporal_stride_augmentation": {
            "enabled": bool(args.temporal_stride_augmentation),
            "choices": list(args.temporal_stride_choices),
            "probabilities": list(args.temporal_stride_probabilities),
            "scope": "training_episode_timeline",
            "resampling": "per_epoch",
            "validation_and_inference_stride": max(
                1, int(args.video_fps) // max(1, int(args.sample_fps))
            ),
        },
        # Consumed by agent_server and evaluation so serving cannot drift from training.
        "image_preprocess": "native_dynamic" if args.image_size is None else "square_resize",
        "image_size": None if args.image_size is None else int(args.image_size),
        "max_length": int(args.max_length),
        "prompt_layout": int(PROMPT_LAYOUT_VERSION),
        "scheduled_sampling": {
            "max_probability": float(args.scheduled_sampling_max_prob),
            "warmup_ratio": float(args.scheduled_sampling_warmup_ratio),
            "threshold": float(args.scheduled_sampling_threshold),
            "protocol": "epoch_autoregressive_memory_conditioned_mixed_feedback",
        },
    }
    # Keep the writer subject to the same strict contract as every reader.
    image_size_from_config(prompt_cfg)

    class KeyframeSidecarCallback(TrainerCallback):
        """Write serving metadata and processor assets into each checkpoint on save."""
        def on_save(self, _args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            ckpt_dir = os.path.join(_args.output_dir, f"checkpoint-{state.global_step}")
            try:
                os.makedirs(ckpt_dir, exist_ok=True)
                processor.save_pretrained(ckpt_dir)
                with open(os.path.join(ckpt_dir, "keyframe_prompt_config.json"), "w") as f:
                    json.dump(prompt_cfg, f, indent=2)
            except Exception as e:  # never let sidecar writing break training
                print(f"[train] WARN: failed to write checkpoint sidecar in {ckpt_dir}: {e}")

    # Ordering matters: first resample this epoch's counterfactual GT labels, then roll out the
    # model against those labels to construct the mixed history seen by training samples.
    callbacks: List[TrainerCallback] = [
        EpochResampleCallback(train_ds),
        ScheduledSamplingCallback(
            processor=processor,
            train_dataset=train_ds,
            max_probability=args.scheduled_sampling_max_prob,
            warmup_ratio=args.scheduled_sampling_warmup_ratio,
            threshold=args.scheduled_sampling_threshold,
            batch_size=args.scheduled_sampling_batch_size,
            buffer_size=args.buffer_size,
            max_history=args.max_history,
            max_length=args.max_length,
            vision_cache_cap=args.scheduled_sampling_vision_cache_cap,
            seed=args.seed,
        ),
        KeyframeSidecarCallback(),
    ]
    if args.validate_on_save:
        callbacks.append(
            AutoregressiveValidationCallback(
                processor=processor,
                contexts=val_contexts,
                threshold=args.eval_threshold,
                batch_size=args.per_device_eval_batch_size,
                buffer_size=args.buffer_size,
                max_history=args.max_history,
                max_length=args.max_length,
                vision_cache_cap=args.eval_vision_cache_cap,
                validation_split_ratio=1.0 - args.train_split_ratio,
            )
        )
    if args.max_train_time_hours and args.max_train_time_hours > 0:
        callbacks.append(TimeStopCallback(args.max_train_time_hours))

    trainer = FocalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=callbacks,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        balance_sampler=args.balance_sampler,
        positive_fraction=args.positive_fraction,
        hard_negative_fraction=args.hard_negative_fraction,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Also store prompt settings at the run root.
    with open(os.path.join(args.output_dir, "keyframe_prompt_config.json"), "w") as f:
        json.dump(prompt_cfg, f, indent=2)
    print(f"[train] wrote keyframe_prompt_config.json: {prompt_cfg}")


if __name__ == "__main__":
    main()
