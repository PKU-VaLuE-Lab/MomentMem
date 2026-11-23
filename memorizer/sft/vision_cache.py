"""Per-frame vision-feature cache for the AUTOREGRESSIVE keyframe rollout."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from prompts import build_user_parts, render_user_content, system_prompt


def _to_rgb_pil(img: Any) -> Image.Image:
    """Normalize an image without importing the training entrypoint."""
    if isinstance(img, torch.Tensor):
        x = img.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0).numpy()
        x = x.clip(0, 1) if x.max() <= 1.0 else (x.clip(0, 255) / 255.0)
        return Image.fromarray((x * 255.0).astype(np.uint8)).convert("RGB")
    if isinstance(img, np.ndarray):
        x = img
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = np.transpose(x, (1, 2, 0))
        if np.issubdtype(x.dtype, np.floating):
            x = (np.clip(x, 0, 1) * 255.0).astype(np.uint8)
        return Image.fromarray(x).convert("RGB")
    return img.convert("RGB")

IMAGE_PAD = "<|image_pad|>"


class VisionCache:
    """LRU cache keyed by episode, original source-frame index, and camera."""

    def __init__(self, processor, cap: int = 20000):
        self.processor = processor
        self.ip = processor.image_processor
        self.tok = processor.tokenizer
        self.cap = cap
        self._pv: "OrderedDict[Tuple[int, int, int], torch.Tensor]" = OrderedDict()
        self._K: int | None = None
        self._grid: torch.Tensor | None = None  # (1,3) constant image_grid_thw
        self.hits = 0
        self.misses = 0

    def _process_single(self, pil) -> torch.Tensor:
        out = self.ip(images=[pil], return_tensors="pt")
        thw = out["image_grid_thw"][0]
        merge = int(getattr(self.ip, "merge_size", 2))
        token_count = int(thw.prod().item() // (merge ** 2))
        if self._K is None:
            self._grid = out["image_grid_thw"][:1].clone()
            self._K = token_count
        elif token_count != self._K or not torch.equal(thw, self._grid[0]):
            raise ValueError(
                "VisionCache requires one constant Qwen image grid per evaluation run; "
                f"got {thw.tolist()} after {self._grid[0].tolist()}. Disable --cache-vision "
                "or evaluate datasets with different native resolutions separately."
            )
        return out["pixel_values"]

    def frame_views(self, ctx, ds, k: int) -> List[torch.Tensor]:
        """Ordered per-camera pixel_values for one timestep (cached). Camera order == video_paths."""
        views: List[torch.Tensor] = []
        for cam, vp in enumerate(ctx.video_paths):
            key = (id(ctx), int(ctx.sampled_orig_indices[k]), cam)
            pv = self._pv.get(key)
            if pv is None:
                self.misses += 1
                pil = _to_rgb_pil(ds._load_frame_by_sub_idx(vp, k))
                pv = self._process_single(pil)
                self._pv[key] = pv
                if len(self._pv) > self.cap:
                    self._pv.popitem(last=False)
            else:
                self.hits += 1
                self._pv.move_to_end(key)
            views.append(pv)
        return views

    def _text_for(
        self,
        instruction,
        num_cameras,
        history_count,
        buffer_flags,
        buffer_indices,
    ) -> str:
        image_content = [{"type": "image"}
                         for _ in range((history_count + len(buffer_flags) + 1) * num_cameras)]
        parts = build_user_parts(instruction, history_count, num_cameras,
                                 buffer_keyframe_flags=buffer_flags,
                                 buffer_frame_indices=buffer_indices)
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt(num_cameras)}]},
            {"role": "user", "content": render_user_content(parts, image_content)},
        ]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text.replace(IMAGE_PAD, IMAGE_PAD * self._K)

    def build_batch(self, cached_items: List[Dict[str, Any]], max_length: int):
        """Assemble a padded model batch from cached items (each carries ordered `pv_blocks`).
        Byte-identical to build_infer_inputs(processor, items)."""
        texts: List[str] = []
        pv_list: List[torch.Tensor] = []
        n_img = 0
        for it in cached_items:
            texts.append(self._text_for(it["instruction"], it["num_cameras"], it["history_count"],
                                        it["buffer_is_keyframe"], it["buffer_frame_indices_sub"]))
            for pv in it["pv_blocks"]:
                pv_list.append(pv)
                n_img += 1
        # NEVER truncate (mirrors eval_utils.build_infer_inputs): a cut sequence would score the
        # Yes/No logits at the wrong position and desync input_ids from the cached pixel_values.
        enc = self.tok(texts, return_tensors="pt", padding=True)
        seq_len = int(enc["input_ids"].shape[1])
        if seq_len > max_length:
            raise ValueError(
                f"VLM prompt is {seq_len} tokens > --max-length={max_length}. Truncating would "
                "corrupt the Yes/No answer position, so this is a hard error: raise --max-length, "
                "or lower --max-history / --buffer-size / --image-size so the prompt fits."
            )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "pixel_values": torch.cat(pv_list, 0),
            "image_grid_thw": self._grid.repeat(n_img, 1),
        }


def build_cached_item(
    ctx,
    ds,
    t_sub: int,
    selected_sub: List[int],
    buffer_size: int,
    vc: VisionCache,
    max_history: int | None = None,
) -> Dict[str, Any]:
    """Cached analogue of eval_keyframe_ar._build_item: same history/buffer semantics, but the
    images are cached pixel_values blocks (ordered history -> buffer -> current) rather than PIL."""
    buffer_sub = list(range(max(0, t_sub - buffer_size), t_sub)) if buffer_size > 0 else []
    buffer_set = set(buffer_sub)
    selected_set = set(selected_sub)
    history_sub = [k for k in selected_sub if k < t_sub and k not in buffer_set]
    if max_history is not None and len(history_sub) > max_history:
        history_sub = history_sub[-max_history:] if max_history > 0 else []
    pv_blocks: List[torch.Tensor] = []
    for k in history_sub:
        pv_blocks += vc.frame_views(ctx, ds, k)
    for k in buffer_sub:
        pv_blocks += vc.frame_views(ctx, ds, k)
    pv_blocks += vc.frame_views(ctx, ds, t_sub)  # current
    return {
        "instruction": ctx.instruction,
        "num_cameras": len(ds._camera_keys),
        "history_count": len(history_sub),
        "buffer_is_keyframe": [k in selected_set for k in buffer_sub],
        "buffer_frame_indices_sub": buffer_sub,
        "pv_blocks": pv_blocks,
    }
