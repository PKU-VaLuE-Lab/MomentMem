#!/usr/bin/env python
"""Offline single-trajectory keyframe inference against a vLLM server.

Usage:
  python inference.py \
      --overview-video  /path/episode_000000_image.mp4 \
      --wrist-video     /path/episode_000000_wrist_image.mp4 \
      --instruction "Read each cube's texture, then pair the two matching cubes." \
      --model qwen3vl-keyframe --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
from PIL import Image

_PROJ_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from prompts import (  # noqa: E402
    ANSWER_NO,
    ANSWER_YES,
    CHOICES,
    PROMPT_LAYOUT_VERSION,
    build_user_parts,
    camera_mode_from_config,
    image_size_from_config,
    render_user_content,
    system_prompt,
)

from openai import OpenAI  # noqa: E402


# Frame handling.
def sample_frames(
    video_path: str,
    stride: int,
    max_subframes: Optional[int],
    image_size: Optional[int] = 224,
) -> List[Image.Image]:
    """Decode every ``stride``-th frame and apply the SFT checkpoint's image contract."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames: List[Image.Image] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            if image_size is not None and image.size != (image_size, image_size):
                image = image.resize((image_size, image_size), Image.BILINEAR)
            frames.append(image)
            if max_subframes is not None and len(frames) >= max_subframes:
                break
        idx += 1
    cap.release()
    return frames


def pil_to_data_url(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# Build messages with the training-time layout.
def build_messages(
    instruction: str,
    history_views: List[List[Image.Image]],
    buffer_views: List[List[Image.Image]],
    buffer_keyframe_flags: List[bool],
    buffer_frame_indices: List[int],
    current_views: List[Image.Image],
) -> List[dict]:
    num_cameras = len(current_views)
    history_count = len(history_views)

    image_items = []
    for views in history_views:          # each history keyframe timestep
        for im in views:                 # overview, then wrist
            image_items.append({"type": "image_url", "image_url": {"url": pil_to_data_url(im)}})
    for views in buffer_views:           # recent buffer timesteps (oldest to newest)
        for im in views:                 # overview, then wrist
            image_items.append({"type": "image_url", "image_url": {"url": pil_to_data_url(im)}})
    for im in current_views:             # current timestep views
        image_items.append({"type": "image_url", "image_url": {"url": pil_to_data_url(im)}})

    user_parts = build_user_parts(
        instruction,
        history_count,
        num_cameras,
        buffer_keyframe_flags=buffer_keyframe_flags,
        buffer_frame_indices=buffer_frame_indices,
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt(num_cameras)}]},
        {"role": "user", "content": render_user_content(user_parts, image_items)},
    ]


# Run one constrained Yes/No decision.
def decide(client: OpenAI, model: str, messages: List[dict]):
    """Return (answer, p_yes). Output is constrained to exactly {"Yes","No"}."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2,
        temperature=0.0,
        logprobs=True,
        top_logprobs=20,
        extra_body={"guided_choice": CHOICES},   # <-- constrained decoding: only Yes/No
    )
    choice = resp.choices[0]
    content = (choice.message.content or "").strip()
    answer = ANSWER_YES if content.lower().startswith("yes") else ANSWER_NO

    # Confidence from logprobs, looking ONLY at the Yes / No tokens.
    p_yes = 1.0 if answer == ANSWER_YES else 0.0
    try:
        top = choice.logprobs.content[0].top_logprobs
        lp = {t.token.strip().lower(): t.logprob for t in top}
        if "yes" in lp or "no" in lp:
            my = math.exp(lp["yes"]) if "yes" in lp else 0.0
            mn = math.exp(lp["no"]) if "no" in lp else 0.0
            if (my + mn) > 0:
                p_yes = my / (my + mn)
    except Exception:
        pass
    return answer, p_yes


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Offline keyframe inference (constrained Yes/No)")
    ap.add_argument("--overview-video", type=str, required=True,
                    help="Overview (base) camera video of one episode.")
    ap.add_argument("--wrist-video", type=str, default=None,
                    help="Wrist video synchronized with --overview-video; required only in "
                    "overview+wrist mode.")
    ap.add_argument(
        "--camera-mode", choices=["auto", "overview", "overview+wrist"], default="auto",
        help="Camera layout. auto uses --prompt-config when available, otherwise infers it from "
        "whether --wrist-video was supplied.",
    )
    ap.add_argument(
        "--prompt-config", default=None,
        help="Training keyframe_prompt_config.json; defaults to MODEL/keyframe_prompt_config.json "
        "when --model is a local checkpoint directory.",
    )
    ap.add_argument("--instruction", type=str, required=True,
                    help="Task description (INSTRUCTION_TASK; same style as the dataset `tasks` field).")
    ap.add_argument("--memory-instruction", type=str, default=None,
                    help="INSTRUCTION_MEMORY appended to the task ('what to remember'). Pass it only if "
                    "the served VLM was trained with --add-memory-instruction, to match training.")
    ap.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    ap.add_argument("--api-key", type=str, default="EMPTY")
    ap.add_argument("--model", type=str, default="qwen3vl-keyframe")
    ap.add_argument("--video-fps", type=int, default=20)
    ap.add_argument("--sample-fps", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap the number of sub-sampled timesteps (for a quick demo).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="If set, mark keyframe when P(Yes) > threshold instead of argmax.")
    ap.add_argument("--max-history", type=int, default=48,
                    help="Cap kept keyframes in the prompt (most recent first-in-memory history) so "
                    "the image count stays under the server's --limit-mm-per-prompt. 0 means no history.")
    ap.add_argument("--buffer-size", type=int, default=10,
                    help="Number of recent sampled timesteps before the current frame to include. "
                    "Keyframes inside this buffer are marked there and not repeated in history.")
    ap.add_argument("--image-size", type=int, default=224,
                    help="Resize every camera frame to the SFT image size before sending it.")
    ap.add_argument(
        "--native-image-resolution",
        action="store_true",
        help="Keep native frame dimensions and use Qwen dynamic resolution when no checkpoint "
        "sidecar overrides the image contract.",
    )
    ap.add_argument("--output", type=str, default=None, help="Optional path to write a JSON result.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.native_image_resolution:
        args.image_size = None

    prompt_path = args.prompt_config
    if prompt_path is None and os.path.isdir(args.model):
        candidate = os.path.join(args.model, "keyframe_prompt_config.json")
        prompt_path = candidate if os.path.exists(candidate) else None
    cfg = {}
    if prompt_path:
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt config not found: {prompt_path}")
        cfg = json.loads(Path(prompt_path).read_text())
        layout = cfg.get("prompt_layout")
        if layout is None:
            raise ValueError(f"{prompt_path}: missing required prompt_layout.")
        if int(layout) != int(PROMPT_LAYOUT_VERSION):
            raise ValueError(
                f"{prompt_path}: prompt_layout={layout}, but this code builds "
                f"layout {PROMPT_LAYOUT_VERSION}."
            )
        for key in ("video_fps", "sample_fps", "buffer_size", "max_history"):
            if key in cfg:
                setattr(args, key, int(cfg[key]))
        try:
            args.image_size = image_size_from_config(cfg)
        except ValueError as exc:
            raise ValueError(f"Invalid prompt config {prompt_path}: {exc}") from exc

    try:
        checkpoint_mode = camera_mode_from_config(cfg) if cfg else None
    except ValueError as exc:
        raise ValueError(f"Invalid prompt config {prompt_path}: {exc}") from exc
    if args.camera_mode != "auto" and checkpoint_mode and args.camera_mode != checkpoint_mode:
        raise ValueError(
            f"--camera-mode={args.camera_mode} contradicts checkpoint camera_mode={checkpoint_mode}."
        )
    camera_mode = (
        checkpoint_mode
        or (args.camera_mode if args.camera_mode != "auto" else None)
        or ("overview+wrist" if args.wrist_video else "overview")
    )
    if camera_mode == "overview+wrist" and not args.wrist_video:
        raise ValueError("overview+wrist mode requires --wrist-video.")
    if camera_mode == "overview" and args.wrist_video:
        print("[camera] overview mode: --wrist-video is intentionally ignored.")

    trained_with_memory = cfg.get("add_memory_instruction") if cfg else None
    if trained_with_memory is True and not args.memory_instruction:
        raise ValueError(
            "The checkpoint was trained with INSTRUCTION_MEMORY; pass --memory-instruction."
        )
    if trained_with_memory is False and args.memory_instruction:
        raise ValueError(
            "The checkpoint was trained without INSTRUCTION_MEMORY; do not pass --memory-instruction."
        )

    if args.video_fps % args.sample_fps != 0:
        raise ValueError("video_fps must be divisible by sample_fps")
    instruction = args.instruction
    if args.memory_instruction:
        instruction = f"{instruction}\n{args.memory_instruction}"
    stride = args.video_fps // args.sample_fps

    overview = sample_frames(args.overview_video, stride, args.max_frames, args.image_size)
    wrist = (
        sample_frames(args.wrist_video, stride, args.max_frames, args.image_size)
        if camera_mode == "overview+wrist" else None
    )
    T = min(len(overview), len(wrist)) if wrist is not None else len(overview)

    def views_at(t: int) -> List[Image.Image]:
        return [overview[t]] + ([wrist[t]] if wrist is not None else [])

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    history: List[int] = []
    records = []
    print(f"[run] {T} timesteps, camera_mode={camera_mode}, model={args.model}")

    for t in range(T):
        buffer_idx = list(range(max(0, t - args.buffer_size), t)) if args.buffer_size > 0 else []
        buffer_set = set(buffer_idx)
        buffer_flags = [k in set(history) for k in buffer_idx]
        # Keep recent history within the server image limit.
        hist_outside_buffer = [k for k in history if k not in buffer_set]
        if len(hist_outside_buffer) > args.max_history:
            kept = hist_outside_buffer[-args.max_history:] if args.max_history > 0 else []
        else:
            kept = hist_outside_buffer
        msgs = build_messages(
            instruction,
            [views_at(k) for k in kept],
            [views_at(k) for k in buffer_idx],
            buffer_flags,
            buffer_idx,
            views_at(t),
        )
        answer, p_yes = decide(client, args.model, msgs)
        is_kf = (p_yes > args.threshold) if args.threshold is not None else (answer == ANSWER_YES)
        if is_kf:
            history.append(t)
        records.append({
            "t": t,
            "answer": answer,
            "p_yes": round(p_yes, 4),
            "keyframe": bool(is_kf),
            "history": kept,
            "buffer": [{"t": k, "keyframe": bool(f)} for k, f in zip(buffer_idx, buffer_flags)],
        })
        mark = "KEYFRAME" if is_kf else "        "
        print(f"  t={t:<4} -> {answer:<3} P(Yes)={p_yes:.3f}  {mark}  (history={len(history)})")

    print(f"\n[result] selected {len(history)} keyframes at sub-timesteps: {history}")
    print(f"         original frame indices: {[t * stride for t in history]}")

    if args.output:
        out = {
            "overview_video": args.overview_video,
            "wrist_video": args.wrist_video,
            "instruction": args.instruction,
            "memory_instruction": args.memory_instruction,
            "effective_instruction": instruction,
            "camera_mode": camera_mode,
            "num_cameras": 2 if wrist is not None else 1,
            "stride": stride,
            "image_size": args.image_size,
            "image_preprocess": "native_dynamic" if args.image_size is None else "square_resize",
            "max_history": args.max_history,
            "buffer_size": args.buffer_size,
            "keyframe_sub_indices": history,
            "keyframe_frame_indices": [t * stride for t in history],
            "frames": records,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"[result] wrote {args.output}")


if __name__ == "__main__":
    main()
