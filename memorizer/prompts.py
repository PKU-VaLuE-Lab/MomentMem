"""Shared prompt construction for the Qwen keyframe-decision task."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

# Answer vocabulary (single tokens in the Qwen tokenizer). The order is fixed.
ANSWER_YES = "Yes"
ANSWER_NO = "No"
CHOICES: List[str] = [ANSWER_YES, ANSWER_NO]

# Prompt layout version stored with checkpoints and checked during evaluation and serving.
PROMPT_LAYOUT_VERSION = 4

CAMERA_MODES = ("overview", "overview+wrist")
IMAGE_PREPROCESS_MODES = ("native_dynamic", "square_resize")
LEGACY_SQUARE_IMAGE_SIZE = 224


def image_size_from_config(config: Dict[str, Any]) -> int | None:
    """Validate and resolve the image contract in a checkpoint sidecar."""
    if "image_size" not in config:
        raise ValueError(
            "checkpoint sidecar is missing mandatory image contract field: image_size"
        )
    raw_size = config["image_size"]
    if "image_preprocess" not in config:
        if type(raw_size) is int and raw_size == LEGACY_SQUARE_IMAGE_SIZE:
            return LEGACY_SQUARE_IMAGE_SIZE
        raise ValueError(
            "legacy checkpoint sidecar without image_preprocess is supported only when "
            f"image_size={LEGACY_SQUARE_IMAGE_SIZE}, got {raw_size!r}."
        )

    mode = config["image_preprocess"]
    if mode not in IMAGE_PREPROCESS_MODES:
        raise ValueError(
            f"image_preprocess must be one of {IMAGE_PREPROCESS_MODES}, got {mode!r}."
        )
    if mode == "native_dynamic":
        if raw_size is not None:
            raise ValueError("native_dynamic requires image_size=null.")
        return None
    if type(raw_size) is not int or raw_size <= 0:
        raise ValueError(
            f"square_resize requires image_size to be a positive integer, got {raw_size!r}."
        )
    return raw_size


def normalize_camera_mode(value: str, *, allow_auto: bool = False) -> str:
    mode = str(value).strip().lower()
    allowed = set(CAMERA_MODES) | ({"auto"} if allow_auto else set())
    if mode not in allowed:
        raise ValueError(f"camera mode must be one of {sorted(allowed)}, got {value!r}.")
    return mode


def camera_mode_from_config(config: Dict[str, Any], default: str | None = None) -> str | None:
    """Resolve new and legacy sidecar camera fields, rejecting contradictory metadata."""
    candidates: List[tuple[str, str]] = []
    if config.get("camera_mode") is not None:
        candidates.append(("camera_mode", normalize_camera_mode(config["camera_mode"])))
    if config.get("use_wrist") is not None:
        candidates.append(("use_wrist", "overview+wrist" if bool(config["use_wrist"]) else "overview"))
    if config.get("num_cameras") is not None:
        nc = int(config["num_cameras"])
        if nc not in (1, 2):
            raise ValueError(f"num_cameras must be 1 or 2, got {nc}.")
        candidates.append(("num_cameras", "overview+wrist" if nc == 2 else "overview"))
    if config.get("camera_keys") is not None:
        keys = list(config["camera_keys"])
        if len(keys) not in (1, 2):
            raise ValueError(f"camera_keys must contain one or two entries, got {keys}.")
        key_mode = "overview+wrist" if len(keys) == 2 else "overview"
        if "wrist" in str(keys[0]).lower() or (len(keys) == 2 and "wrist" not in str(keys[1]).lower()):
            raise ValueError(f"camera_keys must be ordered overview first, wrist second: {keys}.")
        candidates.append(("camera_keys", key_mode))
    if candidates:
        modes = {mode for _, mode in candidates}
        if len(modes) != 1:
            raise ValueError(f"Contradictory camera fields in prompt config: {candidates}.")
        return candidates[0][1]
    return normalize_camera_mode(default) if default is not None else None

# Placeholder mapping: dataset `tasks`/instruction shorthand -> richer description.
ENV_NAME_TO_TASK_SPECIALIZED_DESC: Dict[str, str] = {}

# Redundancy-aware prompt.
VLM_SYSTEM_PROMPT = (
    "You are a robotic memory-keyframe selector. "
    "Each timestep contains two synchronized images in this fixed order: overview camera first, "
    "wrist close-up camera second. "
    "You maintain a COMPACT memory of keyframes that should capture all task-relevant information "
    "with NO redundancy. Given the task description, the keyframes already stored in memory (history, "
    "in chronological order), and the current timestep, decide whether to ADD the current timestep as "
    "a new keyframe. Add it ONLY if it reveals NEW task-relevant information that is not already "
    "captured by the existing keyframes. If the current observation is redundant — it shows "
    "essentially the same information as an existing keyframe and adds nothing new beyond them — do "
    "NOT add it. Compare the current observation against the history keyframes carefully (textures, object "
    "positions, task progress), not pixel-level noise. "
    'Answer strictly with one token: "Yes" (add as a new keyframe) or "No" (redundant / no new information).'
)

# Final question shared by training and inference.
VLM_USER_QUESTION = (
    "Question:\n"
    "Does the current timestep provide NEW task-relevant information beyond the keyframes already in "
    'memory? Answer "Yes" only if it should be added as a new keyframe; answer "No" if it is redundant '
    "with the existing keyframes (no new information)."
)


# Camera-dependent prompt text.
_CAMERA_SENTENCE_2 = (
    "Each timestep contains two synchronized images in this fixed order: overview camera first, "
    "wrist close-up camera second. "
)


def _camera_sentence(num_cameras: int) -> str:
    nc = int(num_cameras)
    if nc == 1:
        return "Each timestep contains one image from the overview camera. "
    if nc == 2:
        return _CAMERA_SENTENCE_2
    raise ValueError(f"num_cameras must be 1 (overview) or 2 (overview+wrist), got {nc}.")


def system_prompt(num_cameras: int = 2) -> str:
    """Return a prompt whose camera description exactly matches the checkpoint layout."""
    sentence = _camera_sentence(num_cameras)  # validates even when num_cameras == 2
    base = VLM_SYSTEM_PROMPT  # the canonical 2-camera system prompt
    if int(num_cameras) == 2:
        return base
    return base.replace(_CAMERA_SENTENCE_2, sentence, 1)


def resolve_task_specific_desc(task_field: str) -> str:
    key = str(task_field).strip()
    return ENV_NAME_TO_TASK_SPECIALIZED_DESC.get(key, key)


def _text_part(s: str) -> Dict[str, str]:
    return {"type": "text", "text": s}


def _image_group(count: int) -> Dict[str, int]:
    """Marker for `count` image slots the caller fills, in order, with its own image dicts."""
    return {"type": "images", "count": int(count)}


def _buffer_label(flag: bool, idx: int | None) -> str:
    # Absolute frame indices are bookkeeping data and must not enter the prompt.
    del idx
    return f"({'keyframe' if bool(flag) else 'not a keyframe'})"


def build_user_parts(
    task_field: str,
    history_count: int,
    num_cameras: int,
    buffer_keyframe_flags: Sequence[bool] | None = None,
    buffer_frame_indices: Sequence[int] | None = None,
) -> List[Dict[str, Any]]:
    """Ordered user-message parts, with each text label placed next to its image group."""
    nc = int(num_cameras)
    camera_sentence = _camera_sentence(nc).strip()
    flags = [bool(x) for x in (buffer_keyframe_flags or [])]
    # Callers use frame indices to retrieve images; prompt construction ignores them.
    del buffer_frame_indices
    parts: List[Dict[str, Any]] = []

    header = (
        "Task description:\n"
        f"{resolve_task_specific_desc(task_field)}\n\n"
        f"Camera input: {camera_sentence}\n\n"
    )
    if history_count > 0:
        parts.append(_text_part(header + "Keyframes already stored in memory (history, in chronological order):"))
        parts.append(_image_group(history_count * nc))
    else:
        parts.append(_text_part(header + "No long-term keyframes stored in memory yet."))

    if flags:
        parts.append(_text_part(
            "\nRecent timesteps just before the current one (oldest to newest), each tagged with "
            'whether it is already a keyframe. A frame tagged "keyframe" is part of the memory too '
            "and is not repeated in the history above:"))
        for flag in flags:
            parts.append(_image_group(nc))
            parts.append(_text_part(" " + _buffer_label(flag, None)))

    parts.append(_text_part("\nCurrent timestep (decide whether to add THIS one):"))
    parts.append(_image_group(nc))
    parts.append(_text_part("\n\n" + VLM_USER_QUESTION))
    return parts


def render_user_content(parts: Sequence[Dict[str, Any]], image_items: Sequence[Any]) -> List[Dict[str, Any]]:
    """Expand ``parts`` into a flat content list, filling each image slot from ``image_items`` in
    order.
    """
    image_items = list(image_items)
    expected = sum(int(p["count"]) for p in parts if p.get("type") == "images")
    if len(image_items) != expected:
        raise ValueError(
            f"render_user_content: got {len(image_items)} images but the prompt layout expects "
            f"{expected} (history + buffer + current image slots). Image/prompt structure has drifted."
        )
    it = iter(image_items)
    content: List[Dict[str, Any]] = []
    for part in parts:
        if part.get("type") == "images":
            for _ in range(int(part["count"])):
                content.append(next(it))
        else:
            content.append({"type": "text", "text": part["text"]})
    return content
