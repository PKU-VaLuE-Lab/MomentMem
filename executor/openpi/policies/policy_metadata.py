from collections.abc import Mapping
from typing import Any


def build_policy_metadata(policy_metadata: Mapping[str, Any] | None, model_config: Any) -> dict[str, Any]:
    """Build server metadata without overriding user-specified values."""
    metadata = dict(policy_metadata or {})
    num_keyframes = getattr(model_config, "num_keyframes", None)
    if num_keyframes is not None:
        metadata.setdefault("num_keyframes", int(num_keyframes))
    keyframe_use_wrist = getattr(
        model_config, "keyframe_use_wrist_camera", None
    )
    if keyframe_use_wrist is not None:
        metadata.setdefault(
            "keyframe_use_wrist_camera", bool(keyframe_use_wrist)
        )
        metadata.setdefault(
            "keyframe_camera_mode",
            "overview+wrist" if keyframe_use_wrist else "overview",
        )
    return metadata
