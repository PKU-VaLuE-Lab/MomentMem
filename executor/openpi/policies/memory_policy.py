import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_memory_example(
    num_keyframes: int = 8, *, keyframe_use_wrist_camera: bool = True
) -> dict:
    """Random input example for the memory (history-keyframe) policy."""
    img = lambda: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)  # noqa: E731
    example = {
        "observation/state": np.random.rand(8),
        "observation/image": img(),
        "observation/wrist_image": img(),
        # Past keyframe views (oldest..newest), zero-padded; keyframe_mask flags the real ones.
        "keyframe_image_base": np.stack([img() for _ in range(num_keyframes)]),
        "keyframe_mask": np.ones((num_keyframes,), dtype=bool),
        "prompt": "do something",
    }
    if keyframe_use_wrist_camera:
        example["keyframe_image_wrist"] = np.stack(
            [img() for _ in range(num_keyframes)]
        )
    return example


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class MemoryInputs(transforms.DataTransformFn):
    """Builds the model image dict with current + history-keyframe camera views."""

    model_type: _model.ModelType
    num_keyframes: int
    keyframe_use_wrist_camera: bool = True

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        image = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image,
            "right_wrist_0_rgb": np.zeros_like(base_image),
        }
        mask_pad = np.False_
        image_mask = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": mask_pad,
        }

        kf_base = data.get("keyframe_image_base")
        kf_wrist = (
            data.get("keyframe_image_wrist")
            if self.keyframe_use_wrist_camera
            else None
        )
        kf_mask = data.get("keyframe_mask")
        # All-or-nothing for the configured camera mode. None at all means no
        # history; overview-only mode neither requires nor emits wrist slots.
        fields = [kf_base, kf_mask]
        field_names = ["keyframe_image_base", "keyframe_mask"]
        if self.keyframe_use_wrist_camera:
            fields.insert(1, kf_wrist)
            field_names.insert(1, "keyframe_image_wrist")
        provided = [value is not None for value in fields]
        have_kf = all(provided)
        if have_kf:
            kf_mask = np.asarray(kf_mask).reshape(-1)
            lengths = [len(kf_base), len(kf_mask)]
            if self.keyframe_use_wrist_camera:
                lengths.insert(1, len(kf_wrist))

        for i in range(self.num_keyframes):
            valid = have_kf and i < len(kf_base) and bool(kf_mask[i])
            b = (
                _parse_image(kf_base[i])
                if valid
                else np.zeros_like(base_image)
            )
            image[f"kf{i}_base_0_rgb"] = b
            image_mask[f"kf{i}_base_0_rgb"] = np.bool_(valid)
            if self.keyframe_use_wrist_camera:
                w = (
                    _parse_image(kf_wrist[i])
                    if valid
                    else np.zeros_like(wrist_image)
                )
                image[f"kf{i}_left_wrist_0_rgb"] = w
                image_mask[f"kf{i}_left_wrist_0_rgb"] = np.bool_(valid)

        inputs = {
            "state": data["observation/state"],
            "image": image,
            "image_mask": image_mask,
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class MemoryOutputs(transforms.DataTransformFn):
    """Return the first `action_dim` actions (rest is padding)."""

    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
