from __future__ import annotations

import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Mapping

import numpy as np


logger = logging.getLogger(__name__)

POLICY_STATE_DIM = 8
POLICY_STATE_REPRESENTATION = "absolute_joint7+physical_gripper1"
POLICY_ACTION_SPACE = "rel_ee"
POLICY_ACTION_DIM = 10
POLICY_ACTION_REPRESENTATION = (
    "base_camera_relative_xyz3+relative_rot6d6+gripper_command1"
)
POLICY_CONTROL_MODE_PROMPT = "<control_mode> end effector <control_mode> "
ACTION_SPACE_DIMS = {
    "delta_ee": 7,
    "rel_ee": 10,
    "abs_joint": 8,
    "delta_joint": 8,
}

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MEMORY_AGENT_ROOT = Path(__file__).resolve().parent
_OPENPI_ROOT = _REPO_ROOT / "executor"
_OPENPI_CLIENT_SRC = _REPO_ROOT / "executor" / "packages" / "openpi-client" / "src"
for _path in (_MEMORY_AGENT_ROOT, _OPENPI_ROOT, _OPENPI_CLIENT_SRC):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

@dataclass(frozen=True)
class RetrievalResult:
    """One online memorizer decision."""

    step_idx: int
    processed: bool
    is_keyframe: bool
    probability: float | None
    threshold: float


class MemoryAgent:
    """Compose a VLM memorizer with a memory-conditioned VLA executor."""

    def __init__(
        self,
        *,
        vlm_client: Any | None = None,
        vla_policy: Any | None = None,
        openpi_config_name: str = "pi05_memory",
        openpi_checkpoint_dir: str | os.PathLike[str] | None = None,
        retrieval_stride: int | None = None,
        num_keyframes: int | None = None,
        threshold: float | None = None,
        task_thresholds: Mapping[int, float] | None = None,
        allow_missing_wrist: bool = False,
        default_prompt: str | None = None,
        openpi_sample_kwargs: dict[str, Any] | None = None,
        openpi_pytorch_device: str | None = None,
        policy_action_space: str | None = None,
        env_action_space: str = "auto",
        action_adapter: str | None = "auto",
        vlm_buffer_size: int = 0,
        add_memory_instruction: bool = False,
        default_memory: str | None = None,
    ) -> None:
        self.repo_root = _REPO_ROOT
        self.allow_missing_wrist = bool(allow_missing_wrist)
        self.default_prompt = default_prompt
        # Only System 2 receives the optional memory instruction.
        self.add_memory_instruction = bool(add_memory_instruction)
        self.default_memory = default_memory
        self.vlm_buffer_size = int(vlm_buffer_size)
        # Warn-once bookkeeping (process lifetime, not per episode).
        self._warned_unknown_task_ids: set[int] = set()

        self.vlm_client = vlm_client
        self.use_wrist_camera = bool(getattr(vlm_client, "use_wrist", True))
        self.retrieval_stride = int(4 if retrieval_stride is None else retrieval_stride)
        self.default_threshold = float(0.5 if threshold is None else threshold)
        self.task_thresholds = {int(k): float(v) for k, v in (task_thresholds or {}).items()}

        self.vla_policy = vla_policy or self._load_openpi_policy(
            config_name=openpi_config_name,
            checkpoint_dir=openpi_checkpoint_dir,
            sample_kwargs=openpi_sample_kwargs,
            default_prompt=default_prompt,
            pytorch_device=openpi_pytorch_device,
        )

        configured_kf_wrist = self._infer_keyframe_use_wrist_camera(
            openpi_config_name
        )
        lower_metadata = dict(
            getattr(self.vla_policy, "metadata", {}) or {}
        )
        metadata_kf_wrist = lower_metadata.get("keyframe_use_wrist_camera")
        if metadata_kf_wrist is not None:
            metadata_kf_wrist = bool(metadata_kf_wrist)
            if metadata_kf_wrist != configured_kf_wrist:
                raise ValueError(
                    "lower-policy keyframe camera mismatch: openpi config "
                    f"{openpi_config_name!r} has "
                    f"keyframe_use_wrist_camera={configured_kf_wrist}, but "
                    f"policy metadata reports {metadata_kf_wrist}."
                )
        self.vla_keyframe_use_wrist_camera = configured_kf_wrist

        self.policy_action_space = self._resolve_policy_action_space(
            openpi_config_name, policy_action_space
        )
        self.env_action_space = self._resolve_env_action_space(
            openpi_config_name=openpi_config_name,
            policy_action_space=self.policy_action_space,
            env_action_space=env_action_space,
        )
        self.action_adapter = self._resolve_action_adapter(
            openpi_config_name=openpi_config_name,
            policy_action_space=self.policy_action_space,
            env_action_space=self.env_action_space,
            action_adapter=action_adapter,
        )
        inferred_num_keyframes = self._infer_num_keyframes(openpi_config_name)
        self.num_keyframes = int(num_keyframes or inferred_num_keyframes or 8)
        # The lower policy reads a FIXED number of keyframe slots (oldest-first). A larger agent
        # memory would silently drop the NEWEST keyframes, so refuse the mismatch outright.
        if inferred_num_keyframes and self.num_keyframes > int(inferred_num_keyframes):
            raise ValueError(
                f"num_keyframes={self.num_keyframes} exceeds the lower policy's fixed keyframe "
                f"slots ({inferred_num_keyframes} in config {openpi_config_name!r}); the policy "
                f"would read only the OLDEST {inferred_num_keyframes} and silently drop the "
                f"newest keyframes. Pass num_keyframes <= {inferred_num_keyframes}."
            )

        self.reset()

    def reset(self) -> None:
        """Clear episode-local visual memory."""

        self.step_idx = 0
        self.keyframes: Deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.num_keyframes)
        self.vlm_substep_idx = 0
        self.vlm_keyframes: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.vlm_recent_frames: Deque[tuple[int, np.ndarray, np.ndarray, bool]] = deque(
            maxlen=self.vlm_buffer_size if self.vlm_buffer_size > 0 else 0
        )
        self.last_retrieval: RetrievalResult | None = None

    def infer(
        self,
        observation: Mapping[str, Any],
        *,
        prompt: str | None = None,
        memory: str | None = None,
        task_id: int | None = None,
        update_memory: bool = True,
        return_info: bool = False,
        noise: np.ndarray | None = None,
        env_step: int | None = None,
    ) -> np.ndarray | dict[str, Any]:
        """Run one online memory update and return the pi0.5 action chunk."""

        vla_prompt, vlm_prompt = self._resolve_routed_prompts(observation, prompt=prompt, memory=memory)
        # Build the VLA input before retrieval so memory contains only past frames.
        vla_observation = self.build_vla_observation(observation, prompt=vla_prompt)
        retrieval = (
            self.update_memory(observation, prompt=vlm_prompt, task_id=task_id, env_step=env_step)
            if update_memory
            else None
        )
        policy_output = self.vla_policy.infer(vla_observation, noise=noise)
        raw_actions = np.asarray(policy_output["actions"], dtype=np.float32)
        actions = self._adapt_policy_actions(raw_actions, observation)

        if return_info:
            return {
                "actions": actions,
                "raw_actions": raw_actions,
                "retrieval": retrieval,
                "num_keyframes": len(self.keyframes),
                "policy_action_space": self.policy_action_space,
                "env_action_space": self.env_action_space,
                "action_adapter": self.action_adapter,
                "policy_output": policy_output,
                "vla_observation": vla_observation,
            }
        return actions

    __call__ = infer

    def observe(
        self,
        observation: Mapping[str, Any],
        *,
        prompt: str | None = None,
        memory: str | None = None,
        task_id: int | None = None,
        env_step: int | None = None,
    ) -> RetrievalResult:
        """Update System 2 from one environment observation without running the lower VLA."""

        _vla_prompt, vlm_prompt = self._resolve_routed_prompts(
            observation, prompt=prompt, memory=memory
        )
        return self.update_memory(
            observation,
            prompt=vlm_prompt,
            task_id=task_id,
            env_step=env_step,
        )

    def update_memory(
        self,
        observation: Mapping[str, Any],
        *,
        prompt: str | None = None,
        task_id: int | None = None,
        env_step: int | None = None,
    ) -> RetrievalResult:
        """Update memory at the configured observation cadence."""
        return self._update_memory_vlm(observation, prompt=prompt, task_id=task_id, env_step=env_step)

    def _update_memory_vlm(
        self,
        observation: Mapping[str, Any],
        *,
        prompt: str | None = None,
        task_id: int | None = None,
        env_step: int | None = None,
    ) -> RetrievalResult:
        """Upper-level keyframe decision via the remote VLM (vLLM HTTP), updating keyframe
        memory.
        """
        prompt_text = self._resolve_prompt(observation, prompt)
        threshold = self._threshold_for_task(task_id)

        base_image = self._extract_image(observation, "observation/image", required=True)
        wrist_image = self._extract_image(
            observation, "observation/wrist_image", required=not self.allow_missing_wrist
        )
        if wrist_image is None and not getattr(self, "_warned_vlm_zero_wrist", False):
            logger.warning(
                "allow_missing_wrist: the VLM receives an all-zero wrist image, which its SFT "
                "training (real dual-camera frames) never saw. Keyframe decisions on such frames "
                "are unreliable -- use this only for debugging, or serve with --vlm-no-wrist and "
                "a single-camera-trained checkpoint."
            )
            self._warned_vlm_zero_wrist = True

        # Only query the VLM on strided frames (it was trained on sub-sampled frames).
        eff_step = int(env_step) if env_step is not None else self.step_idx
        if eff_step % self.retrieval_stride != 0:
            result = RetrievalResult(
                step_idx=eff_step, processed=False, is_keyframe=False, probability=None, threshold=threshold
            )
            self.last_retrieval = result
            self.step_idx = eff_step + 1
            return result

        # Substep = sampled-timeline index; derive from the true env frame when it is known so
        # the buffer/history indices match the training-time sub-sampled numbering.
        current_substep = (eff_step // self.retrieval_stride) if env_step is not None else int(self.vlm_substep_idx)
        buffer_steps = {int(step) for step, _base, _wrist, _is_kf in self.vlm_recent_frames}
        history = [
            (base, wrist)
            for step, base, wrist in self.vlm_keyframes
            if int(step) not in buffer_steps
        ]
        buffer = [
            (base, wrist, bool(is_kf), int(step))
            for step, base, wrist, is_kf in self.vlm_recent_frames
        ]
        prob = float(
            self.vlm_client.predict(
                current_base=base_image,
                current_wrist=wrist_image,
                history=history,
                buffer=buffer,
                prompt=prompt_text,
            )
        )
        pred = int(prob > threshold)  # strict `>`, matching eval_keyframe(_ar)
        wrist_for_history = wrist_image if wrist_image is not None else np.zeros_like(base_image)
        base_u8 = self._ensure_hwc_uint8(base_image)
        wrist_u8 = self._ensure_hwc_uint8(wrist_for_history)
        if pred:
            self.vlm_keyframes.append((current_substep, base_u8, wrist_u8))
            self._append_keyframe(base_image, wrist_for_history)
        if self.vlm_buffer_size > 0:
            self.vlm_recent_frames.append((current_substep, base_u8, wrist_u8, bool(pred)))

        result = RetrievalResult(
            step_idx=eff_step, processed=True, is_keyframe=bool(pred), probability=prob, threshold=threshold
        )
        self.last_retrieval = result
        self.vlm_substep_idx = current_substep + 1
        self.step_idx = eff_step + 1
        return result

    def build_vla_observation(self, observation: Mapping[str, Any], *, prompt: str | None = None) -> dict[str, Any]:
        """Build the unbatched observation consumed by openpi memory policy."""

        base_image = self._extract_image(observation, "observation/image", required=True)
        wrist_image = self._extract_image(observation, "observation/wrist_image", required=not self.allow_missing_wrist)
        if wrist_image is None:
            wrist_image = np.zeros_like(base_image)

        state = np.asarray(
            self._extract_value(observation, "observation/state", required=True),
            dtype=np.float32,
        )
        keyframe_base, keyframe_wrist, keyframe_mask = self._keyframe_arrays(
            base_image, wrist_image
        )
        result = {
            "observation/image": base_image,
            "observation/wrist_image": wrist_image,
            "observation/state": state,
            "keyframe_image_base": keyframe_base,
            "keyframe_mask": keyframe_mask,
            "prompt": self._resolve_prompt(observation, prompt),
        }
        if keyframe_wrist is not None:
            result["keyframe_image_wrist"] = keyframe_wrist
        return result

    @property
    def keyframe_count(self) -> int:
        return len(self.keyframes)


    def _load_openpi_policy(
        self,
        *,
        config_name: str,
        checkpoint_dir: str | os.PathLike[str] | None,
        sample_kwargs: dict[str, Any] | None,
        default_prompt: str | None,
        pytorch_device: str | None,
    ) -> Any:
        if checkpoint_dir is None:
            raise ValueError("Either vla_policy or openpi_checkpoint_dir must be provided.")

        from openpi.policies import policy_config
        from openpi.training import config as openpi_config

        train_config = openpi_config.get_config(config_name)
        return policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            sample_kwargs=sample_kwargs,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
        )

    def _infer_num_keyframes(self, config_name: str) -> int | None:
        try:
            from openpi.training import config as openpi_config

            train_config = openpi_config.get_config(config_name)
            return int(getattr(train_config.model, "num_keyframes", 0) or getattr(train_config.data, "num_keyframes", 0))
        except Exception:
            return None

    def _infer_keyframe_use_wrist_camera(self, config_name: str) -> bool:
        """Read the lower VLA history-camera contract from its train config."""

        try:
            from openpi.training import config as openpi_config

            train_config = openpi_config.get_config(config_name)
        except Exception:
            # Legacy/custom policies predate the camera field and used both
            # keyframe views. Preserve that contract as the safe fallback.
            return True
        return bool(getattr(train_config.model, "keyframe_use_wrist_camera", True))

    def _resolve_policy_action_space(self, config_name: str, policy_action_space: str | None) -> str:
        if policy_action_space not in (None, "auto"):
            result = str(policy_action_space)
            if result not in ACTION_SPACE_DIMS:
                raise ValueError(f"unknown policy_action_space={result!r}")
            return result

        try:
            from openpi.training import config as openpi_config

            train_config = openpi_config.get_config(config_name)
            data = train_config.data
            action_key = str(getattr(data, "action_key", "action"))
            action_dim = int(getattr(data, "action_dim", -1))
            if action_key == "action_native":
                if action_dim != 7:
                    raise ValueError(f"action_native must be 7-D, got {action_dim}")
                return "delta_ee"
            if action_key == "action_rel_ee":
                if action_dim != 10:
                    raise ValueError(f"action_rel_ee must be 10-D, got {action_dim}")
                return "rel_ee"
            if action_key == "action_abs_joint":
                return "abs_joint"
            if action_key == "action_delta_joint":
                return "delta_joint"
            raise ValueError(
                f"cannot infer policy action space from action_key={action_key!r}, "
                f"action_dim={action_dim}"
            )
        except Exception as error:
            raise ValueError(
                f"Cannot infer the lower policy action contract from OpenPI config "
                f"{config_name!r}; pass policy_action_space explicitly."
            ) from error

    @staticmethod
    def _resolve_env_action_space(
        *,
        openpi_config_name: str,
        policy_action_space: str,
        env_action_space: str | None,
    ) -> str:
        requested = "auto" if env_action_space is None else str(env_action_space)
        if requested == "auto":
            return policy_action_space
        if requested not in set(ACTION_SPACE_DIMS):
            raise ValueError(f"unknown env_action_space={requested!r}")
        return requested

    def _resolve_action_adapter(
        self,
        *,
        openpi_config_name: str,
        policy_action_space: str,
        env_action_space: str,
        action_adapter: str | None,
    ) -> str | None:
        if action_adapter in (None, "none"):
            if policy_action_space != env_action_space:
                raise ValueError(
                    f"policy_action_space={policy_action_space!r} cannot be served as "
                    f"env_action_space={env_action_space!r} with action_adapter={action_adapter!r}."
                )
            return None
        if policy_action_space == env_action_space:
            return None
        raise ValueError(
            f"Unsupported action-space conversion: {policy_action_space!r} -> {env_action_space!r}."
        )

    # Env action spaces whose Box is the NORMALIZED [-1, 1]. pi0.5 unnormalizes with quantile
    # stats (linear extrapolation), so a healthy checkpoint can output values slightly outside
    # the Box (the gripper column trains on exactly +/-1); clip here, in one place, so every
    # client receives Box-valid actions.
    _NORMALIZED_ENV_SPACES = ("delta_ee", "delta_joint")

    def _adapt_policy_actions(self, actions: np.ndarray, observation: Mapping[str, Any]) -> np.ndarray:
        arr = np.asarray(actions, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]

        if self.action_adapter is None:
            adapted = arr
        else:
            raise ValueError(f"Unknown action_adapter: {self.action_adapter!r}")

        if self.env_action_space in self._NORMALIZED_ENV_SPACES:
            adapted = np.clip(adapted, -1.0, 1.0)
        return adapted[0] if single else adapted

    def _threshold_for_task(self, task_id: int | None) -> float:
        if task_id is None:
            return self.default_threshold
        tid = int(task_id)
        if tid not in self.task_thresholds:
            # Per-task thresholds are keyed by stable_task_id(task_name); a client task_name
            # that does not exactly match the training dataset_spec name silently hashes to a
            # different id. When overrides exist but this id has none, that mismatch is the
            # most likely cause -- warn once per id instead of silently using the default.
            if self.task_thresholds and tid not in self._warned_unknown_task_ids:
                logger.warning(
                    "task_id %s has no per-task threshold (known ids: %s); falling back to the "
                    "default %.3f. If a per-task threshold was expected, check that the client's "
                    "task_name exactly matches the training dataset_spec task name.",
                    tid, sorted(self.task_thresholds), self.default_threshold,
                )
                self._warned_unknown_task_ids.add(tid)
            return self.default_threshold
        return float(self.task_thresholds[tid])




    def _append_keyframe(self, base_image: np.ndarray, wrist_image: np.ndarray) -> None:
        self.keyframes.append((self._ensure_hwc_uint8(base_image), self._ensure_hwc_uint8(wrist_image)))

    def _keyframe_arrays(
        self, base_like: np.ndarray, wrist_like: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        base_pad = np.zeros_like(self._ensure_hwc_uint8(base_like))
        wrist_pad = (
            np.zeros_like(self._ensure_hwc_uint8(wrist_like))
            if self.vla_keyframe_use_wrist_camera
            else None
        )
        base_frames: list[np.ndarray] = []
        wrist_frames: list[np.ndarray] = []
        mask = np.zeros((self.num_keyframes,), dtype=bool)

        for i in range(self.num_keyframes):
            if i < len(self.keyframes):
                base, wrist = self.keyframes[i]
                base_frames.append(base)
                if self.vla_keyframe_use_wrist_camera:
                    wrist_frames.append(wrist)
                mask[i] = True
            else:
                base_frames.append(base_pad)
                if wrist_pad is not None:
                    wrist_frames.append(wrist_pad)

        wrist_array = (
            np.stack(wrist_frames, axis=0)
            if self.vla_keyframe_use_wrist_camera
            else None
        )
        return np.stack(base_frames, axis=0), wrist_array, mask

    def _resolve_prompt(self, observation: Mapping[str, Any], prompt: str | None) -> str:
        value = prompt
        if value is None:
            value = self._extract_value(observation, "prompt", required=False)
        if value is None:
            value = self.default_prompt
        if value is None:
            raise ValueError("A task prompt is required: pass prompt=... or include observation['prompt'].")
        if isinstance(value, np.ndarray):
            value = value.item() if value.shape == () else value.tolist()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return str(value)

    def _resolve_routed_prompts(
        self,
        observation: Mapping[str, Any],
        *,
        prompt: str | None,
        memory: str | None,
    ) -> tuple[str, str]:
        """Return the task-only System-1 prompt and the configured System-2 prompt."""

        vla_prompt = self._resolve_prompt(observation, prompt)
        vlm_prompt = self._compose_vlm_prompt(vla_prompt, self._resolve_memory(observation, memory))
        return vla_prompt, vlm_prompt

    def _resolve_memory(self, observation: Mapping[str, Any], memory: str | None) -> str:
        """INSTRUCTION_MEMORY for the VLM prompt. Optional: returns '' when none is provided."""
        value = memory
        if value is None:
            value = self._extract_value(observation, "prompt_memory", required=False)
        if value is None:
            value = self.default_memory
        if value is None:
            return ""
        if isinstance(value, np.ndarray):
            value = value.item() if value.shape == () else value.tolist()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return str(value)

    def _compose_vlm_prompt(self, task_prompt: str, memory_text: str) -> str:
        """VLM (System 2) prompt = task (+ memory instruction if the served VLM was trained with it).
        The lower VLA always receives task-only via build_vla_observation(prompt=vla_prompt)."""
        if self.add_memory_instruction:
            if memory_text:
                return f"{task_prompt}\n{memory_text}"
            # Warn once when a required VLM memory instruction is missing.
            if not getattr(self, "_warned_missing_memory", False):
                logger.warning(
                    "add_memory_instruction=True but no memory instruction was provided "
                    "(observation['prompt_memory'] and default_memory are empty). The VLM prompt "
                    "falls back to TASK-ONLY, which does NOT match how a memory-trained VLM was "
                    "trained. Send obs['prompt_memory']=INSTRUCTION_MEMORY (or set default_memory)."
                )
                self._warned_missing_memory = True
        return task_prompt

    def _extract_image(self, observation: Mapping[str, Any], key: str, *, required: bool) -> np.ndarray | None:
        value = self._extract_value(observation, key, required=required)
        if value is None:
            return None
        return self._ensure_hwc_uint8(value)

    def _extract_value(self, observation: Mapping[str, Any], key: str, *, required: bool) -> Any:
        if key in observation:
            return observation[key]

        cur: Any = observation
        for part in key.split("/"):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break
        if cur is not None:
            return cur

        dotted = key.replace("/", ".")
        if dotted in observation:
            return observation[dotted]

        if required:
            raise KeyError(f"Observation is missing required key {key!r}.")
        return None

    @staticmethod
    def _ensure_hwc_uint8(image: Any) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if np.issubdtype(arr.dtype, np.floating):
            if arr.size and float(np.nanmax(arr)) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)
