"""Common task client for GT replay and real MemoryAgent evaluation."""

from __future__ import annotations

from collections import deque
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

try:
    from ..task_env.action_contract import (
        REPLAY_QUANTIZED_TASKS,
        quantize_native_arm_action,
    )
except ImportError:  # direct eval_client launcher
    _MANISKILL_ROOT = Path(__file__).resolve().parents[1]
    if str(_MANISKILL_ROOT) not in sys.path:
        sys.path.insert(0, str(_MANISKILL_ROOT))
    from task_env.action_contract import (
        REPLAY_QUANTIZED_TASKS,
        quantize_native_arm_action,
    )

from .connection import MemoryAgentConnection, MemoryAgentProtocolError
from .protocol import (
    ACTION_SPACE_TO_CONTROL_MODE,
    ACTION_SPACE_TO_DIM,
    POLICY_ACTION_DIM,
    POLICY_ACTION_REPRESENTATION,
    POLICY_ACTION_SPACE,
    POLICY_CONTRACT,
    POLICY_CONTROL_MODE_PROMPT,
    POLICY_STATE_DIM,
    POLICY_STATE_REPRESENTATION,
    REPLAYABLE_SPACES,
    REPLAY_AUDIT_STATE_DIM,
    REPLAY_AUDIT_STATE_REPRESENTATION,
    packb,
    unpackb,
)


DEFAULT_CAMERA_RESOLUTION = 224
POLICY_DIFFUSION_ACTION_DIM = 32
POLICY_DIFFUSION_ACTION_HORIZON = 10


class _PolicyVideoWriter:
    """Stream overview frames to an atomic H.264 MP4 evaluation artifact."""

    def __init__(self, path: Path, *, fps: int, keyframe_hold_seconds: float) -> None:
        import av

        self.path = Path(path)
        self.fps = int(fps)
        self.hold_frames = int(round(float(keyframe_hold_seconds) * self.fps))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial_path = self.path.with_suffix(self.path.suffix + ".partial.mp4")
        self.partial_path.unlink(missing_ok=True)
        self._container = av.open(str(self.partial_path), "w")
        self._stream = self._container.add_stream(
            "libx264", rate=self.fps, options={"crf": "23", "preset": "veryfast"}
        )
        self._stream.pix_fmt = "yuv420p"
        self._stream.height = DEFAULT_CAMERA_RESOLUTION
        self._stream.width = DEFAULT_CAMERA_RESOLUTION
        self._stream.time_base = Fraction(1, self.fps)
        self._closed = False

    @staticmethod
    def _rgb(source: np.ndarray, *, red_border: bool) -> np.ndarray:
        image = np.asarray(source)
        image = np.clip(image, 0, 255).astype(np.uint8, copy=True)
        if red_border:
            width = 6
            image[:width, :, :] = (255, 0, 0)
            image[-width:, :, :] = (255, 0, 0)
            image[:, :width, :] = (255, 0, 0)
            image[:, -width:, :] = (255, 0, 0)
        return np.ascontiguousarray(image)

    def write(self, source: np.ndarray, *, is_keyframe: bool = False) -> None:
        import av

        image = self._rgb(source, red_border=bool(is_keyframe))
        repeats = self.hold_frames if is_keyframe else 1
        for _ in range(repeats):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in self._stream.encode(frame):
                self._container.mux(packet)

    def close(self, *, commit: bool) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        finally:
            self._container.close()
        if commit:
            self.partial_path.replace(self.path)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _bool(value: Any) -> bool:
    array = _numpy(value).reshape(-1)
    return bool(array[0]) if array.size else False


def _quaternion_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return a proper rotation matrix for one wxyz quaternion."""

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rot6d(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return np.concatenate((rotation[:, 0], rotation[:, 1])).astype(np.float32)


def _matrix_from_rot6d(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(6)
    first = value[:3]
    first_norm = float(np.linalg.norm(first))
    first = first / first_norm
    second = value[3:] - first * float(first @ value[3:])
    second_norm = float(np.linalg.norm(second))
    second = second / second_norm
    return np.stack((first, second, np.cross(first, second)), axis=1)


def _metadata_digest(value: Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    payload = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class BaseTaskClient:
    """Per-task subclasses declare names/instructions; all interfaces stay shared."""

    ENV_ID = ""
    TASK_NAME = ""
    INSTRUCTION_TASK = ""
    INSTRUCTION_MEMORY = ""
    ROBOT_UIDS = "panda"
    # Closed-loop policies that have not succeeded after 2000 control steps are
    # treated as timed out.  Individual callers may still request a different
    # limit explicitly through ``max_steps``.
    DEFAULT_MAX_STEPS = 2000
    MIN_KEYFRAMES = 1
    SUCCESS_INFO_KEY = "success"
    EVALUATION_INFO_KEYS: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        camera_resolution: int = DEFAULT_CAMERA_RESOLUTION,
        shader: str | None = "default",
        env_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.camera_resolution = int(camera_resolution)
        self.shader = "default" if shader in (None, "", "default") else str(shader)
        self.env_kwargs = dict(env_kwargs or {})

    def extra_make_kwargs(self) -> dict[str, Any]:
        return {}

    def policy_controls_action(self, task_env: Any) -> bool:
        """Override for target-independent acquisition phases that block the arm."""

        del task_env
        return True

    def observe_only_action(self, task_env: Any) -> np.ndarray:
        """Local simulator no-op used when no VLA action is requested."""

        agent = self._arm(task_env)
        mode = str(getattr(agent, "control_mode", ""))
        try:
            combined = agent.controller
            arm_controller = combined.controllers["arm"]
            gripper_controller = combined.controllers["gripper"]
        except Exception as error:
            raise RuntimeError("cannot inspect active Panda controller") from error

        if mode == "pd_ee_delta_pose":
            arm_action = np.zeros(6, dtype=np.float32)
        elif mode == "pd_joint_delta_pos":
            arm_action = np.zeros(7, dtype=np.float32)
        elif mode == "pd_joint_pos":
            arm_action = _numpy(arm_controller.qpos).reshape(-1).astype(np.float32)
        else:
            raise RuntimeError(
                f"observe-only action is not defined for control_mode={mode!r}"
            )

        gripper_qpos = _numpy(gripper_controller.qpos).reshape(-1)
        gripper_action = float(gripper_qpos[0])
        if bool(getattr(gripper_controller, "_normalize_action", False)):
            low = _numpy(gripper_controller.action_space_low).reshape(-1)
            high = _numpy(gripper_controller.action_space_high).reshape(-1)
            gripper_action = 2.0 * (gripper_action - low[0]) / (high[0] - low[0]) - 1.0

        action = np.concatenate(
            (arm_action, [np.clip(gripper_action, -1.0, 1.0)])
        ).astype(np.float32)
        return action

    def check_success(self, info: Mapping[str, Any]) -> bool:
        return _bool(info.get(self.SUCCESS_INFO_KEY, False))

    def evaluation_diagnostics(self, info: Mapping[str, Any]) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}
        for key in self.EVALUATION_INFO_KEYS:
            if key not in info:
                continue
            value = _numpy(info[key])
            diagnostics[key] = value.item() if value.size == 1 else value.tolist()
        return diagnostics

    @staticmethod
    def _arm(task_env: Any):
        agents = getattr(task_env.agent, "agents", None)
        return agents[0] if agents is not None else task_env.agent

    def get_state(
        self, observation: Mapping[str, Any], task_env: Any
    ) -> np.ndarray:
        """Return the learned policy's absolute joint/gripper state."""

        arm = self._arm(task_env)
        qpos = _numpy(arm.robot.get_qpos()).reshape(-1)
        state = np.concatenate((qpos[:7], [qpos[-1]])).astype(np.float32)
        return state

    @staticmethod
    def _base_camera_extrinsic(observation: Mapping[str, Any]) -> np.ndarray:
        sensor_param = observation.get("sensor_param")
        base = sensor_param.get("base_camera")
        extrinsic = _numpy(base["extrinsic_cv"])
        if extrinsic.shape == (1, 3, 4):
            extrinsic = extrinsic[0]
        return np.asarray(extrinsic, dtype=np.float64)

    def get_replay_state(
        self, observation: Mapping[str, Any], task_env: Any
    ) -> np.ndarray:
        """Build the canonical 17-D LeRobot state for GT replay auditing."""

        arm = self._arm(task_env)
        qpos = _numpy(arm.robot.get_qpos()).reshape(-1)
        world_p = _numpy(arm.tcp.pose.p).reshape(-1)[:3].astype(np.float64)
        world_q = _numpy(arm.tcp.pose.q).reshape(-1)[:4].astype(np.float64)
        extrinsic = self._base_camera_extrinsic(observation)
        camera_world = extrinsic[:, :3]
        camera_p = camera_world @ world_p + extrinsic[:, 3]
        camera_rotation = camera_world @ _quaternion_matrix(world_q)
        state = np.concatenate(
            (
                qpos[:7],
                [qpos[-1]],
                camera_p,
                _rot6d(camera_rotation),
            )
        ).astype(np.float32)
        return state

    @staticmethod
    def _arm_controller(task_env: Any):
        arm = BaseTaskClient._arm(task_env)
        try:
            return arm.controller.controllers["arm"]
        except Exception as error:
            raise RuntimeError("cannot access Panda arm controller") from error

    def _relative_ee_to_native(
        self,
        task_env: Any,
        row: np.ndarray,
        observation: Mapping[str, Any],
    ) -> np.ndarray:
        """Adapt camera-frame relative xyz/Rot6D to native 7-D PDEE."""

        import torch
        from mani_skill.utils.geometry.rotation_conversions import (
            matrix_to_euler_angles,
        )

        value = np.asarray(row, dtype=np.float64).reshape(10)
        extrinsic = self._base_camera_extrinsic(observation)
        camera_world = extrinsic[:, :3]
        world_camera = camera_world.T
        delta_p_world = world_camera @ value[:3]
        delta_rotation_camera = _matrix_from_rot6d(value[3:9])
        delta_rotation_world = (
            world_camera @ delta_rotation_camera @ camera_world
        )

        arm_agent = self._arm(task_env)
        base_rotation = _quaternion_matrix(
            _numpy(arm_agent.robot.pose.q).reshape(-1)[:4]
        )
        delta_p_base = base_rotation.T @ delta_p_world
        delta_rotation_base = (
            base_rotation.T @ delta_rotation_world @ base_rotation
        )
        delta_euler = matrix_to_euler_angles(
            torch.as_tensor(delta_rotation_base, dtype=torch.float64), "XYZ"
        ).cpu().numpy()

        arm_controller = self._arm_controller(task_env)
        low = _numpy(arm_controller.action_space_low).reshape(-1)
        high = _numpy(arm_controller.action_space_high).reshape(-1)
        physical = np.concatenate((delta_p_base, delta_euler))
        normalized = 2.0 * (physical - low) / (high - low) - 1.0
        normalized[3:6] *= -1.0
        if str(getattr(task_env, "TASK_NAME", "")) in REPLAY_QUANTIZED_TASKS:
            normalized[:6] = quantize_native_arm_action(normalized[:6])
        gripper = float(value[-1])
        # The xyz/Rot6D columns are physical camera-relative quantities, but
        # the final gripper column remains a normalized [-1, 1] command.
        # Quantile unnormalization can legitimately extrapolate it by a few
        # thousandths; use the same tolerant clipping convention as the other
        # normalized policy action paths.
        action = np.concatenate((normalized, [np.clip(gripper, -1.0, 1.0)])).astype(
            np.float32
        )
        return np.clip(action, -1.0, 1.0)

    @staticmethod
    def _rgb(sensor_data: Mapping[str, Any], uid: str) -> np.ndarray:
        image = _numpy(sensor_data[uid]["rgb"])
        if image.ndim == 4:
            image = image[0]
        image = image[..., :3]
        if np.issubdtype(image.dtype, np.floating) and image.size:
            if float(image.max()) <= 1.0 + 1e-6:
                image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8, copy=False)

    def get_views(self, observation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        sensor_data = observation.get("sensor_data")
        base = self._rgb(sensor_data, "base_camera")
        wrist = self._rgb(sensor_data, "hand_camera")
        return base, wrist

    def make_policy_observation(
        self,
        observation: Mapping[str, Any],
        task_env: Any,
        *,
        frame: int,
        prompt: str,
        memory_prompt: str,
    ) -> dict[str, Any]:
        base, wrist = self.get_views(observation)
        return {
            "observation/image": base,
            "observation/wrist_image": wrist,
            "observation/state": self.get_state(observation, task_env),
            "state_representation": POLICY_STATE_REPRESENTATION,
            "prompt": str(prompt),
            "prompt_memory": str(memory_prompt),
            "task_name": self.TASK_NAME,
            "frame": int(frame),
        }

    def build_env(
        self,
        *,
        action_space: str = "delta_ee",
        max_episode_steps: int | None = None,
    ):
        requested_cuda_devices = os.getenv("MEMORY_BENCHMARK_CUDA_VISIBLE_DEVICES")
        if requested_cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = requested_cuda_devices
        import gymnasium as gym

        if action_space not in ACTION_SPACE_TO_CONTROL_MODE:
            raise ValueError(
                f"{action_space!r} is not simulator-replayable; choices={REPLAYABLE_SPACES}"
            )
        steps = self.DEFAULT_MAX_STEPS if max_episode_steps is None else int(max_episode_steps)
        make_kwargs: dict[str, Any] = {
            "max_episode_steps": steps,
            "render_mode": "sensors",
            "num_envs": 1,
            "parallel_in_single_scene": False,
            "obs_mode": "rgb",
            "mode": "manual",
            "control_mode": ACTION_SPACE_TO_CONTROL_MODE[action_space],
            "robot_uids": self.ROBOT_UIDS,
            "sensor_configs": {
                "width": self.camera_resolution,
                "height": self.camera_resolution,
            },
        }
        if self.shader != "default":
            make_kwargs["shader_dir"] = self.shader
        task_kwargs = self.extra_make_kwargs()
        protected = dict(make_kwargs)
        protected.update(task_kwargs)
        make_kwargs.update(task_kwargs)
        make_kwargs.update(self.env_kwargs)
        return gym.make(self.ENV_ID, **make_kwargs)

    def run_replay(
        self,
        *,
        action_space: str,
        seed: int,
        episode_metadata: Mapping[str, Any],
        prompt: str | None = None,
        memory_prompt: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8770,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Apply converted GT action chunks and compare canonical state rows."""

        import websockets.sync.client

        limit = self.DEFAULT_MAX_STEPS if max_steps is None else int(max_steps)
        env = self.build_env(action_space=action_space, max_episode_steps=limit)
        task_env = env.unwrapped
        observation, _ = env.reset(
            seed=int(seed),
            options={
                "episode_metadata": dict(episode_metadata),
                "reconfigure": True,
            },
        )
        socket = None
        success_ever = False
        final_success = False
        frame = 0
        state_errors: dict[int, tuple[float, float]] = {}
        mismatch = None
        reference_contract = None
        source_length: int | None = None

        def compare_reference(
            recorded_index: int, actual: np.ndarray, expected: Any
        ) -> None:
            nonlocal mismatch
            expected_array = np.asarray(expected, dtype=np.float32)
            if expected_array.shape != (REPLAY_AUDIT_STATE_DIM,):
                mismatch = (
                    f"reference state {recorded_index} has shape "
                    f"{expected_array.shape}, expected {(REPLAY_AUDIT_STATE_DIM,)}"
                )
                return
            delta = np.asarray(actual, dtype=np.float32) - expected_array
            state_errors[int(recorded_index)] = (
                float(np.linalg.norm(delta)),
                float(np.max(np.abs(delta))),
            )

        try:
            socket = websockets.sync.client.connect(
                f"ws://{host}:{int(port)}", max_size=None, open_timeout=60
            )
            while frame < limit:
                current_state = self.get_replay_state(observation, task_env)
                request = {
                    "action_space": action_space,
                    "frame": frame,
                    "observation/state": current_state,
                    "state_representation": REPLAY_AUDIT_STATE_REPRESENTATION,
                    "prompt": self.INSTRUCTION_TASK if prompt is None else prompt,
                    "prompt_memory": (
                        self.INSTRUCTION_MEMORY if memory_prompt is None else memory_prompt
                    ),
                }
                socket.send(packb(request))
                response = unpackb(socket.recv())
                if response.get("error"):
                    mismatch = str(response["error"])
                    break
                if response.get("action_space") != action_space:
                    mismatch = f"server returned {response.get('action_space')!r}"
                    break
                try:
                    response_length = int(response.get("episode_length"))
                except (TypeError, ValueError):
                    mismatch = "server did not provide a valid episode_length"
                    break
                if source_length is None:
                    source_length = response_length
                elif source_length != response_length:
                    mismatch = "server episode_length changed during replay"
                    break
                actions = np.asarray(response.get("actions"), dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions[None, :]
                expected_action_shape = (len(actions), ACTION_SPACE_TO_DIM[action_space])
                if (
                    actions.shape != expected_action_shape
                    or not len(actions)
                    or not np.all(np.isfinite(actions))
                ):
                    mismatch = (
                        f"invalid {action_space} action chunk {actions.shape}, "
                        f"expected [H,{ACTION_SPACE_TO_DIM[action_space]}]"
                    )
                    break
                current_reference = response.get("ref_current_state")
                if current_reference is not None:
                    compare_reference(
                        int(response.get("ref_current_index", frame)),
                        current_state,
                        current_reference,
                    )
                references = response.get("ref_state")
                reference_indices = response.get("ref_state_indices")
                if references is not None:
                    references = np.asarray(references, dtype=np.float32)
                    if reference_indices is None:
                        reference_indices = np.arange(
                            frame + 1, frame + 1 + len(references), dtype=np.int64
                        )
                    else:
                        reference_indices = np.asarray(
                            reference_indices, dtype=np.int64
                        ).reshape(-1)
                    if len(reference_indices) != len(references):
                        mismatch = "server reference indices/states have different lengths"
                        break
                    references_by_index = {
                        int(index): value
                        for index, value in zip(reference_indices, references)
                    }
                else:
                    references_by_index = {}
                reference_contract = response.get("state_reference_contract")
                done = bool(response.get("done", False))
                for action in actions:
                    controller_action = (
                        self._relative_ee_to_native(
                            task_env, action, observation
                        )
                        if action_space == "rel_ee"
                        else action
                    )
                    observation, _, terminated, truncated, info = env.step(
                        controller_action
                    )
                    frame += 1
                    final_success = self.check_success(info)
                    success_ever = success_ever or final_success
                    if frame in references_by_index:
                        compare_reference(
                            frame,
                            self.get_replay_state(observation, task_env),
                            references_by_index[frame],
                        )
                    if mismatch is not None:
                        done = True
                        break
                    if _bool(terminated) or _bool(truncated):
                        done = True
                        break
                if done:
                    break
        finally:
            try:
                if socket is not None:
                    socket.close()
            finally:
                env.close()
        l2_errors = [value[0] for value in state_errors.values()]
        max_errors = [value[1] for value in state_errors.values()]
        missing_reference_frames = sorted(set(range(frame + 1)) - set(state_errors))
        source_complete = source_length is not None and frame == source_length
        expected_available_references = (
            set(range(source_length)) if source_length is not None else set()
        )
        missing_available_reference_frames = sorted(
            expected_available_references - set(state_errors)
        )
        available_references_complete = bool(
            source_length is not None and not missing_available_reference_frames
        )
        accepted_success = bool(
            final_success
            and source_complete
            and available_references_complete
            and mismatch is None
        )
        return {
            "mode": "replay",
            "task": self.TASK_NAME,
            "action_space": action_space,
            "success": accepted_success,
            "final_state_success": final_success,
            "success_ever": success_ever,
            "steps": frame,
            "source_episode_length": source_length,
            "source_complete": source_complete,
            "available_references_complete": available_references_complete,
            "canonical_state_comparisons": len(state_errors),
            "canonical_state_l2_mean": float(np.mean(l2_errors)) if l2_errors else None,
            "canonical_state_l2_max": float(np.max(l2_errors)) if l2_errors else None,
            "canonical_state_max_abs": float(np.max(max_errors)) if max_errors else None,
            "missing_reference_frames": missing_reference_frames,
            "missing_available_reference_frames": missing_available_reference_frames,
            "state_reference_contract": reference_contract,
            # Deprecated legacy fields intentionally carry no value: the old
            # implementation compared world TCP coordinates with joint values.
            "tcp_drift_mean": None,
            "tcp_drift_max": None,
            "mismatch": mismatch,
        }

    def _validate_policy_metadata(self, metadata: Mapping[str, Any]) -> str:
        errors = []
        if metadata.get("server") != "memory_agent":
            errors.append(f"server={metadata.get('server')!r}")
        if metadata.get("policy_contract_source") != "checkpoint":
            errors.append(
                f"policy_contract_source={metadata.get('policy_contract_source')!r}"
            )
        if metadata.get("policy_contract") != POLICY_CONTRACT:
            errors.append("policy_contract does not match the current benchmark contract")
        env_action_space = str(metadata.get("env_action_space") or "")
        if env_action_space not in ACTION_SPACE_TO_DIM:
            errors.append(f"env_action_space={env_action_space!r}")
        else:
            try:
                env_action_dim = int(metadata.get("env_action_dim", -1))
            except (TypeError, ValueError):
                env_action_dim = -1
            if env_action_dim != ACTION_SPACE_TO_DIM[env_action_space]:
                errors.append(
                    f"env_action_dim={env_action_dim}, expected "
                    f"{ACTION_SPACE_TO_DIM[env_action_space]}"
                )
        if metadata.get("state_representation") != POLICY_STATE_REPRESENTATION:
            errors.append(
                f"state_representation={metadata.get('state_representation')!r}"
            )
        if metadata.get("state_key") != "observation.state_joint":
            errors.append(f"state_key={metadata.get('state_key')!r}")
        try:
            state_dim = int(metadata.get("state_dim", -1))
        except (TypeError, ValueError):
            state_dim = -1
        if state_dim != POLICY_STATE_DIM:
            errors.append(f"state_dim={state_dim}")
        policy_action_space = str(metadata.get("policy_action_space") or "")
        if policy_action_space != POLICY_ACTION_SPACE:
            errors.append(
                f"policy_action_space={policy_action_space!r}, expected "
                f"{POLICY_ACTION_SPACE!r}"
            )
        if env_action_space != POLICY_ACTION_SPACE:
            errors.append(
                f"env_action_space={env_action_space!r}, expected {POLICY_ACTION_SPACE!r}"
            )
        if metadata.get("action_representation") != POLICY_ACTION_REPRESENTATION:
            errors.append(
                f"action_representation={metadata.get('action_representation')!r}"
            )
        if metadata.get("action_key") != "action_rel_ee":
            errors.append(f"action_key={metadata.get('action_key')!r}")
        try:
            policy_action_dim = int(metadata.get("policy_action_dim", -1))
        except (TypeError, ValueError):
            policy_action_dim = -1
        if policy_action_dim != POLICY_ACTION_DIM:
            errors.append(
                f"policy_action_dim={policy_action_dim}, expected {POLICY_ACTION_DIM}"
            )
        try:
            action_dim = int(metadata.get("action_dim", -1))
        except (TypeError, ValueError):
            action_dim = -1
        if action_dim != POLICY_ACTION_DIM:
            errors.append(f"action_dim={action_dim}, expected {POLICY_ACTION_DIM}")
        if metadata.get("control_mode_prompt") != POLICY_CONTROL_MODE_PROMPT:
            errors.append(
                f"control_mode_prompt={metadata.get('control_mode_prompt')!r}"
            )
        if metadata.get("action_adapter") not in (None, "none"):
            errors.append(
                f"action_adapter={metadata.get('action_adapter')!r}, expected 'none'"
            )
        if not bool(metadata.get("supports_observe", False)):
            errors.append("supports_observe is false/missing")
        try:
            slots = int(metadata.get("num_keyframes", -1))
        except (TypeError, ValueError):
            slots = -1
        if slots < int(self.MIN_KEYFRAMES):
            errors.append(
                f"num_keyframes={slots}, task requires at least {self.MIN_KEYFRAMES}"
            )
        if errors:
            raise MemoryAgentProtocolError("incompatible policy server: " + "; ".join(errors))
        return env_action_space

    @staticmethod
    def _policy_actions(
        response: Mapping[str, Any], horizon: int, action_space: str
    ) -> list[np.ndarray]:
        actions = np.asarray(response.get("actions"), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if action_space in {"native", "delta_ee", "delta_joint"}:
            actions = np.clip(actions, -1, 1)
        return [row.copy() for row in actions[:horizon]]

    def run_policy(
        self,
        *,
        seed: int = 0,
        episode_metadata: Mapping[str, Any] | None = None,
        prompt: str | None = None,
        memory_prompt: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8000,
        api_key: str | None = None,
        max_steps: int | None = None,
        execution_horizon: int = 1,
        open_timeout: float = 60.0,
        request_timeout: float | None = 300.0,
        connection: Any | None = None,
        video_path: Path | None = None,
        video_fps: int = 20,
        keyframe_hold_seconds: float = 1.0,
        policy_noise_seed: int | None = None,
    ) -> dict[str, Any]:
        """Run a real closed-loop MemoryAgent episode with live RGB observations."""

        limit = self.DEFAULT_MAX_STEPS if max_steps is None else int(max_steps)
        task_prompt = self.INSTRUCTION_TASK if prompt is None else prompt
        memory = self.INSTRUCTION_MEMORY if memory_prompt is None else memory_prompt
        owns_connection = connection is None
        env = None
        video = None
        generated_episode_metadata = None
        try:
            if connection is None:
                connection = MemoryAgentConnection(
                    host,
                    port,
                    api_key=api_key,
                    open_timeout=open_timeout,
                    request_timeout=request_timeout,
                )
            metadata = dict(connection.metadata)
            wire_action_space = self._validate_policy_metadata(metadata)
            controller_action_space = (
                "delta_ee" if wire_action_space == "rel_ee" else wire_action_space
            )
            env = self.build_env(
                action_space=controller_action_space,
                max_episode_steps=limit,
            )
            task_env = env.unwrapped
            options = (
                {"episode_metadata": dict(episode_metadata)}
                if episode_metadata is not None
                else {}
            )
            options["reconfigure"] = True
            observation, _ = env.reset(seed=int(seed), options=options)
            generated_episode_metadata = task_env.get_episode_metadata()
            connection.reset()
            if video_path is not None:
                video = _PolicyVideoWriter(
                    Path(video_path),
                    fps=int(video_fps),
                    keyframe_hold_seconds=float(keyframe_hold_seconds),
                )
        except BaseException:
            if env is not None:
                env.close()
            if owns_connection and connection is not None:
                connection.close()
            raise
        queued: deque[np.ndarray] = deque()
        success = failure = terminated_flag = truncated_flag = False
        info: Mapping[str, Any] = {}
        frame = infer_requests = observe_requests = blocked_frames = 0
        retrieval_events: list[dict[str, Any]] = []
        try:
            while frame < limit:
                request = self.make_policy_observation(
                    observation,
                    task_env,
                    frame=frame,
                    prompt=str(task_prompt),
                    memory_prompt=str(memory),
                )
                controlled = bool(self.policy_controls_action(task_env))
                if not controlled:
                    queued.clear()
                    response = connection.observe(request)
                    observe_requests += 1
                    blocked_frames += 1
                    action = self.observe_only_action(task_env)
                    request_kind = "observe_only"
                elif queued:
                    response = connection.observe(request)
                    observe_requests += 1
                    action = queued.popleft()
                    request_kind = "observe_chunk"
                else:
                    if policy_noise_seed is not None:
                        noise_rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [int(policy_noise_seed), int(frame)]
                            )
                        )
                        request["noise"] = noise_rng.standard_normal(
                            (
                                POLICY_DIFFUSION_ACTION_HORIZON,
                                POLICY_DIFFUSION_ACTION_DIM,
                            ),
                            dtype=np.float32,
                        )
                    response = connection.infer(request)
                    infer_requests += 1
                    queued.extend(
                        self._policy_actions(
                            response, execution_horizon, wire_action_space
                        )
                    )
                    action = queued.popleft()
                    request_kind = "infer"
                retrieval_events.append(
                    {
                        "frame": frame,
                        "request": request_kind,
                        "policy_controls_action": controlled,
                        "phase_id": int(task_env.phase_id[0].item()),
                        "is_keyframe": bool(response.get("is_keyframe", False)),
                        "num_keyframes": int(response.get("num_keyframes", 0)),
                        "kf_probability": response.get("kf_probability"),
                    }
                )
                if video is not None:
                    video.write(
                        request["observation/image"],
                        is_keyframe=bool(response.get("is_keyframe", False)),
                    )
                controller_action = (
                    self._relative_ee_to_native(task_env, action, observation)
                    if wire_action_space == "rel_ee" and controlled
                    else action
                )
                observation, _, terminated, truncated, info = env.step(
                    controller_action
                )
                frame += 1
                success = success or self.check_success(info)
                failure = failure or _bool(info.get("fail", False))
                terminated_flag = _bool(terminated)
                truncated_flag = _bool(truncated)
                if success or failure or terminated_flag or truncated_flag:
                    break
            if video is not None:
                final_base, _ = self.get_views(observation)
                video.write(final_base)
        finally:
            try:
                if video is not None:
                    video.close(commit=sys.exc_info()[0] is None)
            finally:
                try:
                    env.close()
                finally:
                    if owns_connection:
                        connection.close()
        reason = (
            "success"
            if success
            else "failure"
            if failure
            else "terminated"
            if terminated_flag
            else "truncated"
            if truncated_flag
            else "max_steps"
        )
        return {
            "mode": "policy",
            "task": self.TASK_NAME,
            "env_id": self.ENV_ID,
            "seed": int(seed),
            "success": success,
            "failure": failure,
            "steps": frame,
            "episode_step_limit": limit,
            "termination_reason": reason,
            "evaluation_diagnostics": self.evaluation_diagnostics(info),
            "infer_requests": infer_requests,
            "observe_requests": observe_requests,
            "blocked_frames": blocked_frames,
            "execution_horizon": int(execution_horizon),
            "wire_action_space": wire_action_space,
            "controller_action_space": controller_action_space,
            "episode_metadata_sha256": _metadata_digest(generated_episode_metadata),
            "episode_metadata": generated_episode_metadata,
            "policy_noise_seed": (
                None if policy_noise_seed is None else int(policy_noise_seed)
            ),
            "policy_noise_scheme": (
                None
                if policy_noise_seed is None
                else "numpy.PCG64(SeedSequence([episode_policy_seed, env_frame])); "
                "standard_normal([10,32],float32) on infer requests"
            ),
            "video_path": None if video_path is None else str(Path(video_path).resolve()),
            "video_fps": int(video_fps) if video_path is not None else None,
            "keyframe_hold_seconds": (
                float(keyframe_hold_seconds) if video_path is not None else None
            ),
            "retrieval_events": retrieval_events,
            "server_metadata": metadata,
        }

    # Backward-compatible name used by early v3 task clients.
    run_episode = run_policy


BaseMemoryAgentTaskClient = BaseTaskClient
