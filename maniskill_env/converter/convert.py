"""Convert v3 ManiSkill demonstrations to the canonical LeRobot v2.1 schema."""

from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
from itertools import product
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import av
import gymnasium as gym
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .h5_tools import (
    load_episode_tensors,
    sort_camera_uids,
    sorted_traj_ids,
)
from .keyframes import keyframe_record
from .specs import ALL_TASK_NAMES, ENV_ID_TO_TASK, TASK_SPECS, lerobot_task_path
from .stats_v21 import build_episode_stats


V21 = "v2.1"
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
_V3_ROOT = Path(__file__).resolve().parents[1]
_TASK_ENV_ROOT = _V3_ROOT / "task_env"

VECTOR_DIMS = {
    "observation.state_joint": 8,
    "observation.state_pose": 9,
    "observation.state": 17,
    "action": 17,
    "action_abs_joint": 8,
    "action_rel_ee": 10,
    "action_native": 7,
    "action_delta_joint": 8,
}
VIDEO_MAP = {
    "base_camera": ("image", "observation.images.image"),
    "hand_camera": ("wrist_image", "observation.images.wrist_image"),
}
OBSERVED_TASKS = frozenset(
    {
        "AO_hidden_pattern",

        "SO_spring_compression",
        "ASO_placement_imitation",
    }
)


def register_task_environment(
    *, env_id: str | None = None, task_name: str | None = None
) -> str:
    """Load one v3 task module and return its canonical task name."""

    if task_name is None:
        task_name = ENV_ID_TO_TASK.get(str(env_id))
    if task_name not in TASK_SPECS:
        raise ValueError(
            f"unknown v3 env/task: env_id={env_id!r}, task_name={task_name!r}; "
            f"known env ids={sorted(ENV_ID_TO_TASK)}"
        )
    task_spec = TASK_SPECS[task_name]
    visual_ood = env_id == task_spec["visual_ood_env_id"]
    module_name = task_spec["visual_ood_module"] if visual_ood else task_spec["module"]
    task_dir = _TASK_ENV_ROOT / task_name
    module_path = task_dir / f"{module_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"v3 task module is not implemented yet: {module_path}"
        )
    for path in (task_dir, _TASK_ENV_ROOT, _V3_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if visual_ood:
        # The OOD module subclasses the original class. Import the canonical
        # base module first so it is registered exactly once in this process.
        base_name = task_spec["module"]
        if base_name not in sys.modules:
            base_path = task_dir / f"{base_name}.py"
            base_spec = importlib.util.spec_from_file_location(base_name, base_path)
            base_module = importlib.util.module_from_spec(base_spec)
            sys.modules[base_name] = base_module
            try:
                base_spec.loader.exec_module(base_module)
            except BaseException:
                sys.modules.pop(base_name, None)
                raise
    unique_name = f"_memory_benchmark_v3_{module_name}"
    if unique_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(unique_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(unique_name, None)
            raise
    return task_name


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _task_text(metadata: dict[str, Any], override: str | None) -> str:
    if override:
        return str(override)
    env_info = metadata.get("env_info") or {}
    return str(
        env_info.get("instruction_task")
        or env_info.get("instruction")
        or "maniskill_memory_task"
    )


def _memory_text(metadata: dict[str, Any]) -> str:
    return str((metadata.get("env_info") or {}).get("instruction_memory") or "")


def _kept(episode: dict[str, Any]) -> bool:
    return episode.get("success") is not False and episode.get("fail") is not True


def _even_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    pad_h, pad_w = image.shape[0] % 2, image.shape[1] % 2
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return np.ascontiguousarray(image[..., :3])


def _write_video(path: Path, dataset, n: int, fps: int) -> None:
    frames = np.asarray(dataset[:n])
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream(
            "libx264", rate=int(fps), options={"crf": "23", "preset": "veryfast"}
        )
        stream.pix_fmt = "yuv420p"
        first = _even_rgb(frames[0])
        stream.height, stream.width = first.shape[:2]
        stream.time_base = Fraction(1, int(fps))
        for index, raw in enumerate(frames):
            image = _even_rgb(raw)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _schema() -> pa.Schema:
    fields = [
        pa.field(name, pa.list_(pa.float32(), dimension))
        for name, dimension in VECTOR_DIMS.items()
    ]
    fields.extend(
        [
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("policy_action_mask", pa.bool_()),
            pa.field("task", pa.string()),
        ]
    )
    return pa.schema(fields)


def _policy_action_mask(
    data: dict[str, Any], *, task_name: str, episode_index: int, horizon: int
) -> np.ndarray:
    """Return the lower-policy sampling mask without dropping visual frames."""

    recorded = data.get("policy_action_mask")
    extra = data.get("obs_extra")
    phase = np.asarray(extra["phase_id"][:horizon]).reshape(-1)
    expected = phase == 2 if task_name in OBSERVED_TASKS else np.ones(horizon, dtype=bool)
    if recorded is None:
        # Compatibility for raw episodes generated before this annotation was
        # introduced.  Conversion still writes an explicit mask, derived from
        # the finalized phase labels rather than from action magnitudes.
        mask = expected
    else:
        mask = np.asarray(recorded, dtype=bool).reshape(-1)
    return mask


def _tree_step(tree: Any, index: int) -> Any:
    if isinstance(tree, dict):
        return {key: _tree_step(value, index) for key, value in tree.items()}
    return np.asarray(tree)[int(index)][None, ...]


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(left, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(right, dtype=np.float64)
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _compose_pose(
    base_p: np.ndarray,
    base_q: np.ndarray,
    local_p: np.ndarray,
    local_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_p = np.asarray(base_p) + _quat_matrix(base_q) @ np.asarray(local_p)
    world_q = _quat_multiply(base_q, local_q)
    world_q /= np.linalg.norm(world_q) + 1e-12
    return world_p.astype(np.float32), world_q.astype(np.float32)


def _rot6d(matrix: np.ndarray) -> np.ndarray:
    """Encode a rotation as its first two matrix columns."""

    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return np.concatenate((rotation[:, 0], rotation[:, 1])).astype(np.float32)


def quantize_camera_translation_for_native(
    camera_translation: np.ndarray,
    native_translation: np.ndarray,
    camera_world: np.ndarray,
    base_rotation: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """Choose camera-frame float32 xyz with the best native round trip."""

    camera_world = np.asarray(camera_world, dtype=np.float64).reshape(3, 3)
    world_camera = camera_world.T
    base_rotation = np.asarray(base_rotation, dtype=np.float64).reshape(3, 3)
    low = np.asarray(action_low, dtype=np.float64).reshape(-1)[:3]
    high = np.asarray(action_high, dtype=np.float64).reshape(-1)[:3]
    desired = np.asarray(native_translation, dtype=np.float32).reshape(3)
    candidate = np.asarray(camera_translation, dtype=np.float32).reshape(3)

    def physical_and_decoded(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        physical = (
            base_rotation.T
            @ world_camera
            @ np.asarray(value, dtype=np.float64)
        )
        normalized = 2.0 * (physical - low) / (high - low) - 1.0
        return physical, normalized.astype(np.float32)

    best = candidate.copy()
    _, best_decoded = physical_and_decoded(best)
    best_error = float(
        np.max(
            np.abs(
                best_decoded.astype(np.float64) - desired.astype(np.float64)
            )
        )
    )
    desired_physical = low + (desired.astype(np.float64) + 1.0) * 0.5 * (
        high - low
    )
    for _ in range(8):
        physical, decoded = physical_and_decoded(candidate)
        error = float(
            np.max(
                np.abs(decoded.astype(np.float64) - desired.astype(np.float64))
            )
        )
        if error < best_error:
            best, best_error = candidate.copy(), error
        if np.array_equal(decoded, desired):
            return candidate
        correction_base = desired_physical - physical
        correction_camera = camera_world @ base_rotation @ correction_base
        updated = np.asarray(
            candidate.astype(np.float64) + correction_camera,
            dtype=np.float32,
        )
        if np.array_equal(updated, candidate):
            break
        candidate = updated

    neighbours: list[tuple[np.float32, ...]] = []
    for value in candidate:
        centre = np.float32(value)
        values = [centre]
        down = centre
        up = centre
        for _ in range(4):
            down = np.nextafter(down, np.float32(-np.inf), dtype=np.float32)
            up = np.nextafter(up, np.float32(np.inf), dtype=np.float32)
            values.extend((down, up))
        neighbours.append(tuple(values))
    for values in product(*neighbours):
        trial = np.asarray(values, dtype=np.float32)
        _, decoded = physical_and_decoded(trial)
        error = float(
            np.max(
                np.abs(decoded.astype(np.float64) - desired.astype(np.float64))
            )
        )
        if error < best_error:
            best, best_error = trial.copy(), error
        if np.array_equal(decoded, desired):
            return trial
    return best


def _camera_extrinsic(data: dict[str, Any], frame: int) -> np.ndarray:
    sensor_param = data.get("sensor_param")
    try:
        matrix = np.asarray(
            sensor_param["base_camera"]["extrinsic_cv"][int(frame)],
            dtype=np.float64,
        )
    except KeyError as error:
        raise KeyError(
            "raw H5 is missing obs/sensor_param/base_camera/extrinsic_cv"
        ) from error
    return matrix


def _raw_tcp_pose(data: dict[str, Any], frame: int) -> tuple[np.ndarray, np.ndarray]:
    extra = data.get("obs_extra")
    pose = np.asarray(extra["tcp_pose"][int(frame)], dtype=np.float64).reshape(-1)
    return pose[:3], pose[3:7]


def _world_pose_to_camera(
    world_p: np.ndarray,
    world_q: np.ndarray,
    extrinsic_cv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera_world = np.asarray(extrinsic_cv, dtype=np.float64)[:, :3]
    camera_p = camera_world @ np.asarray(world_p, dtype=np.float64)
    camera_p += np.asarray(extrinsic_cv, dtype=np.float64)[:, 3]
    camera_rotation = camera_world @ _quat_matrix(world_q)
    return camera_p.astype(np.float32), camera_rotation


def _make_state_environment(metadata: dict[str, Any]):
    env_info = metadata.get("env_info") or {}
    env_id = str(env_info.get("env_id") or "")
    task_name = register_task_environment(env_id=env_id)
    recorded = env_info.get("env_kwargs") or {}
    ignored = {
        "max_episode_steps",
        "render_mode",
        "obs_mode",
        "control_mode",
        "sensor_configs",
        "shader_dir",
        "sim_backend",
        "render_backend",
        "num_envs",
        "parallel_in_single_scene",
    }
    make_kwargs = {key: value for key, value in recorded.items() if key not in ignored}
    make_kwargs.update(
        obs_mode="none",
        control_mode="pd_ee_delta_pose",
        num_envs=1,
        robot_uids="panda",
        mode="manual",
        sim_backend="cpu",
        render_backend="cpu",
    )
    try:
        environment = gym.make(env_id, **make_kwargs)
    except Exception as error:
        raise RuntimeError(
            f"failed to create {env_id} for commanded-target conversion with "
            f"kwargs={make_kwargs}: {error}"
        ) from error
    return task_name, environment


def _state_and_action_rows(
    environment,
    data: dict[str, Any],
    *,
    episode_index: int,
) -> dict[str, np.ndarray]:
    """Reconstruct the canonical state and commanded action representations."""

    import torch

    native = np.asarray(data["actions"], dtype=np.float32)
    qpos = np.asarray(data["qpos"], dtype=np.float32)
    env_states = data["env_states"]
    n = len(native)
    task_env = environment.unwrapped
    agent = task_env.agent
    try:
        arm_controller = agent.controller.controllers["arm"]
    except Exception as error:
        raise RuntimeError("cannot access Panda arm controller target state") from error
    arm_action_low = np.asarray(
        arm_controller.action_space_low.detach().cpu().numpy(),
        dtype=np.float64,
    ).reshape(-1)
    arm_action_high = np.asarray(
        arm_controller.action_space_high.detach().cpu().numpy(),
        dtype=np.float64,
    ).reshape(-1)

    state_joint = np.empty((n, 8), dtype=np.float32)
    state_pose = np.empty((n, 9), dtype=np.float32)
    abs_joint = np.empty((n, 8), dtype=np.float32)
    rel_ee = np.empty((n, 10), dtype=np.float32)
    delta_joint = np.empty((n, 8), dtype=np.float32)
    for frame in range(n):
        task_env.set_state_dict(_tree_step(env_states, frame))
        current_p = np.asarray(
            agent.tcp.pose.p.detach().cpu().numpy(), dtype=np.float64
        ).reshape(-1)[:3]
        current_q = np.asarray(
            agent.tcp.pose.q.detach().cpu().numpy(), dtype=np.float64
        ).reshape(-1)[:4]

        raw_p, raw_q = _raw_tcp_pose(data, frame)
        raw_q = np.asarray(raw_q, dtype=np.float64)
        raw_q /= np.linalg.norm(raw_q) + 1e-12
        extrinsic = _camera_extrinsic(data, frame)
        camera_p, camera_rotation = _world_pose_to_camera(raw_p, raw_q, extrinsic)
        state_joint[frame] = np.concatenate((qpos[frame, :7], [qpos[frame, -1]]))
        state_pose[frame] = np.concatenate((camera_p, _rot6d(camera_rotation)))

        agent.set_action(torch.as_tensor(native[frame], dtype=torch.float32)[None, :])
        target_qpos = np.asarray(
            arm_controller._target_qpos.detach().cpu().numpy()
        ).reshape(-1)[:7]
        target_pose = np.asarray(
            arm_controller._target_pose.raw_pose.detach().cpu().numpy()
        ).reshape(-1)
        base_p = np.asarray(agent.robot.pose.p.detach().cpu().numpy()).reshape(-1)[:3]
        base_q = np.asarray(agent.robot.pose.q.detach().cpu().numpy()).reshape(-1)[:4]
        target_p, target_q = _compose_pose(
            base_p, base_q, target_pose[:3], target_pose[3:7]
        )
        camera_world = extrinsic[:, :3]
        world_camera = camera_world.T
        delta_p_camera = quantize_camera_translation_for_native(
            camera_world @ (target_p - current_p),
            native[frame, :3],
            camera_world,
            _quat_matrix(base_q),
            arm_action_low,
            arm_action_high,
        )
        target_rotation = _quat_matrix(target_q)
        current_rotation = _quat_matrix(current_q)
        delta_rotation_camera = (
            camera_world @ target_rotation @ current_rotation.T @ world_camera
        )

        gripper = float(native[frame, -1])
        abs_joint[frame] = np.concatenate((target_qpos, [gripper]))
        normalized_delta = np.clip((target_qpos - qpos[frame, :7]) / 0.1, -1, 1)
        delta_joint[frame] = np.concatenate((normalized_delta, [gripper]))
        rel_ee[frame] = np.concatenate(
            (delta_p_camera, _rot6d(delta_rotation_camera), [gripper])
        )

    state = np.concatenate((state_joint, state_pose), axis=1)
    action = np.concatenate((abs_joint, rel_ee[:, :9]), axis=1)
    result = {
        "observation.state_joint": state_joint,
        "observation.state_pose": state_pose,
        "observation.state": state,
        "action": action,
        "action_abs_joint": abs_joint,
        "action_rel_ee": rel_ee,
        "action_native": native.copy(),
        "action_delta_joint": delta_joint,
    }
    return result


def _timestamps(data: dict[str, Any], n: int, fps: int) -> np.ndarray:
    extra = data.get("obs_extra")
    if isinstance(extra, dict) and "timestamp" in extra:
        timestamp = np.asarray(extra["timestamp"], dtype=np.float32).reshape(-1)[:n]
        return timestamp
    return np.arange(n, dtype=np.float32) / float(fps)


def _features(image_hw: dict[str, tuple[int, int]], first_videos: dict[str, Path]) -> dict:
    result: dict[str, Any] = {}
    for video_key, (height, width) in image_hw.items():
        with av.open(str(first_videos[video_key]), "r") as container:
            stream = container.streams.video[0]
            pix_fmt = str(stream.pix_fmt)
            channels = 1 if "gray" in pix_fmt else 4 if "rgba" in pix_fmt else 3
            rate = stream.base_rate or stream.average_rate
            video_info = {
                "video.height": int(stream.height),
                "video.width": int(stream.width),
                "video.codec": str(stream.codec_context.name),
                "video.pix_fmt": pix_fmt,
                "video.is_depth_map": False,
                "video.fps": int(round(float(rate))),
                "video.channels": channels,
                "has_audio": bool(container.streams.audio),
            }
        result[video_key] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "video_info": video_info,
        }
    joint_names = [f"panda_joint_{index + 1}" for index in range(7)]
    rotation_names = [
        "rot6d_col0_x",
        "rot6d_col0_y",
        "rot6d_col0_z",
        "rot6d_col1_x",
        "rot6d_col1_y",
        "rot6d_col1_z",
    ]
    pose_names = ["camera_x", "camera_y", "camera_z", *rotation_names]
    names = {
        "observation.state_joint": [*joint_names, "gripper_position"],
        "observation.state_pose": [f"tcp_{name}" for name in pose_names],
        "observation.state": [
            *joint_names,
            "gripper_position",
            *[f"tcp_{name}" for name in pose_names],
        ],
        "action": [
            *[f"target_{name}" for name in joint_names],
            "gripper_command",
            *[f"delta_tcp_{name}" for name in pose_names],
        ],
        "action_abs_joint": [
            *[f"target_{name}" for name in joint_names],
            "gripper_command",
        ],
        "action_rel_ee": [
            *[f"delta_tcp_{name}" for name in pose_names],
            "gripper_command",
        ],
        "action_native": [
            "native_delta_x",
            "native_delta_y",
            "native_delta_z",
            "native_delta_rot_x",
            "native_delta_rot_y",
            "native_delta_rot_z",
            "gripper_command",
        ],
        "action_delta_joint": [
            *[f"delta_{name}" for name in joint_names],
            "gripper_command",
        ],
    }
    for name, dimension in VECTOR_DIMS.items():
        result[name] = {
            "dtype": "float32",
            "shape": [dimension],
            "names": names[name],
        }
    result.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "policy_action_mask": {
                "dtype": "bool",
                "shape": [1],
                "names": None,
            },
        }
    )
    return result


def _modality() -> dict[str, Any]:
    return {
        "policy_training": {
            "schema_version": 1,
            "state_source_column": "observation.state_joint",
            "state_dimension": 8,
            "action_source_column": "action_rel_ee",
            "action_dimension": 10,
            "action_coordinate_frame": "base_camera_opencv",
            "rotation_representation": "rot6d_first_two_matrix_columns",
            "rotation_convention": "spatial_relative_rotation_expressed_in_base_camera",
            "legacy_combined_columns": {
                "observation.state": "compatibility_and_replay_audit_only",
                "action": "compatibility_and_replay_audit_only",
            },
        },
        "state": {
            "joint": {"start": 0, "end": 7, "absolute": True},
            "gripper": {"start": 7, "end": 8, "absolute": True},
            "position": {
                "start": 8,
                "end": 11,
                "frame": "base_camera",
                "absolute": True,
            },
            "rotation": {
                "start": 11,
                "end": 17,
                "rotation_type": "rotation_6d",
                "frame": "base_camera",
                "absolute": True,
            },
            "source_column": "observation.state",
        },
        "action": {
            "joints": {"start": 0, "end": 7, "absolute": True},
            "gripper": {"start": 7, "end": 8, "absolute": False},
            "position": {
                "start": 8,
                "end": 11,
                "frame": "base_camera",
                "absolute": False,
            },
            "rotation": {
                "start": 11,
                "end": 17,
                "rotation_type": "rotation_6d",
                "frame": "base_camera",
                "absolute": False,
                "relative_convention": "spatial_root_aligned",
            },
            "source_column": "action",
            "simulator_replayable": False,
        },
        "observation.state_joint": {
            "joint": {"start": 0, "end": 7, "absolute": True},
            "gripper": {"start": 7, "end": 8, "absolute": True},
        },
        "observation.state_pose": {
            "position": {
                "start": 0,
                "end": 3,
                "frame": "base_camera",
                "absolute": True,
            },
            "rotation": {
                "start": 3,
                "end": 9,
                "rotation_type": "rotation_6d",
                "frame": "base_camera",
                "absolute": True,
            },
        },
        "action_abs_joint": {
            "joints": {"start": 0, "end": 7, "absolute": True},
            "gripper": {"start": 7, "end": 8},
            "control_mode": "pd_joint_pos",
            "simulator_replayable": True,
        },
        "action_rel_ee": {
            "position": {
                "start": 0,
                "end": 3,
                "frame": "base_camera",
                "absolute": False,
            },
            "rotation": {
                "start": 3,
                "end": 9,
                "rotation_type": "rotation_6d",
                "frame": "base_camera",
                "absolute": False,
                "relative_convention": "spatial_root_aligned",
            },
            "gripper": {"start": 9, "end": 10},
            "simulator_replayable": False,
            "adapter": "camera_relative_rot6d_to_pd_ee_delta_pose",
        },
        "action_native": {
            "position": {"start": 0, "end": 3, "absolute": False},
            "rotation": {
                "start": 3,
                "end": 6,
                "rotation_type": "euler_angles",
                "absolute": False,
            },
            "gripper": {"start": 6, "end": 7},
            "control_mode": "pd_ee_delta_pose",
            "simulator_replayable": True,
        },
        "action_delta_joint": {
            "joints": {"start": 0, "end": 7, "absolute": False, "scale": 0.1},
            "gripper": {"start": 7, "end": 8},
            "control_mode": "pd_joint_delta_pos",
            "simulator_replayable": True,
        },
        "video": {
            "image": {"original_key": "observation.images.image"},
            "wrist_image": {"original_key": "observation.images.wrist_image"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
            "policy_action_mask": {
                "original_key": "policy_action_mask",
                "meaning": "true iff the row is valid lower-VLA action supervision",
            },
        },
    }


def convert(
    *,
    input_dir: Path,
    stem: str,
    output_root: Path,
    fps: int = 20,
    task_name: str | None = None,
    expected_task: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert one paired ``stem.h5`` / ``stem.json`` dataset."""

    input_dir = Path(input_dir).resolve()
    output_root = Path(output_root).resolve()
    h5_path = input_dir / f"{stem}.h5"
    json_path = input_dir / f"{stem}.json"
    if not h5_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"expected paired files {h5_path} and {json_path}")
    metadata = json.loads(json_path.read_text())
    episodes = metadata.get("episodes")
    env_id = str((metadata.get("env_info") or {}).get("env_id") or "")
    canonical_task, state_environment = _make_state_environment(metadata)
    if expected_task is not None and canonical_task != expected_task:
        state_environment.close()
        raise ValueError(
            f"dataset env_id {env_id!r} resolves to {canonical_task!r}, not "
            f"requested task {expected_task!r}"
        )
    requested_task = canonical_task if task_name is None else task_name
    if task_name is not None and task_name not in (canonical_task, _task_text(metadata, None)):
        # ``--task-name`` historically overrides text, so allow arbitrary prose;
        # only canonical task stems are checked when one is supplied.
        if task_name in TASK_SPECS:
            raise ValueError(
                f"--task-name {task_name!r} conflicts with env_id {env_id!r}"
            )
    task_text = _task_text(metadata, None if requested_task == canonical_task else requested_task)
    memory_text = _memory_text(metadata)

    root = lerobot_task_path(output_root, canonical_task)
    if root.exists():
        if not overwrite:
            state_environment.close()
            raise FileExistsError(f"{root} exists (pass --overwrite)")
        shutil.rmtree(root)
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    for _, video_key in VIDEO_MAP.values():
        (root / "videos" / "chunk-000" / video_key).mkdir(parents=True)
    _jsonl(
        root / "meta" / "tasks.jsonl",
        [{"task_index": 0, "task": task_text, "task_memory": memory_text}],
    )

    episode_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    keyframe_rows: list[dict[str, Any]] = []
    image_hw: dict[str, tuple[int, int]] = {}
    first_videos: dict[str, Path] = {}
    global_index = 0
    output_episode = 0
    try:
        with h5py.File(h5_path, "r") as h5:
            trajectory_ids = sorted_traj_ids(h5)
            for source_index, episode in enumerate(episodes):
                if not _kept(episode):
                    continue
                data = load_episode_tensors(h5[trajectory_ids[source_index]])
                n = int(np.asarray(data["actions"]).shape[0])
                camera_uids = sort_camera_uids(list(data["rgb_by_camera"]))
                vectors = _state_and_action_rows(
                    state_environment, data, episode_index=output_episode
                )
                policy_action_mask = _policy_action_mask(
                    data,
                    task_name=canonical_task,
                    episode_index=output_episode,
                    horizon=n,
                )
                timestamps = _timestamps(data, n, int(fps))
                frame_indices = np.arange(n, dtype=np.int64)
                episode_indices = np.full(n, output_episode, dtype=np.int64)
                indices = np.arange(global_index, global_index + n, dtype=np.int64)
                task_indices = np.zeros(n, dtype=np.int64)

                rgb_by_video_key: dict[str, Any] = {}
                for camera_uid in camera_uids:
                    _, video_key = VIDEO_MAP[camera_uid]
                    dataset = data["rgb_by_camera"][camera_uid]
                    first_frame = np.asarray(dataset[0])
                    current_hw = (int(first_frame.shape[0]), int(first_frame.shape[1]))
                    image_hw[video_key] = current_hw
                    video_path = (
                        root
                        / VIDEO_PATH.format(
                            episode_chunk=0,
                            video_key=video_key,
                            episode_index=output_episode,
                        )
                    )
                    _write_video(video_path, dataset, n, int(fps))
                    first_videos.setdefault(video_key, video_path)
                    rgb_by_video_key[video_key] = dataset

                rows = {
                    **{
                        name: [row.tolist() for row in vectors[name]]
                        for name in VECTOR_DIMS
                    },
                    "timestamp": timestamps,
                    "frame_index": frame_indices,
                    "episode_index": episode_indices,
                    "index": indices,
                    "task_index": task_indices,
                    "policy_action_mask": policy_action_mask,
                    "task": [task_text] * n,
                }
                table = pa.Table.from_pandas(
                    pd.DataFrame(rows), schema=_schema(), preserve_index=False
                )
                pq.write_table(
                    table,
                    root
                    / DATA_PATH.format(episode_chunk=0, episode_index=output_episode),
                )
                episode_rows.append(
                    {
                        "episode_index": output_episode,
                        "tasks": [task_text],
                        "length": n,
                        "policy_action_count": int(np.count_nonzero(policy_action_mask)),
                    }
                )
                stats_rows.append(
                    {
                        "episode_index": output_episode,
                        "stats": build_episode_stats(
                            actions=vectors["action"],
                            extra_actions={
                                name: vectors[name]
                                for name in (
                                    "action_abs_joint",
                                    "action_rel_ee",
                                    "action_native",
                                    "action_delta_joint",
                                )
                            },
                            states=vectors["observation.state"],
                            extra_states={
                                name: vectors[name]
                                for name in (
                                    "observation.state_joint",
                                    "observation.state_pose",
                                )
                            },
                            rgb_by_video_key=rgb_by_video_key,
                            episode_index=output_episode,
                            frame_index=frame_indices,
                            timestamp=timestamps,
                            index=indices,
                            task_index=task_indices,
                            policy_action_mask=policy_action_mask,
                        ),
                    }
                )
                keyframe_rows.append(
                    keyframe_record(
                        episode, episode_index=output_episode, episode_length=n
                    )
                )
                global_index += n
                output_episode += 1
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    finally:
        state_environment.close()

    if output_episode == 0:
        raise ValueError("no successful episodes were converted")
    _jsonl(root / "meta" / "episodes.jsonl", episode_rows)
    _jsonl(root / "meta" / "episodes_stats.jsonl", stats_rows)
    (root / "meta" / "keyframes.json").write_text(
        json.dumps({"episodes": keyframe_rows}, indent=2, ensure_ascii=False)
    )
    (root / "meta" / "modality.json").write_text(
        json.dumps(_modality(), indent=2, ensure_ascii=False)
    )
    info = {
        "codebase_version": V21,
        "robot_type": "panda",
        "total_episodes": output_episode,
        "total_frames": global_index,
        "total_tasks": 1,
        "total_videos": output_episode * len(VIDEO_MAP),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": int(fps),
        "splits": {"train": f"0:{output_episode}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": _features(image_hw, first_videos),
        # LeRobot's conventional observation.state/action names are retained
        # for compatibility; consumers should use these explicit policy columns.
        "policy_training": _modality()["policy_training"],
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False)
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="input .h5/.json filename stem")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="dataset root; output is written to lerobot_format/<task>",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--task-name")
    parser.add_argument("--expected-task", choices=ALL_TASK_NAMES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = convert(
        input_dir=args.input_dir,
        stem=args.name,
        output_root=args.output_root,
        fps=args.fps,
        task_name=args.task_name,
        expected_task=args.expected_task,
        overwrite=args.overwrite,
    )
    print(f"CONVERT_DONE: {result}")


if __name__ == "__main__":
    main()
