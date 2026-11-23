"""Shared single-process sanity-trajectory recorder for v3 tasks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import av
import gymnasium as gym
import h5py
import numpy as np
import torch
from mani_skill.utils.wrappers.record import RecordEpisode

try:
    from .camera_config import DEFAULT_CAMERA_RESOLUTION
except ImportError:
    from camera_config import DEFAULT_CAMERA_RESOLUTION


# These tasks acquire their memory while the scene moves autonomously and the
# Panda stays at home.  The simulator still needs a local no-op on every step,
# but those no-ops are replay data, not demonstrations for the learned policy.
OBSERVED_TASKS = frozenset(
    {
        "AO_hidden_pattern",

        "SO_spring_compression",
        "ASO_placement_imitation",
    }
)

# These environments expose the benchmark's closed-loop success contract
# separately from the stricter expert-termination condition.  Recorded demos
# must satisfy both so every accepted trajectory is valid under evaluation.
EVALUATION_SUCCESS_TASKS = frozenset(
    {
        "AD_scroll_discovery",

        "SO_spring_compression",
        "SD_explore_cube",
        "ST_flash_beacon",
    }
)

def _scalar_bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value).reshape(-1)
    return bool(array[0]) if array.size else False


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def keyframe_pairs(events: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    pairs = []
    for event in events:
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        pairs.append((start, end))
    return pairs


def draw_red_keyframe_borders(
    images: Sequence[np.ndarray],
    events: Sequence[Mapping[str, Any]],
    *,
    thickness: int = 6,
) -> list[np.ndarray]:
    """Return video frames with an RGB red border on inclusive GT intervals."""

    pairs = keyframe_pairs(events)
    marked: list[np.ndarray] = []
    for frame_index, source in enumerate(images):
        frame = np.asarray(source).copy()
        if any(start <= frame_index <= end for start, end in pairs):
            width = min(int(thickness), frame.shape[0] // 2, frame.shape[1] // 2)
            red = np.asarray(
                [1.0, 0.0, 0.0] if np.issubdtype(frame.dtype, np.floating) else [255, 0, 0],
                dtype=frame.dtype,
            )
            frame[:width, :, :3] = red
            frame[-width:, :, :3] = red
            frame[:, :width, :3] = red
            frame[:, -width:, :3] = red
        marked.append(frame)
    return marked


def _even_rgb(image: np.ndarray) -> np.ndarray:
    """Return contiguous uint8 RGB with dimensions accepted by yuv420p."""

    frame = np.asarray(image)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    pad_h, pad_w = frame.shape[0] % 2, frame.shape[1] % 2
    if pad_h or pad_w:
        frame = np.pad(
            frame,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="edge",
        )
    return np.ascontiguousarray(frame[..., :3])


def _write_marked_video(
    path: Path,
    images: Sequence[np.ndarray],
    *,
    fps: int,
) -> None:
    """Encode marked frames with the same PyAV/libx264 path as LeRobot."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream(
            "libx264",
            rate=int(fps),
            options={"crf": "23", "preset": "veryfast"},
        )
        stream.pix_fmt = "yuv420p"
        first = _even_rgb(images[0])
        stream.height, stream.width = first.shape[:2]
        stream.time_base = Fraction(1, int(fps))
        for frame_index, raw in enumerate(images):
            image = _even_rgb(raw)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = frame_index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@dataclass(frozen=True)
class GenerationResult:
    h5_path: Path
    json_path: Path
    video_paths: tuple[Path, ...]
    episodes: int
    attempts: int


def _patch_json(
    path: Path,
    *,
    task_name: str,
    env_id: str,
    instruction: str,
    instruction_memory: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    data = json.loads(path.read_text())
    episodes = data.get("episodes") or []
    env_info = data.setdefault("env_info", {})
    env_info.update(
        {
            "env_id": env_id,
            "task_name": task_name,
            "instruction": instruction,
            "instruction_memory": instruction_memory,
            "robot_uids": "panda",
            "control_mode": "pd_ee_delta_pose",
            "policy_action_supervision": (
                "execution_phase_only"
                if task_name in OBSERVED_TASKS
                else "all_recorded_steps"
            ),
        }
    )
    for episode_id, (episode, record) in enumerate(zip(episodes, records)):
        episode.update(_json_safe(record))
        episode["episode_id"] = episode_id
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _phase_ids_from_intervals(
    intervals: Mapping[str, Sequence[int]],
    *,
    horizon: int,
) -> np.ndarray:
    """Materialize the final, inclusive phase schedule for observations s_0..s_T."""

    memory = [int(x) for x in intervals["memory"]]
    execution = [int(x) for x in intervals["execution"]]

    phase = np.empty((horizon + 1, 1), dtype=np.int64)
    phase[: memory[0], 0] = 0
    phase[memory[0] : execution[0], 0] = 1
    phase[execution[0] :, 0] = 2
    return phase


def _patch_h5_phase_ids(
    h5_path: Path,
    *,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Finalize per-observation phase labels without changing rollout data."""

    with h5py.File(h5_path, "r+") as h5:
        trajectory_names = sorted(
            (name for name in h5 if name.startswith("traj_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        for trajectory_name, record in zip(trajectory_names, records):
            trajectory = h5[trajectory_name]
            horizon = int(trajectory["actions"].shape[0])
            dataset = trajectory["obs/extra/phase_id"]
            finalized = _phase_ids_from_intervals(
                record["phase_intervals"],
                horizon=horizon,
            )
            dataset[...] = finalized.astype(dataset.dtype, copy=False)


def _patch_h5_policy_action_masks(
    h5_path: Path,
    *,
    task_name: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Mark which recorded controls are valid lower-policy supervision."""

    with h5py.File(h5_path, "r+") as h5:
        trajectory_names = sorted(
            (name for name in h5 if name.startswith("traj_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        for trajectory_name in trajectory_names:
            trajectory = h5[trajectory_name]
            horizon = int(trajectory["actions"].shape[0])
            phase = np.asarray(trajectory["obs/extra/phase_id"][:horizon]).reshape(-1)
            if task_name in OBSERVED_TASKS:
                mask = phase == 2
            else:
                mask = np.ones(horizon, dtype=bool)
            if "policy_action_mask" in trajectory:
                del trajectory["policy_action_mask"]
            dataset = trajectory.create_dataset(
                "policy_action_mask",
                data=mask.astype(np.bool_),
                dtype=np.bool_,
            )
            dataset.attrs["meaning"] = (
                "true iff this frame is valid lower-VLA action supervision; "
                "false actions are retained only for exact simulator replay"
            )


def _encode_h5_base_camera_videos(
    h5_path: Path,
    *,
    output_dir: Path,
    task_name: str,
    records: Sequence[Mapping[str, Any]],
    fps: int,
) -> tuple[Path, ...]:
    """Encode exactly T base-camera frames for each T-action trajectory."""

    paths: list[Path] = []
    with h5py.File(h5_path, "r") as h5:
        trajectory_names = sorted(
            (name for name in h5 if name.startswith("traj_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        for episode_id, (trajectory_name, record) in enumerate(
            zip(trajectory_names, records)
        ):
            trajectory = h5[trajectory_name]
            action_count = int(trajectory["actions"].shape[0])
            rgb_path = "obs/sensor_data/base_camera/rgb"
            rgb = trajectory[rgb_path]
            # ManiSkill observations contain s_0 ... s_T.  Dataset/video contract
            # uses the first T images paired with actions a_0 ... a_{T-1}.
            frames = np.asarray(rgb[:action_count])
            if frames.ndim == 5 and frames.shape[1] == 1:
                frames = frames[:, 0]
            frames = frames[..., :3].astype(np.uint8, copy=False)
            marked = draw_red_keyframe_borders(
                frames,
                record["keyframe_metadata"],
            )
            video_name = f"{task_name}_episode_{episode_id:03d}"
            path = output_dir / f"{video_name}.mp4"
            _write_marked_video(path, marked, fps=int(fps))
            paths.append(path)
    return tuple(paths)


def record_sanity_episodes(
    *,
    env_id: str,
    task_name: str,
    output_dir: Path,
    episodes: int = 5,
    base_seed: int = 0,
    max_episode_steps: int = 4000,
    max_attempts: int | None = None,
    camera_resolution: int = DEFAULT_CAMERA_RESOLUTION,
    shader: str | None = "default",
    save_video: bool = True,
    env_kwargs: Mapping[str, Any] | None = None,
) -> GenerationResult:
    """Generate successful physical expert episodes and rich GT metadata."""

    requested_cuda_devices = os.getenv(
        "MEMORY_BENCHMARK_CUDA_VISIBLE_DEVICES"
    )
    if requested_cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = requested_cuda_devices

    max_attempts = int(max_attempts or (4 * episodes + 8))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    make_kwargs: dict[str, Any] = {
        "max_episode_steps": int(max_episode_steps),
        "render_mode": "sensors",
        "num_envs": 1,
        "parallel_in_single_scene": False,
        "obs_mode": "rgb",
        "control_mode": "pd_ee_delta_pose",
        "robot_uids": "panda",
        "mode": "auto",
        "sensor_configs": {
            "width": int(camera_resolution),
            "height": int(camera_resolution),
        },
    }
    if shader not in (None, "", "default"):
        make_kwargs["shader_dir"] = str(shader)
    for key, value in dict(env_kwargs or {}).items():
        make_kwargs[key] = value

    env = gym.make(env_id, **make_kwargs)
    env = RecordEpisode(
        env,
        output_dir=str(output_dir),
        trajectory_name=task_name,
        save_trajectory=True,
        # Videos are intentionally encoded after H5 is finalized from the exact
        # base-camera tensor used by downstream conversion (not render_camera).
        save_video=False,
        save_on_reset=False,
        info_on_video=False,
        max_steps_per_video=int(max_episode_steps) + 1,
        record_env_state=True,
        video_fps=int(env.unwrapped.control_freq),
        avoid_overwriting_video=False,
        source_type="expert",
        source_desc="v3 closed-loop physical waypoint expert",
    )
    task_env = env.unwrapped
    if task_env.TASK_NAME != task_name:
        env.close()
        raise ValueError(
            f"environment TASK_NAME={task_env.TASK_NAME!r}, expected {task_name!r}"
        )
    records: list[dict[str, Any]] = []
    kept = 0
    attempts = 0
    seed = int(base_seed)
    try:
        while kept < episodes and attempts < max_attempts:
            # A recorded episode must be reproducible from its own seed and
            # metadata, independent of every episode attempted before it.
            # Rebuilding the scene clears PhysX contact/solver caches that can
            # otherwise leak millimetre-scale history into long manipulation
            # trajectories and break fresh-environment replay fidelity.
            reset_options: dict[str, Any] = {"reconfigure": True}
            env.reset(seed=seed, options=reset_options)
            attempts += 1
            final_info: Mapping[str, Any] = {}
            steps = 0
            while steps < max_episode_steps:
                action = task_env.get_expert_action()
                _, _, terminated, truncated, final_info = env.step(action)
                steps += 1
                if (
                    _scalar_bool(final_info.get("success", False))
                    or _scalar_bool(final_info.get("fail", False))
                    or _scalar_bool(terminated)
                    or _scalar_bool(truncated)
                ):
                    break
            success = _scalar_bool(final_info.get("success", False))
            if task_name in EVALUATION_SUCCESS_TASKS:
                success = success and _scalar_bool(
                    final_info.get("evaluation_success", False)
                )
            if not success:
                env.flush_trajectory(save=False)
                seed += 1
                continue

            keyframes = task_env.get_keyframe_metadata(episode_id=kept)
            episode_metadata = task_env.get_episode_metadata()
            episode_metadata["keyframe_metadata"] = keyframes
            episode_metadata["mem_boundary"] = (
                task_env.mem_boundary_tensor()[0].detach().cpu().tolist()
            )
            phase_intervals = task_env.get_phase_intervals()
            phase_intervals["execution"][1] = steps
            # Keep the schema's nested replay metadata consistent with the
            # finalized top-level annotation.  The environment cannot know T
            # until rollout termination, so get_episode_metadata() initially
            # carries execution=[start, -1].
            episode_metadata["phase_intervals"] = {
                name: [int(bounds[0]), int(bounds[1])]
                for name, bounds in phase_intervals.items()
            }
            record = {
                "episode_seed": seed,
                "episode_metadata": episode_metadata,
                "mem_boundary": episode_metadata["mem_boundary"],
                "keyframe_metadata": keyframes,
                "phase_intervals": phase_intervals,
                # Observation timestamps (s_0 through s_T); the same values are
                # stored framewise in H5 obs/extra/timestamp.
                "timestamps": [i / float(task_env.control_freq) for i in range(steps + 1)],
            }
            env.flush_trajectory()
            records.append(record)
            kept += 1
            seed += 1
    finally:
        h5_path = Path(env._h5_file.filename)
        env.close()

    if kept != episodes:
        raise RuntimeError(
            f"only generated {kept}/{episodes} successful {task_name} episodes "
            f"after {attempts} attempts"
        )
    json_path = h5_path.with_suffix(".json")
    _patch_h5_phase_ids(h5_path, records=records)
    _patch_h5_policy_action_masks(
        h5_path,
        task_name=task_name,
        records=records,
    )
    _patch_json(
        json_path,
        task_name=task_name,
        env_id=env_id,
        instruction=str(task_env.INSTRUCTION_TASK),
        instruction_memory=str(task_env.INSTRUCTION_MEMORY),
        records=records,
    )
    video_paths = (
        _encode_h5_base_camera_videos(
            h5_path,
            output_dir=output_dir,
            task_name=task_name,
            records=records,
            fps=int(task_env.control_freq),
        )
        if save_video
        else ()
    )
    return GenerationResult(
        h5_path=h5_path,
        json_path=json_path,
        video_paths=tuple(video_paths),
        episodes=kept,
        attempts=attempts,
    )


def task_generation_main(
    *, task_name: str, env_id: str, visual_ood_env_id: str | None = None
) -> None:
    parser = argparse.ArgumentParser(description=f"Generate {task_name} v3 demos")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--camera-resolution", type=int, default=DEFAULT_CAMERA_RESOLUTION)
    parser.add_argument("--shader", default="default")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--visual-ood",
        action="store_true",
        help="generate the registered visual-OOD environment instead of the standard one",
    )
    parser.add_argument("--env-kwargs-json", default="{}")
    args = parser.parse_args()
    env_kwargs = json.loads(args.env_kwargs_json)
    selected_env_id = str(visual_ood_env_id if args.visual_ood else env_id)
    result = record_sanity_episodes(
        env_id=selected_env_id,
        task_name=task_name,
        output_dir=args.output_dir,
        episodes=args.episodes,
        base_seed=args.base_seed,
        max_episode_steps=args.steps,
        max_attempts=args.max_attempts,
        camera_resolution=args.camera_resolution,
        shader=args.shader,
        save_video=not args.no_video,
        env_kwargs=env_kwargs,
    )
    print(
        json.dumps(
            {
                "task": task_name,
                "visual_ood": bool(args.visual_ood),
                "env_id": selected_env_id,
                "episodes": result.episodes,
                "attempts": result.attempts,
                "h5": str(result.h5_path),
                "json": str(result.json_path),
                "videos": [str(x) for x in result.video_paths],
            },
            ensure_ascii=False,
        )
    )


__all__ = [
    "GenerationResult",
    "draw_red_keyframe_borders",
    "record_sanity_episodes",
    "task_generation_main",
]
