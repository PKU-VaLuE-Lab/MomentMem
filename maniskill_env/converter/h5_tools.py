"""Strict readers for ManiSkill ``RecordEpisode`` HDF5 trajectories."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np


def sorted_traj_ids(h5: h5py.File) -> list[str]:
    keys = [key for key in h5.keys() if key.startswith("traj_")]
    return sorted(keys, key=lambda key: int(key.split("_", 1)[1]))


def list_rgb_camera_streams(obs: h5py.Group) -> dict[str, h5py.Dataset]:
    streams: dict[str, h5py.Dataset] = {}
    if "sensor_data" not in obs:
        return streams
    for uid, group in obs["sensor_data"].items():
        if isinstance(group, h5py.Group) and "rgb" in group:
            streams[str(uid)] = group["rgb"]
    return streams


def sort_camera_uids(uids: list[str]) -> list[str]:
    remaining = set(uids)
    ordered: list[str] = []
    for required in ("base_camera", "hand_camera"):
        if required in remaining:
            ordered.append(required)
            remaining.remove(required)
    ordered.extend(sorted(remaining))
    return ordered


def _read_tree(item: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(item, h5py.Dataset):
        return item[:]
    return {name: _read_tree(value) for name, value in item.items()}


def _load_qpos(obs: h5py.Group) -> np.ndarray | None:
    if "agent" not in obs:
        return None
    agent = obs["agent"]
    if "qpos" in agent:
        return agent["qpos"][:].astype(np.float32)
    # Kept for compatibility with old multi-agent H5s; all v3 tasks are single Panda.
    for uid in sorted(agent.keys()):
        group = agent[uid]
        if isinstance(group, h5py.Group) and "qpos" in group:
            return group["qpos"][:].astype(np.float32)
    return None


def load_episode_tensors(trajectory: h5py.Group) -> dict[str, Any]:
    result: dict[str, Any] = {
        "actions": trajectory["actions"][:].astype(np.float32)
    }
    if "policy_action_mask" in trajectory:
        result["policy_action_mask"] = trajectory["policy_action_mask"][:].astype(bool)
    if "obs" not in trajectory:
        return result
    obs = trajectory["obs"]
    streams = list_rgb_camera_streams(obs)
    if streams:
        result["rgb_by_camera"] = streams
    qpos = _load_qpos(obs)
    if qpos is not None:
        result["qpos"] = qpos
    if "extra" in obs:
        result["obs_extra"] = _read_tree(obs["extra"])
    if "sensor_param" in obs:
        result["sensor_param"] = _read_tree(obs["sensor_param"])
    if "env_states" in trajectory:
        result["env_states"] = _read_tree(trajectory["env_states"])
    return result
