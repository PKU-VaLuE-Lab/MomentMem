"""Canonical cameras and embodiment geometry for every v3 task."""

from __future__ import annotations

from typing import Any

import sapien

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils


DEFAULT_CAMERA_RESOLUTION = 224

BASE_CAMERA_UID = "base_camera"
BASE_CAMERA_EYE = (0.56, -0.64, 0.68)
BASE_CAMERA_TARGET = (-0.02, 0.0, 0.08)
BASE_CAMERA_FOV = 1.05
BASE_CAMERA_NEAR = 0.01
BASE_CAMERA_FAR = 100.0

HAND_CAMERA_UID = "hand_camera"
HAND_CAMERA_MOUNT_LINK = "panda_hand"
HAND_CAMERA_POSITION = (0.0464982, -0.0200011, 0.0360011)
HAND_CAMERA_QUATERNION_WXYZ = (0.0, 0.70710678, 0.0, 0.70710678)
HAND_CAMERA_FOV = 1.4
HAND_CAMERA_NEAR = 0.01
HAND_CAMERA_FAR = 100.0

HUMAN_CAMERA_UID = "render_camera"
HUMAN_CAMERA_RESOLUTION = 512


def _single_panda(agent: Any):
    agents = getattr(agent, "agents", None)
    if agents is not None:
        return agents[0]
    return agent


def make_base_camera(
    *,
    width: int = DEFAULT_CAMERA_RESOLUTION,
    height: int = DEFAULT_CAMERA_RESOLUTION,
) -> CameraConfig:
    return CameraConfig(
        BASE_CAMERA_UID,
        sapien_utils.look_at(eye=BASE_CAMERA_EYE, target=BASE_CAMERA_TARGET),
        int(width),
        int(height),
        BASE_CAMERA_FOV,
        BASE_CAMERA_NEAR,
        BASE_CAMERA_FAR,
    )


def make_hand_camera(
    agent: Any,
    *,
    width: int = DEFAULT_CAMERA_RESOLUTION,
    height: int = DEFAULT_CAMERA_RESOLUTION,
) -> CameraConfig:
    panda = _single_panda(agent)
    try:
        mount = panda.robot.links_map[HAND_CAMERA_MOUNT_LINK]
    except KeyError as exc:
        raise RuntimeError(
            f"ordinary Panda must expose link {HAND_CAMERA_MOUNT_LINK!r}"
        ) from exc
    return CameraConfig(
        HAND_CAMERA_UID,
        sapien.Pose(
            p=HAND_CAMERA_POSITION,
            q=HAND_CAMERA_QUATERNION_WXYZ,
        ),
        int(width),
        int(height),
        HAND_CAMERA_FOV,
        HAND_CAMERA_NEAR,
        HAND_CAMERA_FAR,
        mount=mount,
    )


def make_human_camera() -> CameraConfig:
    return CameraConfig(
        HUMAN_CAMERA_UID,
        sapien_utils.look_at(eye=BASE_CAMERA_EYE, target=BASE_CAMERA_TARGET),
        HUMAN_CAMERA_RESOLUTION,
        HUMAN_CAMERA_RESOLUTION,
        BASE_CAMERA_FOV,
        BASE_CAMERA_NEAR,
        BASE_CAMERA_FAR,
    )


def camera_contract(
    *,
    width: int = DEFAULT_CAMERA_RESOLUTION,
    height: int = DEFAULT_CAMERA_RESOLUTION,
) -> dict[str, object]:
    """Return a JSON-serializable contract saved in every episode."""

    return {
        "base_camera": {
            "uid": BASE_CAMERA_UID,
            "width": int(width),
            "height": int(height),
            "eye": list(BASE_CAMERA_EYE),
            "target": list(BASE_CAMERA_TARGET),
            "fov": BASE_CAMERA_FOV,
            "near": BASE_CAMERA_NEAR,
            "far": BASE_CAMERA_FAR,
        },
        "hand_camera": {
            "uid": HAND_CAMERA_UID,
            "mount_link": HAND_CAMERA_MOUNT_LINK,
            "width": int(width),
            "height": int(height),
            "position": list(HAND_CAMERA_POSITION),
            "quaternion_wxyz": list(HAND_CAMERA_QUATERNION_WXYZ),
            "fov": HAND_CAMERA_FOV,
            "near": HAND_CAMERA_NEAR,
            "far": HAND_CAMERA_FAR,
        },
    }


__all__ = [
    "DEFAULT_CAMERA_RESOLUTION",
    "BASE_CAMERA_UID",
    "HAND_CAMERA_UID",
    "make_base_camera",
    "make_hand_camera",
    "make_human_camera",
    "camera_contract",
]
