"""Wire protocol and safe action-space registry shared by benchmark test clients."""

from __future__ import annotations

import functools

import msgpack
import numpy as np


POLICY_STATE_DIM = 8
POLICY_STATE_REPRESENTATION = "absolute_joint7+physical_gripper1"
POLICY_ACTION_SPACE = "rel_ee"
POLICY_ACTION_DIM = 10
POLICY_ACTION_REPRESENTATION = (
    "base_camera_relative_xyz3+relative_rot6d6+gripper_command1"
)
POLICY_CONTROL_MODE_PROMPT = "<control_mode> end effector <control_mode> "
POLICY_ACTION_HORIZON = 10
POLICY_CONTRACT_SCHEMA_VERSION = 1
POLICY_CONTRACT = {
    "schema_version": POLICY_CONTRACT_SCHEMA_VERSION,
    "state_key": "observation.state_joint",
    "state_dim": POLICY_STATE_DIM,
    "state_representation": POLICY_STATE_REPRESENTATION,
    "state_coordinate_frame": "joint_space",
    "action_key": "action_rel_ee",
    "action_dim": POLICY_ACTION_DIM,
    "policy_action_space": POLICY_ACTION_SPACE,
    "action_representation": POLICY_ACTION_REPRESENTATION,
    "action_coordinate_frame": "base_camera_opencv",
    "rotation_representation": "rot6d_first_two_matrix_columns",
    "rotation_convention": "spatial_relative_rotation_expressed_in_base_camera",
    "control_mode_prompt": POLICY_CONTROL_MODE_PROMPT,
    "action_horizon": POLICY_ACTION_HORIZON,
}

REPLAY_AUDIT_STATE_DIM = 17
REPLAY_AUDIT_STATE_REPRESENTATION = (
    "joint7+physical_gripper1+base_camera_xyz3+rot6d6"
)

ACTION_SPACE_TO_CONTROL_MODE = {
    # ``native`` is the unambiguous name used by the fidelity validator.  Keep
    # ``delta_ee`` as a wire-compatible alias for older task-client scripts.
    "native": "pd_ee_delta_pose",
    "delta_ee": "pd_ee_delta_pose",
    "abs_joint": "pd_joint_pos",
    # Camera-relative Rot6D is converted back to a native PDEE command by the
    # task client before it reaches ManiSkill; it is never passed through as a
    # ten-dimensional controller action.
    "rel_ee": "pd_ee_delta_pose",
    "delta_joint": "pd_joint_delta_pos",
}
ACTION_SPACE_TO_COLUMN = {
    # The 17-D ``action`` compatibility/audit column must never be passed to a
    # ManiSkill controller. Training uses the 10-D ``action_rel_ee`` column;
    # exact pd_ee replay uses the lossless 7-D native column.
    "native": "action_native",
    "delta_ee": "action_native",
    "abs_joint": "action_abs_joint",
    "rel_ee": "action_rel_ee",
    "delta_joint": "action_delta_joint",
}
ACTION_SPACE_TO_DIM = {
    "native": 7,
    "delta_ee": 7,
    "abs_joint": 8,
    "rel_ee": 10,
    "delta_joint": 8,
}
REPLAYABLE_SPACES = tuple(ACTION_SPACE_TO_CONTROL_MODE)


def _pack_numpy(value):
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _unpack_numpy(value):
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


packb = functools.partial(msgpack.packb, default=_pack_numpy)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_numpy, raw=False)
