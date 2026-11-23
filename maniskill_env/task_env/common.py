"""Shared runtime and real-physics construction helpers for all v3 tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.sapien_env import BaseEnv
import mani_skill.agents.robots.panda  # noqa: F401 - register ordinary Panda
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.types import (
    DefaultMaterialsConfig,
    SceneConfig,
    SimConfig,
)

try:  # package import (task_env.common)
    from .action_contract import REPLAY_QUANTIZED_TASKS, quantize_native_arm_action
    from .camera_config import (
        DEFAULT_CAMERA_RESOLUTION,
        camera_contract,
        make_base_camera,
        make_hand_camera,
        make_human_camera,
    )
except ImportError:  # flat task-local launcher with task_env on PYTHONPATH
    from action_contract import REPLAY_QUANTIZED_TASKS, quantize_native_arm_action
    from camera_config import (
        DEFAULT_CAMERA_RESOLUTION,
        camera_contract,
        make_base_camera,
        make_hand_camera,
        make_human_camera,
    )


TASK_NAMES = (
    "AO_hidden_pattern",
    "SO_spring_compression",
    "ASO_placement_imitation",
    "AT_drawer_match",
    "ST_flash_beacon",
    "AST_mine_sweeper",
    "AD_scroll_discovery",
    "SD_explore_cube",
    "ASD_seal_checking",
)
ALL_TASK_NAMES = TASK_NAMES
TASK_ENV_IDS = {name: f"{name}-v3" for name in ALL_TASK_NAMES}
TASK_OOD_ENV_IDS = {name: f"{name}-visual-ood-v3" for name in ALL_TASK_NAMES}


def _quat2euler_sxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """NumPy-2-compatible static-XYZ quaternion to Euler conversion."""

    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= np.finfo(np.float64).eps:
        return np.zeros(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    sin_pitch = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    return np.asarray(
        (
            math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch),
            math.asin(sin_pitch),
            math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch),
        ),
        dtype=np.float64,
    )

# One embodiment and one reachable tabletop region for the complete suite.
ROBOT_UID = "panda"
ROBOT_BASE_POSITION = (-0.615, 0.0, 0.0)
ROBOT_BASE_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)
ROBOT_INITIAL_QPOS = (
    -0.00355173,
    -0.30373093,
    0.00345598,
    -2.47937608,
    0.00126873,
    2.17971253,
    0.78442591,
    0.04,
    0.04,
)
HOME_TCP_POSITION = (-0.18, 0.0, 0.30)
DOWNWARD_TCP_QUATERNION_WXYZ = (0.0, 1.0, 0.0, 0.0)
WORKSPACE_XY_BOUNDS = ((-0.34, 0.30), (-0.34, 0.34))
WORKSPACE_Z_BOUNDS = (0.0, 0.42)

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0
DELTA_POSITION_SCALE_METERS = 0.1
DELTA_ROTATION_SCALE_RADIANS = 0.1
DEFAULT_MAX_POSITION_COMMAND = 0.22
DEFAULT_MAX_ROTATION_COMMAND = 0.28

# Shared high-contrast appearance bank.  Appearance tasks sample two entries
# without replacement per episode; logical color slots remain 0/1 so the H5
# actor/articulation names and task interfaces do not change.
COLOR_NAMES = (
    "red", "blue", "green", "yellow", "magenta", "cyan", "orange", "violet",
)
COLOR_RGBA = np.asarray(
    [
        (0.95, 0.035, 0.025, 1.0),
        (0.025, 0.10, 0.98, 1.0),
        (0.035, 0.72, 0.12, 1.0),
        (0.98, 0.82, 0.025, 1.0),
        (0.92, 0.025, 0.78, 1.0),
        (0.025, 0.82, 0.88, 1.0),
        (0.98, 0.34, 0.025, 1.0),
        (0.42, 0.08, 0.92, 1.0),
    ],
    dtype=np.float32,
)


def sample_distinct_color_ids(rng: Any, count: int = 2) -> np.ndarray:
    """Sample visually separated colors without replacement."""

    # Exclude pairs that are technically unequal but too similar at 224 px.
    pairs = [
        (i, j)
        for i in range(len(COLOR_NAMES))
        for j in range(i + 1, len(COLOR_NAMES))
        if float(np.linalg.norm(COLOR_RGBA[i, :3] - COLOR_RGBA[j, :3])) >= 0.72
    ]
    pair = pairs[int(rng.randint(len(pairs)))]
    return np.asarray(pair, dtype=np.int64)[rng.permutation(2)]


class TaskPhase(IntEnum):
    """Per-frame phase stored in H5 ``obs/extra/phase_id``."""

    OBSERVATION = 0
    MEMORY = 1
    DELAY = 1  # descriptive alias
    EXECUTION = 2


@dataclass(frozen=True)
class PhysicsMaterial:
    static_friction: float = 0.9
    dynamic_friction: float = 0.7
    restitution: float = 0.02
    density: float = 650.0

    def sapien(self) -> physx.PhysxMaterial:
        return physx.PhysxMaterial(
            float(self.static_friction),
            float(self.dynamic_friction),
            float(self.restitution),
        )


DEFAULT_PHYSICS_MATERIAL = PhysicsMaterial()


def render_material(color: Sequence[float]) -> sapien.render.RenderMaterial:
    rgba = list(float(x) for x in color)
    if len(rgba) == 3:
        rgba.append(1.0)
    return sapien.render.RenderMaterial(
        base_color=rgba,
        roughness=0.65,
        specular=0.15,
    )


def _pose(value: sapien.Pose | Sequence[float] | None) -> sapien.Pose:
    if value is None:
        return sapien.Pose()
    if isinstance(value, sapien.Pose):
        return value
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (3,):
        return sapien.Pose(p=array)
    if array.shape == (7,):
        return sapien.Pose(p=array[:3], q=array[3:])
    raise ValueError("pose must be sapien.Pose, xyz, or xyz+wxyz")


def _half_size(value: Sequence[float]) -> np.ndarray:
    out = np.asarray(value, dtype=np.float32)
    return out


def _finish_actor(builder, *, name: str, pose: sapien.Pose, body_type: str):
    builder.initial_pose = pose
    if body_type == "dynamic":
        return builder.build_dynamic(name)
    if body_type == "static":
        return builder.build_static(name)
    if body_type == "kinematic":
        return builder.build_kinematic(name)
    raise ValueError(f"unknown body type {body_type!r}")


def build_dynamic_box(
    scene,
    name: str,
    half_size: Sequence[float],
    pose: sapien.Pose | Sequence[float],
    *,
    color: Sequence[float] = (0.8, 0.8, 0.8, 1.0),
    physics: PhysicsMaterial = DEFAULT_PHYSICS_MATERIAL,
    mass: float | None = None,
):
    """Build a manipulable box with matching visual and collision geometry."""

    hs = _half_size(half_size)
    builder = scene.create_actor_builder()
    builder.add_box_collision(
        half_size=hs,
        material=physics.sapien(),
        density=float(physics.density),
    )
    builder.add_box_visual(half_size=hs, material=render_material(color))
    actor = _finish_actor(builder, name=name, pose=_pose(pose), body_type="dynamic")
    if mass is not None:
        actor.set_mass(float(mass))
    return actor


def build_static_box(
    scene,
    name: str,
    half_size: Sequence[float],
    pose: sapien.Pose | Sequence[float],
    *,
    color: Sequence[float] = (0.5, 0.5, 0.5, 1.0),
    physics: PhysicsMaterial = DEFAULT_PHYSICS_MATERIAL,
):
    """Build collision-bearing immovable scenery (rail, cabinet, tray)."""

    hs = _half_size(half_size)
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=hs, material=physics.sapien())
    builder.add_box_visual(half_size=hs, material=render_material(color))
    return _finish_actor(builder, name=name, pose=_pose(pose), body_type="static")


def build_visual_marker_box(
    scene,
    name: str,
    half_size: Sequence[float],
    pose: sapien.Pose | Sequence[float],
    *,
    color: Sequence[float],
):
    """Build a non-manipulable visual marker."""

    hs = _half_size(half_size)
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=hs, material=render_material(color))
    return _finish_actor(builder, name=name, pose=_pose(pose), body_type="kinematic")


def build_dynamic_cylinder(
    scene,
    name: str,
    *,
    radius: float,
    half_length: float,
    pose: sapien.Pose | Sequence[float],
    color: Sequence[float] = (0.8, 0.8, 0.8, 1.0),
    physics: PhysicsMaterial = DEFAULT_PHYSICS_MATERIAL,
    mass: float | None = None,
):
    builder = scene.create_actor_builder()
    builder.add_cylinder_collision(
        radius=float(radius),
        half_length=float(half_length),
        material=physics.sapien(),
        density=float(physics.density),
    )
    builder.add_cylinder_visual(
        radius=float(radius),
        half_length=float(half_length),
        material=render_material(color),
    )
    actor = _finish_actor(builder, name=name, pose=_pose(pose), body_type="dynamic")
    if mass is not None:
        actor.set_mass(float(mass))
    return actor


def vertical_cylinder_pose(
    xyz: Sequence[float],
) -> sapien.Pose:
    """Pose a SAPIEN cylinder (whose local axis is X) vertically along world Z."""

    return sapien.Pose(p=xyz, q=euler2quat(0.0, math.pi / 2, 0.0))


def build_dynamic_open_box(
    scene,
    name: str,
    *,
    inner_half_size: Sequence[float],
    wall_height: float,
    wall_thickness: float,
    pose: sapien.Pose | Sequence[float],
    color: Sequence[float] = (0.70, 0.52, 0.30, 1.0),
    physics: PhysicsMaterial = DEFAULT_PHYSICS_MATERIAL,
    mass: float | None = None,
):
    """Build one compound dynamic actor with a floor and four physical walls."""

    inner = np.asarray(inner_half_size, dtype=np.float32)
    hx, hy = float(inner[0]), float(inner[1])
    t = float(wall_thickness)
    h = float(wall_height)
    mat = physics.sapien()
    visual = render_material(color)
    builder = scene.create_actor_builder()

    def part(half, pos):
        local = sapien.Pose(p=pos)
        builder.add_box_collision(
            half_size=half,
            pose=local,
            material=mat,
            density=float(physics.density),
        )
        builder.add_box_visual(half_size=half, pose=local, material=visual)

    part([hx + t, hy + t, t / 2], [0, 0, t / 2])
    part([t / 2, hy + t, h / 2], [-(hx + t / 2), 0, h / 2])
    part([t / 2, hy + t, h / 2], [hx + t / 2, 0, h / 2])
    part([hx, t / 2, h / 2], [0, -(hy + t / 2), h / 2])
    part([hx, t / 2, h / 2], [0, hy + t / 2, h / 2])
    actor = _finish_actor(builder, name=name, pose=_pose(pose), body_type="dynamic")
    if mass is not None:
        actor.set_mass(float(mass))
    return actor


def _axis_quaternion(axis: str) -> np.ndarray:
    if axis == "x":
        return np.asarray([1.0, 0.0, 0.0, 0.0])
    if axis == "y":
        return np.asarray(euler2quat(0.0, 0.0, math.pi / 2))
    if axis == "z":
        return np.asarray(euler2quat(0.0, -math.pi / 2, 0.0))
    raise ValueError("axis must be 'x', 'y', or 'z'")


@dataclass
class JointAssembly:
    articulation: Any
    joint: Any
    moving_link: Any
    lower: float
    upper: float

    def set_qpos(self, value: float | Sequence[float] | torch.Tensor) -> None:
        """Set joint state for reset/replay initialization only."""

        target = torch.as_tensor(
            value,
            dtype=torch.float32,
            device=self.articulation.device,
        )
        if target.ndim == 0:
            target = target.repeat(self.articulation.scene.num_envs)
        self.articulation.set_qpos(target.reshape(-1, 1))

    def drive_to(self, target: float | Sequence[float] | torch.Tensor) -> None:
        """Move through the joint drive; this never teleports the moving link."""

        target_t = torch.as_tensor(
            target,
            dtype=torch.float32,
            device=self.articulation.device,
        )
        if target_t.ndim == 0:
            target_t = target_t.repeat(self.articulation.scene.num_envs)
        target_t = torch.clamp(target_t, self.lower, self.upper).reshape(-1, 1)
        if self.articulation.scene.gpu_sim_enabled:
            indices = torch.tensor([0], dtype=torch.int64, device=target_t.device)
            self.articulation.set_joint_drive_targets(
                target_t,
                joint_indices=indices,
            )
            self.articulation.scene.px.gpu_apply_articulation_target_position()
        else:
            self.articulation.set_joint_drive_targets(
                target_t,
                joints=[self.joint],
            )

    @property
    def qpos(self) -> torch.Tensor:
        return self.articulation.get_qpos()[..., 0]

    @property
    def qvel(self) -> torch.Tensor:
        return self.articulation.get_qvel()[..., 0]


def build_prismatic_articulation(
    scene,
    name: str,
    *,
    root_half_size: Sequence[float],
    moving_half_size: Sequence[float],
    pose: sapien.Pose | Sequence[float],
    moving_offset: Sequence[float] = (0.0, 0.0, 0.0),
    axis: str = "x",
    limits: tuple[float, float] = (0.0, 0.20),
    root_color: Sequence[float] = (0.25, 0.25, 0.25, 1.0),
    moving_color: Sequence[float] = (0.75, 0.75, 0.75, 1.0),
    physics: PhysicsMaterial = DEFAULT_PHYSICS_MATERIAL,
    drive_stiffness: float = 120.0,
    drive_damping: float = 20.0,
    drive_force_limit: float = 80.0,
) -> JointAssembly:
    """Build a fixed-root, collision-bearing one-DOF physical slider."""

    root_hs = _half_size(root_half_size)
    moving_hs = _half_size(moving_half_size)
    lower, upper = (float(limits[0]), float(limits[1]))
    axis_q = _axis_quaternion(axis)
    builder = scene.create_articulation_builder()
    builder.initial_pose = _pose(pose)
    root = builder.create_link_builder(parent=None)
    root.set_name(f"{name}_root")
    root.add_box_collision(
        half_size=root_hs,
        material=physics.sapien(),
        density=float(physics.density),
    )
    root.add_box_visual(half_size=root_hs, material=render_material(root_color))
    moving = builder.create_link_builder(parent=root)
    moving.set_name(f"{name}_moving")
    moving.add_box_collision(
        half_size=moving_hs,
        material=physics.sapien(),
        density=float(physics.density),
    )
    moving.add_box_visual(
        half_size=moving_hs,
        material=render_material(moving_color),
    )
    moving.set_joint_name(f"{name}_joint")
    moving.set_joint_properties(
        type="prismatic",
        limits=[[lower, upper]],
        pose_in_parent=sapien.Pose(p=moving_offset, q=axis_q),
        pose_in_child=sapien.Pose(q=axis_q),
        friction=0.05,
        damping=1.0,
    )
    articulation = builder.build(name, fix_root_link=True)
    joint = articulation.get_active_joints()[0]
    joint.set_drive_properties(
        stiffness=float(drive_stiffness),
        damping=float(drive_damping),
        force_limit=float(drive_force_limit),
    )
    return JointAssembly(
        articulation=articulation,
        joint=joint,
        moving_link=articulation.links_map[f"{name}_moving"],
        lower=lower,
        upper=upper,
    )


def build_revolute_articulation(
    scene,
    name: str,
    *,
    root_half_size: Sequence[float],
    moving_half_size: Sequence[float],
    pose: sapien.Pose | Sequence[float],
    hinge_in_parent: Sequence[float],
    hinge_in_child: Sequence[float],
    axis: str = "y",
    limits: tuple[float, float] = (0.0, math.pi / 2),
    root_color: Sequence[float] = (0.45, 0.32, 0.20, 1.0),
    moving_color: Sequence[float] = (0.70, 0.52, 0.30, 1.0),
    physics: PhysicsMaterial = DEFAULT_PHYSICS_MATERIAL,
    drive_stiffness: float = 40.0,
    drive_damping: float = 8.0,
    drive_force_limit: float = 30.0,
) -> JointAssembly:
    """Build a collision-bearing hinged panel/lid driven by real joint forces."""

    root_hs = _half_size(root_half_size)
    moving_hs = _half_size(moving_half_size)
    lower, upper = (float(limits[0]), float(limits[1]))
    axis_q = _axis_quaternion(axis)
    builder = scene.create_articulation_builder()
    builder.initial_pose = _pose(pose)
    root = builder.create_link_builder(parent=None)
    root.set_name(f"{name}_root")
    root.add_box_collision(
        half_size=root_hs,
        material=physics.sapien(),
        density=float(physics.density),
    )
    root.add_box_visual(half_size=root_hs, material=render_material(root_color))
    moving = builder.create_link_builder(parent=root)
    moving.set_name(f"{name}_moving")
    moving.add_box_collision(
        half_size=moving_hs,
        material=physics.sapien(),
        density=float(physics.density),
    )
    moving.add_box_visual(
        half_size=moving_hs,
        material=render_material(moving_color),
    )
    moving.set_joint_name(f"{name}_joint")
    moving.set_joint_properties(
        type="revolute",
        limits=[[lower, upper]],
        pose_in_parent=sapien.Pose(p=hinge_in_parent, q=axis_q),
        pose_in_child=sapien.Pose(p=hinge_in_child, q=axis_q),
        friction=0.03,
        damping=0.5,
    )
    articulation = builder.build(name, fix_root_link=True)
    joint = articulation.get_active_joints()[0]
    joint.set_drive_properties(
        stiffness=float(drive_stiffness),
        damping=float(drive_damping),
        force_limit=float(drive_force_limit),
    )
    return JointAssembly(
        articulation=articulation,
        joint=joint,
        moving_link=articulation.links_map[f"{name}_moving"],
        lower=lower,
        upper=upper,
    )


WaypointTarget = Sequence[float] | Callable[[Any], Sequence[float]]
WaypointCondition = Callable[[Any], bool]


@dataclass(frozen=True)
class PhysicalWaypoint:
    """One closed-loop Cartesian command; it never moves an object directly."""

    position: WaypointTarget
    gripper: float = GRIPPER_OPEN
    quaternion_wxyz: WaypointTarget = DOWNWARD_TCP_QUATERNION_WXYZ
    position_tolerance: float = 0.012
    rotation_tolerance: float = 0.12
    dwell_steps: int = 2
    label: str = ""
    require: WaypointCondition | None = None
    # The default component-wise clipping matches ManiSkill's native action
    # convention.  A direct carry can opt into Euclidean-norm clipping so a
    # long diagonal move stays on one straight segment without intermediate
    # waypoints (and therefore without waypoint hand-off speed ripples).
    isotropic_position_command: bool = False
    # Optional per-waypoint translation-command cap. This lets a long motion
    # segment use one constant speed without slowing unrelated transfers.
    max_position_command: float | None = None


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


class PhysicalWaypointFSM:
    """Closed-loop Panda expert controller for one physical environment."""

    def __init__(
        self,
        env: Any,
        waypoints: Sequence[PhysicalWaypoint] = (),
        *,
        max_position_command: float = DEFAULT_MAX_POSITION_COMMAND,
        max_rotation_command: float = DEFAULT_MAX_ROTATION_COMMAND,
    ) -> None:
        self.env = env
        self.max_position_command = float(max_position_command)
        self.max_rotation_command = float(max_rotation_command)
        self.set_waypoints(waypoints)

    def set_waypoints(self, waypoints: Sequence[PhysicalWaypoint]) -> None:
        self.waypoints = list(waypoints)
        self.index = 0
        self._dwell = 0
        self._last_gripper = (
            float(self.waypoints[0].gripper) if self.waypoints else GRIPPER_OPEN
        )

    def append(self, *waypoints: PhysicalWaypoint) -> None:
        self.waypoints.extend(waypoints)

    @property
    def complete(self) -> bool:
        return self.index >= len(self.waypoints)

    @property
    def label(self) -> str:
        return "complete" if self.complete else self.waypoints[self.index].label

    def _resolve(self, value: WaypointTarget) -> np.ndarray:
        if callable(value):
            value = value(self.env)
        return np.asarray(value, dtype=np.float32)

    def action(self) -> torch.Tensor:
        device = self.env.agent.tcp.pose.p.device
        if self.complete:
            action = torch.zeros((1, 7), dtype=torch.float32, device=device)
            action[:, 6] = self._last_gripper
            return action
        waypoint = self.waypoints[self.index]
        self._last_gripper = float(waypoint.gripper)
        current_p = self.env.agent.tcp.pose.p[0].detach().cpu().numpy()
        current_q = self.env.agent.tcp.pose.q[0].detach().cpu().numpy()
        target_p = self._resolve(waypoint.position)
        target_q = self._resolve(waypoint.quaternion_wxyz)
        target_q = target_q / np.linalg.norm(target_q)
        delta_p = target_p - current_p
        q_error = _quat_multiply(target_q, current_q * np.asarray([1, -1, -1, -1]))
        if q_error[0] < 0:
            q_error = -q_error
        delta_euler = np.asarray(_quat2euler_sxyz(q_error), dtype=np.float32)
        reached = (
            float(np.linalg.norm(delta_p)) <= waypoint.position_tolerance
            and float(np.linalg.norm(delta_euler)) <= waypoint.rotation_tolerance
        )
        if reached:
            self._dwell += 1
            condition_ok = waypoint.require is None or bool(waypoint.require(self.env))
            if self._dwell >= max(1, int(waypoint.dwell_steps)) and condition_ok:
                self.index += 1
                self._dwell = 0
        else:
            self._dwell = 0
        raw_pos_cmd = delta_p / DELTA_POSITION_SCALE_METERS
        position_command_limit = (
            self.max_position_command
            if waypoint.max_position_command is None
            else float(waypoint.max_position_command)
        )
        if waypoint.isotropic_position_command:
            command_norm = float(np.linalg.norm(raw_pos_cmd))
            if command_norm > position_command_limit:
                pos_cmd = raw_pos_cmd * (
                    position_command_limit / command_norm
                )
            else:
                pos_cmd = raw_pos_cmd
        else:
            pos_cmd = np.clip(
                raw_pos_cmd,
                -position_command_limit,
                position_command_limit,
            )
        # ManiSkill 3.0.0b21's normalized Panda PDEEPoseController multiplies
        # rotation actions by ``rot_lower`` (-0.1), unlike translation's
        # symmetric clip/scale path.  Negate here so a positive world-frame
        # quaternion error produces the intended positive physical rotation.
        rot_cmd = np.clip(
            -delta_euler / DELTA_ROTATION_SCALE_RADIANS,
            -self.max_rotation_command,
            self.max_rotation_command,
        )
        arm_cmd = np.concatenate((pos_cmd, rot_cmd)).astype(np.float32)
        if str(getattr(self.env, "TASK_NAME", "")) in REPLAY_QUANTIZED_TASKS:
            arm_cmd = quantize_native_arm_action(arm_cmd)
        action = np.concatenate(
            (arm_cmd, [float(np.clip(waypoint.gripper, -1.0, 1.0))])
        ).astype(np.float32)
        return torch.as_tensor(action, device=device).unsqueeze(0)


@dataclass
class _KeyframeEvent:
    event_name: str
    start_frames: torch.Tensor
    end_frames: torch.Tensor
    required_information: tuple[str, ...]
    valid_frames: tuple[int, ...] | None = None


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
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"episode metadata contains non-JSON value {type(value).__name__}")


class UnifiedPandaEnv(BaseEnv):
    """Base class fixing the table, Panda, cameras, phases, and GT schema."""

    SUPPORTED_ROBOTS = [ROBOT_UID]
    SUPPORTED_MODES = ("auto", "manual")
    TASK_NAME = ""
    INSTRUCTION_TASK = ""
    INSTRUCTION_MEMORY = ""
    MAX_KEYFRAME_WINDOWS = 3
    # OOD subclasses only flip this flag. Task geometry, dynamics, expert
    # control, phase logic, and success criteria stay inherited unchanged.
    VISUAL_OOD = False

    def __init__(
        self,
        *args,
        robot_uids: str = ROBOT_UID,
        robot_init_qpos_noise: float = 0.0,
        mode: str = "auto",
        **kwargs,
    ) -> None:
        self.mode = mode
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self._manipulable_actors: list[Any] = []
        self._keyframe_events: list[_KeyframeEvent] = []
        self._task_episode_metadata: dict[str, Any] = {}
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        return [make_base_camera(), make_hand_camera(self.agent)]

    @property
    def _default_human_render_camera_configs(self):
        return make_human_camera()

    @property
    def _default_sim_config(self):
        # 10 substeps per 20-Hz control step, CCD and a stronger contact solve
        # materially reduce fingertip tunnelling and stacked-object jitter.
        return SimConfig(
            sim_freq=200,
            control_freq=20,
            scene_config=SceneConfig(
                contact_offset=0.005,
                rest_offset=0.0,
                solver_position_iterations=30,
                solver_velocity_iterations=4,
                enable_pcm=True,
                enable_tgs=True,
                enable_ccd=True,
                enable_friction_every_iteration=True,
            ),
            default_materials_config=DefaultMaterialsConfig(
                static_friction=0.9,
                dynamic_friction=0.7,
                restitution=0.02,
            ),
        )

    def _load_scene(self, options: dict) -> None:
        # ``reconfigure=True`` rebuilds every PhysX actor while keeping this
        # Python environment instance. Do not retain wrappers from old scenes.
        self._manipulable_actors.clear()
        self.table_scene = TableSceneBuilder(
            self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()
        if self.VISUAL_OOD:
            try:
                from .visual_ood import load_visual_ood_scene
            except ImportError:  # flat task-local launcher
                from visual_ood import load_visual_ood_scene
            load_visual_ood_scene(self, options)
        self._load_task_scene(options)

    def _load_lighting(self, options: dict) -> None:
        if not self.VISUAL_OOD:
            super()._load_lighting(options)
            return
        try:
            from .visual_ood import load_visual_ood_lighting
        except ImportError:  # flat task-local launcher
            from visual_ood import load_visual_ood_lighting
        load_visual_ood_lighting(self)

    def _load_task_scene(self, options: dict) -> None:
        raise NotImplementedError

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        self.table_scene.initialize(env_idx)
        count = len(env_idx)
        qpos = torch.as_tensor(
            ROBOT_INITIAL_QPOS,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).repeat(count, 1)
        self.agent.robot.set_qpos(qpos)
        self.agent.robot.set_qvel(torch.zeros_like(qpos))
        self.agent.robot.set_pose(
            sapien.Pose(
                p=ROBOT_BASE_POSITION,
                q=ROBOT_BASE_QUATERNION_WXYZ,
            )
        )
        self._observation_end = torch.zeros(
            self.num_envs,
            dtype=torch.int64,
            device=self.device,
        )
        self._execution_start = torch.ones(
            self.num_envs,
            dtype=torch.int64,
            device=self.device,
        )
        self._keyframe_events = []
        self._task_episode_metadata = {}
        replay = (options or {}).get("episode_metadata")
        if replay is not None:
            replay = dict(replay.get("task", replay))
        self._initialize_task_episode(env_idx, replay)
        if self.VISUAL_OOD:
            visual_metadata = getattr(self, "_visual_ood_metadata", None)
            self.set_task_episode_metadata(visual_ood=visual_metadata)

    def _initialize_task_episode(
        self,
        env_idx: torch.Tensor,
        episode_metadata: Mapping[str, Any] | None,
    ) -> None:
        raise NotImplementedError

    def register_manipulable(self, *actors: Any) -> None:
        """Declare contact-manipulated rigid bodies."""

        self._manipulable_actors.extend(actors)

    def set_task_episode_metadata(self, **values: Any) -> None:
        self._task_episode_metadata.update(_json_safe(values))

    def set_phase_schedule(
        self,
        observation_end_step: int | Sequence[int] | torch.Tensor,
        execution_start_step: int | Sequence[int] | torch.Tensor,
    ) -> None:
        obs_end = torch.as_tensor(
            observation_end_step,
            dtype=torch.int64,
            device=self.device,
        ).reshape(-1)
        exe_start = torch.as_tensor(
            execution_start_step,
            dtype=torch.int64,
            device=self.device,
        ).reshape(-1)
        if obs_end.numel() == 1:
            obs_end = obs_end.repeat(self.num_envs)
        if exe_start.numel() == 1:
            exe_start = exe_start.repeat(self.num_envs)
        self._observation_end = obs_end
        self._execution_start = exe_start

    @property
    def phase_id(self) -> torch.Tensor:
        step = self.elapsed_steps
        return torch.where(
            step <= self._observation_end,
            int(TaskPhase.OBSERVATION),
            torch.where(
                step < self._execution_start,
                int(TaskPhase.MEMORY),
                int(TaskPhase.EXECUTION),
            ),
        ).to(torch.int64)

    def define_keyframe_event(
        self,
        event_name: str,
        start_frame: int | Sequence[int] | torch.Tensor,
        end_frame: int | Sequence[int] | torch.Tensor,
        *,
        required_information: Sequence[str],
        valid_frames: Sequence[int] | None = None,
    ) -> None:
        start = torch.as_tensor(start_frame, dtype=torch.int64, device=self.device).reshape(-1)
        end = torch.as_tensor(end_frame, dtype=torch.int64, device=self.device).reshape(-1)
        if start.numel() == 1:
            start = start.repeat(self.num_envs)
        if end.numel() == 1:
            end = end.repeat(self.num_envs)
        existing = next(
            (i for i, event in enumerate(self._keyframe_events) if event.event_name == event_name),
            None,
        )
        event = _KeyframeEvent(
            event_name=str(event_name),
            start_frames=start,
            end_frames=end,
            required_information=tuple(str(x) for x in required_information),
            valid_frames=(None if valid_frames is None else tuple(int(x) for x in valid_frames)),
        )
        if existing is None:
            self._keyframe_events.append(event)
        else:
            self._keyframe_events[existing] = event

    def mem_boundary_tensor(self) -> torch.Tensor:
        out = torch.full(
            (self.num_envs, 2 * self.MAX_KEYFRAME_WINDOWS),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        for i, event in enumerate(self._keyframe_events):
            out[:, 2 * i] = event.start_frames
            out[:, 2 * i + 1] = event.end_frames
        return out

    def get_keyframe_metadata(self, episode_id: int = 0, env_index: int = 0) -> list[dict[str, Any]]:
        fps = float(self.control_freq)
        result = []
        for event in self._keyframe_events:
            start = int(event.start_frames[env_index].item())
            end = int(event.end_frames[env_index].item())
            valid = (
                list(range(start, end + 1))
                if event.valid_frames is None
                else [x for x in event.valid_frames if start <= x <= end]
            )
            result.append(
                {
                    "task_name": self.TASK_NAME,
                    "episode_id": int(episode_id),
                    "event": event.event_name,
                    "event_name": event.event_name,
                    "start_frame": start,
                    "end_frame": end,
                    "start_time": start / fps,
                    "end_time": end / fps,
                    "valid_frames": valid,
                    "required_information": list(event.required_information),
                }
            )
        return result

    def get_phase_intervals(self, env_index: int = 0) -> dict[str, list[int]]:
        obs_end = int(self._observation_end[env_index].item())
        execute = int(self._execution_start[env_index].item())
        return {
            "observation": [0, obs_end],
            "memory": [obs_end + 1, execute - 1],
            "execution": [execute, -1],
        }

    def get_episode_metadata(self, env_index: int = 0) -> dict[str, Any]:
        width = DEFAULT_CAMERA_RESOLUTION
        height = DEFAULT_CAMERA_RESOLUTION
        sensors = getattr(self, "_sensors", {})
        if "base_camera" in sensors:
            width = int(sensors["base_camera"].config.width)
            height = int(sensors["base_camera"].config.height)
        return {
            "schema_version": 3,
            "task_name": self.TASK_NAME,
            "env_id": (
                TASK_OOD_ENV_IDS[self.TASK_NAME]
                if self.VISUAL_OOD
                else TASK_ENV_IDS[self.TASK_NAME]
            ),
            "robot_uids": ROBOT_UID,
            "control_mode": "pd_ee_delta_pose",
            "camera_resolution": width if width == height else [width, height],
            "cameras": camera_contract(width=width, height=height),
            "phase_intervals": self.get_phase_intervals(env_index),
            "mem_boundary": self.mem_boundary_tensor()[env_index].detach().cpu().tolist(),
            "keyframe_metadata": self.get_keyframe_metadata(
                episode_id=0,
                env_index=env_index,
            ),
            "task": _json_safe(self._task_episode_metadata),
        }

    def _get_obs_extra(self, info: dict[str, Any]):
        extra = {
            "step_index": self.elapsed_steps.to(torch.int64).unsqueeze(-1),
            "timestamp": (self.elapsed_steps.to(torch.float32) / float(self.control_freq)).unsqueeze(-1),
            "phase_id": self.phase_id.unsqueeze(-1),
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "gripper": self.agent.robot.get_qpos()[..., -1:].to(torch.float32),
        }
        task_extra = self._get_task_obs_extra(info)
        if task_extra:
            extra.update(task_extra)
        return extra

    def _get_task_obs_extra(self, info: dict[str, Any]) -> dict[str, Any]:
        return {}

    def evaluate(self) -> dict[str, Any]:
        info = dict(self._evaluate_task())
        info.update(
            {
                "mem_boundary": self.mem_boundary_tensor(),
                "phase_id": self.phase_id,
                "timestamp": self.elapsed_steps.to(torch.float32) / float(self.control_freq),
            }
        )
        return info

    def _evaluate_task(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict[str, Any]):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict[str, Any],
    ):
        return self.compute_dense_reward(obs, action, info)

    def _before_control_step(self) -> None:
        self._before_task_control_step()

    def _after_control_step(self) -> None:
        self._after_task_control_step()

    def _before_task_control_step(self) -> None:
        pass

    def _after_task_control_step(self) -> None:
        pass

    def get_expert_action(self) -> torch.Tensor:
        """Return the actual action that the recorder must pass to ``env.step``."""

        action = self._expert_action()
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.shape == (7,):
            action = action.unsqueeze(0)
        return torch.clamp(action, -1.0, 1.0)

    def _expert_action(self) -> torch.Tensor:
        raise NotImplementedError

    def zero_action(self, gripper: float = GRIPPER_OPEN) -> torch.Tensor:
        action = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        action[:, 6] = float(np.clip(gripper, -1.0, 1.0))
        return action


def assert_points_reachable(points: Sequence[Sequence[float]], *, margin: float = 0.0) -> None:
    """Fast shared layout guard; runtime smoke additionally checks Panda IK."""

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    (xmin, xmax), (ymin, ymax) = WORKSPACE_XY_BOUNDS
    zmin, zmax = WORKSPACE_Z_BOUNDS
    ok = (
        (pts[:, 0] >= xmin + margin)
        & (pts[:, 0] <= xmax - margin)
        & (pts[:, 1] >= ymin + margin)
        & (pts[:, 1] <= ymax - margin)
        & (pts[:, 2] >= zmin + margin)
        & (pts[:, 2] <= zmax - margin)
    )
    if not np.all(ok):
        raise ValueError(f"points outside canonical Panda workspace: {pts[~ok].tolist()}")


def dump_jsonable(value: Any) -> str:
    """Used by smoke tests to fail on non-replayable episode metadata."""

    return json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False)


__all__ = [
    "ALL_TASK_NAMES",
    "TASK_NAMES",
    "TASK_ENV_IDS",
    "TASK_OOD_ENV_IDS",
    "ROBOT_UID",
    "ROBOT_BASE_POSITION",
    "ROBOT_INITIAL_QPOS",
    "HOME_TCP_POSITION",
    "DOWNWARD_TCP_QUATERNION_WXYZ",
    "DEFAULT_MAX_POSITION_COMMAND",
    "DEFAULT_MAX_ROTATION_COMMAND",
    "COLOR_NAMES",
    "COLOR_RGBA",
    "sample_distinct_color_ids",
    "GRIPPER_OPEN",
    "GRIPPER_CLOSED",
    "TaskPhase",
    "PhysicsMaterial",
    "JointAssembly",
    "PhysicalWaypoint",
    "PhysicalWaypointFSM",
    "UnifiedPandaEnv",
    "render_material",
    "build_dynamic_box",
    "build_static_box",
    "build_visual_marker_box",
    "build_dynamic_cylinder",
    "vertical_cylinder_pose",
    "build_dynamic_open_box",
    "build_prismatic_articulation",
    "build_revolute_articulation",
    "assert_points_reachable",
    "dump_jsonable",
]
