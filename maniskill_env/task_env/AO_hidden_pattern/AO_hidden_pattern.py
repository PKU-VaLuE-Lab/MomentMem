"""Observed+Appearance task with real hinged, pushable boxes."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import sapien
import torch

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

_TASK_ENV_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ENV_ROOT))

from common import (  # noqa: E402
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_TCP_POSITION,
    JointAssembly,
    PhysicalWaypoint,
    PhysicalWaypointFSM,
    PhysicsMaterial,
    TASK_ENV_IDS,
    UnifiedPandaEnv,
    build_static_box,
    render_material,
)
try:  # package import and direct-script import are both supported
    from .AO_hidden_pattern_cfg import CFG
except ImportError:  # pragma: no cover - direct gen_data.py execution
    from AO_hidden_pattern_cfg import CFG  # type: ignore


TASK_NAME = "AO_hidden_pattern"
ENV_ID = TASK_ENV_IDS[TASK_NAME]

INSTRUCTION_TASK = (
    "Three boxes open together to show three different patterns, then close. "
    "A target card appears briefly with one of those patterns and then disappears. "
    "Then push forward the box that showed the same pattern as the target card."
)
INSTRUCTION_MEMORY = (
    "Remember the moment when all three boxes are open and their patterns are visible, "
    "and the later moment when the target card appears."
)

PATTERN_NAMES = (
    "triangle", "plus", "horizontal_bars", "x", "diamond",
    "center_square", "corners", "checker",
)
PATTERN_MASKS = np.asarray(
    [
        [[0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 0, 0, 1],
         [1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0]],
        [[0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]],
        [[1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1]],
        [[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]],
        [[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
        [[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]],
        [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
    ],
    dtype=np.bool_,
)


def _add_pattern(builder, *, horizontal: bool, extent: float) -> list[Any]:
    """Add the same binary high-contrast mark to a box insert or query card."""

    grid = int(CFG["texture_grid_size"])
    pitch = 2.0 * extent / grid
    half = pitch * 0.45
    materials = []
    for row in range(grid):
        for col in range(grid):
            a = -extent + (col + 0.5) * pitch
            b = extent - (row + 0.5) * pitch
            if horizontal:
                pose, half_size = sapien.Pose([a, b, 0.010]), [half, half, 0.0008]
            else:
                pose, half_size = sapien.Pose([a, -0.0035, b]), [half, 0.0008, half]
            material = render_material((0.96, 0.96, 0.91, 1.0))
            materials.append(material)
            builder.add_box_visual(
                pose=pose,
                half_size=half_size,
                material=material,
            )
    return materials


def _set_pattern(materials: list[Any], pattern_id: int) -> None:
    cells = PATTERN_MASKS[int(pattern_id)]
    for index, material in enumerate(materials):
        row, col = divmod(index, 5)
        material.set_base_color(
            (0.015, 0.015, 0.015, 1.0)
            if bool(cells[row, col])
            else (0.96, 0.96, 0.91, 1.0)
        )


@dataclass
class _HingedBox:
    assembly: JointAssembly
    root_link: Any
    cell_materials: list[Any]


def _build_hinged_box(scene, name: str, slot: int) -> _HingedBox:
    """Build a movable free-root box and a force-driven revolute lid."""

    hx, hy = (float(x) for x in CFG["box_inner_half_xy"])
    h = float(CFG["box_wall_height"])
    t = float(CFG["box_wall_thickness"])
    lid_t = float(CFG["lid_thickness"])
    physics = PhysicsMaterial(static_friction=1.25, dynamic_friction=0.95, density=900)
    phys_mat = physics.sapien()
    box_mat = render_material((0.66, 0.45, 0.24, 1.0))
    lid_mat = render_material((0.76, 0.57, 0.33, 1.0))

    builder = scene.create_articulation_builder()
    builder.initial_pose = sapien.Pose(
        [float(CFG["slot_x"]), float(CFG["slot_y"][slot]), 0.0]
    )
    root = builder.create_link_builder(parent=None)
    root.set_name(f"{name}_root")

    def add_root_part(half_size, position):
        pose = sapien.Pose(position)
        root.add_box_collision(
            pose=pose,
            half_size=half_size,
            material=phys_mat,
            density=physics.density,
        )
        root.add_box_visual(pose=pose, half_size=half_size, material=box_mat)

    add_root_part([hx + t, hy + t, t / 2], [0, 0, t / 2])
    add_root_part([t / 2, hy + t, h / 2], [-(hx + t / 2), 0, h / 2])
    add_root_part([t / 2, hy + t, h / 2], [hx + t / 2, 0, h / 2])
    add_root_part([hx, t / 2, h / 2], [0, -(hy + t / 2), h / 2])
    add_root_part([hx, t / 2, h / 2], [0, hy + t / 2, h / 2])
    root.add_box_visual(
        pose=sapien.Pose([0, 0, 0.008]),
        half_size=[hx * 0.90, hy * 0.90, 0.002],
        material=render_material((0.92, 0.92, 0.88, 1.0)),
    )
    cell_materials = _add_pattern(root, horizontal=True, extent=min(hx, hy) * 0.82)

    lid = builder.create_link_builder(parent=root)
    lid.set_name(f"{name}_lid")
    outer_x, outer_y = hx + t, hy + t
    lid.add_box_collision(
        half_size=[outer_x, outer_y, lid_t / 2],
        material=phys_mat,
        density=physics.density,
    )
    lid.add_box_visual(
        half_size=[outer_x, outer_y, lid_t / 2],
        material=lid_mat,
    )
    # Put the hinge on the camera-far (-X) edge.  Positive rotation about -Y
    # lifts the lid behind the box, leaving the complete pattern visible from
    # the shared base camera instead of placing a vertical panel in front of
    # the card.
    axis_q = np.asarray([math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)], dtype=np.float32)
    lid.set_joint_name(f"{name}_hinge")
    lid.set_joint_properties(
        type="revolute",
        limits=[[0.0, float(CFG["lid_open_angle"])]],
        pose_in_parent=sapien.Pose([-outer_x, 0, h + lid_t / 2], axis_q),
        pose_in_child=sapien.Pose([-outer_x, 0, 0], axis_q),
        friction=0.06,
        damping=0.8,
    )
    articulation = builder.build(name, fix_root_link=False)
    joint = articulation.get_active_joints()[0]
    joint.set_drive_properties(stiffness=55.0, damping=10.0, force_limit=18.0)
    root_link = articulation.links_map[f"{name}_root"]
    return _HingedBox(
        assembly=JointAssembly(
            articulation=articulation,
            joint=joint,
            moving_link=articulation.links_map[f"{name}_lid"],
            lower=0.0,
            upper=float(CFG["lid_open_angle"]),
        ),
        root_link=root_link,
        cell_materials=cell_materials,
    )


def _build_query_lift(scene, name: str) -> tuple[JointAssembly, list[Any]]:
    """Collision-free cue card on a real prismatic display mechanism."""

    builder = scene.create_articulation_builder()
    # Keep the display mechanism clear of the enlarged front box.  Its card is
    # still centred in the base-camera view once it rises, but neither the
    # screen nor the prismatic assembly intersects a randomized box envelope.
    builder.initial_pose = sapien.Pose([0.16, -0.315, 0.055])
    root = builder.create_link_builder(parent=None)
    root.set_name(f"{name}_root")
    moving = builder.create_link_builder(parent=root)
    moving.set_name(f"{name}_card")
    moving.add_box_visual(
        half_size=[0.075, 0.003, 0.052],
        material=render_material((0.96, 0.96, 0.92, 1.0)),
    )
    cell_materials = _add_pattern(moving, horizontal=False, extent=0.045)
    moving.set_joint_name(f"{name}_joint")
    # Joint native X axis is rotated onto world Z.
    axis_q = np.asarray([math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0], dtype=np.float32)
    moving.set_joint_properties(
        type="prismatic",
        limits=[[0.0, 0.17]],
        pose_in_parent=sapien.Pose(q=axis_q),
        pose_in_child=sapien.Pose(q=axis_q),
        friction=0.02,
        damping=0.5,
    )
    articulation = builder.build(name, fix_root_link=True)
    joint = articulation.get_active_joints()[0]
    joint.set_drive_properties(stiffness=90.0, damping=18.0, force_limit=10.0)
    return JointAssembly(
        articulation=articulation,
        joint=joint,
        moving_link=articulation.links_map[f"{name}_card"],
        lower=0.0,
        upper=0.17,
    ), cell_materials


@register_env(ENV_ID, max_episode_steps=800)
class AOHiddenPattern(UnifiedPandaEnv):
    TASK_NAME = TASK_NAME
    INSTRUCTION_TASK = INSTRUCTION_TASK
    INSTRUCTION_MEMORY = INSTRUCTION_MEMORY

    def _load_task_scene(self, options: dict) -> None:
        self.boxes = [
            _build_hinged_box(self.scene, f"texture_box_{i}", i)
            for i in range(3)
        ]
        self.query_lift, self.query_cell_materials = _build_query_lift(
            self.scene, "query_lift"
        )
        # The opaque screen physically occludes every retracted cue card.
        build_static_box(
            self.scene,
            "query_card_screen",
            [0.092, 0.010, 0.075],
            [0.16, -0.335, 0.075],
            color=(0.30, 0.34, 0.38, 1.0),
        )
        self.register_manipulable(*(box.root_link for box in self.boxes))

    @staticmethod
    def _randint(rng, bounds) -> int:
        return int(rng.randint(int(bounds[0]), int(bounds[1]) + 1))

    def _initialize_task_episode(
        self,
        env_idx: torch.Tensor,
        episode_metadata: Mapping[str, Any] | None,
    ) -> None:
        n = self.num_envs
        device = self.device
        self.texture_ids = torch.zeros((n, 3), dtype=torch.int64, device=device)
        self.target_slot = torch.zeros(n, dtype=torch.int64, device=device)
        self.slot_xy = torch.zeros((n, 3, 2), dtype=torch.float32, device=device)
        timing_names = (
            "open_settle", "close_start", "close_end", "query_raise_start",
            "query_stable_start", "query_lower_start", "query_lower_end", "execution_start",
        )
        timings = {name: torch.zeros(n, dtype=torch.int64, device=device) for name in timing_names}

        for env_i in range(n):
            if episode_metadata is None:
                # Sample exactly three distinct identities from the larger
                # bank, then bind them to the three physical box slots.
                ids = self._episode_rng.permutation(
                    int(CFG["texture_bank_size"])
                )[:3]
                target = int(self._episode_rng.randint(3))
                jitter = self._episode_rng.uniform(-CFG["slot_jitter"], CFG["slot_jitter"], size=(3, 2))
                open_hold = self._randint(self._episode_rng, CFG["open_hold_steps"])
                gap = self._randint(self._episode_rng, CFG["closed_gap_steps"])
                query_hold = self._randint(self._episode_rng, CFG["query_hold_steps"])
                delay = self._randint(self._episode_rng, CFG["memory_delay_steps"])
            else:
                ids = np.asarray(episode_metadata["texture_ids"], dtype=np.int64).reshape(-1)[:3]
                target = int(np.asarray(episode_metadata["target_slot"]).reshape(-1)[0])
                jitter = np.asarray(
                    episode_metadata["slot_jitter_xy"], dtype=np.float32
                ).reshape(-1, 3, 2)[0]
                open_hold = int(np.asarray(episode_metadata["open_hold_steps"]).reshape(-1)[0])
                gap = int(np.asarray(episode_metadata["closed_gap_steps"]).reshape(-1)[0])
                query_hold = int(np.asarray(episode_metadata["query_hold_steps"]).reshape(-1)[0])
                delay = int(np.asarray(episode_metadata["memory_delay_steps"]).reshape(-1)[0])
            base = np.stack(
                [
                    np.full(3, float(CFG["slot_x"])),
                    np.asarray(CFG["slot_y"], dtype=np.float32),
                ],
                axis=1,
            )
            xy = base + jitter
            self.texture_ids[env_i] = torch.as_tensor(ids, device=device)
            self.target_slot[env_i] = target
            self.slot_xy[env_i] = torch.as_tensor(xy, device=device)

            open_settle = int(CFG["open_settle_steps"])
            close_start = open_settle + open_hold
            close_end = close_start + int(CFG["close_steps"])
            query_raise_start = close_end + gap
            query_stable_start = query_raise_start + int(CFG["query_raise_steps"])
            query_lower_start = query_stable_start + query_hold
            query_lower_end = query_lower_start + int(CFG["query_lower_steps"])
            # observation_end is inclusive.  Leave exactly ``delay`` saved
            # memory frames before the first execution frame.
            execution_start = query_lower_end + 1 + delay
            values = (open_settle, close_start, close_end, query_raise_start,
                      query_stable_start, query_lower_start, query_lower_end, execution_start)
            for name, value in zip(timing_names, values):
                timings[name][env_i] = value
        for name, value in timings.items():
            setattr(self, name, value)

        # Reset the three physical pattern boxes into their randomized slots.
        for slot, box in enumerate(self.boxes):
            positions = torch.zeros((n, 3), dtype=torch.float32, device=device)
            positions[:, :2] = self.slot_xy[:, slot]
            quats = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device).repeat(n, 1)
            box.assembly.articulation.set_pose(Pose.create_from_pq(p=positions, q=quats))
            box.assembly.set_qpos(torch.zeros(n, device=device))
            box.assembly.articulation.set_qvel(torch.zeros((n, 1), device=device))
            if hasattr(box.assembly.articulation, "set_root_linear_velocity"):
                zeros = torch.zeros((n, 3), device=device)
                box.assembly.articulation.set_root_linear_velocity(zeros)
                box.assembly.articulation.set_root_angular_velocity(zeros)
            _set_pattern(box.cell_materials, int(self.texture_ids[0, slot].item()))
        self.query_lift.set_qpos(torch.zeros(n, device=device))
        self.query_lift.articulation.set_qvel(torch.zeros((n, 1), device=device))
        _set_pattern(
            self.query_cell_materials,
            int(self.texture_ids[0, self.target_slot[0]].item()),
        )

        self.execution_box_xy = self.slot_xy.clone()
        self.wrong_box_violation = torch.zeros(n, dtype=torch.bool, device=device)
        self.set_phase_schedule(self.query_lower_end, self.execution_start)
        self.define_keyframe_event(
            "box_texture_observation",
            self.open_settle,
            self.close_start - 1,
            required_information=("three_pattern_identities", "pattern_to_box_locations"),
        )
        self.define_keyframe_event(
            "target_texture_observation",
            self.query_stable_start,
            self.query_lower_start - 1,
            required_information=("target_pattern_identity",),
        )
        self.set_task_episode_metadata(
            texture_ids=self.texture_ids,
            selected_pattern_ids=self.texture_ids,
            pattern_candidate_count=len(PATTERN_NAMES),
            pattern_bank_names=PATTERN_NAMES,
            pattern_names=[
                [PATTERN_NAMES[int(x)] for x in self.texture_ids[i].tolist()]
                for i in range(n)
            ],
            target_slot=self.target_slot,
            slot_jitter_xy=self.slot_xy
            - torch.tensor(
                [[CFG["slot_x"], y] for y in CFG["slot_y"]],
                dtype=torch.float32,
                device=device,
            ),
            slot_xy=self.slot_xy,
            open_hold_steps=self.close_start - self.open_settle,
            closed_gap_steps=self.query_raise_start - self.close_end,
            query_hold_steps=self.query_lower_start - self.query_stable_start,
            memory_delay_steps=self.execution_start - self.query_lower_end - 1,
            target_texture_ids=self.texture_ids[
                torch.arange(n, device=device), self.target_slot
            ],
            target_pattern_names=[
                PATTERN_NAMES[
                    int(self.texture_ids[i, self.target_slot[i]].item())
                ]
                for i in range(n)
            ],
        )

        if n == 1:
            self._expert_fsm = PhysicalWaypointFSM(self, self._expert_waypoints())
        else:
            self._expert_fsm = None

    def _texture_actor(self, env_i: int, slot: int):
        del env_i
        return self.boxes[slot].root_link

    def _expert_waypoints(self):
        def target_root(env):
            slot = int(env.target_slot[0].item())
            xy = env.execution_box_xy[0, slot].detach().cpu().numpy()
            return np.asarray([xy[0], xy[1], 0.0], dtype=np.float32)

        def point(dx, z):
            def resolve(env):
                p = target_root(env).copy()
                return (float(p[0] + dx), float(p[1]), float(z))
            return resolve

        return [
            PhysicalWaypoint(point(-0.16, 0.17), GRIPPER_CLOSED,
                             rotation_tolerance=3.2, label="above_push_face"),
            PhysicalWaypoint(point(-0.145, 0.036), GRIPPER_CLOSED,
                             rotation_tolerance=3.2, dwell_steps=4, label="contact_box"),
            PhysicalWaypoint(point(0.22, 0.036), GRIPPER_CLOSED, position_tolerance=0.018,
                             rotation_tolerance=3.2, dwell_steps=5, label="push_matching_box"),
            PhysicalWaypoint(HOME_TCP_POSITION, GRIPPER_OPEN,
                             rotation_tolerance=3.2, label="retract_home"),
        ]

    def _before_task_control_step(self) -> None:
        step = self.elapsed_steps.to(torch.int64)
        for box in self.boxes:
            open_now = step < self.close_start
            box.assembly.drive_to(
                torch.where(
                    open_now,
                    torch.full_like(step, float(CFG["lid_open_angle"]), dtype=torch.float32),
                    torch.zeros_like(step, dtype=torch.float32),
                )
            )
        visible = (step >= self.query_raise_start) & (step < self.query_lower_start)
        self.query_lift.drive_to(
            torch.where(
                visible,
                torch.full_like(step, 0.17, dtype=torch.float32),
                torch.zeros_like(step, dtype=torch.float32),
            )
        )

    def _after_task_control_step(self) -> None:
        step = self.elapsed_steps.to(torch.int64)
        just_before_execution = step == (self.execution_start - 1)
        for env_i in range(self.num_envs):
            if bool(just_before_execution[env_i].item()):
                for slot in range(3):
                    self.execution_box_xy[env_i, slot] = self._texture_actor(env_i, slot).pose.p[env_i, :2]
            if int(step[env_i].item()) >= int(self.execution_start[env_i].item()):
                target = int(self.target_slot[env_i].item())
                for slot in range(3):
                    if slot == target:
                        continue
                    disp = torch.linalg.norm(
                        self._texture_actor(env_i, slot).pose.p[env_i, :2]
                        - self.execution_box_xy[env_i, slot]
                    )
                    self.wrong_box_violation[env_i] |= disp > float(CFG["wrong_box_tolerance"])

    def _evaluate_task(self) -> Mapping[str, Any]:
        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        displacement = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for env_i in range(self.num_envs):
            slot = int(self.target_slot[env_i].item())
            actor = self._texture_actor(env_i, slot)
            displacement[env_i] = actor.pose.p[env_i, 0] - self.execution_box_xy[env_i, slot, 0]
            success[env_i] = (
                int(self.elapsed_steps[env_i].item()) >= int(self.execution_start[env_i].item())
                and float(displacement[env_i].item()) >= float(CFG["push_distance"])
                and not bool(self.wrong_box_violation[env_i].item())
            )
        return {
            "success": success,
            "target_displacement": displacement,
            "wrong_box_violation": self.wrong_box_violation,
        }

    def _expert_action(self) -> torch.Tensor:
        if torch.any(self.elapsed_steps < self.execution_start):
            return self.zero_action()
        if self._expert_fsm is None:
            return self.zero_action()
        return self._expert_fsm.action()


__all__ = [
    "AOHiddenPattern",
    "ENV_ID",
    "INSTRUCTION_TASK",
    "INSTRUCTION_MEMORY",
]
