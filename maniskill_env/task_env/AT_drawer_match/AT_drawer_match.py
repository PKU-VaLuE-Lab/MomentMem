"""Triggered+Appearance memory task with physical drawers and spring buttons."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
    build_prismatic_articulation,
    render_material,
)
from AT_drawer_match_cfg import CFG  # noqa: E402

TASK_NAME = "AT_drawer_match"
ENV_ID = TASK_ENV_IDS[TASK_NAME]
INSTRUCTION_TASK = (
    "A three-level cabinet hides a pattern in each drawer. Press the upper, middle, and bottom "
    "drawer buttons in that order; each drawer opens once and then closes. Then press the upper "
    "or middle button for the drawer whose pattern matches the bottom drawer. "
    "Finish when the selected drawer is fully open."
)
INSTRUCTION_MEMORY = (
    "Remember when each of the upper, middle, and bottom drawers is open and its pattern is visible."
)


_PATTERN_NAMES = (
    "x", "plus", "t", "h", "horizontal_bars", "corners", "checker", "diamond",
)
_PATTERN_MASKS = np.asarray(
    [
        # X, plus, and diamond: simple high-contrast identities in the same
        # spirit as hidden_object's black geometric symbols.
        [
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        [[1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
        [[1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]],
        [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
        [[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]],
        [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
        [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
        ],
    ],
    dtype=np.bool_,
)
_PATTERN_BLACK = (0.015, 0.015, 0.015, 1.0)
_CARD_WHITE = (0.96, 0.96, 0.91, 1.0)


def _pattern(texture_id: int, row: int, col: int) -> tuple[float, float, float, float]:
    mask = _PATTERN_MASKS[int(texture_id) % len(_PATTERN_MASKS)]
    return _PATTERN_BLACK if bool(mask[row, col]) else _CARD_WHITE


def _build_cabinet(scene):
    """One fixed root with three independently driven collision-bearing trays."""

    physics = PhysicsMaterial(static_friction=1.0, dynamic_friction=0.8, density=850)
    builder = scene.create_articulation_builder()
    # Face the open side toward the canonical +X base camera.  The local
    # drawer joint still uses its tested negative-X travel, while this rigid
    # 180-degree root rotation maps that travel to camera-facing world +X.
    builder.initial_pose = sapien.Pose(CFG["cabinet_pose"], q=[0, 0, 0, 1])
    root = builder.create_link_builder(parent=None)
    root.set_name("drawer_cabinet_root")
    cabinet_mat = render_material((0.36, 0.39, 0.43, 1.0))
    # Back, sides, top, bottom and separators are genuine cabinet collisions.
    parts = [
        ([0.025, 0.12, 0.14], [0.085, 0.0, 0.0]),
        ([0.11, 0.012, 0.14], [0.0, -0.112, 0.0]),
        ([0.11, 0.012, 0.14], [0.0, 0.112, 0.0]),
        ([0.11, 0.12, 0.010], [0.0, 0.0, 0.14]),
        ([0.11, 0.12, 0.010], [0.0, 0.0, -0.14]),
        ([0.11, 0.12, 0.007], [0.0, 0.0, 0.043]),
        ([0.11, 0.12, 0.007], [0.0, 0.0, -0.043]),
    ]
    for half, pos in parts:
        pose = sapien.Pose(pos)
        root.add_box_collision(pose=pose, half_size=half, material=physics.sapien())
        root.add_box_visual(pose=pose, half_size=half, material=cabinet_mat)

    drawers = []
    texture_materials = []
    axis_q = [1.0, 0.0, 0.0, 0.0]
    dh = CFG["drawer_half_size"]
    for index, level in enumerate(CFG["drawer_levels"]):
        link = builder.create_link_builder(parent=root)
        link.set_name(f"drawer_{index}")
        tray_mat = render_material((0.72, 0.74, 0.76, 1.0))
        # Bottom, front and low side lips; cards sit on the physical tray bottom.
        for half, pos in (
            ([dh[0], dh[1], 0.006], [0, 0, -dh[2] + 0.006]),
            ([0.008, dh[1], dh[2]], [-dh[0] + 0.008, 0, 0]),
            ([dh[0], 0.006, dh[2]], [0, -dh[1] + 0.006, 0]),
            ([dh[0], 0.006, dh[2]], [0, dh[1] - 0.006, 0]),
        ):
            pose = sapien.Pose(pos)
            link.add_box_collision(pose=pose, half_size=half, material=physics.sapien())
            link.add_box_visual(pose=pose, half_size=half, material=tray_mat)
        # A real, thin card insert sits toward the drawer front (negative local
        # X is the opening direction).  At the stable open threshold the full
        # card is outside the cabinet face instead of remaining under the next
        # shelf.  The printed cells are slightly above this collision-bearing
        # insert, so they do not flicker/z-fight.
        card_pose = sapien.Pose([CFG["card_center_x"], 0.0, 0.004])
        link.add_box_collision(
            pose=card_pose,
            half_size=[0.047, 0.076, 0.0015],
            material=physics.sapien(),
        )
        link.add_box_visual(
            pose=card_pose,
            half_size=[0.047, 0.076, 0.0015],
            material=render_material(_CARD_WHITE),
        )
        mats = []
        for row in range(5):
            for col in range(5):
                mat = render_material(_pattern(index, row, col))
                mats.append(mat)
                link.add_box_visual(
                    pose=sapien.Pose([
                        CFG["card_center_x"] - 0.032 + row * 0.016,
                        -0.052 + col * 0.026,
                        0.007,
                    ]),
                    half_size=[0.0068, 0.0105, 0.0010],
                    material=mat,
                )
        texture_materials.append(mats)
        link.set_joint_name(f"drawer_joint_{index}")
        link.set_joint_properties(
            type="prismatic",
            limits=[[-float(CFG["drawer_travel"]), 0.0]],
            pose_in_parent=sapien.Pose([0, 0, level], axis_q),
            pose_in_child=sapien.Pose(q=axis_q),
            friction=0.08,
            damping=1.0,
        )
    articulation = builder.build("three_drawer_cabinet", fix_root_link=True)
    for index, joint in enumerate(articulation.get_active_joints()):
        joint.set_drive_properties(stiffness=110.0, damping=22.0, force_limit=55.0)
        drawers.append(
            JointAssembly(
                articulation=articulation,
                joint=joint,
                moving_link=articulation.links_map[f"drawer_{index}"],
                lower=-float(CFG["drawer_travel"]),
                upper=0.0,
            )
        )
    return articulation, drawers, texture_materials


@register_env(ENV_ID, max_episode_steps=1300)
class ATDrawerMatch(UnifiedPandaEnv):
    TASK_NAME = TASK_NAME
    INSTRUCTION_TASK = INSTRUCTION_TASK
    INSTRUCTION_MEMORY = INSTRUCTION_MEMORY

    def _load_task_scene(self, options: dict) -> None:
        del options
        self.cabinet, self.drawers, self.texture_materials = _build_cabinet(self.scene)
        self.buttons = []
        for index, xy in enumerate(CFG["button_xy"]):
            self.buttons.append(
                build_prismatic_articulation(
                    self.scene,
                    f"drawer_button_{index}",
                    root_half_size=[0.035, 0.035, 0.008],
                    moving_half_size=[0.024, 0.024, 0.010],
                    pose=sapien.Pose([xy[0], xy[1], 0.008]),
                    moving_offset=[0, 0, 0.025],
                    axis="z",
                    limits=(-float(CFG["button_travel"]), 0.0),
                    root_color=(0.18, 0.20, 0.22, 1),
                    moving_color=((0.86, 0.24, 0.18, 1), (0.92, 0.67, 0.12, 1), (0.22, 0.48, 0.86, 1))[index],
                    drive_stiffness=180.0,
                    drive_damping=16.0,
                    drive_force_limit=24.0,
                )
            )
        self.register_manipulable(*(button.moving_link for button in self.buttons))

    def _initialize_task_episode(self, env_idx, episode_metadata: Mapping[str, Any] | None) -> None:
        del env_idx
        if episode_metadata is None:
            match_slot = int(self._episode_rng.randint(2))
            base_texture = int(self._episode_rng.randint(len(_PATTERN_MASKS)))
            other_texture = int(
                self._episode_rng.choice(
                    [index for index in range(len(_PATTERN_MASKS)) if index != base_texture]
                )
            )
            order = [0, 1, 2]
            hold_steps = int(self._episode_rng.randint(CFG["open_hold_steps"][0], CFG["open_hold_steps"][1] + 1))
            delay_steps = int(self._episode_rng.randint(CFG["memory_delay_steps"][0], CFG["memory_delay_steps"][1] + 1))
        else:
            match_slot = int(episode_metadata["match_slot"])
            base_texture = int(episode_metadata["base_texture"])
            other_texture = int(episode_metadata["other_texture"])
            order = [int(x) for x in episode_metadata["demonstration_order"]]
            hold_steps = int(episode_metadata["open_hold_steps"])
            delay_steps = int(episode_metadata["memory_delay_steps"])
        self.match_slot = match_slot
        self.texture_ids = [base_texture if i in (match_slot, 2) else other_texture for i in range(3)]
        self.demonstration_order = order
        self.open_hold_steps = hold_steps
        self.memory_delay_steps = delay_steps
        self.cabinet.set_qpos(torch.zeros((1, 3), device=self.device))
        self.cabinet.set_qvel(torch.zeros((1, 3), device=self.device))
        for button in self.buttons:
            button.set_qpos(0.0)
            button.articulation.set_qvel(torch.zeros((1, 1), device=self.device))
        for drawer_index, mats in enumerate(self.texture_materials):
            for cell_index, mat in enumerate(mats):
                row, col = divmod(cell_index, 5)
                mat.set_base_color(_pattern(self.texture_ids[drawer_index], row, col))
        self.active_drawer = -1
        self.active_trigger_step = -1
        self.stable_start = -1
        self.observed = []
        self.answer_pressed = -1
        self.answer_trigger_step = -1
        self.answer_drawer_opened = False
        self.answer_drawer_displacement = 0.0
        self._button_latched = [False, False, False]
        self._observation_complete_step = -1
        self._execution_step = 10_000
        self.set_phase_schedule(9_998, 9_999)
        self.set_task_episode_metadata(
            match_slot=match_slot,
            base_texture=base_texture,
            other_texture=other_texture,
            selected_pattern_ids=[base_texture, other_texture],
            pattern_candidate_count=len(_PATTERN_MASKS),
            pattern_bank_names=_PATTERN_NAMES,
            drawer_texture_ids=self.texture_ids,
            demonstration_order=order,
            observation_order_roles=("upper", "middle", "bottom"),
            open_hold_steps=hold_steps,
            memory_delay_steps=delay_steps,
            answer_drawer_pull=CFG["answer_drawer_pull"],
            answer_drawer_success_threshold=CFG["answer_drawer_success_threshold"],
            answer_completion="selected_drawer_reaches_full_travel_immediate_success",
        )
        self._press_fsm = None

    def _button_waypoints(self, index: int):
        def point(dz):
            def resolve(env):
                del env
                # Use the unpressed button centre in world coordinates.  A
                # target derived from moving_link.pose would retreat together
                # with the spring button and could never be reached.
                xy = CFG["button_xy"][index]
                return np.asarray([xy[0], xy[1], 0.033 + dz], dtype=np.float32)
            return resolve
        return [
            PhysicalWaypoint(point(0.11), GRIPPER_CLOSED, label=f"button_{index}_above"),
            PhysicalWaypoint(point(-0.012), GRIPPER_CLOSED, position_tolerance=0.020, dwell_steps=5, label=f"button_{index}_press"),
            PhysicalWaypoint(point(0.12), GRIPPER_OPEN, dwell_steps=3, label=f"button_{index}_release"),
            PhysicalWaypoint(HOME_TCP_POSITION, GRIPPER_OPEN, label="home"),
        ]

    def _before_task_control_step(self) -> None:
        for button in self.buttons:
            button.drive_to(0.0)
        targets = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        if self.answer_pressed >= 0:
            targets[0, self.answer_pressed] = -float(CFG["answer_drawer_pull"])
        elif self.active_drawer >= 0:
            elapsed = int(self.elapsed_steps[0].item()) - self.active_trigger_step
            if elapsed < 26 + self.open_hold_steps:
                targets[0, self.active_drawer] = -float(CFG["drawer_travel"])
        if self.scene.gpu_sim_enabled:
            indices = torch.arange(3, dtype=torch.int64, device=self.device)
            self.cabinet.set_joint_drive_targets(targets, joint_indices=indices)
            self.scene.px.gpu_apply_articulation_target_position()
        else:
            self.cabinet.set_joint_drive_targets(
                targets,
                joints=self.cabinet.get_active_joints(),
            )

    def _after_task_control_step(self) -> None:
        step = int(self.elapsed_steps[0].item())
        for index, button in enumerate(self.buttons):
            pressed = float(button.qpos[0].item()) <= float(CFG["button_threshold"])
            if pressed and not self._button_latched[index]:
                self._button_latched[index] = True
                if self._observation_complete_step >= 0:
                    if step >= self._execution_step and self.answer_pressed < 0:
                        self.answer_pressed = index
                        self.answer_trigger_step = step
                elif self.active_drawer < 0 and index == self.demonstration_order[len(self.observed)]:
                    self.active_drawer = index
                    self.active_trigger_step = step
                    self.stable_start = -1
            elif not pressed:
                self._button_latched[index] = False
        if self.active_drawer >= 0:
            q = float(self.cabinet.get_qpos()[0, self.active_drawer].item())
            if q <= float(CFG["drawer_open_threshold"]) and self.stable_start < 0:
                self.stable_start = step
                self.define_keyframe_event(
                    f"drawer_{self.active_drawer}_pattern_observation",
                    step + 1,
                    step + self.open_hold_steps - 1,
                    required_information=(f"drawer_{self.active_drawer}_pattern_identity", "drawer_level"),
                )
            closing = step - self.active_trigger_step >= 26 + self.open_hold_steps
            if closing and q >= float(CFG["drawer_close_threshold"]):
                if self.active_drawer not in self.observed:
                    self.observed.append(self.active_drawer)
                self.active_drawer = -1
                self._press_fsm = None
                if len(self.observed) == 3:
                    self._observation_complete_step = step
                    self._execution_step = step + 1 + self.memory_delay_steps
                    self.set_phase_schedule(step, self._execution_step)
        if self.answer_pressed >= 0:
            q = float(self.cabinet.get_qpos()[0, self.answer_pressed].item())
            self.answer_drawer_displacement = max(0.0, -q)
            self.answer_drawer_opened = (
                self.answer_drawer_displacement
                >= float(CFG["answer_drawer_success_threshold"])
            )

    def _evaluate_task(self):
        success = (
            self.answer_pressed == self.match_slot
            and self.answer_pressed >= 0
            and self.answer_drawer_opened
        )
        wrong = self.answer_pressed >= 0 and self.answer_pressed != self.match_slot
        return {
            "success": torch.tensor([success], dtype=torch.bool, device=self.device),
            "fail": torch.tensor([wrong], dtype=torch.bool, device=self.device),
            "observed_drawers": torch.tensor([len(self.observed)], device=self.device),
            "answer_drawer_displacement": torch.tensor(
                [self.answer_drawer_displacement], dtype=torch.float32, device=self.device
            ),
            "answer_drawer_opened": torch.tensor(
                [self.answer_drawer_opened], dtype=torch.bool, device=self.device
            ),
        }

    def _expert_action(self):
        if self.active_drawer >= 0:
            # Finish the button-press trajectory while the drawer animation
            # continues, then wait at the common home pose.
            if self._press_fsm is not None and not self._press_fsm.complete:
                return self._press_fsm.action()
            return self.zero_action()
        if len(self.observed) < 3:
            target = self.demonstration_order[len(self.observed)]
        else:
            if int(self.elapsed_steps[0].item()) < self._execution_step:
                return self.zero_action()
            if self.answer_pressed >= 0:
                if self._press_fsm is not None and not self._press_fsm.complete:
                    return self._press_fsm.action()
                return self.zero_action()
            target = self.match_slot
        if self._press_fsm is None or self._press_fsm.complete:
            self._press_fsm = PhysicalWaypointFSM(self, self._button_waypoints(target))
        return self._press_fsm.action()


__all__ = ["ATDrawerMatch", "ENV_ID", "INSTRUCTION_TASK", "INSTRUCTION_MEMORY"]
