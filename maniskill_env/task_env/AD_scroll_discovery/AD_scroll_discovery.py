"""Direct+Appearance task with physical spring-return patterned strips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

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
    render_material,
)
try:  # package import and direct-script execution are both supported
    from .AD_scroll_discovery_cfg import CFG
except ImportError:  # pragma: no cover - direct gen_data.py execution
    from AD_scroll_discovery_cfg import CFG  # type: ignore

TASK_NAME = "AD_scroll_discovery"
ENV_ID = TASK_ENV_IDS[TASK_NAME]
EVALUATION_STABLE_STEPS = 4
INSTRUCTION_TASK = (
    "Three narrow scrolls hide patterned strips. From left to right, pull each strip until "
    "the whole pattern is visible, then release it so it rolls back up. After the third "
    "strip rolls up, immediately grasp one of the two scrolls that showed the same pattern "
    "and stack it on top of the other, without touching the scroll with the different pattern."
)
INSTRUCTION_MEMORY = (
    "Remember when each strip is fully pulled down and its whole pattern is visible."
)


_PATTERN_NAMES = (
    "x", "plus", "diamond", "horizontal_bars",
    "corners", "checker", "t", "h",
)
_PATTERN_MASKS = np.asarray(
    [
        # Large 5x5 X, T and H marks.  Their silhouettes differ globally and
        # remain readable in the shared 224x224 camera at all three slots.
        [[1, 0, 0, 0, 1],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 0, 0, 1]],
        [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
        [[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]],
        [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
        [[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1]],
        [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
        [[1, 1, 1, 1, 1],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0]],
        [[1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1],
         [1, 1, 1, 1, 1],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 1]],
    ],
    dtype=np.bool_,
)
_PATTERN_INK = (0.025, 0.025, 0.025, 1.0)
_PATTERN_PAPER = (0.91, 0.89, 0.82, 1.0)


def _cell_color(texture_id: int, cell: int) -> tuple[float, float, float, float]:
    row, col = divmod(int(cell), 5)
    return _PATTERN_INK if _PATTERN_MASKS[int(texture_id), row, col] else _PATTERN_PAPER


@dataclass
class _Scroll:
    assembly: JointAssembly
    root_link: Any
    cell_materials: list[Any]


def _build_scroll(scene, name: str) -> _Scroll:
    physics = PhysicsMaterial(static_friction=1.35, dynamic_friction=1.05, density=1100)
    builder = scene.create_articulation_builder()
    builder.initial_pose = sapien.Pose([-0.5, 0.8, 0.0])
    root = builder.create_link_builder(parent=None)
    root.set_name(f"{name}_root")
    housing = render_material((0.34, 0.38, 0.43, 1))
    for half, pos in (
        # The complete body is only 68 mm wide, so the Panda can grasp the
        # housing itself.  There is intentionally no relocation handle.
        ([0.070, 0.034, 0.006], [0.018, 0, 0.006]),
        ([0.070, 0.034, 0.006], [0.018, 0, 0.052]),
        ([0.070, 0.004, 0.029], [0.018, -0.030, 0.029]),
        ([0.070, 0.004, 0.029], [0.018, 0.030, 0.029]),
        ([0.008, 0.034, 0.029], [0.080, 0, 0.029]),
    ):
        pose = sapien.Pose(pos)
        root.add_box_collision(pose=pose, half_size=half, material=physics.sapien(), density=physics.density)
        root.add_box_visual(pose=pose, half_size=half, material=housing)
    moving = builder.create_link_builder(parent=root)
    moving.set_name(f"{name}_strip")
    strip_pose = sapien.Pose([0, 0, 0.036])
    moving.add_box_collision(
        pose=strip_pose,
        half_size=[0.072, 0.025, 0.004],
        material=physics.sapien(),
        density=420,
    )
    moving.add_box_visual(
        pose=strip_pose,
        half_size=[0.072, 0.025, 0.004],
        material=render_material(_PATTERN_PAPER),
    )
    mats = []
    for row in range(5):
        for col in range(5):
            mat = render_material(_cell_color(0, row * 5 + col))
            mats.append(mat)
            moving.add_box_visual(
                pose=sapien.Pose([-0.048 + col * 0.024, -0.020 + row * 0.010, 0.041]),
                half_size=[0.0105, 0.0042, 0.0012],
                material=mat,
            )
    # Collision-bearing pull tab is part of the moving strip.
    moving.add_box_collision(
        pose=sapien.Pose([CFG["pull_tab_x"], 0, 0.044]),
        half_size=[0.015, 0.018, 0.018],
        material=physics.sapien(),
        density=600,
    )
    moving.add_box_visual(
        pose=sapien.Pose([CFG["pull_tab_x"], 0, 0.044]),
        half_size=[0.015, 0.018, 0.018],
        material=render_material((0.68, 0.70, 0.73, 1)),
    )
    moving.set_joint_name(f"{name}_spring_joint")
    moving.set_joint_properties(
        type="prismatic",
        limits=[[-float(CFG["strip_travel"]), 0.0]],
        pose_in_parent=sapien.Pose(),
        pose_in_child=sapien.Pose(),
        friction=0.04,
        damping=0.8,
    )
    art = builder.build(name, fix_root_link=False)
    joint = art.get_active_joints()[0]
    joint.set_drive_properties(stiffness=75.0, damping=12.0, force_limit=24.0)
    root_link = art.links_map[f"{name}_root"]
    # During reveal the housing is a heavy, high-friction counter-body, so the
    # gripper compresses the prismatic spring instead of dragging the device.
    # It remains below Panda's physical payload for the final relocation.
    root_link.set_mass(1.60)
    return _Scroll(
        JointAssembly(art, joint, art.links_map[f"{name}_strip"], -float(CFG["strip_travel"]), 0.0),
        root_link,
        mats,
    )


@register_env(ENV_ID, max_episode_steps=1800)
class ADScrollDiscovery(UnifiedPandaEnv):
    TASK_NAME = TASK_NAME
    INSTRUCTION_TASK = INSTRUCTION_TASK
    INSTRUCTION_MEMORY = INSTRUCTION_MEMORY

    def _load_task_scene(self, options: dict) -> None:
        del options
        self.scrolls = [_build_scroll(self.scene, f"scroll_{i}") for i in range(3)]
        self.register_manipulable(
            *(s.root_link for s in self.scrolls),
            *(s.assembly.moving_link for s in self.scrolls),
        )

    def _initialize_task_episode(self, env_idx, episode_metadata: Mapping[str, Any] | None) -> None:
        del env_idx
        if episode_metadata is None:
            matching = sorted(self._episode_rng.choice(3, 2, replace=False).astype(int).tolist())
            common = int(self._episode_rng.randint(len(_PATTERN_NAMES)))
            odd = int((common + self._episode_rng.randint(1, len(_PATTERN_NAMES))) % len(_PATTERN_NAMES))
            textures = [common if i in matching else odd for i in range(3)]
            # With the canonical camera, increasing world y maps from visual
            # left to visual right for these equal-x slots.  Jitter is far
            # smaller than the 24 cm spacing, so this order cannot invert.
            order = [0, 1, 2]
            jitter = self._episode_rng.uniform(-CFG["slot_jitter"], CFG["slot_jitter"], size=(3, 2)).astype(np.float32)
            holds = [int(self._episode_rng.randint(CFG["reveal_hold_steps"][0], CFG["reveal_hold_steps"][1] + 1)) for _ in range(3)]
            delay = int(self._episode_rng.randint(CFG["memory_delay_steps"][0], CFG["memory_delay_steps"][1] + 1))
        else:
            matching = [int(x) for x in episode_metadata["matching_scrolls"]]
            textures = [int(x) for x in episode_metadata["texture_ids"]]
            order = [int(x) for x in episode_metadata["observation_order"]]
            jitter = np.asarray(episode_metadata["slot_jitter_xy"], dtype=np.float32)
            holds = [int(x) for x in episode_metadata["reveal_hold_steps"]]
            delay = int(episode_metadata["memory_delay_steps"])
        self.matching_scrolls = matching
        self.texture_ids = textures
        common = int(textures[matching[0]])
        odd_slots = [index for index in range(3) if index not in matching]
        odd = int(textures[odd_slots[0]])
        self.observation_order = order
        self.reveal_hold_steps = holds
        self.memory_delay_steps = delay
        base = np.asarray([[CFG["scroll_x"], y] for y in CFG["scroll_y"]], dtype=np.float32)
        self.slot_xy = base + jitter
        qroot = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=self.device)
        for i, scroll in enumerate(self.scrolls):
            scroll.assembly.articulation.set_pose(Pose.create_from_pq(
                p=torch.tensor([[self.slot_xy[i, 0], self.slot_xy[i, 1], 0.0]], dtype=torch.float32, device=self.device),
                q=qroot,
            ))
            scroll.assembly.set_qpos(0.0)
            scroll.assembly.articulation.set_qvel(torch.zeros((1, 1), device=self.device))
            if hasattr(scroll.assembly.articulation, "set_root_linear_velocity"):
                scroll.assembly.articulation.set_root_linear_velocity(torch.zeros((1, 3), device=self.device))
                scroll.assembly.articulation.set_root_angular_velocity(torch.zeros((1, 3), device=self.device))
            for cell, mat in enumerate(scroll.cell_materials):
                mat.set_base_color(_cell_color(textures[i], cell))
        self.observed = [False, False, False]
        self.wrong_pair_ever_touched = False
        self.matching_contact_force = 0.0
        self.matching_contact_force_max = 0.0
        self.matching_pair_ever_touched = False
        self.wrong_contact_force_max = 0.0
        self._fsm_index = 0
        self._observation_complete = False
        self._transition_stable_steps = 0
        self._evaluation_stable_steps = 0
        self._last_evaluation_gate_step = -1
        self._final_home_stable_steps = 0
        self._last_final_home_gate_step = -1
        self.execution_start = 10_000
        self.stack_target_root_xyz = np.asarray(
            [
                self.slot_xy[matching[1], 0],
                self.slot_xy[matching[1], 1],
                CFG["stack_root_offset_z"],
            ],
            dtype=np.float32,
        )
        self.set_phase_schedule(9_998, 9_999)
        self.set_task_episode_metadata(
            matching_scrolls=matching,
            texture_ids=textures,
            selected_pattern_ids=[common, odd],
            pattern_candidate_count=len(_PATTERN_NAMES),
            pattern_bank_names=_PATTERN_NAMES,
            texture_patterns=[_PATTERN_NAMES[x] for x in textures],
            observation_order=order,
            slot_jitter_xy=jitter,
            slot_xy=self.slot_xy,
            reveal_hold_steps=holds,
            reveal_threshold=CFG["reveal_threshold"],
            memory_delay_steps=delay,
            memory_reset="none_continuous_pair_approach",
            blank_delay_motion=(
                "continuous_open_gripper_matching_scroll_approach_without_reset"
            ),
            post_observation_transition=(
                "third_scroll_vertical_retract_direct_to_matching_scroll"
            ),
            transition_retract_tolerance=0.010,
            stack_target_root_xyz=self.stack_target_root_xyz,
            observation_order_semantics="canonical_camera_visual_left_to_right",
            final_home_tolerance=CFG["final_home_tolerance"],
            final_home_stable_steps=CFG["final_home_stable_steps"],
            terminal_success_gate=(
                "physical_matching_contact_and_stack_released_without_wrong_"
                "contact_plus_consecutive_common_home_states"
            ),
        )
        self._fsm = self._slow_fsm(self._reveal_waypoints(order[0]))
        self._pair_fsm = None

    def _slow_fsm(self, waypoints):
        return PhysicalWaypointFSM(
            self,
            waypoints,
            max_position_command=CFG["max_position_command"],
            max_rotation_command=CFG["max_rotation_command"],
        )

    def _handle_point(self, index, dz=0.0):
        def resolve(env):
            p = env.scrolls[index].assembly.moving_link.pose.p[0].detach().cpu().numpy().copy()
            p += np.asarray([CFG["pull_tab_x"], 0, 0.044 + dz], dtype=np.float32)
            return p
        return resolve

    def _pull_point(self, index, dz=0.0):
        def resolve(env):
            xy = env.slot_xy[index]
            return [
                float(
                    xy[0]
                    - CFG["strip_travel"]
                    - CFG["pull_overtravel"]
                    + CFG["pull_tab_x"]
                ),
                float(xy[1]),
                float(0.044 + dz),
            ]
        return resolve

    def _reveal_waypoints(self, index):
        hold = self.reveal_hold_steps[index]
        waypoints = [
            PhysicalWaypoint(self._handle_point(index, CFG["reveal_clearance"]), GRIPPER_OPEN, label=f"scroll_{index}_above"),
            PhysicalWaypoint(self._handle_point(index), GRIPPER_OPEN, dwell_steps=3, label=f"scroll_{index}_handle"),
            PhysicalWaypoint(self._handle_point(index), GRIPPER_CLOSED, dwell_steps=8, require=lambda env, i=index: bool(env.agent.is_grasping(env.scrolls[i].assembly.moving_link)[0].item()), label=f"scroll_{index}_grasp"),
            PhysicalWaypoint(self._pull_point(index), GRIPPER_CLOSED, position_tolerance=0.010, dwell_steps=hold, label=f"scroll_{index}_reveal"),
            PhysicalWaypoint(self._pull_point(index), GRIPPER_OPEN, dwell_steps=6, label=f"scroll_{index}_release"),
            # First lift vertically at the just-released location.  The next
            # FSM then travels horizontally at the same safe height before it
            # descends onto the next pull tab.
            PhysicalWaypoint(self._pull_point(index, CFG["reveal_clearance"]), GRIPPER_OPEN, label=f"scroll_{index}_vertical_retract"),
        ]
        # After the third reveal, stop at this collision-safe vertical retract
        # and let the pairing FSM immediately continue toward the selected
        # housing.  There is deliberately no observation-home detour.
        return waypoints

    def _pair_waypoints(self):
        moving = self.matching_scrolls[0]
        actor = self.scrolls[moving].root_link
        target_actor = self.scrolls[self.matching_scrolls[1]].root_link
        root_xyz = actor.pose.p[0].detach().cpu().numpy().copy()
        grasp = np.asarray(
            [root_xyz[0] + 0.018, root_xyz[1], root_xyz[2] + CFG["housing_grasp_z"]],
            dtype=np.float32,
        )

        def stack_goal(env, clearance=0.0):
            target = target_actor.pose.p[0].detach().cpu().numpy()
            return np.asarray(
                [
                    target[0] + 0.018,
                    target[1],
                    target[2]
                    + CFG["stack_root_offset_z"]
                    + CFG["housing_grasp_z"]
                    + clearance,
                ],
                dtype=np.float32,
            )

        lift = grasp.copy()
        lift[2] = CFG["pair_transfer_height"]

        def transfer(env):
            goal = stack_goal(env)
            goal[2] = CFG["pair_transfer_height"]
            return goal

        return [
            PhysicalWaypoint(grasp + np.asarray([0, 0, CFG["housing_approach_height"]]), GRIPPER_OPEN, label="pair_scroll_above"),
            PhysicalWaypoint(grasp, GRIPPER_OPEN, label="pair_scroll_body_contact"),
            PhysicalWaypoint(grasp, GRIPPER_CLOSED, dwell_steps=10,
                             require=lambda env: bool(env.agent.is_grasping(actor)[0].item()),
                             label="pair_scroll_body_grasp"),
            # Explicit vertical lift before any horizontal transfer prevents
            # the carried housing from striking either stationary scroll.
            PhysicalWaypoint(lift, GRIPPER_CLOSED, position_tolerance=0.018,
                             label="pair_vertical_lift"),
            PhysicalWaypoint(transfer, GRIPPER_CLOSED, position_tolerance=0.018,
                             label="pair_high_transfer"),
            # Lower vertically onto the stationary matching housing.  The
            # contact-force gate proves physical support instead of accepting
            # an oracle pose or a commanded waypoint alone.
            PhysicalWaypoint(
                stack_goal,
                GRIPPER_CLOSED,
                position_tolerance=0.032,
                require=lambda env: env.matching_contact_force
                > CFG["contact_force_threshold"],
                label="pair_vertical_stack_contact",
            ),
            PhysicalWaypoint(stack_goal, GRIPPER_OPEN, position_tolerance=0.038,
                             dwell_steps=14, label="pair_release_on_top"),
            PhysicalWaypoint(lambda env: stack_goal(env, 0.15), GRIPPER_OPEN,
                             label="pair_vertical_retract"),
            # This is an expert-demonstration tail only. Formal task success is
            # already determined by the stable released stack and never gates
            # on reaching this pose.
            PhysicalWaypoint(
                HOME_TCP_POSITION,
                GRIPPER_OPEN,
                position_tolerance=0.004,
                dwell_steps=4,
                label="pair_finish_home",
            ),
        ]

    def _scroll_pair_contact_force(self, first: int, second: int) -> float:
        links_first = (
            self.scrolls[first].root_link,
            self.scrolls[first].assembly.moving_link,
        )
        links_second = (
            self.scrolls[second].root_link,
            self.scrolls[second].assembly.moving_link,
        )
        forces = []
        for link_first in links_first:
            for link_second in links_second:
                force = self.scene.get_pairwise_contact_forces(link_first, link_second)
                forces.append(float(torch.linalg.norm(force, dim=-1)[0].item()))
        return max(forces)

    def _update_scroll_contacts(self) -> None:
        # BaseEnv evaluates once while producing reset observation s_0.  A GPU
        # pairwise impulse query can still contain the terminal impulse from the
        # preceding episode at that instant, so never write episode history
        # before the first new PhysX control step has actually run.
        if int(self.elapsed_steps[0].item()) == 0:
            self.matching_contact_force = 0.0
            return
        matching_pair = tuple(sorted(self.matching_scrolls))
        matching_force = 0.0
        wrong_force = 0.0
        for first in range(3):
            for second in range(first + 1, 3):
                force = self._scroll_pair_contact_force(first, second)
                if (first, second) == matching_pair:
                    matching_force = force
                else:
                    wrong_force = max(wrong_force, force)
        self.matching_contact_force = matching_force
        self.matching_contact_force_max = max(
            self.matching_contact_force_max, matching_force
        )
        if matching_force > CFG["contact_force_threshold"]:
            self.matching_pair_ever_touched = True
        self.wrong_contact_force_max = max(self.wrong_contact_force_max, wrong_force)
        if wrong_force > CFG["contact_force_threshold"]:
            self.wrong_pair_ever_touched = True

    def _before_task_control_step(self) -> None:
        for scroll in self.scrolls:
            scroll.assembly.drive_to(0.0)

    def _after_task_control_step(self) -> None:
        step = int(self.elapsed_steps[0].item())
        for index, scroll in enumerate(self.scrolls):
            if not self.observed[index] and float(scroll.assembly.qpos[0].item()) <= CFG["reveal_threshold"]:
                self.observed[index] = True
                # This hook observes s_{step+1} while elapsed_steps names s_step.
                start_frame = step + 1
                self.define_keyframe_event(
                    f"scroll_{index}_fully_revealed",
                    start_frame,
                    start_frame + self.reveal_hold_steps[index] - 1,
                    required_information=(f"scroll_{index}_texture_identity", "scroll_location"),
                )
        if self._fsm.complete and self._fsm_index < 2:
            current = self.observation_order[self._fsm_index]
            if float(self.scrolls[current].assembly.qpos[0].item()) >= CFG["closed_threshold"]:
                self._fsm_index += 1
                self._fsm = self._slow_fsm(self._reveal_waypoints(self.observation_order[self._fsm_index]))
        final_index = self.observation_order[-1]
        final_xy = self.slot_xy[final_index]
        final_retract = torch.as_tensor(
            [
                float(
                    final_xy[0]
                    - CFG["strip_travel"]
                    - CFG["pull_overtravel"]
                    + CFG["pull_tab_x"]
                ),
                float(final_xy[1]),
                float(0.044 + CFG["reveal_clearance"]),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        tcp_transition_error = torch.linalg.norm(
            self.agent.tcp.pose.p[0]
            - final_retract
        )
        transition_ready_now = (
            all(self.observed)
            and all(
                float(s.assembly.qpos[0].item()) >= CFG["closed_threshold"]
                for s in self.scrolls
            )
            and float(tcp_transition_error.item()) <= 0.010
        )
        self._transition_stable_steps = (
            self._transition_stable_steps + 1 if transition_ready_now else 0
        )
        observation_transition_ready = self._transition_stable_steps >= 1
        # Do not couple environment phase/success state to the expert-only FSM:
        # raw action replay and a VLA client intentionally never call it.
        if observation_transition_ready and not self._observation_complete:
            self._observation_complete = True
            observation_end = step + 1
            self.execution_start = observation_end + 1 + self.memory_delay_steps
            self.set_phase_schedule(observation_end, self.execution_start)

    def _evaluate_task(self):
        self._update_scroll_contacts()
        moving = self.scrolls[self.matching_scrolls[0]].root_link
        target = self.scrolls[self.matching_scrolls[1]].root_link
        released = not bool(self.agent.is_grasping(moving)[0].item())
        matching_touching = self.matching_contact_force > CFG["contact_force_threshold"]
        stack_xy_error = float(
            torch.linalg.norm(moving.pose.p[0, :2] - target.pose.p[0, :2]).item()
        )
        stack_root_dz = float((moving.pose.p[0, 2] - target.pose.p[0, 2]).item())
        stacked_above = (
            stack_xy_error <= CFG["stack_xy_tolerance"]
            and CFG["stack_min_root_dz"]
            <= stack_root_dz
            <= CFG["stack_max_root_dz"]
        )
        tcp_home_error = float(
            torch.linalg.norm(
                self.agent.tcp.pose.p[0]
                - torch.as_tensor(HOME_TCP_POSITION, device=self.device)
            ).item()
        )
        robot_clear = tcp_home_error <= CFG["final_home_tolerance"]
        terminal_physics_ready = (
            self._observation_complete
            and matching_touching
            and self.matching_pair_ever_touched
            and not self.wrong_pair_ever_touched
            and released
            and stacked_above
        )
        # Formal evaluation requires a stable released stack. Expert recording
        # additionally waits for the robot to remain at home.
        step = int(self.elapsed_steps[0].item())
        if step != self._last_evaluation_gate_step:
            self._evaluation_stable_steps = (
                self._evaluation_stable_steps + 1
                if terminal_physics_ready
                else 0
            )
            self._last_evaluation_gate_step = step
        evaluation_success = (
            terminal_physics_ready
            and self._evaluation_stable_steps >= EVALUATION_STABLE_STEPS
        )
        expert_terminal_ready = terminal_physics_ready and robot_clear
        if step != self._last_final_home_gate_step:
            self._final_home_stable_steps = (
                self._final_home_stable_steps + 1
                if expert_terminal_ready
                else 0
            )
            self._last_final_home_gate_step = step
        final_home_stable = (
            self._final_home_stable_steps >= CFG["final_home_stable_steps"]
        )
        success = expert_terminal_ready and final_home_stable
        return {
            "success": torch.tensor([success], dtype=torch.bool, device=self.device),
            "evaluation_success": torch.tensor(
                [evaluation_success], dtype=torch.bool, device=self.device
            ),
            "revealed_scrolls": torch.tensor([sum(self.observed)], device=self.device),
            "matching_scrolls_touching": torch.tensor([matching_touching], dtype=torch.bool, device=self.device),
            "matching_contact_force": torch.tensor([self.matching_contact_force], dtype=torch.float32, device=self.device),
            "matching_contact_force_max": torch.tensor([self.matching_contact_force_max], dtype=torch.float32, device=self.device),
            "matching_pair_ever_touched": torch.tensor([self.matching_pair_ever_touched], dtype=torch.bool, device=self.device),
            "stack_xy_error": torch.tensor([stack_xy_error], dtype=torch.float32, device=self.device),
            "stack_root_dz": torch.tensor([stack_root_dz], dtype=torch.float32, device=self.device),
            "stacked_above": torch.tensor([stacked_above], dtype=torch.bool, device=self.device),
            "paired_scroll_released": torch.tensor([released], dtype=torch.bool, device=self.device),
            "evaluation_stable_steps": torch.tensor(
                [self._evaluation_stable_steps], dtype=torch.int64, device=self.device
            ),
            "final_tcp_home_error": torch.tensor(
                [tcp_home_error], dtype=torch.float32, device=self.device
            ),
            "robot_clear_of_stack": torch.tensor(
                [robot_clear], dtype=torch.bool, device=self.device
            ),
            "final_home_stable_steps": torch.tensor(
                [self._final_home_stable_steps], dtype=torch.int64, device=self.device
            ),
            "final_home_stable": torch.tensor(
                [final_home_stable], dtype=torch.bool, device=self.device
            ),
            "wrong_pair_ever_touched": torch.tensor([self.wrong_pair_ever_touched], dtype=torch.bool, device=self.device),
            "wrong_contact_force_max": torch.tensor([self.wrong_contact_force_max], dtype=torch.float32, device=self.device),
        }

    def _expert_action(self):
        if not self._fsm.complete:
            return self._fsm.action()
        if not self._observation_complete:
            return self.zero_action()
        if self._pair_fsm is None:
            self._pair_fsm = self._slow_fsm(self._pair_waypoints())
        return self._pair_fsm.action()


__all__ = ["ADScrollDiscovery", "ENV_ID", "INSTRUCTION_TASK", "INSTRUCTION_MEMORY"]
