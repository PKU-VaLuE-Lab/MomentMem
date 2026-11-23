"""Direct+Spatial task with three genuinely hollow dynamic cup covers."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common import (  # noqa: E402
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_TCP_POSITION,
    PhysicalWaypoint,
    PhysicalWaypointFSM,
    PhysicsMaterial,
    TASK_ENV_IDS,
    UnifiedPandaEnv,
    assert_points_reachable,
    build_dynamic_box,
    render_material,
)
try:  # package import and direct-script execution are both supported
    from .SD_explore_cube_cfg import CFG
except ImportError:  # pragma: no cover - direct gen_data.py execution
    from SD_explore_cube_cfg import CFG  # type: ignore

TASK_NAME = "SD_explore_cube"
ENV_ID = TASK_ENV_IDS[TASK_NAME]
INSTRUCTION_TASK = (
    "Three identical opaque cups are spread across the table, and one cup hides a "
    "green cube. Inspect all three cups in any order. Each time, lift a cup high enough to see "
    "under it, then put it back upside down in the same place before checking the next. "
    "Once all three cups have been checked, move the cup hiding the cube aside, then grasp and "
    "lift the cube. Leave the other cups in place and leave every cup upside down on the table."
)
INSTRUCTION_MEMORY = (
    "Remember each moment when a cup is lifted high enough to see clearly whether the green cube is underneath."
)


def _build_cup(scene, name: str) -> Any:
    """A round open-bottom cup made from physical wall segments and a top."""

    inner = float(CFG["cup_inner_radius"])
    outer = float(CFG["cup_outer_radius"])
    height = float(CFG["cup_height"])
    t = float(CFG["cup_wall"])
    physics = PhysicsMaterial(static_friction=1.2, dynamic_friction=0.9, density=650)
    body_material = render_material((0.84, 0.76, 0.62, 1.0))
    rim_material = render_material((0.35, 0.24, 0.15, 1.0))
    builder = scene.create_actor_builder()
    segment_count = int(CFG["cup_wall_segments"])
    radial_center = 0.5 * (inner + outer)
    tangent_half = outer * math.pi / segment_count * 1.10
    for segment in range(segment_count):
        angle = 2 * math.pi * segment / segment_count
        pose = sapien.Pose(
            p=[radial_center * math.cos(angle), radial_center * math.sin(angle), height / 2],
            q=euler2quat(0.0, 0.0, angle + math.pi / 2),
        )
        half = [tangent_half, t / 2, height / 2]
        builder.add_box_collision(
            pose=pose,
            half_size=half,
            material=physics.sapien(),
            density=physics.density,
        )
        builder.add_box_visual(pose=pose, half_size=half, material=body_material)
    top_pose = sapien.Pose(
        p=[0, 0, height + t / 2],
        q=euler2quat(0.0, math.pi / 2, 0.0),
    )
    builder.add_cylinder_collision(
        pose=top_pose,
        radius=outer,
        half_length=t / 2,
        material=physics.sapien(),
        density=physics.density,
    )
    builder.add_cylinder_visual(
        pose=top_pose,
        radius=outer,
        half_length=t / 2,
        material=body_material,
    )
    # A real bridge handle: two collision posts plus one graspable bar.
    post_half = [0.006, 0.010, float(CFG["handle_post_half_height"])]
    post_center_z = height + t + post_half[2]
    for x in (-0.026, 0.026):
        pose = sapien.Pose(p=[x, 0, post_center_z])
        builder.add_box_collision(pose=pose, half_size=post_half, material=physics.sapien(), density=physics.density)
        builder.add_box_visual(pose=pose, half_size=post_half, material=rim_material)
    bridge_pose = sapien.Pose(p=[0, 0, float(CFG["handle_tcp_height"])])
    bridge_half = [0.033, 0.010, 0.005]
    builder.add_box_collision(pose=bridge_pose, half_size=bridge_half, material=physics.sapien(), density=physics.density)
    builder.add_box_visual(pose=bridge_pose, half_size=bridge_half, material=rim_material)
    builder.initial_pose = sapien.Pose([-0.5, 0.8, 0])
    actor = builder.build_dynamic(name)
    actor.set_mass(0.18)
    return actor


@register_env(ENV_ID, max_episode_steps=1800)
class SDExploreCube(UnifiedPandaEnv):
    TASK_NAME = TASK_NAME
    INSTRUCTION_TASK = INSTRUCTION_TASK
    INSTRUCTION_MEMORY = INSTRUCTION_MEMORY

    @staticmethod
    def _inside_carried_workspace(xy: np.ndarray) -> bool:
        """Guard the empirically robust top-down carried-object workspace."""

        points = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        base = np.asarray(CFG["robot_base_xy"], dtype=np.float32)
        radii = np.linalg.norm(points - base[None, :], axis=1)
        low, high = [float(x) for x in CFG["carried_horizontal_reach_range"]]
        return bool(np.all((radii >= low) & (radii <= high)))

    def _load_task_scene(self, options: dict) -> None:
        del options
        # Identical appearance prevents cup identity from replacing spatial memory.
        self.cups = [_build_cup(self.scene, f"inspection_cup_{i}") for i in range(3)]
        h = float(CFG["cube_half_size"])
        self.hidden_cube = build_dynamic_box(
            self.scene,
            "hidden_cube",
            [h, h, h],
            [-0.5, 0.8, h],
            color=(0.035, 0.72, 0.12, 1),
            mass=0.045,
        )
        self.register_manipulable(*self.cups, self.hidden_cube)

    @staticmethod
    def _sample_non_collinear_layout(rng) -> np.ndarray:
        low = np.asarray(CFG["layout_xy_low"], dtype=np.float32)
        high = np.asarray(CFG["layout_xy_high"], dtype=np.float32)
        for _ in range(1000):
            xy = rng.uniform(low, high, size=(3, 2)).astype(np.float32)
            pairwise = [
                float(np.linalg.norm(xy[i] - xy[j]))
                for i in range(3)
                for j in range(i + 1, 3)
            ]
            twice_area = abs(float(np.cross(xy[1] - xy[0], xy[2] - xy[0])))
            if (
                min(pairwise) >= float(CFG["minimum_cup_separation"])
                and twice_area >= float(CFG["minimum_twice_triangle_area"])
                and SDExploreCube._inside_carried_workspace(xy)
            ):
                return xy
        raise RuntimeError("failed to sample a separated non-collinear cup layout")

    @staticmethod
    def _sample_side_goal(rng, xy: np.ndarray) -> np.ndarray:
        low = np.asarray(CFG["cup_side_goal_low"], dtype=np.float32)
        high = np.asarray(CFG["cup_side_goal_high"], dtype=np.float32)
        for _ in range(2000):
            point = rng.uniform(low, high).astype(np.float32)
            clearance = np.linalg.norm(xy - point[None, :], axis=1)
            if (
                float(clearance.min()) >= float(CFG["cup_side_goal_min_clearance"])
                and SDExploreCube._inside_carried_workspace(point)
            ):
                return point
        raise RuntimeError("failed to sample a collision-free random cup side goal")

    def _initialize_task_episode(self, env_idx, episode_metadata: Mapping[str, Any] | None) -> None:
        del env_idx
        if episode_metadata is None:
            target = int(self._episode_rng.randint(3))
            order = self._episode_rng.permutation(3).astype(int).tolist()
            cup_xy = self._sample_non_collinear_layout(self._episode_rng)
            holds = [int(self._episode_rng.randint(CFG["reveal_hold_steps"][0], CFG["reveal_hold_steps"][1] + 1)) for _ in range(3)]
            delay = int(self._episode_rng.randint(CFG["memory_delay_steps"][0], CFG["memory_delay_steps"][1] + 1))
            side_goal = self._sample_side_goal(self._episode_rng, cup_xy)
        else:
            target = int(episode_metadata["target_cup"])
            order = [int(x) for x in episode_metadata["inspection_order"]]
            cup_xy = np.asarray(episode_metadata.get("cup_xy", episode_metadata.get("slot_xy")), dtype=np.float32)
            holds = [int(x) for x in episode_metadata["reveal_hold_steps"]]
            delay = int(episode_metadata["memory_delay_steps"])
            side_goal = np.asarray(episode_metadata["cup_side_goal"], dtype=np.float32)
        twice_area = abs(float(np.cross(cup_xy[1] - cup_xy[0], cup_xy[2] - cup_xy[0])))
        self.slot_xy = cup_xy
        self.cup_side_goal = side_goal
        self.target_cup = target
        self.inspection_order = order
        self.reveal_hold_steps = holds
        self.memory_delay_steps = delay
        q = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=self.device)
        for i, cup in enumerate(self.cups):
            cup.set_pose(Pose.create_from_pq(
                p=torch.tensor([[self.slot_xy[i, 0], self.slot_xy[i, 1], 0.002]], dtype=torch.float32, device=self.device),
                q=q,
            ))
            cup.set_linear_velocity(torch.zeros((1, 3), device=self.device))
            cup.set_angular_velocity(torch.zeros((1, 3), device=self.device))
        h = float(CFG["cube_half_size"])
        target_xy = self.slot_xy[target]
        self.hidden_cube.set_pose(Pose.create_from_pq(
            p=torch.tensor([[target_xy[0], target_xy[1], h]], dtype=torch.float32, device=self.device),
            q=q,
        ))
        self.hidden_cube.set_linear_velocity(torch.zeros((1, 3), device=self.device))
        self.hidden_cube.set_angular_velocity(torch.zeros((1, 3), device=self.device))
        self.observed = [False, False, False]
        self._inspection_index = 0
        self._observation_complete = False
        self._memory_home_stable_steps = 0
        self.execution_start = 10_000
        self.set_phase_schedule(9_998, 9_999)
        reach_z = float(CFG["handle_tcp_height"])
        assert_points_reachable(
            np.concatenate(
                [
                    np.column_stack((self.slot_xy, np.full(3, reach_z))),
                    np.asarray([[side_goal[0], side_goal[1], reach_z]], dtype=np.float32),
                ],
                axis=0,
            )
        )
        self.set_task_episode_metadata(
            target_cup=target,
            inspection_order=order,
            cup_xy=self.slot_xy,
            reveal_hold_steps=holds,
            memory_delay_steps=delay,
            memory_home_tolerance=0.006,
            memory_home_stable_steps=4,
            cup_side_goal=self.cup_side_goal,
            cup_side_goal_sampling="episode_random_collision_free_side_position",
            cup_side_goal_min_clearance=CFG["cup_side_goal_min_clearance"],
            carried_workspace_base_xy=CFG["robot_base_xy"],
            carried_horizontal_reach_range=CFG["carried_horizontal_reach_range"],
            carried_workspace_contract="conservative_top_down_carry_with_elbow_bend",
            cup_appearance="identical_round_segmented_cups_with_bridge_handles",
            cup_resting_contract=(
                "mouth_down_table_contact_when_not_grasped_during_memory_reset_and_success"
            ),
            cup_mouth_down_min_axis_z=CFG["cup_mouth_down_min_axis_z"],
            cup_table_contact_force_threshold=CFG[
                "cup_table_contact_force_threshold"
            ],
            cup_table_root_z_bounds=CFG["cup_table_root_z_bounds"],
            cup_replacement_xy_tolerance=CFG["cup_replacement_xy_tolerance"],
            hidden_cube_color_name="green",
            hidden_cube_color_rgba=(0.035, 0.72, 0.12, 1),
            evaluation_cube_table_clearance=CFG["cube_table_clearance"],
            evaluation_success_condition=(
                "all_cups_inspected_and_reset_then_green_cube_grasped_with_"
                "bottom_face_clear_of_table_while_all_cups_rest_upside_down_"
                "and_non_target_cups_remain_in_place"
            ),
            layout_twice_triangle_area=twice_area,
        )
        self._inspect_fsm = PhysicalWaypointFSM(self, self._inspect_waypoints(order[0]))
        self._execute_fsm = None

    def _cup_handle(self, index, lift=0.0):
        def resolve(env):
            p = env.cups[index].pose.p[0].detach().cpu().numpy().copy()
            p[2] += float(CFG["handle_tcp_height"] + lift)
            return p
        return resolve

    def _cup_handle_at_height(self, index, tcp_z):
        """Track cup XY while using a fixed world-frame lift height."""

        def resolve(env):
            p = env.cups[index].pose.p[0].detach().cpu().numpy().copy()
            p[2] = float(tcp_z)
            return p

        return resolve

    def _cup_grasped(self, index):
        return bool(self.agent.is_grasping(self.cups[index])[0].item())

    def _cup_resting_on_table(self, index: int) -> tuple[bool, float, float]:
        """Return physical mouth-down tabletop state for one released cup."""

        cup = self.cups[index]
        quaternion = cup.pose.q[0]
        _, qx, qy, _ = quaternion.unbind()
        # The cup opens along local -Z.  Its mouth therefore points down when
        # local +Z remains aligned with world +Z.
        local_z_world_z = float(
            (1.0 - 2.0 * (qx * qx + qy * qy)).item()
        )
        contact_force = float(
            torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(
                    cup, self.table_scene.table
                ),
                dim=-1,
            )[0].item()
        )
        root_z = float(cup.pose.p[0, 2].item())
        z_low, z_high = CFG["cup_table_root_z_bounds"]
        resting = (
            not self._cup_grasped(index)
            and float(z_low) <= root_z <= float(z_high)
            and local_z_world_z >= float(CFG["cup_mouth_down_min_axis_z"])
            and contact_force >= float(CFG["cup_table_contact_force_threshold"])
        )
        return resting, local_z_world_z, contact_force

    def _inspect_waypoints(self, index):
        hold = self.reveal_hold_steps[index]
        return [
            PhysicalWaypoint(self._cup_handle(index, 0.12), GRIPPER_OPEN, label=f"cup_{index}_above"),
            PhysicalWaypoint(self._cup_handle(index), GRIPPER_OPEN, dwell_steps=3, label=f"cup_{index}_handle"),
            PhysicalWaypoint(self._cup_handle(index), GRIPPER_CLOSED, dwell_steps=9, require=lambda env, i=index: env._cup_grasped(i), label=f"cup_{index}_grasp"),
            PhysicalWaypoint(self._cup_handle_at_height(index, 0.255), GRIPPER_CLOSED, dwell_steps=hold, require=lambda env, i=index: env._cup_grasped(i), label=f"cup_{index}_reveal"),
            PhysicalWaypoint(lambda env, i=index: [float(env.slot_xy[i, 0]), float(env.slot_xy[i, 1]), float(CFG["handle_tcp_height"])], GRIPPER_CLOSED, dwell_steps=4, label=f"cup_{index}_replace"),
            PhysicalWaypoint(lambda env, i=index: [float(env.slot_xy[i, 0]), float(env.slot_xy[i, 1]), float(CFG["handle_tcp_height"])], GRIPPER_OPEN, dwell_steps=8, label=f"cup_{index}_release"),
            PhysicalWaypoint(self._cup_handle(index, 0.13), GRIPPER_OPEN, label=f"cup_{index}_retract"),
            PhysicalWaypoint(
                HOME_TCP_POSITION,
                GRIPPER_OPEN,
                position_tolerance=0.004,
                dwell_steps=4,
                label="home",
            ),
        ]

    def _execute_waypoints(self):
        i = self.target_cup
        side = self.cup_side_goal
        htcp = float(CFG["handle_tcp_height"])
        cube = self.hidden_cube

        def cube_at_height(env, z):
            del env
            p = cube.pose.p[0].detach().cpu().numpy()
            return np.asarray([float(p[0]), float(p[1]), float(z)], dtype=np.float32)

        return [
            PhysicalWaypoint(self._cup_handle(i, 0.12), GRIPPER_OPEN, label="target_cup_above"),
            PhysicalWaypoint(self._cup_handle(i), GRIPPER_OPEN, label="target_cup_handle"),
            PhysicalWaypoint(self._cup_handle(i), GRIPPER_CLOSED, dwell_steps=9, require=lambda env: env._cup_grasped(i), label="target_cup_grasp"),
            # Lift vertically clear of the two other physical covers before
            # translating across their footprints.  A diagonal lift can make
            # the carried cover collide with the middle cover while both are
            # still at almost the same height.
            PhysicalWaypoint(self._cup_handle_at_height(i, 0.30), GRIPPER_CLOSED, require=lambda env: env._cup_grasped(i), label="target_cup_vertical_clearance"),
            PhysicalWaypoint(lambda env: [side[0], side[1], 0.30], GRIPPER_CLOSED, label="target_cup_side_high"),
            PhysicalWaypoint(lambda env: [side[0], side[1], htcp], GRIPPER_CLOSED, label="target_cup_side_low"),
            PhysicalWaypoint(lambda env: [side[0], side[1], htcp], GRIPPER_OPEN, dwell_steps=8, label="target_cup_side_release"),
            # The direct low diagonal from the released cover to the cube can
            # cross another randomized cup.  Retract vertically and traverse
            # above every handle before descending at the cube footprint.
            PhysicalWaypoint(lambda env: [side[0], side[1], 0.30], GRIPPER_OPEN, label="target_cup_side_retract"),
            PhysicalWaypoint(lambda env: cube_at_height(env, 0.30), GRIPPER_OPEN, label="cube_transfer_high"),
            PhysicalWaypoint(lambda env: cube_at_height(env, 0.17), GRIPPER_OPEN, label="cube_above"),
            PhysicalWaypoint(lambda env: cube_at_height(env, 0.032), GRIPPER_OPEN,
                             position_tolerance=0.012, rotation_tolerance=10.0,
                             dwell_steps=3, label="cube_contact"),
            PhysicalWaypoint(lambda env: cube_at_height(env, 0.032), GRIPPER_CLOSED,
                             position_tolerance=0.012, rotation_tolerance=10.0,
                             dwell_steps=10,
                             require=lambda env: bool(env.agent.is_grasping(cube)[0].item()),
                             label="cube_grasp"),
            PhysicalWaypoint(lambda env: cube_at_height(env, 0.20), GRIPPER_CLOSED,
                             rotation_tolerance=10.0, dwell_steps=8,
                             require=lambda env: bool(env.agent.is_grasping(cube)[0].item()),
                             label="cube_lift"),
        ]

    def _after_task_control_step(self) -> None:
        step = int(self.elapsed_steps[0].item())
        for index, cup in enumerate(self.cups):
            if not self.observed[index] and float(cup.pose.p[0, 2].item()) > 0.105:
                self.observed[index] = True
                self.define_keyframe_event(
                    f"cup_{index}_footprint_observation",
                    step + 1,
                    step + min(11, self.reveal_hold_steps[index] - 1),
                    required_information=("cup_location", "cube_present_or_absent"),
                )
        if self._inspect_fsm.complete and self._inspection_index < 2:
            current = self.inspection_order[self._inspection_index]
            position_error = torch.linalg.norm(
                self.cups[current].pose.p[0, :2] - torch.as_tensor(self.slot_xy[current], device=self.device)
            )
            current_resting, _, _ = self._cup_resting_on_table(current)
            if float(position_error.item()) < 0.025 and current_resting:
                self._inspection_index += 1
                self._inspect_fsm = PhysicalWaypointFSM(self, self._inspect_waypoints(self.inspection_order[self._inspection_index]))
        cups_replaced = True
        for index, cup in enumerate(self.cups):
            position_error = torch.linalg.norm(
                cup.pose.p[0, :2]
                - torch.as_tensor(self.slot_xy[index], device=self.device)
            )
            resting, _, _ = self._cup_resting_on_table(index)
            cups_replaced = cups_replaced and (
                float(position_error.item())
                < float(CFG["cup_replacement_xy_tolerance"])
                and resting
            )
        tcp_home_error = torch.linalg.norm(
            self.agent.tcp.pose.p[0]
            - torch.as_tensor(HOME_TCP_POSITION, device=self.device)
        )
        home_reset_now = (
            all(self.observed)
            and cups_replaced
            and float(tcp_home_error.item()) <= 0.006
        )
        self._memory_home_stable_steps = (
            self._memory_home_stable_steps + 1 if home_reset_now else 0
        )
        observation_physically_reset = self._memory_home_stable_steps >= 4
        if observation_physically_reset and not self._observation_complete:
            self._observation_complete = True
            observation_end = step + 3
            self.execution_start = observation_end + 1 + self.memory_delay_steps
            self.set_phase_schedule(observation_end, self.execution_start)

    def _evaluate_task(self):
        grasped = bool(self.agent.is_grasping(self.hidden_cube)[0].item())
        cube_center_z = float(self.hidden_cube.pose.p[0, 2].item())
        cube_bottom_z = cube_center_z - float(CFG["cube_half_size"])
        expert_lifted = cube_center_z > 0.105
        evaluation_lifted = (
            cube_bottom_z > float(CFG["cube_table_clearance"])
        )
        cup_states = [self._cup_resting_on_table(i) for i in range(3)]
        cups_resting = [state[0] for state in cup_states]
        mouth_down_scores = [state[1] for state in cup_states]
        table_contact_forces = [state[2] for state in cup_states]
        replacement_errors = [
            float(
                torch.linalg.norm(
                    cup.pose.p[0, :2]
                    - torch.as_tensor(self.slot_xy[index], device=self.device)
                ).item()
            )
            for index, cup in enumerate(self.cups)
        ]
        non_target_cups_replaced = all(
            index == self.target_cup
            or replacement_errors[index]
            < float(CFG["cup_replacement_xy_tolerance"])
            for index in range(3)
        )
        all_cups_resting = all(cups_resting)
        return {
            "success": torch.tensor(
                [
                    self._observation_complete
                    and grasped
                    and expert_lifted
                    and all_cups_resting
                    and non_target_cups_replaced
                ],
                dtype=torch.bool,
                device=self.device,
            ),
            "evaluation_success": torch.tensor(
                [
                    self._observation_complete
                    and grasped
                    and evaluation_lifted
                    and all_cups_resting
                    and non_target_cups_replaced
                ],
                dtype=torch.bool,
                device=self.device,
            ),
            "inspected_cups": torch.tensor([sum(self.observed)], device=self.device),
            "cube_lifted": torch.tensor([expert_lifted], device=self.device),
            "evaluation_cube_lifted": torch.tensor(
                [evaluation_lifted], device=self.device
            ),
            "cube_grasped": torch.tensor([grasped], device=self.device),
            "cube_center_z": torch.tensor(
                [cube_center_z], dtype=torch.float32, device=self.device
            ),
            "cube_bottom_table_clearance": torch.tensor(
                [cube_bottom_z], dtype=torch.float32, device=self.device
            ),
            "cups_resting_on_table": torch.tensor(
                [cups_resting], dtype=torch.bool, device=self.device
            ),
            "cup_mouth_down_axis_z": torch.tensor(
                [mouth_down_scores], dtype=torch.float32, device=self.device
            ),
            "cup_table_contact_force": torch.tensor(
                [table_contact_forces], dtype=torch.float32, device=self.device
            ),
            "cup_replacement_error_xy": torch.tensor(
                [replacement_errors], dtype=torch.float32, device=self.device
            ),
            "non_target_cups_replaced": torch.tensor(
                [non_target_cups_replaced], dtype=torch.bool, device=self.device
            ),
        }

    def _expert_action(self):
        step = int(self.elapsed_steps[0].item())
        if not self._inspect_fsm.complete:
            return self._inspect_fsm.action()
        if not self._observation_complete or step < self.execution_start:
            return self.zero_action()
        if self._execute_fsm is None:
            self._execute_fsm = PhysicalWaypointFSM(self, self._execute_waypoints())
        return self._execute_fsm.action()


__all__ = ["SDExploreCube", "ENV_ID", "INSTRUCTION_TASK", "INSTRUCTION_MEMORY"]
