"""Triggered Appearance+Spatial task using a physically grasped scan probe."""

from __future__ import annotations

import math
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
    COLOR_NAMES,
    COLOR_RGBA,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_TCP_POSITION,
    PhysicalWaypoint,
    PhysicalWaypointFSM,
    PhysicsMaterial,
    TASK_ENV_IDS,
    UnifiedPandaEnv,
    assert_points_reachable,
    build_visual_marker_box,
    render_material,
)
from AST_mine_sweeper_cfg import CFG  # noqa: E402

TASK_NAME = "AST_mine_sweeper"
ENV_ID = TASK_ENV_IDS[TASK_NAME]
INSTRUCTION_TASK = (
    "Two parallel black lines, a white probe, and two cubes of different colors are on the table. "
    "Move the probe directly to one end of the nearer line and scan "
    "once to the other end. Move directly to the farther line and scan it once in the opposite "
    "direction. The probe briefly changes color at one point on each line. Then put the probe back, "
    "return the arm to its starting pose, and place each cube where the probe showed the same color. "
    "Release each cube on the table."
)
INSTRUCTION_MEMORY = "Remember each brief moment when the probe changes color while crossing a line."

AST_COLOR_CANDIDATE_IDS = np.asarray(
    [index for index, name in enumerate(COLOR_NAMES) if name != "yellow"],
    dtype=np.int64,
)


def _sample_ast_color_ids(rng) -> np.ndarray:
    """Sample a high-contrast AST pair while excluding low-visibility yellow."""

    pairs = [
        (int(i), int(j))
        for offset, i in enumerate(AST_COLOR_CANDIDATE_IDS)
        for j in AST_COLOR_CANDIDATE_IDS[offset + 1 :]
        if float(np.linalg.norm(COLOR_RGBA[i, :3] - COLOR_RGBA[j, :3])) >= 0.72
    ]
    pair = pairs[int(rng.randint(len(pairs)))]
    return np.asarray(pair, dtype=np.int64)[rng.permutation(2)]

def _build_probe(scene):
    physics = PhysicsMaterial(static_friction=1.25, dynamic_friction=0.95, density=600)
    builder = scene.create_actor_builder()
    pose = sapien.Pose(q=[math.sqrt(0.5), 0, math.sqrt(0.5), 0])
    builder.add_cylinder_collision(
        radius=float(CFG["probe_radius"]),
        half_length=float(CFG["probe_half_height"]),
        pose=pose,
        material=physics.sapien(),
        density=physics.density,
    )
    material = render_material((0.93, 0.93, 0.93, 1))
    builder.add_cylinder_visual(
        radius=float(CFG["probe_radius"]),
        half_length=float(CFG["probe_half_height"]),
        pose=pose,
        material=material,
    )
    builder.initial_pose = sapien.Pose(
        [CFG["probe_spawn"][0], CFG["probe_spawn"][1], CFG["probe_half_height"]]
    )
    actor = builder.build_dynamic("scan_probe")
    actor.set_mass(float(CFG["probe_mass"]))
    return actor, material


def _build_color_cube(scene, name: str):
    physics = PhysicsMaterial(static_friction=0.9, dynamic_friction=0.7, density=650)
    builder = scene.create_actor_builder()
    material = render_material((0.8, 0.8, 0.8, 1.0))
    builder.add_box_collision(
        half_size=[0.023] * 3, material=physics.sapien(), density=physics.density
    )
    builder.add_box_visual(half_size=[0.023] * 3, material=material)
    builder.initial_pose = sapien.Pose([-0.25, 0.0, 0.023])
    actor = builder.build_dynamic(name)
    actor.set_mass(0.065)
    return actor, material


@register_env(ENV_ID, max_episode_steps=1500)
class ASTMineSweeper(UnifiedPandaEnv):
    TASK_NAME = TASK_NAME
    INSTRUCTION_TASK = INSTRUCTION_TASK
    INSTRUCTION_MEMORY = INSTRUCTION_MEMORY

    def _load_task_scene(self, options: dict) -> None:
        del options
        for line_index, x in enumerate(CFG["line_x"]):
            build_visual_marker_box(
                self.scene,
                f"black_scan_rail_{line_index}",
                [0.010, CFG["rail_half_length"], 0.001],
                [x, CFG["rail_center_y"], 0.002],
                color=(0.025, 0.025, 0.025, 1),
            )
        self.probe, self.probe_material = _build_probe(self.scene)
        built = [
            _build_color_cube(self.scene, "red_cube" if i == 0 else "blue_cube")
            for i in range(2)
        ]
        self.cubes = [item[0] for item in built]
        self.cube_materials = [item[1] for item in built]
        self.register_manipulable(self.probe, *self.cubes)

    @staticmethod
    def _cube_spawn_geometry_valid(cube_spawn_xy: np.ndarray) -> bool:
        points = np.asarray(cube_spawn_xy, dtype=np.float32)
        if points.shape != (2, 2) or not np.all(np.isfinite(points)):
            return False
        low = np.asarray(CFG["cube_spawn_xy_low"], dtype=np.float32)
        high = np.asarray(CFG["cube_spawn_xy_high"], dtype=np.float32)
        if np.any(points < low) or np.any(points > high):
            return False
        if float(np.linalg.norm(points[0] - points[1])) < float(
            CFG["cube_spawn_min_separation"]
        ):
            return False
        probe_xy = np.asarray(CFG["probe_spawn"], dtype=np.float32)
        if np.any(
            np.linalg.norm(points - probe_xy[None, :], axis=1)
            < float(CFG["cube_spawn_probe_clearance"])
        ):
            return False
        line_x = np.asarray(CFG["line_x"], dtype=np.float32)
        if np.any(
            np.min(np.abs(points[:, 0, None] - line_x[None, :]), axis=1)
            < float(CFG["cube_spawn_rail_clearance"])
        ):
            return False
        y0, y1 = (float(x) for x in CFG["scan_y_endpoints"])
        endpoints = np.asarray(
            [[x, y] for x in line_x for y in (y0, y1)], dtype=np.float32
        )
        if np.any(
            np.min(
                np.linalg.norm(points[:, None, :] - endpoints[None, :, :], axis=2),
                axis=1,
            )
            < float(CFG["cube_spawn_endpoint_clearance"])
        ):
            return False
        return True

    def _sample_cube_spawns(self) -> np.ndarray:
        low = np.asarray(CFG["cube_spawn_xy_low"], dtype=np.float32)
        high = np.asarray(CFG["cube_spawn_xy_high"], dtype=np.float32)
        for _ in range(1000):
            points = self._episode_rng.uniform(low, high, size=(2, 2)).astype(
                np.float32
            )
            if self._cube_spawn_geometry_valid(points):
                return points
        raise RuntimeError("failed to sample two safe AST cube spawn positions")

    def _initialize_task_episode(self, env_idx, episode_metadata: Mapping[str, Any] | None) -> None:
        del env_idx
        if episode_metadata is None:
            trigger_y = self._episode_rng.uniform(*CFG["trigger_y_range"], size=2).astype(np.float32)
            while abs(float(trigger_y[0] - trigger_y[1])) < CFG["trigger_min_separation"]:
                trigger_y = self._episode_rng.uniform(*CFG["trigger_y_range"], size=2).astype(np.float32)
            line_colors = self._episode_rng.permutation(2).astype(int).tolist()
            selected_color_ids = _sample_ast_color_ids(self._episode_rng)
            cube_spawn_xy = self._sample_cube_spawns()
            delay_steps = int(self._episode_rng.randint(CFG["memory_delay_steps"][0], CFG["memory_delay_steps"][1] + 1))
        else:
            trigger_y = np.asarray(episode_metadata["trigger_y"], dtype=np.float32)
            line_colors = [int(x) for x in episode_metadata["line_color_ids"]]
            selected_color_ids = np.asarray(
                episode_metadata["selected_color_ids"], dtype=np.int64
            )
            cube_spawn_xy = np.asarray(
                episode_metadata["cube_spawn_xy"], dtype=np.float32
            )
            delay_steps = int(episode_metadata["memory_delay_steps"])
        assert_points_reachable(
            np.column_stack(
                (cube_spawn_xy, np.full(2, 0.023, dtype=np.float32))
            )
        )
        self.trigger_xy = torch.tensor(
            [[CFG["line_x"][i], trigger_y[i]] for i in range(2)],
            dtype=torch.float32,
            device=self.device,
        )
        self.line_colors = line_colors
        self.selected_color_ids = selected_color_ids.tolist()
        self.episode_colors = COLOR_RGBA[selected_color_ids]
        self.cube_spawn_xy = cube_spawn_xy
        self.memory_delay_steps = delay_steps
        p = torch.tensor(
            [[CFG["probe_spawn"][0], CFG["probe_spawn"][1], CFG["probe_half_height"]]],
            dtype=torch.float32,
            device=self.device,
        )
        q = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=self.device)
        self.probe.set_pose(Pose.create_from_pq(p=p, q=q))
        self.probe.set_linear_velocity(torch.zeros((1, 3), device=self.device))
        self.probe.set_angular_velocity(torch.zeros((1, 3), device=self.device))
        for i, cube in enumerate(self.cubes):
            self.cube_materials[i].set_base_color(
                tuple(float(x) for x in self.episode_colors[i])
            )
            xy = self.cube_spawn_xy[i]
            cube.set_pose(Pose.create_from_pq(
                p=torch.tensor([[xy[0], xy[1], 0.023]], dtype=torch.float32, device=self.device),
                q=q,
            ))
            cube.set_linear_velocity(torch.zeros((1, 3), device=self.device))
            cube.set_angular_velocity(torch.zeros((1, 3), device=self.device))
        self.probe_material.set_base_color((0.93, 0.93, 0.93, 1))
        self.triggered = [False, False]
        self.active_reveal_line = -1
        self.reveal_start_frames = [-1, -1]
        self.placed = [False, False]
        self._placement_stable_counts = [0, 0]
        self.placement_contact_forces = [0.0, 0.0]
        self.observation_complete = False
        self._memory_home_stable_steps = 0
        self.execution_start = 10_000
        self.set_phase_schedule(9_998, 9_999)
        self.set_task_episode_metadata(
            trigger_y=trigger_y,
            trigger_xy=self.trigger_xy,
            line_color_ids=line_colors,
            selected_color_ids=selected_color_ids,
            selected_color_names=[COLOR_NAMES[int(x)] for x in selected_color_ids],
            selected_color_rgba=self.episode_colors,
            color_candidate_count=len(AST_COLOR_CANDIDATE_IDS),
            color_candidate_names=[COLOR_NAMES[int(x)] for x in AST_COLOR_CANDIDATE_IDS],
            scan_target_speed_mps=CFG["scan_target_speed_mps"],
            scan_max_position_command=CFG["scan_max_position_command"],
            memory_delay_steps=delay_steps,
            memory_home_tolerance=0.006,
            memory_home_stable_steps=4,
            cube_spawn_xy=self.cube_spawn_xy,
            line_x_near_to_far=CFG["line_x"],
            scan_order=[0, 1],
            scan_order_roles=["near", "far"],
            scan_direction_y=[
                [CFG["scan_y_endpoints"][0], CFG["scan_y_endpoints"][1]],
                [CFG["scan_y_endpoints"][1], CFG["scan_y_endpoints"][0]],
            ],
            first_endpoint_approach="direct_collinear_high_carry_then_vertical_descent",
            first_endpoint_direct_segments=1,
            first_endpoint_direct_tolerance=CFG["first_endpoint_direct_tolerance"],
            direct_carry_motion="single_norm_clipped_cartesian_segment_without_handoff",
            reveal_trigger_gate="physical_low_probe_crossing_only",
            reveal_motion="continuous_constant_speed_pass_without_pause",
            reveal_visual_rule="probe_colored_only_while_inside_hidden_point_radius",
            probe_spawn_xy=CFG["probe_spawn"],
            placement_contact_force_threshold=CFG[
                "placement_contact_force_threshold"
            ],
            placement_linear_speed_tolerance=CFG[
                "placement_linear_speed_tolerance"
            ],
            placement_angular_speed_tolerance=CFG[
                "placement_angular_speed_tolerance"
            ],
            placement_stable_steps=CFG["placement_stable_steps"],
        )
        self._scan_fsm = PhysicalWaypointFSM(self, self._scan_waypoints())
        self._execute_fsm = None

    def _probe_grasped(self, _env=None):
        return bool(self.agent.is_grasping(self.probe)[0].item())

    def _object_point(self, actor, dz=0.0):
        def resolve(env):
            p = actor.pose.p[0].detach().cpu().numpy().copy()
            p[2] += dz
            return p
        return resolve

    def _scan_waypoints(self):
        def probe_at_height(env, z=0.14):
            p = env.probe.pose.p[0].detach().cpu().numpy()
            return [float(p[0]), float(p[1]), float(z)]

        wp = [
            PhysicalWaypoint(self._object_point(self.probe, 0.12), GRIPPER_OPEN, label="probe_above"),
            PhysicalWaypoint(self._object_point(self.probe, 0.002), GRIPPER_OPEN, label="probe_contact"),
            PhysicalWaypoint(self._object_point(self.probe, 0.002), GRIPPER_CLOSED, dwell_steps=8, require=lambda env: env._probe_grasped(), label="probe_grasp"),
            PhysicalWaypoint(probe_at_height, GRIPPER_CLOSED,
                             require=lambda env: env._probe_grasped(), label="probe_lift"),
        ]
        y_low, y_high = CFG["scan_y_endpoints"]
        # One norm-clipped target produces a truly straight, uninterrupted
        # carry.  It neither aligns with the rail early nor decelerates at a
        # chain of intermediate waypoints, so this approach cannot look like
        # a premature scan or exhibit periodic hand-off stutter.
        wp.append(
            PhysicalWaypoint(
                [CFG["line_x"][0], y_low, 0.14],
                GRIPPER_CLOSED,
                position_tolerance=float(CFG["first_endpoint_direct_tolerance"]),
                require=lambda env: env._probe_grasped(),
                label="near_endpoint_direct_carry",
                isotropic_position_command=True,
            )
        )
        scan_specs = (
            (0, CFG["line_x"][0], y_low, y_high, "near"),
            (1, CFG["line_x"][1], y_high, y_low, "far"),
        )
        for index, x, start_y, end_y, role in scan_specs:
            endpoint_waypoints = []
            if index != 0:
                endpoint_waypoints.append(
                    PhysicalWaypoint(
                        [x, start_y, 0.10], GRIPPER_CLOSED,
                        require=lambda env: env._probe_grasped(),
                        label=f"{role}_rail_endpoint_start_high",
                    )
                )
            else:
                endpoint_waypoints.append(
                    PhysicalWaypoint(
                        [x, start_y, 0.10], GRIPPER_CLOSED,
                        require=lambda env: env._probe_grasped(),
                        label="near_endpoint_vertical_descent_high",
                    )
                )
            wp.extend(endpoint_waypoints + [
                PhysicalWaypoint(
                    [x, start_y, 0.040], GRIPPER_CLOSED,
                    dwell_steps=3, require=lambda env: env._probe_grasped(),
                    label=f"{role}_rail_endpoint_start",
                ),
                PhysicalWaypoint(
                    [x, end_y, 0.040], GRIPPER_CLOSED,
                    position_tolerance=max(0.006, CFG["reveal_distance"] / 2),
                    dwell_steps=5, require=lambda env: env._probe_grasped(),
                    label=f"{role}_rail_single_pass_scan",
                    isotropic_position_command=True,
                    max_position_command=float(CFG["scan_max_position_command"]),
                ),
                PhysicalWaypoint(
                    [x, end_y, 0.10], GRIPPER_CLOSED,
                    require=lambda env: env._probe_grasped(),
                    label=f"{role}_rail_endpoint_finish_high",
                ),
            ])
        wp.extend([
            PhysicalWaypoint(
                [CFG["probe_spawn"][0], CFG["probe_spawn"][1], 0.14], GRIPPER_CLOSED,
                require=lambda env: env._probe_grasped(), label="probe_return_high",
            ),
            PhysicalWaypoint(
                [CFG["probe_spawn"][0], CFG["probe_spawn"][1], 0.040], GRIPPER_CLOSED,
                require=lambda env: env._probe_grasped(), label="probe_return",
            ),
            PhysicalWaypoint([CFG["probe_spawn"][0], CFG["probe_spawn"][1], 0.040], GRIPPER_OPEN, dwell_steps=7, label="probe_release"),
            PhysicalWaypoint(
                HOME_TCP_POSITION,
                GRIPPER_OPEN,
                position_tolerance=0.004,
                dwell_steps=4,
                label="observation_home",
            ),
        ])
        return wp

    def _target_for_color(self, color_index):
        line = self.line_colors.index(color_index)
        xy = self.trigger_xy[line].detach().cpu().numpy()
        return xy

    def _execute_waypoints(self):
        wp = []
        for color_index, cube in enumerate(self.cubes):
            def target(env, ci=color_index, z=0.14):
                xy = env._target_for_color(ci); return [float(xy[0]), float(xy[1]), z]
            wp.extend([
                PhysicalWaypoint(lambda env, a=cube: (a.pose.p[0].detach().cpu().numpy() + np.array([0, 0, 0.12], np.float32)), GRIPPER_OPEN, label=f"cube_{color_index}_above"),
                PhysicalWaypoint(lambda env, a=cube: (a.pose.p[0].detach().cpu().numpy() + np.array([0, 0, 0.002], np.float32)), GRIPPER_OPEN, label=f"cube_{color_index}_touch"),
                PhysicalWaypoint(lambda env, a=cube: (a.pose.p[0].detach().cpu().numpy() + np.array([0, 0, 0.002], np.float32)), GRIPPER_CLOSED, dwell_steps=8, require=lambda env, a=cube: bool(env.agent.is_grasping(a)[0].item()), label=f"cube_{color_index}_grasp"),
                PhysicalWaypoint(target, GRIPPER_CLOSED, label=f"cube_{color_index}_target_high"),
                PhysicalWaypoint(lambda env, ci=color_index: [*env._target_for_color(ci), 0.031], GRIPPER_CLOSED, label=f"cube_{color_index}_target_low"),
                PhysicalWaypoint(lambda env, ci=color_index: [*env._target_for_color(ci), 0.031], GRIPPER_OPEN, dwell_steps=7, label=f"cube_{color_index}_release"),
                PhysicalWaypoint(lambda env, ci=color_index: [*env._target_for_color(ci), 0.15], GRIPPER_OPEN, label=f"cube_{color_index}_retract"),
            ])
        wp.append(PhysicalWaypoint(HOME_TCP_POSITION, GRIPPER_OPEN, label="finish_home"))
        return wp

    def _after_task_control_step(self) -> None:
        step = int(self.elapsed_steps[0].item())
        grasped = self._probe_grasped()
        probe_xy = self.probe.pose.p[0, :2]

        # End a reveal as soon as the continuously moving physical probe exits
        # the hidden-point radius. The material update made in this post-step
        # hook is visible from s_(t+1), so the last colored recorded frame is
        # s_t. Defining the event on exit gives exact visual bounds without
        # pausing or letting the colored probe drift away from the target.
        if self.active_reveal_line >= 0:
            line = self.active_reveal_line
            still_inside = (
                grasped
                and float(self.probe.pose.p[0, 2].item())
                <= float(CFG["reveal_max_probe_z"])
                and float(
                    torch.linalg.norm(probe_xy - self.trigger_xy[line]).item()
                )
                <= float(CFG["reveal_distance"])
            )
            if not still_inside:
                color_index = self.line_colors[line]
                start_frame = int(self.reveal_start_frames[line])
                self.define_keyframe_event(
                    f"{COLOR_NAMES[self.selected_color_ids[color_index]]}_probe_reveal",
                    start_frame,
                    max(start_frame, step),
                    required_information=(
                        "revealed_color",
                        "probe_trigger_position",
                        "rail_identity",
                    ),
                )
                self.probe_material.set_base_color((0.93, 0.93, 0.93, 1))
                self.active_reveal_line = -1

        for line in range(2):
            if (
                self.active_reveal_line >= 0
                or self.triggered[line]
                or not grasped
                or float(self.probe.pose.p[0, 2].item())
                > float(CFG["reveal_max_probe_z"])
                or abs(
                    float(probe_xy[0].item())
                    - float(CFG["line_x"][line])
                )
                > float(CFG["reveal_distance"])
            ):
                continue
            if (
                float(torch.linalg.norm(probe_xy - self.trigger_xy[line]).item())
                <= float(CFG["reveal_distance"])
            ):
                self.triggered[line] = True
                self.active_reveal_line = line
                self.reveal_start_frames[line] = step + 1
                color_index = self.line_colors[line]
                self.probe_material.set_base_color(
                    tuple(float(x) for x in self.episode_colors[color_index])
                )
                break
        probe_home_error = torch.linalg.norm(
            self.probe.pose.p[0, :2]
            - torch.as_tensor(CFG["probe_spawn"], device=self.device)
        )
        tcp_home_error = torch.linalg.norm(
            self.agent.tcp.pose.p[0]
            - torch.as_tensor(HOME_TCP_POSITION, device=self.device)
        )
        home_reset_now = (
            all(self.triggered)
            and not grasped
            and float(probe_home_error.item()) <= 0.040
            and float(self.probe.pose.p[0, 2].item()) < 0.070
            and float(tcp_home_error.item()) <= 0.006
        )
        self._memory_home_stable_steps = (
            self._memory_home_stable_steps + 1 if home_reset_now else 0
        )
        observation_physically_reset = self._memory_home_stable_steps >= 4
        # Phase progression is part of the environment contract.  It must be
        # inferred from physical events so saved actions and policy clients do
        # not need to advance the expert's private waypoint FSM.
        if observation_physically_reset and not self.observation_complete:
            self.observation_complete = True
            observation_end = step + 3
            self.execution_start = observation_end + 1 + self.memory_delay_steps
            self.set_phase_schedule(observation_end, self.execution_start)
        for color_index, cube in enumerate(self.cubes):
            xy = torch.as_tensor(self._target_for_color(color_index), device=self.device)
            distance = torch.linalg.norm(cube.pose.p[0, :2] - xy)
            contact_force = float(
                torch.linalg.norm(
                    self.scene.get_pairwise_contact_forces(
                        cube, self.table_scene.table
                    ),
                    dim=-1,
                )[0].item()
            )
            linear_speed = float(
                torch.linalg.norm(cube.linear_velocity[0]).item()
            )
            angular_speed = float(
                torch.linalg.norm(cube.angular_velocity[0]).item()
            )
            placement_now = (
                float(distance.item()) <= CFG["placement_tolerance"]
                and float(cube.pose.p[0, 2].item()) < 0.06
                and not bool(self.agent.is_grasping(cube)[0].item())
                and contact_force
                >= float(CFG["placement_contact_force_threshold"])
                and linear_speed
                <= float(CFG["placement_linear_speed_tolerance"])
                and angular_speed
                <= float(CFG["placement_angular_speed_tolerance"])
            )
            self._placement_stable_counts[color_index] = (
                self._placement_stable_counts[color_index] + 1
                if placement_now
                else 0
            )
            self.placement_contact_forces[color_index] = contact_force
            self.placed[color_index] = (
                placement_now
                and self._placement_stable_counts[color_index]
                >= int(CFG["placement_stable_steps"])
            )

    def _evaluate_task(self):
        probe_home_error = torch.linalg.norm(
            self.probe.pose.p[0, :2]
            - torch.as_tensor(CFG["probe_spawn"], device=self.device)
        )
        tcp_home_error = torch.linalg.norm(
            self.agent.tcp.pose.p[0]
            - torch.as_tensor(HOME_TCP_POSITION, device=self.device)
        )
        return {
            "success": torch.tensor([all(self.placed) and self.observation_complete], dtype=torch.bool, device=self.device),
            "color_reveals": torch.tensor([sum(self.triggered)], device=self.device),
            "placements": torch.tensor([sum(self.placed)], device=self.device),
            "placement_contact_force": torch.tensor(
                [self.placement_contact_forces],
                dtype=torch.float32,
                device=self.device,
            ),
            "placement_stable_steps": torch.tensor(
                [self._placement_stable_counts],
                dtype=torch.int64,
                device=self.device,
            ),
            "observation_complete": torch.tensor(
                [self.observation_complete], dtype=torch.bool, device=self.device
            ),
            "probe_home_error": probe_home_error.unsqueeze(0),
            "tcp_home_error": tcp_home_error.unsqueeze(0),
            "probe_grasped": torch.tensor(
                [self._probe_grasped()], dtype=torch.bool, device=self.device
            ),
            "probe_height": self.probe.pose.p[:, 2],
        }

    def _expert_action(self):
        step = int(self.elapsed_steps[0].item())
        if not self._scan_fsm.complete:
            return self._scan_fsm.action()
        if not self.observation_complete or step < self.execution_start:
            return self.zero_action()
        if self._execute_fsm is None:
            self._execute_fsm = PhysicalWaypointFSM(self, self._execute_waypoints())
        return self._execute_fsm.action()


__all__ = ["ASTMineSweeper", "ENV_ID", "INSTRUCTION_TASK", "INSTRUCTION_MEMORY"]
