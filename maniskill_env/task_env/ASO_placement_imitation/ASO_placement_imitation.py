"""Observed Appearance+Spatial placement imitation on a free 2-D tabletop."""

from __future__ import annotations

import sys
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
    COLOR_NAMES,
    COLOR_RGBA,
    HOME_TCP_POSITION,
    PhysicalWaypoint,
    PhysicalWaypointFSM,
    PhysicsMaterial,
    TASK_ENV_IDS,
    UnifiedPandaEnv,
    assert_points_reachable,
    render_material,
    sample_distinct_color_ids,
)

try:
    from .ASO_placement_imitation_cfg import CFG
except ImportError:  # direct gen_data.py execution
    from ASO_placement_imitation_cfg import CFG  # type: ignore


TASK_NAME = "ASO_placement_imitation"
ENV_ID = TASK_ENV_IDS[TASK_NAME]

INSTRUCTION_TASK = (
    "Two cubes of different colors start at separate positions on the table. One at a time, "
    "each cube moves to an unmarked target position, pauses, and returns. Once both "
    "demonstrations are complete, place each cube at the position where it paused, then release it on the table."
)
INSTRUCTION_MEMORY = (
    "Remember each moment when a cube has stopped at its demonstrated target position."
)


def _build_color_cube(scene, name: str, half_size: float):
    """One solid-color dynamic cube with a mutable actor-local material."""

    physics = PhysicsMaterial(
        static_friction=1.15,
        dynamic_friction=0.72,
        restitution=0.0,
        density=650,
    )
    builder = scene.create_actor_builder()
    builder.add_box_collision(
        half_size=[half_size] * 3,
        material=physics.sapien(),
        density=physics.density,
    )
    material = render_material((0.82, 0.78, 0.68, 1.0))
    builder.add_box_visual(
        half_size=[half_size] * 3,
        material=material,
    )
    builder.initial_pose = sapien.Pose(p=[-0.24, 0.0, half_size])
    actor = builder.build_dynamic(name)
    actor.set_mass(float(CFG["cube_mass"]))
    return actor, material


@register_env(ENV_ID, max_episode_steps=1300)
class ASOPlacementImitation(UnifiedPandaEnv):
    TASK_NAME = TASK_NAME
    INSTRUCTION_TASK = INSTRUCTION_TASK
    INSTRUCTION_MEMORY = INSTRUCTION_MEMORY

    def _load_task_scene(self, options: dict) -> None:
        del options
        half = float(CFG["cube_half_size"])
        built = [_build_color_cube(self.scene, f"texture_cube_{i}", half) for i in range(2)]
        self.cubes = [item[0] for item in built]
        self.color_materials = [item[1] for item in built]
        self.register_manipulable(*self.cubes)

    @staticmethod
    def _sample_pair(rng, low, high, min_separation: float) -> np.ndarray:
        for _ in range(500):
            points = rng.uniform(low, high, size=(2, 2)).astype(np.float32)
            if float(np.linalg.norm(points[0] - points[1])) >= min_separation:
                return points
        raise RuntimeError("failed to sample two separated 2-D positions")

    def _initialize_task_episode(
        self,
        env_idx: torch.Tensor,
        episode_metadata: Mapping[str, Any] | None,
    ) -> None:
        del env_idx
        rng = self._episode_rng
        if episode_metadata is None:
            initial_slots = self._sample_pair(
                rng,
                CFG["initial_xy_low"],
                CFG["initial_xy_high"],
                CFG["minimum_initial_separation"],
            )
            initial_order = rng.permutation(2).astype(np.int64)
            initial_xy = initial_slots[initial_order]
            target_xy = self._sample_pair(
                rng,
                CFG["target_xy_low"],
                CFG["target_xy_high"],
                CFG["minimum_target_separation"],
            )
            demonstration_order = rng.permutation(2).astype(np.int64)
            execution_order = rng.permutation(2).astype(np.int64)
            hold_steps = rng.randint(
                CFG["hold_steps"][0],
                CFG["hold_steps"][1] + 1,
                size=2,
            ).astype(np.int64)
            gap_steps = int(rng.randint(CFG["between_demo_steps"][0], CFG["between_demo_steps"][1] + 1))
            delay_steps = int(rng.randint(CFG["memory_delay_steps"][0], CFG["memory_delay_steps"][1] + 1))
            color_ids = sample_distinct_color_ids(rng)
        else:
            initial_xy = np.asarray(episode_metadata["initial_xy"], dtype=np.float32)
            target_xy = np.asarray(episode_metadata["target_xy"], dtype=np.float32)
            initial_order = np.asarray(episode_metadata["initial_texture_order"], dtype=np.int64)
            demonstration_order = np.asarray(episode_metadata["demonstration_order"], dtype=np.int64)
            execution_order = np.asarray(episode_metadata["execution_order"], dtype=np.int64)
            hold_steps = np.asarray(episode_metadata["hold_steps"], dtype=np.int64)
            gap_steps = int(episode_metadata["between_demo_steps"])
            delay_steps = int(episode_metadata["memory_delay_steps"])
            color_ids = np.asarray(
                episode_metadata.get("selected_color_ids", episode_metadata["pattern_ids"]),
                dtype=np.int64,
            )

        z = float(CFG["cube_half_size"])
        assert_points_reachable(
            np.concatenate(
                [
                    np.column_stack((initial_xy, np.full(2, z))),
                    np.column_stack((target_xy, np.full(2, z))),
                ],
                axis=0,
            )
        )
        self.initial_xy = initial_xy
        self.target_xy = target_xy
        self.initial_texture_order = initial_order
        self.demonstration_order = demonstration_order.tolist()
        self.execution_order = execution_order.tolist()
        self.hold_steps = hold_steps.tolist()
        self.between_demo_steps = gap_steps
        self.memory_delay_steps = delay_steps
        self.pattern_ids = color_ids.tolist()
        identity_q = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=self.device)
        for cube_id, cube in enumerate(self.cubes):
            self.color_materials[cube_id].set_base_color(
                tuple(float(x) for x in COLOR_RGBA[int(color_ids[cube_id])])
            )
            cube.set_pose(
                Pose.create_from_pq(
                    p=torch.tensor(
                        [[initial_xy[cube_id, 0], initial_xy[cube_id, 1], z]],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    q=identity_q,
                )
            )
            cube.set_linear_velocity(torch.zeros((1, 3), device=self.device))
            cube.set_angular_velocity(torch.zeros((1, 3), device=self.device))

        self._demo_cursor = 0
        self._demo_state = "outbound"
        self._demo_dwell = 0
        self._teacher_force = torch.zeros(3, dtype=torch.float32, device=self.device)
        self._observation_complete = False
        self._placement_stable_counts = torch.zeros(
            (1, 2), dtype=torch.int64, device=self.device
        )
        self.execution_start = 100_000
        self.set_phase_schedule(99_998, 99_999)
        self.set_task_episode_metadata(
            pattern_ids=color_ids,
            selected_pattern_ids=color_ids,
            pattern_names=[COLOR_NAMES[int(x)] for x in color_ids],
            pattern_candidate_count=len(COLOR_NAMES),
            selected_color_ids=color_ids,
            selected_color_names=[COLOR_NAMES[int(x)] for x in color_ids],
            selected_color_rgba=COLOR_RGBA[color_ids],
            color_candidate_count=len(COLOR_NAMES),
            initial_xy=initial_xy,
            initial_texture_order=initial_order,
            target_xy=target_xy,
            demonstration_order=demonstration_order,
            execution_order=execution_order,
            hold_steps=hold_steps,
            keyframe_settle_steps=CFG["keyframe_settle_steps"],
            keyframe_semantics="begins_after_continuous_stable_target_dwell",
            between_demo_steps=gap_steps,
            memory_delay_steps=delay_steps,
            demonstration_dynamics="external_planar_PD_force_on_dynamic_cube",
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
        self._expert_fsm = None

    def _active_cube_id(self) -> int:
        return int(self.demonstration_order[self._demo_cursor])

    def _teacher_goal(self) -> np.ndarray:
        cube_id = self._active_cube_id()
        return self.target_xy[cube_id] if self._demo_state in ("outbound", "settling", "hold") else self.initial_xy[cube_id]

    def _before_task_control_step(self) -> None:
        if self._observation_complete:
            self._teacher_force.zero_()
            return
        cube = self.cubes[self._active_cube_id()]
        goal = torch.as_tensor(self._teacher_goal(), dtype=torch.float32, device=self.device)
        position = cube.pose.p[0, :2]
        velocity = cube.linear_velocity[0, :2]
        error = goal - position
        force_xy = float(CFG["teacher_kp"]) * error - float(CFG["teacher_kd"]) * velocity
        # Coulomb friction can otherwise hold the cube at a non-zero PD
        # equilibrium (observed at ~3.5 cm with seed 0).  Apply a bounded
        # breakaway term only while outside the true stable-position window;
        # inside it, damping plus table friction settles the dynamic body.
        error_norm = torch.linalg.norm(error)
        if float(error_norm.item()) > float(CFG["teacher_position_tolerance"]):
            force_xy = force_xy + (
                error / error_norm * float(CFG["teacher_breakaway_force"])
            )
        norm = torch.linalg.norm(force_xy)
        max_force = float(CFG["teacher_max_force"])
        if float(norm.item()) > max_force:
            force_xy = force_xy / norm * max_force
        self._teacher_force = torch.cat((force_xy, torch.zeros(1, device=self.device)))

    def _before_simulation_step(self) -> None:
        if not self._observation_complete:
            # Apply on every 200-Hz physics substep. This is a real force on a
            # free dynamic body; no pose/velocity overwrite or rail constraint.
            self.cubes[self._active_cube_id()].apply_force(self._teacher_force)

    def _teacher_at_goal(self) -> bool:
        cube = self.cubes[self._active_cube_id()]
        goal = torch.as_tensor(self._teacher_goal(), dtype=torch.float32, device=self.device)
        error = torch.linalg.norm(cube.pose.p[0, :2] - goal)
        speed = torch.linalg.norm(cube.linear_velocity[0, :2])
        return (
            float(error.item()) <= float(CFG["teacher_position_tolerance"])
            and float(speed.item()) <= float(CFG["teacher_speed_tolerance"])
        )

    def _after_task_control_step(self) -> None:
        if self._observation_complete:
            _, _, placement_now = self._placement_state()
            self._placement_stable_counts = torch.where(
                placement_now,
                self._placement_stable_counts + 1,
                torch.zeros_like(self._placement_stable_counts),
            )
            return
        next_frame = int(self.elapsed_steps[0].item()) + 1
        cube_id = self._active_cube_id()
        if self._demo_state == "outbound":
            if self._teacher_at_goal():
                self._demo_state = "settling"
                self._demo_dwell = 0
        elif self._demo_state == "settling":
            if not self._teacher_at_goal():
                self._demo_state = "outbound"
                self._demo_dwell = 0
            else:
                self._demo_dwell += 1
            if self._demo_dwell >= int(CFG["keyframe_settle_steps"]):
                self._demo_state = "hold"
                self._demo_dwell = 0
                start = next_frame
                self.define_keyframe_event(
                    f"color_position_observation_{self._demo_cursor + 1}",
                    start,
                    start + int(self.hold_steps[cube_id]) - 1,
                    required_information=(
                        "cube_color_identity",
                        "free_2d_demonstrated_position",
                    ),
                )
        elif self._demo_state == "hold":
            self._demo_dwell += 1
            if self._demo_dwell >= int(self.hold_steps[cube_id]):
                self._demo_state = "returning"
                self._demo_dwell = 0
        elif self._demo_state == "returning":
            if self._teacher_at_goal():
                self._demo_state = "gap"
                self._demo_dwell = 0
        else:  # gap
            self._demo_dwell += 1
            if self._demo_dwell >= self.between_demo_steps:
                if self._demo_cursor == 0:
                    self._demo_cursor = 1
                    self._demo_state = "outbound"
                    self._demo_dwell = 0
                else:
                    self._observation_complete = True
                    observation_end = next_frame
                    self.execution_start = (
                        observation_end + 1 + self.memory_delay_steps
                    )
                    self.set_phase_schedule(observation_end, self.execution_start)

    def _expert_waypoints(self) -> list[PhysicalWaypoint]:
        z = float(CFG["cube_half_size"])
        waypoints: list[PhysicalWaypoint] = []
        for cube_id in self.execution_order:
            def live(env, cube_id=cube_id, height=0.16):
                p = env.cubes[cube_id].pose.p[0].detach().cpu().numpy().copy()
                p[2] = height
                return p

            def target(env, cube_id=cube_id, height=z):
                del env
                xy = self.target_xy[cube_id]
                return [float(xy[0]), float(xy[1]), float(height)]

            grasped = lambda env, c=cube_id: bool(env.agent.is_grasping(env.cubes[c])[0].item())
            waypoints.extend(
                [
                    PhysicalWaypoint(lambda env, c=cube_id: live(env, c, 0.16), GRIPPER_OPEN,
                                     label=f"cube_{cube_id}_above"),
                    PhysicalWaypoint(lambda env, c=cube_id: live(env, c, z + 0.008), GRIPPER_OPEN,
                                     position_tolerance=0.014, rotation_tolerance=10.0,
                                     dwell_steps=3, label=f"cube_{cube_id}_contact"),
                    PhysicalWaypoint(lambda env, c=cube_id: live(env, c, z + 0.008), GRIPPER_CLOSED,
                                     position_tolerance=0.014, rotation_tolerance=10.0,
                                     dwell_steps=10, require=grasped,
                                     label=f"cube_{cube_id}_grasp"),
                    PhysicalWaypoint(lambda env, c=cube_id: live(env, c, 0.17), GRIPPER_CLOSED,
                                     rotation_tolerance=10.0, require=grasped,
                                     label=f"cube_{cube_id}_lift"),
                    PhysicalWaypoint(lambda env, c=cube_id: target(env, c, 0.17), GRIPPER_CLOSED,
                                     require=grasped, label=f"cube_{cube_id}_above_target"),
                    PhysicalWaypoint(lambda env, c=cube_id: target(env, c, z + 0.008), GRIPPER_CLOSED,
                                     position_tolerance=0.015, rotation_tolerance=10.0,
                                     dwell_steps=3, label=f"cube_{cube_id}_lower"),
                    PhysicalWaypoint(lambda env, c=cube_id: target(env, c, z + 0.008), GRIPPER_OPEN,
                                     position_tolerance=0.015, rotation_tolerance=10.0,
                                     dwell_steps=8, label=f"cube_{cube_id}_release"),
                    PhysicalWaypoint(lambda env, c=cube_id: target(env, c, 0.16), GRIPPER_OPEN,
                                     label=f"cube_{cube_id}_clear"),
                    PhysicalWaypoint(HOME_TCP_POSITION, GRIPPER_OPEN,
                                     label=f"home_after_cube_{cube_id}"),
                ]
            )
        return waypoints

    def _placement_state(self):
        errors = torch.stack(
            [
                torch.linalg.norm(
                    cube.pose.p[:, :2]
                    - torch.as_tensor(self.target_xy[cube_id], device=self.device).unsqueeze(0),
                    dim=1,
                )
                for cube_id, cube in enumerate(self.cubes)
            ],
            dim=1,
        )
        contact_forces = torch.stack(
            [
                torch.linalg.norm(
                    self.scene.get_pairwise_contact_forces(
                        cube, self.table_scene.table
                    ),
                    dim=-1,
                )
                for cube in self.cubes
            ],
            dim=1,
        )
        released = torch.stack(
            [~self.agent.is_grasping(cube) for cube in self.cubes], dim=1
        )
        low = torch.stack(
            [cube.pose.p[:, 2] < 0.075 for cube in self.cubes], dim=1
        )
        linear_speed = torch.stack(
            [torch.linalg.norm(cube.linear_velocity, dim=-1) for cube in self.cubes],
            dim=1,
        )
        angular_speed = torch.stack(
            [torch.linalg.norm(cube.angular_velocity, dim=-1) for cube in self.cubes],
            dim=1,
        )
        placement_now = (
            (errors <= float(CFG["place_tolerance"]))
            & low
            & released
            & (
                contact_forces
                >= float(CFG["placement_contact_force_threshold"])
            )
            & (
                linear_speed
                <= float(CFG["placement_linear_speed_tolerance"])
            )
            & (
                angular_speed
                <= float(CFG["placement_angular_speed_tolerance"])
            )
        )
        return errors, contact_forces, placement_now

    def _evaluate_task(self) -> Mapping[str, Any]:
        errors, contact_forces, placement_now = self._placement_state()
        stable = self._placement_stable_counts >= int(
            CFG["placement_stable_steps"]
        )
        success = (
            self._observation_complete
            & (self.elapsed_steps >= int(self.execution_start))
            & torch.all(placement_now & stable, dim=1)
        )
        return {
            "success": success,
            "placement_error_xy": errors,
            "placement_contact_force": contact_forces,
            "placement_stable_steps": self._placement_stable_counts,
        }

    def _expert_action(self) -> torch.Tensor:
        if not self._observation_complete or int(self.elapsed_steps[0].item()) < self.execution_start:
            return self.zero_action()
        if self._expert_fsm is None:
            self._expert_fsm = PhysicalWaypointFSM(self, self._expert_waypoints())
        return self._expert_fsm.action()


__all__ = [
    "ASOPlacementImitation",
    "ENV_ID",
    "TASK_NAME",
    "INSTRUCTION_TASK",
    "INSTRUCTION_MEMORY",
]
