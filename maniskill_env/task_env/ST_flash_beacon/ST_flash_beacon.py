"""Triggered+Spatial task with two retractable physical beacon towers."""

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
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_TCP_POSITION,
    JointAssembly,
    PhysicalWaypoint,
    PhysicalWaypointFSM,
    PhysicsMaterial,
    TASK_ENV_IDS,
    UnifiedPandaEnv,
    build_dynamic_box,
    render_material,
)
from ST_flash_beacon_cfg import CFG  # noqa: E402

TASK_NAME = "ST_flash_beacon"
ENV_ID = TASK_ENV_IDS[TASK_NAME]
INSTRUCTION_TASK = (
    "A white cube and two beacons are on the table. Lift the cube. When the first beacon "
    "appears, start moving toward it at a moderate speed so that it disappears before you arrive. "
    "Keep moving without returning the arm to its starting pose, and place the cube at that spot. "
    "The second beacon appears as soon as the cube touches the table. Keep holding the cube and "
    "continue directly toward the second beacon, then place and release the cube at that spot."
)
INSTRUCTION_MEMORY = (
    "Remember the first beacon while it is visible, and later remember the second beacon while it is visible."
)


def _build_beacon_lift(scene, name: str, color) -> JointAssembly:
    builder = scene.create_articulation_builder()
    builder.initial_pose = sapien.Pose(
        [0.0, 0.0, float(CFG["beacon_hidden_root_z"])]
    )
    root = builder.create_link_builder(parent=None)
    root.set_name(f"{name}_root")
    moving = builder.create_link_builder(parent=root)
    moving.set_name(f"{name}_tower")
    vertical_q = [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]
    dark = render_material((0.055, 0.065, 0.075, 1.0))
    metal = render_material((0.42, 0.46, 0.50, 1.0))
    lens = render_material(color)
    emission = np.asarray(color[:3], dtype=np.float32) * 4.0
    lens.set_emission(
        (float(emission[0]), float(emission[1]), float(emission[2]), 0.0)
    )

    # A recognizable beacon rather than a highlighted target disk: a broad
    # pedestal, slender mast, luminous cylindrical lens, dark lens guards and
    # a small glowing crown.  Every component belongs to the same actuated
    # rigid link and rises vertically from below the table.
    moving.add_cylinder_visual(
        radius=0.024,
        half_length=0.006,
        pose=sapien.Pose(p=[0.0, 0.0, -0.018], q=vertical_q),
        material=dark,
    )
    moving.add_cylinder_visual(
        radius=0.007,
        half_length=0.027,
        pose=sapien.Pose(p=[0.0, 0.0, 0.010], q=vertical_q),
        material=metal,
    )
    moving.add_cylinder_visual(
        radius=0.020,
        half_length=0.004,
        pose=sapien.Pose(p=[0.0, 0.0, 0.039], q=vertical_q),
        material=dark,
    )
    moving.add_cylinder_visual(
        radius=0.017,
        half_length=0.012,
        pose=sapien.Pose(p=[0.0, 0.0, 0.055], q=vertical_q),
        material=lens,
    )
    moving.add_cylinder_visual(
        radius=0.020,
        half_length=0.004,
        pose=sapien.Pose(p=[0.0, 0.0, 0.071], q=vertical_q),
        material=dark,
    )
    moving.add_sphere_visual(
        radius=0.009,
        pose=sapien.Pose(p=[0.0, 0.0, 0.078]),
        material=lens,
    )
    # Give the signal disk an explicit, physically plausible lightweight
    # inertia.  The fast response below therefore comes from an ordinary
    # prismatic actuator rather than teleporting its qpos during rollout.
    moving_mass = float(CFG["beacon_moving_mass"])
    moving.set_mass_and_inertia(
        moving_mass,
        sapien.Pose(),
        np.asarray([3.2e-5, 3.2e-5, 1.8e-5], dtype=np.float32),
    )
    moving.set_joint_name(f"{name}_joint")
    axis_q = [math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0]
    moving.set_joint_properties(
        type="prismatic",
        limits=[[0.0, float(CFG["beacon_lift_travel"])]],
        pose_in_parent=sapien.Pose(q=axis_q),
        pose_in_child=sapien.Pose(q=axis_q),
        friction=0.01,
        damping=0.4,
    )
    articulation = builder.build(name, fix_root_link=True)
    joint = articulation.get_active_joints()[0]
    joint.set_drive_properties(
        stiffness=float(CFG["beacon_drive_stiffness"]),
        damping=float(CFG["beacon_drive_damping"]),
        force_limit=float(CFG["beacon_drive_force_limit"]),
    )
    return JointAssembly(
        articulation,
        joint,
        articulation.links_map[f"{name}_tower"],
        0.0,
        float(CFG["beacon_lift_travel"]),
    )


@register_env(ENV_ID, max_episode_steps=1100)
class STFlashBeacon(UnifiedPandaEnv):
    TASK_NAME = TASK_NAME
    INSTRUCTION_TASK = INSTRUCTION_TASK
    INSTRUCTION_MEMORY = INSTRUCTION_MEMORY

    def _load_task_scene(self, options: dict) -> None:
        del options
        h = float(CFG["cube_half_size"])
        self.cube = build_dynamic_box(
            self.scene,
            "flash_cube",
            [h, h, h],
            [-0.12, 0.0, h],
            color=(0.92, 0.93, 0.94, 1),
            physics=PhysicsMaterial(static_friction=1.1, dynamic_friction=0.8, density=520),
            mass=0.075,
        )
        self.register_manipulable(self.cube)
        self.beacons = [
            _build_beacon_lift(self.scene, "first_beacon", (1.0, 0.58, 0.04, 1)),
            _build_beacon_lift(self.scene, "second_beacon", (0.04, 0.72, 1.0, 1)),
        ]

    def _sample_targets(self, spawn):
        for _ in range(5000):
            points = self._episode_rng.uniform(
                [CFG["target_x"][0], CFG["target_y"][0]],
                [CFG["target_x"][1], CFG["target_y"][1]],
                size=(2, 2),
            )
            unsafe_corner = (
                (points[:, 0] > CFG["target_far_x_at_corner"])
                & (
                    np.abs(points[:, 1])
                    > CFG["target_corner_y_threshold"]
                )
            )
            if np.any(unsafe_corner):
                continue
            if (
                np.linalg.norm(points[0] - points[1])
                < CFG["minimum_target_separation"]
            ):
                continue
            # Explicitly randomize temporal identity before validating the
            # first cue.  This keeps first/second independent of image
            # left/right while ensuring the *actual* first target—not the
            # pre-shuffle row—is sufficiently far from the cube spawn.
            if int(self._episode_rng.randint(2)):
                points = points[::-1]
            if (
                np.linalg.norm(points[0] - spawn)
                >= CFG["minimum_spawn_target_separation"]
            ):
                return points.astype(np.float32)
        raise RuntimeError("failed to sample separated beacon targets")

    def _initialize_task_episode(self, env_idx, episode_metadata: Mapping[str, Any] | None) -> None:
        del env_idx
        if episode_metadata is None:
            spawn = self._episode_rng.uniform(
                [CFG["cube_spawn_x"][0], CFG["cube_spawn_y"][0]],
                [CFG["cube_spawn_x"][1], CFG["cube_spawn_y"][1]],
            ).astype(np.float32)
            targets = self._sample_targets(spawn)
            hold = [int(self._episode_rng.randint(*[CFG["beacon_hold_steps"][0], CFG["beacon_hold_steps"][1] + 1])) for _ in range(2)]
            delay = [int(self._episode_rng.randint(CFG["memory_delay_steps"][0], CFG["memory_delay_steps"][1] + 1)) for _ in range(2)]
        else:
            spawn = np.asarray(episode_metadata["cube_spawn_xy"], dtype=np.float32)
            targets = np.asarray(episode_metadata["target_xy"], dtype=np.float32)
            hold = [int(x) for x in episode_metadata["beacon_hold_steps"]]
            delay = [int(x) for x in episode_metadata["memory_delay_steps"]]
        h = float(CFG["cube_half_size"])
        self.cube.set_pose(Pose.create_from_pq(
            p=torch.tensor([[spawn[0], spawn[1], h]], dtype=torch.float32, device=self.device),
            q=torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=self.device),
        ))
        self.cube.set_linear_velocity(torch.zeros((1, 3), device=self.device))
        self.cube.set_angular_velocity(torch.zeros((1, 3), device=self.device))
        self.target_xy = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        self.beacon_hold_steps = hold
        self.memory_delay_steps = delay
        for i, beacon in enumerate(self.beacons):
            pose = torch.tensor(
                [[targets[i, 0], targets[i, 1], float(CFG["beacon_hidden_root_z"])]],
                dtype=torch.float32,
                device=self.device,
            )
            beacon.articulation.set_pose(Pose.create_from_pq(
                p=pose,
                q=torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=self.device),
            ))
            beacon.set_qpos(0.0)
            beacon.articulation.set_qvel(torch.zeros((1, 1), device=self.device))
        self.reveal_index = -1
        self.reveal_trigger_step = -1
        self.reveal_complete = [False, False]
        self.placement_complete = [False, False]
        self.evaluation_placement_complete = [False, False]
        self.execution_ready_steps = [-1, -1]
        self.memory_observation_end_steps = [-1, -1]
        self.memory_transition_scheduled = [False, False]
        self.first_target_contact_force = 0.0
        self.first_target_bottom_clearance = float("inf")
        self.second_target_contact_force = 0.0
        self.second_target_bottom_clearance = float("inf")
        self.second_target_linear_speed = float("inf")
        self.second_target_angular_speed = float("inf")
        self.second_target_stable_steps = 0
        self.set_phase_schedule(9_998, 9_999)
        self.set_task_episode_metadata(
            cube_spawn_xy=spawn,
            target_xy=targets,
            beacon_hold_steps=hold,
            beacon_raise_steps=CFG["beacon_raise_steps"],
            beacon_lower_steps=CFG["beacon_lower_steps"],
            beacon_lift_travel=CFG["beacon_lift_travel"],
            beacon_clear_qpos=CFG["beacon_clear_qpos"],
            beacon_visual_form="retractable_pedestal_mast_luminous_lens_tower",
            memory_delay_steps=delay,
            grasp_sequence="single_continuous_grasp_until_second_target_release",
            second_beacon_trigger="closed_gripper_contact_at_first_target",
            second_beacon_contact_test="oriented_cube_bottom_clearance",
            reveal_motion="continuous_same_speed_target_approach_during_both_beacons",
            beacon_visible_motion="continuous_motion_toward_active_beacon",
            reveal_approach_starts="immediately_on_beacon_trigger",
            reveal_approach_speed="same_as_all_other_task_motion",
            task_max_position_command=CFG["task_max_position_command"],
            temporal_target_order="randomized_independently_of_left_right_position",
            cube_spawn_sampling="uniform_expanded_safe_workspace",
            cube_spawn_bounds=[CFG["cube_spawn_x"], CFG["cube_spawn_y"]],
            target_sampling_bounds=[CFG["target_x"], CFG["target_y"]],
            target_sampling_exclusion={
                "positive_x_above": CFG["target_far_x_at_corner"],
                "when_abs_y_above": CFG["target_corner_y_threshold"],
            },
            minimum_target_separation=CFG["minimum_target_separation"],
            memory_reset="none_continuous_target_approach",
            blank_delay_motion="continuous_closed_gripper_target_approach_without_reset",
        )
        self._expert_stage = "pick_and_lift"
        self._fsm = self._make_fsm(self._pick_lift_waypoints())

    def _make_fsm(self, waypoints):
        return PhysicalWaypointFSM(
            self,
            waypoints,
            max_position_command=float(CFG["task_max_position_command"]),
        )

    def _cube_point(self, dz=0.0):
        def resolve(env):
            p = env.cube.pose.p[0].detach().cpu().numpy().copy()
            p[2] += dz
            return p
        return resolve

    def _target_point(self, index, z):
        def resolve(env):
            xy = env.target_xy[index].detach().cpu().numpy()
            return [float(xy[0]), float(xy[1]), float(z)]
        return resolve

    def _cube_at_height(self, z):
        def resolve(env):
            p = env.cube.pose.p[0].detach().cpu().numpy()
            return [float(p[0]), float(p[1]), float(z)]
        return resolve

    def _grasped(self, _env=None):
        return bool(self.agent.is_grasping(self.cube)[0].item())

    def _pick_lift_waypoints(self):
        return [
            PhysicalWaypoint(self._cube_point(0.12), GRIPPER_OPEN, label="above_cube"),
            PhysicalWaypoint(self._cube_point(0.002), GRIPPER_OPEN, dwell_steps=3, label="at_cube"),
            PhysicalWaypoint(self._cube_point(0.002), GRIPPER_CLOSED, dwell_steps=8, require=lambda env: env._grasped(), label="grasp_cube"),
            PhysicalWaypoint(self._cube_at_height(0.155), GRIPPER_CLOSED,
                             dwell_steps=1, require=lambda env: env._grasped(),
                             label="lift_cube"),
        ]

    def _reveal_approach_waypoints(self, target_index, prefix):
        return [
            PhysicalWaypoint(
                self._target_point(target_index, float(CFG["carry_height"])),
                GRIPPER_CLOSED,
                dwell_steps=1,
                require=lambda env: env._grasped(),
                label=f"{prefix}_moderate_speed_target_approach",
            ),
        ]

    def _first_target_waypoints(self):
        return [
            PhysicalWaypoint(
                self._target_point(0, float(CFG["carry_height"])),
                GRIPPER_CLOSED,
                require=lambda env: env._grasped(),
                label="first_target_above",
            ),
            PhysicalWaypoint(
                self._target_point(0, float(CFG["first_contact_tcp_height"])),
                GRIPPER_CLOSED,
                position_tolerance=float(
                    CFG["first_contact_position_tolerance"]
                ),
                dwell_steps=1,
                require=lambda env: env._grasped(),
                label="first_target_closed_contact",
            ),
        ]

    def _final_place_waypoints(self):
        return [
            PhysicalWaypoint(
                self._target_point(1, float(CFG["carry_height"])),
                GRIPPER_CLOSED,
                require=lambda env: env._grasped(),
                label="second_target_above",
            ),
            PhysicalWaypoint(
                self._target_point(1, float(CFG["placement_tcp_height"])),
                GRIPPER_CLOSED,
                position_tolerance=float(CFG["placement_position_tolerance"]),
                dwell_steps=4,
                require=lambda env: env._grasped(),
                label="second_target_lower",
            ),
            PhysicalWaypoint(
                self._target_point(1, float(CFG["placement_tcp_height"])),
                GRIPPER_OPEN,
                position_tolerance=float(CFG["placement_position_tolerance"]),
                dwell_steps=8,
                label="second_target_final_release",
            ),
            PhysicalWaypoint(
                self._target_point(1, 0.16),
                GRIPPER_OPEN,
                label="final_retract",
            ),
            PhysicalWaypoint(HOME_TCP_POSITION, GRIPPER_OPEN, label="home"),
        ]

    def _trigger_beacon(self, index: int, step: int) -> None:
        self.reveal_index = int(index)
        self.reveal_trigger_step = int(step)
        # The driven tower crosses ``beacon_clear_qpos`` on the final
        # transition of the configured rise, so that state frame is already
        # fully visible.  Start the keyframe there instead of one frame later;
        # the arm remains free to move immediately when the beacon triggers.
        stable = int(step) + int(CFG["beacon_raise_steps"]) - 1
        # The drive remains raised through the final hold transition, so the
        # resulting observation is still part of the inclusive keyframe.
        end = stable + int(self.beacon_hold_steps[index])
        self.define_keyframe_event(
            f"beacon_{index + 1}_visible",
            stable,
            end,
            required_information=(
                f"beacon_{index + 1}_target_position",
                "occurrence_order",
            ),
        )

    def _before_task_control_step(self) -> None:
        step = int(self.elapsed_steps[0].item())
        for index, beacon in enumerate(self.beacons):
            visible = False
            if self.reveal_index == index:
                elapsed = step - self.reveal_trigger_step
                visible = elapsed < int(CFG["beacon_raise_steps"]) + self.beacon_hold_steps[index]
            beacon.drive_to(float(CFG["beacon_lift_travel"]) if visible else 0.0)

    def _after_task_control_step(self) -> None:
        step = int(self.elapsed_steps[0].item())
        grasped = self._grasped()
        cube_z = float(self.cube.pose.p[0, 2].item())
        if (
            self.reveal_index < 0
            and not self.reveal_complete[0]
            and grasped
            and cube_z > 0.075
        ):
            self._trigger_beacon(0, step)
        if self.reveal_index >= 0:
            active_index = self.reveal_index
            total = (
                int(CFG["beacon_raise_steps"])
                + self.beacon_hold_steps[active_index]
                + int(CFG["beacon_lower_steps"])
            )
            if step - self.reveal_trigger_step >= total:
                self.reveal_complete[active_index] = True
                # The cue is now physically absent.  Keep the same target-directed
                # carry instead of inserting a home-reset detour.  The sampled
                # blank interval remains explicit in the phase annotation.
                observation_end = step
                ready = observation_end + 1 + self.memory_delay_steps[active_index]
                self.execution_ready_steps[active_index] = ready
                self.memory_observation_end_steps[active_index] = observation_end
                self.memory_transition_scheduled[active_index] = True
                self.set_task_episode_metadata(
                    memory_observation_end_steps=self.memory_observation_end_steps,
                    execution_ready_steps=self.execution_ready_steps,
                )
                if active_index == 1:
                    self.set_phase_schedule(observation_end, ready)
                self.reveal_index = -1

        first_distance = torch.linalg.norm(
            self.cube.pose.p[0, :2] - self.target_xy[0]
        )
        table_contact = self.scene.get_pairwise_contact_forces(
            self.cube, self.table_scene.table
        )
        current_table_contact_force = float(
            torch.linalg.norm(table_contact, dim=-1)[0].item()
        )
        cube_q = self.cube.pose.q[0]
        qw, qx, qy, qz = cube_q.unbind()
        rotation_z_row = torch.stack(
            (
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            )
        )
        vertical_half_extent = float(CFG["cube_half_size"]) * torch.abs(
            rotation_z_row
        ).sum()
        current_bottom_clearance = float(
            (self.cube.pose.p[0, 2] - vertical_half_extent).item()
        )
        # Formal evaluation keeps the deliberately loose sequential target-hit
        # contract, but a fly-over is not a placement: the cube bottom must be
        # close to the table, and the first placement must keep the grasp.
        if (
            grasped
            and float(first_distance.item()) <= float(CFG["placement_tolerance"])
            and current_bottom_clearance <= float(CFG["final_bottom_clearance"])
        ):
            self.evaluation_placement_complete[0] = True
        first_contact = (
            self.reveal_complete[0]
            and step >= self.execution_ready_steps[0]
            and grasped
            and float(first_distance.item()) < float(CFG["placement_tolerance"])
            and current_bottom_clearance
            <= float(CFG["first_target_contact_clearance"])
        )
        if first_contact and not self.placement_complete[0]:
            self.first_target_contact_force = current_table_contact_force
            self.first_target_bottom_clearance = current_bottom_clearance
            self.placement_complete[0] = True
            # Trigger on the very first control callback with real cube/table
            # contact at target one.  The beacon drive starts on the next
            # physics substep, while the gripper remains continuously closed.
            self._trigger_beacon(1, step)

        second_distance = torch.linalg.norm(
            self.cube.pose.p[0, :2] - self.target_xy[1]
        )
        if (
            self.evaluation_placement_complete[0]
            and float(second_distance.item()) <= float(CFG["placement_tolerance"])
            and current_bottom_clearance <= float(CFG["final_bottom_clearance"])
        ):
            self.evaluation_placement_complete[1] = True
        self.second_target_contact_force = current_table_contact_force
        self.second_target_bottom_clearance = current_bottom_clearance
        self.second_target_linear_speed = float(
            torch.linalg.norm(self.cube.linear_velocity[0]).item()
        )
        self.second_target_angular_speed = float(
            torch.linalg.norm(self.cube.angular_velocity[0]).item()
        )
        second_physically_placed = (
            self.placement_complete[0]
            and self.reveal_complete[1]
            and step >= self.execution_ready_steps[1]
            and not grasped
            and float(second_distance.item()) < float(CFG["placement_tolerance"])
            and self.second_target_contact_force
            >= float(CFG["final_table_contact_force_threshold"])
            and self.second_target_bottom_clearance
            <= float(CFG["final_bottom_clearance"])
            and self.second_target_linear_speed
            <= float(CFG["final_linear_speed_threshold"])
            and self.second_target_angular_speed
            <= float(CFG["final_angular_speed_threshold"])
        )
        if second_physically_placed:
            self.second_target_stable_steps += 1
        else:
            self.second_target_stable_steps = 0
        # Unlike the first target (which must be remembered while carrying on),
        # final placement is a current physical state, not a permanent latch.
        self.placement_complete[1] = (
            self.second_target_stable_steps >= int(CFG["final_stable_steps"])
        )

    def _evaluate_task(self):
        gripper_open = (
            float(self.agent.robot.get_qpos()[0, -2:].sum().item())
            >= float(CFG["final_gripper_open_sum"])
        )
        tcp_clear = (
            float(self.agent.tcp.pose.p[0, 2].item())
            >= float(CFG["final_clear_tcp_height"])
        )
        success = (
            all(self.reveal_complete)
            and all(self.placement_complete)
            and not self._grasped()
            and gripper_open
            and tcp_clear
        )
        evaluation_success = all(self.evaluation_placement_complete)
        return {
            "success": torch.tensor([success], dtype=torch.bool, device=self.device),
            "evaluation_success": torch.tensor(
                [evaluation_success], dtype=torch.bool, device=self.device
            ),
            "beacons_observed": torch.tensor([sum(self.reveal_complete)], device=self.device),
            "placements_complete": torch.tensor([sum(self.placement_complete)], device=self.device),
            "evaluation_placements_complete": torch.tensor(
                [sum(self.evaluation_placement_complete)], device=self.device
            ),
            "first_target_position_reached": torch.tensor(
                [self.evaluation_placement_complete[0]],
                dtype=torch.bool,
                device=self.device,
            ),
            "second_target_position_reached": torch.tensor(
                [self.evaluation_placement_complete[1]],
                dtype=torch.bool,
                device=self.device,
            ),
            "first_target_closed_gripper_contact": torch.tensor(
                [self.placement_complete[0]], dtype=torch.bool, device=self.device
            ),
            "first_target_contact_force": torch.tensor(
                [self.first_target_contact_force], dtype=torch.float32, device=self.device
            ),
            "first_target_bottom_clearance": torch.tensor(
                [self.first_target_bottom_clearance],
                dtype=torch.float32,
                device=self.device,
            ),
            "second_target_contact_force": torch.tensor(
                [self.second_target_contact_force], dtype=torch.float32, device=self.device
            ),
            "second_target_bottom_clearance": torch.tensor(
                [self.second_target_bottom_clearance], dtype=torch.float32, device=self.device
            ),
            "second_target_linear_speed": torch.tensor(
                [self.second_target_linear_speed], dtype=torch.float32, device=self.device
            ),
            "second_target_angular_speed": torch.tensor(
                [self.second_target_angular_speed], dtype=torch.float32, device=self.device
            ),
            "second_target_stable_steps": torch.tensor(
                [self.second_target_stable_steps], dtype=torch.int64, device=self.device
            ),
        }

    def _expert_action(self):
        step = int(self.elapsed_steps[0].item())
        if self._expert_stage == "pick_and_lift" and self.reveal_index == 0:
            self._expert_stage = "approach_first_beacon"
            self._fsm = self._make_fsm(
                self._reveal_approach_waypoints(0, "first_reveal"),
            )

        if (
            self._expert_stage == "approach_first_beacon"
            and self.reveal_complete[0]
            and step >= self.execution_ready_steps[0]
        ):
            self._expert_stage = "closed_contact_at_first_target"
            self._fsm = self._make_fsm(self._first_target_waypoints())

        if (
            self._expert_stage == "closed_contact_at_first_target"
            and self.placement_complete[0]
        ):
            self._expert_stage = "approach_second_beacon"
            self._fsm = self._make_fsm(
                self._reveal_approach_waypoints(1, "second_reveal"),
            )

        if (
            self._expert_stage == "approach_second_beacon"
            and self.reveal_complete[1]
            and step >= self.execution_ready_steps[1]
        ):
            self._expert_stage = "final_place_and_release"
            self._fsm = self._make_fsm(self._final_place_waypoints())

        if self._expert_stage == "final_place_and_release" and self._fsm.complete:
            return self.zero_action(GRIPPER_OPEN)

        return self._fsm.action()


__all__ = ["STFlashBeacon", "ENV_ID", "INSTRUCTION_TASK", "INSTRUCTION_MEMORY"]
