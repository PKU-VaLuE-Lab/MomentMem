"""Observed+Spatial task implemented as a real one-DOF spring slider."""

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
    PhysicalWaypoint,
    PhysicalWaypointFSM,
    PhysicsMaterial,
    TASK_ENV_IDS,
    UnifiedPandaEnv,
    assert_points_reachable,
    build_prismatic_articulation,
    build_static_box,
    build_visual_marker_box,
)
try:  # package import and direct-script import are both supported
    from .SO_spring_compression_cfg import CFG
except ImportError:  # pragma: no cover - direct gen_data.py execution
    from SO_spring_compression_cfg import CFG  # type: ignore


TASK_NAME = "SO_spring_compression"
ENV_ID = TASK_ENV_IDS[TASK_NAME]

INSTRUCTION_TASK = (
    "A white cube sits on a long horizontal rail connected to a spring. Before the robot "
    "moves, the cube slides to a compressed position, pauses, and returns to its "
    "starting position. As soon as it returns, close the gripper and push the cube at a "
    "steady speed to the same compressed position, then lift the closed gripper while the "
    "cube stays there."
)
INSTRUCTION_MEMORY = "Remember the moment when the cube has stopped at the compressed position."


@register_env(ENV_ID, max_episode_steps=700)
class SOSpringCompression(UnifiedPandaEnv):
    TASK_NAME = TASK_NAME
    INSTRUCTION_TASK = INSTRUCTION_TASK
    INSTRUCTION_MEMORY = INSTRUCTION_MEMORY

    def _load_task_scene(self, options: dict) -> None:
        self.slider = build_prismatic_articulation(
            self.scene,
            "spring_slider",
            root_half_size=CFG["joint_root_half_size"],
            moving_half_size=CFG["slider_half_size"],
            pose=sapien.Pose(
                CFG["slider_root_pose"],
                CFG["slider_root_quaternion_wxyz"],
            ),
            moving_offset=CFG["slider_offset"],
            axis=CFG["rail_axis"],
            limits=CFG["joint_limits"],
            root_color=(0.20, 0.22, 0.24, 1.0),
            moving_color=(0.93, 0.93, 0.91, 1.0),
            physics=PhysicsMaterial(
                static_friction=1.2,
                dynamic_friction=0.9,
                restitution=0.0,
                density=750.0,
            ),
            drive_stiffness=CFG["spring_stiffness"],
            drive_damping=CFG["spring_damping"],
            drive_force_limit=CFG["spring_force_limit"],
        )
        build_static_box(
            self.scene,
            "long_spring_rail",
            CFG["rail_half_size"],
            CFG["rail_pose"],
            color=(0.20, 0.22, 0.24, 1.0),
        )
        # Fixed anchor and zig-zag rail marks make the spring direction legible;
        # the actual restoring force is exclusively the prismatic joint drive.
        build_static_box(
            self.scene,
            "spring_anchor",
            CFG["anchor_half_size"],
            CFG["anchor_pose"],
            color=(0.16, 0.18, 0.20, 1.0),
        )
        # A longer transverse zig-zag makes the spring direction and expanded
        # choice space legible.  These are non-contact visual marks; the real
        # restoring force comes exclusively from the collision-bearing joint.
        rail_x = float(CFG["slider_root_pose"][0])
        direction = float(CFG["rail_direction_world"])
        for i in range(int(CFG["spring_marker_count"])):
            x = rail_x + (
                float(CFG["spring_marker_lateral_offset"])
                if i % 2 == 0
                else -float(CFG["spring_marker_lateral_offset"])
            )
            y = float(CFG["spring_marker_start_y"]) + direction * i * float(
                CFG["spring_marker_spacing"]
            )
            build_visual_marker_box(
                self.scene,
                f"spring_zigzag_{i}",
                [0.004, 0.011, 0.001],
                [x, y, 0.026],
                color=(0.65, 0.67, 0.70, 1.0),
            )
        self.register_manipulable(self.slider.moving_link)

    @staticmethod
    def _randint(rng, bounds) -> int:
        return int(rng.randint(int(bounds[0]), int(bounds[1]) + 1))

    def _initialize_task_episode(
        self,
        env_idx: torch.Tensor,
        episode_metadata: Mapping[str, Any] | None,
    ) -> None:
        n, device = self.num_envs, self.device
        if episode_metadata is None:
            target = self._episode_rng.uniform(*CFG["target_range"], size=n)
            rail_jitter = self._episode_rng.uniform(
                -float(CFG["rail_jitter"]),
                float(CFG["rail_jitter"]),
                size=(n, 2),
            )
            hold = np.asarray(
                [self._randint(self._episode_rng, CFG["demonstration_hold_steps"]) for _ in range(n)]
            )
        else:
            target = np.asarray(episode_metadata["target_compression"], dtype=np.float32).reshape(-1)
            rail_jitter = np.asarray(episode_metadata["rail_jitter_xy"], dtype=np.float32).reshape(n, 2)
            hold = np.asarray(episode_metadata["demonstration_hold_steps"], dtype=np.int64).reshape(-1)

        self.target_compression = torch.as_tensor(target, dtype=torch.float32, device=device)
        # The physical drive settles a few millimetres away from its nominal
        # command under spring/contact load.  The benchmark asks the robot to
        # reproduce what was actually observed, so accumulate the stable
        # keyframe position and use that measured value during execution.
        self.observed_compression = self.target_compression.clone()
        self._observed_compression_sum = torch.zeros_like(self.target_compression)
        self._observed_compression_count = torch.zeros_like(
            self.target_compression, dtype=torch.int64
        )
        self.rail_jitter_xy = torch.as_tensor(rail_jitter, dtype=torch.float32, device=device)
        self.show_start = torch.full(
            (n,), int(CFG["demonstration_settle_steps"]), dtype=torch.int64, device=device
        )
        self.release_start = self.show_start + torch.as_tensor(hold, dtype=torch.int64, device=device)
        self.return_end = self.release_start + int(CFG["return_steps"])
        # The expert already starts its approach on the first control step after
        # the spring has physically returned. Keep that exact behavior in the
        # phase contract: the return-completion frame is the one-frame memory
        # transition, and execution starts immediately on the next frame.
        self.execution_start = self.return_end + 1

        root_positions = torch.tensor(
            CFG["slider_root_pose"], dtype=torch.float32, device=device
        ).repeat(n, 1)
        root_positions[:, :2] += self.rail_jitter_xy
        quats = torch.tensor(
            CFG["slider_root_quaternion_wxyz"],
            dtype=torch.float32,
            device=device,
        ).repeat(n, 1)
        self.slider.articulation.set_pose(Pose.create_from_pq(p=root_positions, q=quats))
        self.slider.set_qpos(torch.zeros(n, dtype=torch.float32, device=device))
        self.slider.articulation.set_qvel(torch.zeros((n, 1), device=device))
        self.reproduced = torch.zeros(n, dtype=torch.bool, device=device)
        self.push_target_reached = torch.zeros(n, dtype=torch.bool, device=device)
        self.evaluation_push_started = torch.zeros(
            n, dtype=torch.bool, device=device
        )
        self.evaluation_push_released = torch.zeros(
            n, dtype=torch.bool, device=device
        )
        self.evaluation_max_compression = torch.zeros_like(
            self.target_compression
        )
        self._pre_lift_contact_dwell_count = torch.zeros(
            n, dtype=torch.int64, device=device
        )
        self.pre_lift_contact_dwell_complete = torch.zeros(
            n, dtype=torch.bool, device=device
        )
        self._post_push_lift_stable_count = torch.zeros(
            n, dtype=torch.int64, device=device
        )
        self.slider_grasp_seen = torch.zeros(n, dtype=torch.bool, device=device)

        target_world = root_positions.clone()
        target_world[:, 1] += (
            float(CFG["rail_direction_world"]) * self.target_compression
        )
        target_world[:, 2] += float(CFG["slider_offset"][2])
        natural_world = root_positions.clone()
        natural_world[:, 2] += float(CFG["slider_offset"][2])
        assert_points_reachable(
            torch.cat((natural_world, target_world), dim=0).detach().cpu().numpy()
        )
        self.set_phase_schedule(self.return_end - 1, self.execution_start)
        self.define_keyframe_event(
            "maximum_compression_observation",
            self.show_start,
            self.release_start - 1,
            required_information=("slider_position_along_rail", "compression_amount"),
        )
        self.set_task_episode_metadata(
            target_compression=self.target_compression,
            nominal_target_compression=self.target_compression,
            observed_compression=self.observed_compression,
            nominal_target_world_position=target_world,
            target_world_position=target_world,
            observed_target_world_position=target_world,
            rail_jitter_xy=self.rail_jitter_xy,
            rail_axis_world="-y",
            rail_total_length=2.0 * float(CFG["rail_half_size"][1]),
            demonstration_hold_steps=self.release_start - self.show_start,
            memory_delay_steps=self.execution_start - self.return_end - 1,
            memory_reset="spring_return_completion_transition_frame",
            blank_delay_motion="none_no_random_wait",
            post_observation_transition=(
                "spring_return_complete_immediate_slider_push_approach"
            ),
            execution_interaction="closed_gripper_contact_push_then_vertical_lift",
            push_command_preload=CFG["push_command_preload"],
            post_target_push_margin=CFG["post_target_push_margin"],
            post_target_push_dwell_steps=CFG[
                "post_target_push_dwell_steps"
            ],
            execution_spring_damping=CFG["execution_spring_damping"],
            execution_spring_damping_gain_per_meter=CFG[
                "execution_spring_damping_gain_per_meter"
            ],
            execution_spring_force_limit=CFG["execution_spring_force_limit"],
            pusher_face_offset=CFG["pusher_face_offset"],
            push_tcp_height=CFG["push_tcp_height"],
            push_contact_tcp_max_height=CFG["push_contact_tcp_max_height"],
            post_push_lift_height=CFG["post_push_lift_height"],
            post_push_lift_command_height=CFG[
                "post_push_lift_command_height"
            ],
            post_push_lift_position_tolerance=CFG[
                "post_push_lift_position_tolerance"
            ],
            post_push_lift_success_margin=CFG["post_push_lift_success_margin"],
            pre_lift_contact_dwell_steps=CFG[
                "pre_lift_contact_dwell_steps"
            ],
            post_push_lift_stable_steps=CFG["post_push_lift_stable_steps"],
            success_min_execution_steps=CFG["success_min_execution_steps"],
            success_position_check=(
                "current_slider_qpos_during_consecutive_raised_arm_states"
            ),
            success_position_tolerance=CFG["position_tolerance"],
            evaluation_success_position_check=(
                "maximum_slider_qpos_from_low_push_start_until_gripper_release"
            ),
            evaluation_success_position_tolerance=CFG[
                "evaluation_position_tolerance"
            ],
            evaluation_push_release_tcp_height=CFG[
                "push_contact_tcp_max_height"
            ],
            evaluation_push_start_min_compression=(
                2.0 * float(CFG["position_tolerance"])
            ),
        )
        self._expert_fsm = (
            PhysicalWaypointFSM(
                self,
                self._expert_waypoints(),
                max_position_command=float(CFG["max_position_command"]),
            )
            if n == 1
            else None
        )

    def _slider_position(self, env):
        return env.slider.moving_link.pose.p[0].detach().cpu().numpy()

    def _expert_waypoints(self):
        offset = float(CFG["pusher_face_offset"])

        def live(z, face_offset=0.0):
            def resolve(env):
                p = self._slider_position(env).copy()
                p[1] -= float(CFG["rail_direction_world"]) * face_offset
                p[2] = z
                return p
            return resolve

        def target(z, face_offset=0.0, extra_compression=0.0):
            def resolve(env):
                root = env.slider.articulation.pose.p[0].detach().cpu().numpy()
                # The spring produces a real position-controller compliance
                # error.  Command a small preload past the remembered point;
                # _after_task_control_step flips immediately to the lift once
                # the physical qpos enters the original 5 mm acceptance band.
                commanded_compression = (
                    env.observed_compression[0].item()
                    + float(CFG["push_command_preload"])
                    + float(extra_compression)
                )
                return (
                    float(root[0]),
                    float(
                        root[1]
                        + float(CFG["rail_direction_world"])
                        * commanded_compression
                        - float(CFG["rail_direction_world"]) * face_offset
                    ),
                    float(z),
                )
            return resolve

        return [
            PhysicalWaypoint(live(0.17, offset), GRIPPER_OPEN, label="above_slider_push_face"),
            PhysicalWaypoint(
                live(0.17, offset), GRIPPER_CLOSED, position_tolerance=0.018,
                rotation_tolerance=3.2, dwell_steps=7, label="close_gripper_above_slider",
            ),
            PhysicalWaypoint(
                live(float(CFG["push_tcp_height"]), offset), GRIPPER_CLOSED,
                # The closed Panda fingertips make physical side contact while
                # the TCP remains ~3.6 cm above the requested face centre.
                # Accept that collision-limited Z offset and advance into the
                # horizontal push instead of chasing an unreachable pose.
                position_tolerance=0.045, rotation_tolerance=3.2,
                dwell_steps=3, label="closed_gripper_contact_slider_face",
            ),
            PhysicalWaypoint(
                target(float(CFG["push_tcp_height"]), offset),
                GRIPPER_CLOSED,
                # As in the preceding contact waypoint, the real fingertips
                # seat on the slider face with the TCP about 3.6 cm above the
                # nominal face-centre Z.  Use the tested collision-limited
                # tolerance for low pushing; the final raised-arm waypoint
                # keeps the stricter 8 mm tolerance.
                position_tolerance=0.045,
                rotation_tolerance=3.2,
                dwell_steps=1,
                max_position_command=float(CFG["push_max_position_command"]),
                require=lambda env: bool(
                    env.pre_lift_contact_dwell_complete[0].item()
                ),
                label="closed_gripper_push_to_observed_band",
            ),
            PhysicalWaypoint(
                target(
                    float(CFG["push_tcp_height"]),
                    offset,
                    float(CFG["post_target_push_margin"]),
                ),
                GRIPPER_CLOSED,
                position_tolerance=0.045,
                rotation_tolerance=3.2,
                dwell_steps=int(CFG["post_target_push_dwell_steps"]),
                max_position_command=float(CFG["push_max_position_command"]),
                label="closed_gripper_post_target_horizontal_margin",
            ),
            PhysicalWaypoint(
                target(
                    float(CFG["post_push_lift_command_height"]),
                    offset,
                    float(CFG["post_target_push_margin"]),
                ),
                GRIPPER_CLOSED,
                position_tolerance=float(
                    CFG["post_push_lift_position_tolerance"]
                ),
                rotation_tolerance=3.2,
                dwell_steps=int(CFG["post_push_lift_stable_steps"]) + 2,
                max_position_command=float(CFG["push_max_position_command"]),
                label="closed_gripper_vertical_lift_after_margin",
            ),
        ]

    def _before_task_control_step(self) -> None:
        step = self.elapsed_steps.to(torch.int64)
        executing = bool(torch.all(step > self.return_end))
        post_push = executing and bool(
            torch.all(self.pre_lift_contact_dwell_complete).item()
        )
        if post_push:
            damping_base = float(CFG["post_push_spring_damping"])
            damping_gain = float(CFG["post_push_spring_damping_gain_per_meter"])
        elif executing:
            damping_base = float(CFG["execution_spring_damping"])
            damping_gain = float(CFG["execution_spring_damping_gain_per_meter"])
        else:
            damping_base = float(CFG["spring_damping"])
            damping_gain = 0.0
        damping = damping_base + damping_gain * float(
            torch.max(torch.clamp(self.slider.qpos, min=0.0)).item()
        )
        force_limit = (
            float(CFG["execution_spring_force_limit"])
            if executing
            else float(CFG["spring_force_limit"])
        )
        self.slider.joint.set_drive_properties(
            stiffness=float(CFG["spring_stiffness"]),
            damping=damping,
            force_limit=force_limit,
        )
        observing_or_holding = step < self.release_start
        # Preserve the physical task used by expert generation.  Closed-loop
        # evaluation separately records the maximum compression before the
        # gripper clears the low pushing region, so a later target retarget
        # cannot erase an overshoot that already happened.
        target = torch.where(
            observing_or_holding,
            self.target_compression,
            torch.where(
                self.pre_lift_contact_dwell_complete,
                self.observed_compression,
                torch.zeros_like(self.target_compression),
            ),
        )
        self.slider.drive_to(target)

    def _after_task_control_step(self) -> None:
        step = self.elapsed_steps.to(torch.int64)
        keyframe_stable = (step >= self.show_start) & (step < self.release_start)
        self._observed_compression_sum += torch.where(
            keyframe_stable, self.slider.qpos, torch.zeros_like(self.slider.qpos)
        )
        self._observed_compression_count += keyframe_stable.to(torch.int64)
        have_observation = self._observed_compression_count > 0
        self.observed_compression = torch.where(
            have_observation,
            self._observed_compression_sum
            / torch.clamp(self._observed_compression_count, min=1).to(torch.float32),
            self.target_compression,
        )
        observed_world = self.slider.articulation.pose.p.clone()
        observed_world[:, 1] += (
            float(CFG["rail_direction_world"]) * self.observed_compression
        )
        observed_world[:, 2] += float(CFG["slider_offset"][2])
        self.set_task_episode_metadata(
            observed_compression=self.observed_compression,
            target_world_position=observed_world,
            observed_target_world_position=observed_world,
            observed_position_source="mean_physical_qpos_over_keyframe_interval",
        )
        execution = self.elapsed_steps >= self.execution_start
        q = self.slider.qpos
        grasped = self.agent.is_grasping(self.slider.moving_link)
        # Record transient grasp detections for diagnostics. Side-pushing with
        # closed fingers can satisfy ManiSkill's contact-based grasp heuristic.
        self.slider_grasp_seen |= grasped
        near = torch.abs(q - self.observed_compression) <= float(
            CFG["position_tolerance"]
        )
        finger_qpos = self.agent.robot.get_qpos()[..., -2:]
        closed = torch.sum(finger_qpos, dim=-1) < 0.02
        tcp_z = self.agent.tcp.pose.p[..., 2]
        pushing_low = tcp_z < float(CFG["push_contact_tcp_max_height"])
        evaluation_push_now = (
            execution
            & closed
            & pushing_low
            & (q > 2.0 * float(CFG["position_tolerance"]))
        )
        self.evaluation_push_started |= evaluation_push_now
        evaluation_window_active = (
            self.evaluation_push_started & ~self.evaluation_push_released
        )
        self.evaluation_max_compression = torch.where(
            evaluation_window_active,
            torch.maximum(self.evaluation_max_compression, q),
            self.evaluation_max_compression,
        )
        gripper_cleared_slider = tcp_z >= float(
            CFG["push_contact_tcp_max_height"]
        )
        self.evaluation_push_released |= (
            self.evaluation_push_started & gripper_cleared_slider & closed
        )
        self.push_target_reached |= (
            execution
            & near
            & closed
            & pushing_low
        )
        contact_dwell_ready = (
            execution
            # This latch advances the scripted expert from horizontal contact
            # into the lift.  Closed-loop evaluation does not use the latch as
            # success; it compares the complete push-window maximum instead.
            & self.push_target_reached
            & closed
            & pushing_low
        )
        self._pre_lift_contact_dwell_count = torch.where(
            contact_dwell_ready,
            self._pre_lift_contact_dwell_count + 1,
            torch.zeros_like(self._pre_lift_contact_dwell_count),
        )
        self.pre_lift_contact_dwell_complete |= (
            self._pre_lift_contact_dwell_count
            >= int(CFG["pre_lift_contact_dwell_steps"])
        )
        lifted = (
            tcp_z
            >= float(CFG["post_push_lift_height"])
            - float(CFG["post_push_lift_success_margin"])
        )
        lift_ready = (
            execution
            & (
                self.elapsed_steps
                >= self.execution_start
                + int(CFG["success_min_execution_steps"])
            )
            & self.pre_lift_contact_dwell_complete
            & self.push_target_reached
            # Expert termination requires the current compression to match.
            # Formal evaluation checks the maximum over the complete push.
            & near
            & closed
            & lifted
            & ~grasped
        )
        self._post_push_lift_stable_count = torch.where(
            lift_ready,
            self._post_push_lift_stable_count + 1,
            torch.zeros_like(self._post_push_lift_stable_count),
        )
        self.reproduced |= (
            self._post_push_lift_stable_count
            >= int(CFG["post_push_lift_stable_steps"])
        )

    def _evaluate_task(self) -> Mapping[str, Any]:
        compression_error = torch.abs(
            self.slider.qpos - self.observed_compression
        )
        evaluation_max_compression_error = torch.abs(
            self.evaluation_max_compression - self.observed_compression
        )
        lifted = (
            self.agent.tcp.pose.p[..., 2]
            >= float(CFG["post_push_lift_height"])
            - float(CFG["post_push_lift_success_margin"])
        )
        evaluation_success = (
            self.evaluation_push_released
            & (
                evaluation_max_compression_error
                <= float(CFG["evaluation_position_tolerance"])
            )
        )
        return {
            "success": self.reproduced,
            "slider_qpos": self.slider.qpos,
            "target_compression": self.target_compression,
            "observed_compression": self.observed_compression,
            "compression_error": compression_error,
            "lift_compression_in_tolerance": (
                lifted & (compression_error <= float(CFG["position_tolerance"]))
            ),
            "evaluation_success": evaluation_success,
            "evaluation_push_started": self.evaluation_push_started,
            "evaluation_push_released": self.evaluation_push_released,
            "evaluation_max_compression": self.evaluation_max_compression,
            "evaluation_max_compression_error": (
                evaluation_max_compression_error
            ),
            "post_push_lift_stable_steps": self._post_push_lift_stable_count,
            "slider_grasp_seen": self.slider_grasp_seen,
            "slider_grasped": self.agent.is_grasping(self.slider.moving_link),
            "push_target_reached": self.push_target_reached,
            "pre_lift_contact_dwell_complete": (
                self.pre_lift_contact_dwell_complete
            ),
        }

    def _get_task_obs_extra(self, info: dict[str, Any]) -> dict[str, Any]:
        del info
        return {
            "slider_grasp_seen": self.slider_grasp_seen.unsqueeze(-1),
            "push_target_reached": self.push_target_reached.unsqueeze(-1),
            "pre_lift_contact_dwell_complete": (
                self.pre_lift_contact_dwell_complete.unsqueeze(-1)
            ),
            "observed_compression": self.observed_compression.unsqueeze(-1),
        }

    def _expert_action(self) -> torch.Tensor:
        if self._expert_fsm is None or torch.any(self.elapsed_steps <= self.return_end):
            return self.zero_action()
        return self._expert_fsm.action()


__all__ = [
    "SOSpringCompression",
    "ENV_ID",
    "INSTRUCTION_TASK",
    "INSTRUCTION_MEMORY",
]
