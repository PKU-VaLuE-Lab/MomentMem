"""Direct Appearance+Spatial task with graspable stamps and fading table LEDs."""

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
    JointAssembly,
    PhysicalWaypoint,
    PhysicalWaypointFSM,
    PhysicsMaterial,
    TASK_ENV_IDS,
    UnifiedPandaEnv,
    build_static_box,
    render_material,
    vertical_cylinder_pose,
    sample_distinct_color_ids,
)
from ASD_seal_checking_cfg import CFG  # noqa: E402

TASK_NAME = "ASD_seal_checking"
ENV_ID = TASK_ENV_IDS[TASK_NAME]
INSTRUCTION_TASK = (
    "Two white stamps cover two colored spots. A tray and two cubes of different colors are also "
    "on the table. Lift each stamp by its square handle, "
    "place it fully inside the tray, and watch the spot underneath fade away. After both stamps "
    "have been checked, return the arm to its starting pose. Then place each cube where the spot "
    "showed the same color and release it on the table."
)
INSTRUCTION_MEMORY = (
    "Remember each moment just after a stamp is lifted, while the colored spot underneath is still clearly visible."
)

SEAL_BODY_COLOR = (0.91, 0.92, 0.93, 1)
SEAL_HANDLE_COLOR = (0.82, 0.84, 0.87, 1)


def _set_led_appearance(material, source_rgba, fade_progress: float) -> None:
    """Continuously remove hue, opacity and emission without whitening."""

    source = np.asarray(source_rgba, dtype=np.float32)
    progress = float(np.clip(fade_progress, 0.0, 1.0))
    neutral = np.asarray(CFG["fade_neutral_rgb"], dtype=np.float32)
    rgb = (1.0 - progress) * source[:3] + progress * neutral
    remaining = 1.0 - progress
    material.set_base_color(
        (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(remaining))
    )
    emission = source[:3] * float(CFG["led_emission_strength"]) * remaining
    material.set_emission(
        (float(emission[0]), float(emission[1]), float(emission[2]), 0.0)
    )


def _seal_geometry() -> dict[str, Any]:
    base_half = float(CFG["seal_base_half_height"])
    stem_half = float(CFG["seal_stem_half_height"])
    handle_half = np.asarray(CFG["seal_handle_half_size"], dtype=np.float32)
    stem_center_z = base_half + stem_half
    handle_center_z = base_half + 2.0 * stem_half + float(handle_half[2])
    return {
        "base_half": base_half,
        "stem_half": stem_half,
        "handle_half": handle_half,
        "stem_center_z": stem_center_z,
        "handle_center_z": handle_center_z,
    }


def _build_seal(scene, name: str):
    """Build one stamp as a single compound dynamic rigid body."""

    geometry = _seal_geometry()
    physics = PhysicsMaterial(
        static_friction=1.2,
        dynamic_friction=0.9,
        density=680,
    )
    body_material = render_material(SEAL_BODY_COLOR)
    handle_material = render_material(SEAL_HANDLE_COLOR)
    builder = scene.create_actor_builder()
    base_pose = vertical_cylinder_pose([0.0, 0.0, 0.0])
    builder.add_cylinder_collision(
        pose=base_pose,
        radius=float(CFG["seal_base_radius"]),
        half_length=geometry["base_half"],
        material=physics.sapien(),
        density=physics.density,
    )
    builder.add_cylinder_visual(
        pose=base_pose,
        radius=float(CFG["seal_base_radius"]),
        half_length=geometry["base_half"],
        material=body_material,
    )
    stem_pose = vertical_cylinder_pose(
        [0.0, 0.0, geometry["stem_center_z"]]
    )
    builder.add_cylinder_collision(
        pose=stem_pose,
        radius=float(CFG["seal_stem_radius"]),
        half_length=geometry["stem_half"],
        material=physics.sapien(),
        density=physics.density,
    )
    builder.add_cylinder_visual(
        pose=stem_pose,
        radius=float(CFG["seal_stem_radius"]),
        half_length=geometry["stem_half"],
        material=body_material,
    )
    handle_pose = sapien.Pose(
        p=[0.0, 0.0, geometry["handle_center_z"]]
    )
    builder.add_box_collision(
        pose=handle_pose,
        half_size=geometry["handle_half"],
        material=physics.sapien(),
        density=physics.density,
    )
    builder.add_box_visual(
        pose=handle_pose,
        half_size=geometry["handle_half"],
        material=handle_material,
    )
    builder.initial_pose = sapien.Pose(
        [-0.5, 0.8, geometry["base_half"]]
    )
    seal = builder.build_dynamic(name)
    seal.set_mass(float(CFG["seal_mass"]))
    return seal


def _build_led(scene, name: str, color) -> tuple[JointAssembly, Any]:
    builder = scene.create_articulation_builder()
    builder.initial_pose = sapien.Pose([0, 0, -0.018])
    root = builder.create_link_builder(parent=None)
    root.set_name(f"{name}_root")
    moving = builder.create_link_builder(parent=root)
    moving.set_name(f"{name}_disk")
    material = render_material(color)
    _set_led_appearance(material, color, 0.0)
    moving.add_cylinder_visual(
        radius=float(CFG["led_radius"]),
        half_length=0.0015,
        pose=sapien.Pose(q=[math.sqrt(0.5), 0, math.sqrt(0.5), 0]),
        material=material,
    )
    moving.set_joint_name(f"{name}_joint")
    axis_q = [math.sqrt(0.5), 0, -math.sqrt(0.5), 0]
    moving.set_joint_properties(
        type="prismatic",
        limits=[[0.0, float(CFG["led_travel"])]],
        pose_in_parent=sapien.Pose(q=axis_q),
        pose_in_child=sapien.Pose(q=axis_q),
        friction=0.01,
        damping=0.4,
    )
    art = builder.build(name, fix_root_link=True)
    joint = art.get_active_joints()[0]
    joint.set_drive_properties(stiffness=120, damping=18, force_limit=5)
    return (
        JointAssembly(
            art,
            joint,
            art.links_map[f"{name}_disk"],
            0.0,
            float(CFG["led_travel"]),
        ),
        material,
    )


def _build_color_cube(scene, name: str):
    physics = PhysicsMaterial(static_friction=0.9, dynamic_friction=0.7, density=650)
    builder = scene.create_actor_builder()
    material = render_material((0.8, 0.8, 0.8, 1.0))
    builder.add_box_collision(
        half_size=[0.023] * 3, material=physics.sapien(), density=physics.density
    )
    builder.add_box_visual(half_size=[0.023] * 3, material=material)
    builder.initial_pose = sapien.Pose([-0.15, 0.0, 0.023])
    actor = builder.build_dynamic(name)
    actor.set_mass(0.065)
    return actor, material


@register_env(ENV_ID, max_episode_steps=1700)
class ASDSealChecking(UnifiedPandaEnv):
    TASK_NAME = TASK_NAME
    INSTRUCTION_TASK = INSTRUCTION_TASK
    INSTRUCTION_MEMORY = INSTRUCTION_MEMORY

    def _load_task_scene(self, options: dict) -> None:
        del options
        self.seal_geometry = _seal_geometry()
        self.seals = [
            _build_seal(self.scene, f"white_seal_{i}") for i in range(2)
        ]
        led_parts = [
            _build_led(self.scene, f"covered_led_{i}", COLOR_RGBA[i])
            for i in range(2)
        ]
        self.leds = [parts[0] for parts in led_parts]
        self.led_materials = [parts[1] for parts in led_parts]
        built = [
            _build_color_cube(self.scene, "red_cube" if i == 0 else "blue_cube")
            for i in range(2)
        ]
        self.cubes = [item[0] for item in built]
        self.cube_materials = [item[1] for item in built]
        tx, ty = (-0.225, 0.0)
        tray_color = (0.28, 0.34, 0.38, 1)
        # Real tray floor and walls, sized to hold both seals without overlap.
        self.tray_parts = []
        self.tray_part_offsets = []
        for suffix, half, pos in (
            (
                "floor",
                [*CFG["tray_floor_half_xy"], 0.006],
                [tx, ty, 0.006],
            ),
            ("back", [0.007, 0.125, 0.030], [tx + 0.076, ty, 0.030]),
            ("near", [0.083, 0.007, 0.030], [tx, ty - 0.118, 0.030]),
            ("far", [0.083, 0.007, 0.030], [tx, ty + 0.118, 0.030]),
        ):
            self.tray_parts.append(
                build_static_box(self.scene, f"seal_tray_{suffix}", half, pos, color=tray_color)
            )
            self.tray_part_offsets.append(np.asarray(pos, dtype=np.float32) - np.asarray([tx, ty, 0], dtype=np.float32))
        self.register_manipulable(*self.seals, *self.cubes)

    def _initialize_task_episode(self, env_idx, episode_metadata: Mapping[str, Any] | None) -> None:
        del env_idx
        if episode_metadata is None:
            seal_colors = self._episode_rng.permutation(2).astype(int).tolist()
            selected_color_ids = sample_distinct_color_ids(self._episode_rng)
            order = self._episode_rng.permutation(2).astype(int).tolist()
            # Joint sampling avoids disconnected regions under tray-clearance
            # and object-separation constraints.
            for _ in range(20_000):
                tray_center = self._episode_rng.uniform(
                    [CFG["tray_center_x"][0], CFG["tray_center_y"][0]],
                    [CFG["tray_center_x"][1], CFG["tray_center_y"][1]],
                ).astype(np.float32)
                points = self._episode_rng.uniform(
                    CFG["object_xy_low"], CFG["object_xy_high"], size=(4, 2)
                ).astype(np.float32)
                seal_xy, cube_xy = points[:2], points[2:]
                cross = np.linalg.norm(
                    seal_xy[:, None, :] - cube_xy[None, :, :], axis=2
                )
                relative = np.abs(points - tray_center[None, :])
                clear_of_tray = np.all(
                    (relative[:, 0] >= CFG["object_tray_clearance_xy"][0])
                    | (relative[:, 1] >= CFG["object_tray_clearance_xy"][1])
                )
                if (
                    np.linalg.norm(seal_xy[0] - seal_xy[1]) >= CFG["seal_min_separation"]
                    and np.linalg.norm(cube_xy[0] - cube_xy[1]) >= CFG["cube_spawn_min_separation"]
                    and float(cross.min()) >= CFG["seal_cube_min_separation"]
                    and clear_of_tray
                ):
                    break
            else:
                raise RuntimeError("failed to sample collision-free global ASD layout")
            reveal_steps = [int(self._episode_rng.randint(CFG["reveal_steps"][0], CFG["reveal_steps"][1] + 1)) for _ in range(2)]
            delay = int(self._episode_rng.randint(CFG["memory_delay_steps"][0], CFG["memory_delay_steps"][1] + 1))
        else:
            seal_colors = [int(x) for x in episode_metadata["seal_color_ids"]]
            selected_color_ids = np.asarray(
                episode_metadata["selected_color_ids"], dtype=np.int64
            )
            order = [int(x) for x in episode_metadata["observation_order"]]
            seal_xy = np.asarray(episode_metadata["seal_xy"], dtype=np.float32)
            tray_center = np.asarray(episode_metadata["tray_center"], dtype=np.float32)
            cube_xy = np.asarray(episode_metadata["cube_spawn_xy"], dtype=np.float32)
            reveal_steps = [int(x) for x in episode_metadata["reveal_steps"]]
            delay = int(episode_metadata["memory_delay_steps"])
        self.seal_xy = seal_xy
        self.tray_center = tray_center
        self.cube_spawn_xy = cube_xy
        self.seal_colors = seal_colors
        self.selected_color_ids = selected_color_ids.tolist()
        self.episode_colors = COLOR_RGBA[selected_color_ids]
        self.observation_order = order
        self.reveal_steps = reveal_steps
        self.memory_delay_steps = delay
        q_identity = torch.tensor(
            [[1, 0, 0, 0]], dtype=torch.float32, device=self.device
        )
        for actor, offset in zip(self.tray_parts, self.tray_part_offsets):
            p = np.asarray([tray_center[0], tray_center[1], 0], dtype=np.float32) + offset
            actor.set_pose(sapien.Pose(p=p))
        for i, seal in enumerate(self.seals):
            seal.set_pose(Pose.create_from_pq(
                p=torch.tensor(
                    [[
                        self.seal_xy[i, 0],
                        self.seal_xy[i, 1],
                        self.seal_geometry["base_half"],
                    ]],
                    dtype=torch.float32,
                    device=self.device,
                ),
                q=q_identity,
            ))
            seal.set_linear_velocity(torch.zeros((1, 3), device=self.device))
            seal.set_angular_velocity(torch.zeros((1, 3), device=self.device))
        for color, led in enumerate(self.leds):
            seal_index = seal_colors.index(color)
            xy = self.seal_xy[seal_index]
            led.articulation.set_pose(Pose.create_from_pq(
                p=torch.tensor([[xy[0], xy[1], -0.018]], dtype=torch.float32, device=self.device),
                q=q_identity,
            ))
            # The coloured disk physically starts at the tabletop and is
            # hidden only by the opaque seal above it.  Revealing therefore
            # happens at the instant the seal clears the disk; the mechanism
            # is not raised in response to that observation event.
            led.set_qpos(float(CFG["led_travel"]))
            led.articulation.set_qvel(torch.zeros((1, 1), device=self.device))
            _set_led_appearance(
                self.led_materials[color], self.episode_colors[color], 0.0
            )
        for i, cube in enumerate(self.cubes):
            self.cube_materials[i].set_base_color(
                tuple(float(x) for x in self.episode_colors[i])
            )
            xy = self.cube_spawn_xy[i]
            cube.set_pose(Pose.create_from_pq(
                p=torch.tensor([[xy[0], xy[1], 0.023]], dtype=torch.float32, device=self.device),
                q=q_identity,
            ))
            cube.set_linear_velocity(torch.zeros((1, 3), device=self.device))
            cube.set_angular_velocity(torch.zeros((1, 3), device=self.device))
        self.observed = [False, False]
        self.reveal_start = [-1, -1]
        self.fade_start = [-1, -1]
        self.fade_progress = [0.0, 0.0]
        self._observation_index = 0
        self._observation_complete = False
        self._memory_home_stable_steps = 0
        self.execution_start = 10_000
        self.placed = [False, False]
        self._placement_stable_counts = [0, 0]
        self.placement_contact_forces = [0.0, 0.0]
        self.set_phase_schedule(9_998, 9_999)
        self.set_task_episode_metadata(
            seal_color_ids=seal_colors,
            selected_color_ids=selected_color_ids,
            selected_color_names=[COLOR_NAMES[int(x)] for x in selected_color_ids],
            selected_color_rgba=self.episode_colors,
            color_candidate_count=len(COLOR_NAMES),
            observation_order=order,
            seal_xy=self.seal_xy,
            tray_center=self.tray_center,
            cube_spawn_xy=self.cube_spawn_xy,
            tray_transfer_motion="continuous_approach_and_descent_without_high_dwell",
            between_stamp_motion="direct_tray_to_next_stamp_without_home",
            initial_stamp_approach_clearance=CFG["initial_stamp_approach_clearance"],
            inter_stamp_approach_clearance=CFG["inter_stamp_approach_clearance"],
            post_observation_reset="home_only_after_both_stamps",
            reveal_steps=reveal_steps,
            memory_delay_steps=delay,
            memory_home_tolerance=0.006,
            memory_home_stable_steps=4,
            tray_slots_y=CFG["tray_slots_y"],
            seal_geometry="wide_round_base_with_square_grasp_handle",
            seal_base_radius=CFG["seal_base_radius"],
            seal_base_half_height=CFG["seal_base_half_height"],
            seal_handle_half_size=CFG["seal_handle_half_size"],
            seal_handle_center_z=self.seal_geometry["handle_center_z"],
            tray_floor_top=CFG["tray_floor_top"],
            seal_release_dwell_steps=CFG["seal_release_dwell_steps"],
            reveal_transition="linear_color_alpha_fade_then_physical_retraction",
            reveal_timing="first_edge_keyframe_hold_then_full_visibility_fade",
            initial_led_rgba=self.episode_colors,
            led_radius=CFG["led_radius"],
            led_emission_strength=CFG["led_emission_strength"],
            reveal_lift_clearance=CFG["reveal_lift_clearance"],
            reveal_uncover_distance=CFG["reveal_uncover_distance"],
            fade_lift_clearance=CFG["fade_lift_clearance"],
            fade_uncover_distance=CFG["fade_uncover_distance"],
            reveal_start_steps=self.reveal_start,
            fade_start_steps=self.fade_start,
            fade_neutral_rgb=CFG["fade_neutral_rgb"],
            keyframe_end_fade_progress=CFG["keyframe_end_fade_progress"],
            keyframe_end_appearance_progress=CFG["keyframe_end_appearance_progress"],
            led_retract_start_progress=CFG["led_retract_start_progress"],
            final_led_state="retracted_below_table_no_visible_trace",
            keyframe_semantics="uncover_start_through_last_readable_faint_color_frame",
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
            seal_tray_contact_force_threshold=CFG[
                "seal_tray_contact_force_threshold"
            ],
            tray_floor_half_xy=CFG["tray_floor_half_xy"],
            seal_tray_contract=(
                "complete_base_inside_tray_and_real_tray_contact"
            ),
        )
        self._observe_fsm = PhysicalWaypointFSM(
            self,
            self._seal_waypoints(
                order[0],
                0,
                return_home=False,
                approach_clearance=CFG["initial_stamp_approach_clearance"],
                isotropic_approach=False,
            ),
        )
        self._execute_fsm = None

    def _seal_handle_point(self, index, dz=0.0):
        def resolve(env):
            p = env.seals[index].pose.p[0].detach().cpu().numpy().copy()
            p[2] += float(env.seal_geometry["handle_center_z"]) + float(dz)
            return p
        return resolve

    def _seal_tray_states(self):
        """Return current physical containment/contact state for both seals."""

        tx, ty = [float(value) for value in self.tray_center]
        floor_half_x, floor_half_y = [
            float(value) for value in CFG["tray_floor_half_xy"]
        ]
        radius = float(CFG["seal_base_radius"])
        x_limit = floor_half_x - radius
        y_limit = floor_half_y - radius
        states = []
        for seal in self.seals:
            dx = abs(float(seal.pose.p[0, 0].item()) - tx)
            dy = abs(float(seal.pose.p[0, 1].item()) - ty)
            contained = dx <= x_limit and dy <= y_limit
            contact_force = max(
                float(
                    torch.linalg.norm(
                        self.scene.get_pairwise_contact_forces(seal, tray_part),
                        dim=-1,
                    )[0].item()
                )
                for tray_part in self.tray_parts
            )
            released = not bool(self.agent.is_grasping(seal)[0].item())
            linear_speed = float(
                torch.linalg.norm(seal.linear_velocity[0]).item()
            )
            angular_speed = float(
                torch.linalg.norm(seal.angular_velocity[0]).item()
            )
            valid = (
                contained
                and released
                and contact_force
                >= float(CFG["seal_tray_contact_force_threshold"])
                and linear_speed
                <= float(CFG["placement_linear_speed_tolerance"])
                and angular_speed
                <= float(CFG["placement_angular_speed_tolerance"])
            )
            states.append((valid, contact_force, contained))
        return states

    def _seal_at_height(self, index, tcp_z):
        def resolve(env):
            p = env.seals[index].pose.p[0].detach().cpu().numpy().copy()
            p[2] = float(tcp_z)
            return p
        return resolve

    def _seal_waypoints(
        self,
        seal_index,
        tray_slot,
        *,
        return_home,
        approach_clearance,
        isotropic_approach,
    ):
        tx = float(self.tray_center[0])
        ty = float(self.tray_center[1]) + float(CFG["tray_slots_y"][tray_slot])
        tray_tcp_z = (
            float(CFG["tray_floor_top"])
            + float(CFG["tray_release_clearance"])
            + float(self.seal_geometry["base_half"])
            + float(self.seal_geometry["handle_center_z"])
        )
        waypoints = [
            PhysicalWaypoint(
                self._seal_handle_point(seal_index, approach_clearance),
                GRIPPER_OPEN,
                isotropic_position_command=isotropic_approach,
                label=f"seal_{seal_index}_above",
            ),
            PhysicalWaypoint(self._seal_handle_point(seal_index), GRIPPER_OPEN, dwell_steps=3, label=f"seal_{seal_index}_handle"),
            PhysicalWaypoint(self._seal_handle_point(seal_index), GRIPPER_CLOSED, dwell_steps=9, require=lambda env, i=seal_index: bool(env.agent.is_grasping(env.seals[i])[0].item()), label=f"seal_{seal_index}_grasp"),
            PhysicalWaypoint(self._seal_at_height(seal_index, 0.20), GRIPPER_CLOSED,
                             require=lambda env, i=seal_index: bool(env.agent.is_grasping(env.seals[i])[0].item()),
                             label=f"seal_{seal_index}_vertical_clearance"),
            PhysicalWaypoint(lambda env: [tx + 0.065, ty, 0.20], GRIPPER_CLOSED, label=f"seal_{seal_index}_continuous_tray_approach"),
            PhysicalWaypoint(lambda env: [tx, ty, 0.155], GRIPPER_CLOSED, label=f"seal_{seal_index}_continuous_tray_high"),
            PhysicalWaypoint(lambda env: [tx, ty, tray_tcp_z], GRIPPER_CLOSED, label=f"seal_{seal_index}_tray_low"),
            PhysicalWaypoint(
                lambda env: [tx, ty, tray_tcp_z],
                GRIPPER_OPEN,
                dwell_steps=CFG["seal_release_dwell_steps"],
                label=f"seal_{seal_index}_release",
            ),
        ]
        if return_home:
            waypoints.extend(
                [
                    PhysicalWaypoint(
                        lambda env: [tx, ty, 0.20],
                        GRIPPER_OPEN,
                        label="tray_retract",
                    ),
                    PhysicalWaypoint(
                        HOME_TCP_POSITION,
                        GRIPPER_OPEN,
                        position_tolerance=0.004,
                        dwell_steps=4,
                        label="home_after_both_stamps",
                    ),
                ]
            )
        return waypoints

    def _target_for_color(self, color):
        return self.seal_xy[self.seal_colors.index(color)]

    def _execute_waypoints(self):
        wp = []
        for color, cube in enumerate(self.cubes):
            target = self._target_for_color(color)

            def cube_at_height(env, z, actor=cube):
                del env
                p = actor.pose.p[0].detach().cpu().numpy()
                return np.asarray([float(p[0]), float(p[1]), float(z)], dtype=np.float32)

            wp.extend([
                PhysicalWaypoint(lambda env, a=cube: np.asarray([float(a.pose.p[0, 0].item()), float(a.pose.p[0, 1].item()), 0.16], np.float32), GRIPPER_OPEN, label=f"cube_{color}_above"),
                PhysicalWaypoint(lambda env, f=cube_at_height: f(env, 0.032), GRIPPER_OPEN,
                                 position_tolerance=0.012, rotation_tolerance=10.0,
                                 dwell_steps=3, label=f"cube_{color}_contact"),
                PhysicalWaypoint(lambda env, f=cube_at_height: f(env, 0.032), GRIPPER_CLOSED,
                                 position_tolerance=0.012, rotation_tolerance=10.0,
                                 dwell_steps=10,
                                 require=lambda env, a=cube: bool(env.agent.is_grasping(a)[0].item()),
                                 label=f"cube_{color}_grasp"),
                PhysicalWaypoint(lambda env, f=cube_at_height: f(env, 0.16), GRIPPER_CLOSED,
                                 rotation_tolerance=10.0,
                                 require=lambda env, a=cube: bool(env.agent.is_grasping(a)[0].item()),
                                 label=f"cube_{color}_vertical_clearance"),
                PhysicalWaypoint([float(target[0]), float(target[1]), 0.15], GRIPPER_CLOSED, label=f"cube_{color}_target_high"),
                PhysicalWaypoint([float(target[0]), float(target[1]), 0.031], GRIPPER_CLOSED, label=f"cube_{color}_target_low"),
                PhysicalWaypoint([float(target[0]), float(target[1]), 0.031], GRIPPER_OPEN, dwell_steps=8, label=f"cube_{color}_release"),
                PhysicalWaypoint([float(target[0]), float(target[1]), 0.15], GRIPPER_OPEN, label=f"cube_{color}_retract"),
            ])
        return wp

    def _before_task_control_step(self) -> None:
        # Keep the disk at the tabletop through the readable part of the fade,
        # then continuously lower the real prismatic disk below the surface.
        retract_start = float(CFG["led_retract_start_progress"])
        for color_id, led in enumerate(self.leds):
            seal_index = self.seal_colors.index(color_id)
            progress = float(self.fade_progress[seal_index])
            retract = float(
                np.clip((progress - retract_start) / (1.0 - retract_start), 0.0, 1.0)
            )
            led.drive_to(float(CFG["led_travel"]) * (1.0 - retract))

    def _after_task_control_step(self) -> None:
        step = int(self.elapsed_steps[0].item())
        for seal_index, seal in enumerate(self.seals):
            if self.fade_start[seal_index] >= 0:
                continue
            distance = torch.linalg.norm(
                seal.pose.p[0, :2] - torch.as_tensor(self.seal_xy[seal_index], device=self.device)
            )
            # Compute the broad base disk's exact lowest point under arbitrary
            # tilt.  The handle and stem do not determine whether the covered
            # LED footprint has physically cleared.
            quaternion = seal.pose.q[0]
            qw, qx, qy, qz = quaternion.unbind()
            rotation_z_row = torch.stack(
                (
                    2.0 * (qx * qz - qw * qy),
                    2.0 * (qy * qz + qw * qx),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                )
            )
            axis_z = torch.abs(rotation_z_row[2])
            radial_z = torch.linalg.norm(rotation_z_row[:2])
            vertical_extent = (
                float(CFG["seal_base_half_height"]) * axis_z
                + float(CFG["seal_base_radius"]) * radial_z
            )
            lowest_point = seal.pose.p[0, 2] - vertical_extent
            lowest = float(lowest_point.item())
            horizontal = float(distance.item())
            reveal_clear = (
                lowest >= float(CFG["reveal_lift_clearance"])
                or horizontal >= float(CFG["reveal_uncover_distance"])
            )
            if not self.observed[seal_index] and reveal_clear:
                self.observed[seal_index] = True
                self.reveal_start[seal_index] = step

            fade_clear = (
                lowest >= float(CFG["fade_lift_clearance"])
                or horizontal >= float(CFG["fade_uncover_distance"])
            )
            if self.observed[seal_index] and fade_clear:
                self.fade_start[seal_index] = step
                color = self.seal_colors[seal_index]
                self.define_keyframe_event(
                    f"seal_{seal_index}_{COLOR_NAMES[self.selected_color_ids[color]]}_reveal",
                    self.reveal_start[seal_index] + 1,
                    # End while the colour has become too faint to read.  The
                    # material continues fading and the physical disk retracts
                    # after this memory interval, leaving no visible trace.
                    step
                    + int(
                        np.ceil(
                            self.reveal_steps[seal_index]
                            * float(CFG["keyframe_end_fade_progress"])
                        )
                    )
                    + 1,
                    required_information=("revealed_color", "original_seal_position"),
                )
                self.set_task_episode_metadata(
                    reveal_start_steps=self.reveal_start,
                    fade_start_steps=self.fade_start,
                )
        for seal_index, fade_start in enumerate(self.fade_start):
            if fade_start < 0:
                continue
            fade_steps = int(self.reveal_steps[seal_index])
            progress = float(np.clip((step - fade_start) / fade_steps, 0.0, 1.0))
            color_id = self.seal_colors[seal_index]
            appearance_progress = float(
                np.clip(
                    progress
                    * float(CFG["keyframe_end_appearance_progress"])
                    / float(CFG["keyframe_end_fade_progress"]),
                    0.0,
                    1.0,
                )
            )
            _set_led_appearance(
                self.led_materials[color_id],
                self.episode_colors[color_id],
                appearance_progress,
            )
            self.fade_progress[seal_index] = progress
        if self._observe_fsm.complete and self._observation_index < 1:
            self._observation_index += 1
            self._observe_fsm = PhysicalWaypointFSM(
                self,
                self._seal_waypoints(
                    self.observation_order[self._observation_index],
                    self._observation_index,
                    return_home=True,
                    approach_clearance=CFG["inter_stamp_approach_clearance"],
                    isotropic_approach=True,
                ),
            )
        seal_tray_states = self._seal_tray_states()
        seals_stowed = all(state[0] for state in seal_tray_states)
        tcp_home_error = torch.linalg.norm(
            self.agent.tcp.pose.p[0]
            - torch.as_tensor(HOME_TCP_POSITION, device=self.device)
        )
        home_reset_now = (
            all(self.observed)
            and seals_stowed
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
        for color, cube in enumerate(self.cubes):
            target = torch.as_tensor(self._target_for_color(color), device=self.device)
            distance = torch.linalg.norm(cube.pose.p[0, :2] - target)
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
                and float(cube.pose.p[0, 2].item()) < 0.060
                and not bool(self.agent.is_grasping(cube)[0].item())
                and contact_force
                >= float(CFG["placement_contact_force_threshold"])
                and linear_speed
                <= float(CFG["placement_linear_speed_tolerance"])
                and angular_speed
                <= float(CFG["placement_angular_speed_tolerance"])
            )
            self._placement_stable_counts[color] = (
                self._placement_stable_counts[color] + 1
                if placement_now
                else 0
            )
            self.placement_contact_forces[color] = contact_force
            self.placed[color] = (
                placement_now
                and self._placement_stable_counts[color]
                >= int(CFG["placement_stable_steps"])
            )

    def _evaluate_task(self):
        seal_tray_states = self._seal_tray_states()
        seals_stowed = all(state[0] for state in seal_tray_states)
        return {
            "success": torch.tensor(
                [
                    self._observation_complete
                    and all(self.placed)
                    and seals_stowed
                ],
                dtype=torch.bool,
                device=self.device,
            ),
            "seal_reveals": torch.tensor([sum(self.observed)], device=self.device),
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
            "seals_stowed_in_tray": torch.tensor(
                [[state[0] for state in seal_tray_states]],
                dtype=torch.bool,
                device=self.device,
            ),
            "seal_tray_contact_force": torch.tensor(
                [[state[1] for state in seal_tray_states]],
                dtype=torch.float32,
                device=self.device,
            ),
            "seal_fully_contained_in_tray": torch.tensor(
                [[state[2] for state in seal_tray_states]],
                dtype=torch.bool,
                device=self.device,
            ),
        }

    def _expert_action(self):
        step = int(self.elapsed_steps[0].item())
        if not self._observe_fsm.complete:
            return self._observe_fsm.action()
        if not self._observation_complete or step < self.execution_start:
            return self.zero_action()
        if self._execute_fsm is None:
            self._execute_fsm = PhysicalWaypointFSM(self, self._execute_waypoints())
        return self._execute_fsm.action()


__all__ = ["ASDSealChecking", "ENV_ID", "INSTRUCTION_TASK", "INSTRUCTION_MEMORY"]
