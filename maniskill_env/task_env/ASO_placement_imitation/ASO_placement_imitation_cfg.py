"""Configuration for free-planar ASO placement imitation."""

CFG = {
    "cube_half_size": 0.025,
    "cube_mass": 0.075,
    # Keep tabletop grasps away from the Panda's folded-arm joint limit.
    "initial_xy_low": (-0.220, -0.20),
    "initial_xy_high": (-0.120, 0.20),
    # Keep both planar coordinates genuinely random while staying inside the
    # Panda's empirically stable grasped-object workspace.  The broader table
    # bounds are collision-reachable at tabletop height, but a 17-cm lift at
    # the far positive-x / |y| corners drives the arm into a kinematic limit.
    "target_xy_low": (0.015, -0.17),
    "target_xy_high": (0.17, 0.17),
    "minimum_initial_separation": 0.115,
    "minimum_target_separation": 0.13,
    # The teacher is a real planar force held over each 20-Hz control step.
    # Low stiffness avoids consecutive saturated impulses on long moves, while
    # matched velocity damping keeps the free dynamic cube below 0.8 m/s.
    "teacher_kp": 3.0,
    "teacher_kd": 3.0,
    "teacher_max_force": 0.98,
    "teacher_breakaway_force": 0.90,
    "teacher_position_tolerance": 0.014,
    "teacher_speed_tolerance": 0.035,
    # Require 0.4 s of continuous low-speed target occupancy before the
    # memory interval begins.  Arrival/transient frames are not keyframes.
    "keyframe_settle_steps": 8,
    "hold_steps": (26, 38),
    "between_demo_steps": (10, 16),
    "memory_delay_steps": (16, 24),
    "place_tolerance": 0.045,
    "placement_contact_force_threshold": 0.05,
    "placement_linear_speed_tolerance": 0.03,
    "placement_angular_speed_tolerance": 1.0,
    "placement_stable_steps": 4,
}

__all__ = ["CFG"]
