"""Configuration for AD_scroll_discovery."""

CFG = {
    "scroll_x": 0.075,
    "scroll_y": (-0.24, 0.0, 0.24),
    "slot_jitter": 0.006,
    "strip_travel": 0.150,
    "pull_tab_x": -0.092,
    # The freely moving housing yields slightly on the tabletop.  This extra
    # commanded travel brings the real prismatic joint to its mechanical stop
    # without assigning either link's pose directly.
    "pull_overtravel": 0.040,
    # At -0.112 m the right edge of the enlarged 5x5 mark has completely
    # cleared the housing's -X lip; the freely sliding housing yields before
    # the joint reaches its nominal -0.150 m hard stop.
    "reveal_threshold": -0.112,
    "reveal_hold_steps": (22, 30),
    "closed_threshold": -0.012,
    "memory_delay_steps": (16, 24),
    # Stack one complete housing vertically on the other.  The root geometry
    # occupies z=[0, 0.058], so a 59 mm root offset creates contact without a
    # commanded interpenetration.
    "stack_root_offset_z": 0.059,
    "stack_xy_tolerance": 0.038,
    "stack_min_root_dz": 0.045,
    "stack_max_root_dz": 0.082,
    "contact_force_threshold": 0.02,
    "reveal_clearance": 0.150,
    "housing_grasp_z": 0.043,
    "housing_approach_height": 0.120,
    "pair_transfer_height": 0.270,
    # Expert recording ends after four consecutive states at the home pose.
    # Formal closed-loop success does not use these parameters.
    "final_home_tolerance": 0.006,
    "final_home_stable_steps": 4,
    "max_position_command": 0.18,
    "max_rotation_command": 0.28,
}

__all__ = ["CFG"]
