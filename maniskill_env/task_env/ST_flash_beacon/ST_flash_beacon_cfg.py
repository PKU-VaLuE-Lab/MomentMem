"""Configuration for ST_flash_beacon."""

CFG = {
    "cube_half_size": 0.025,
    # Randomized spawn region inside the robust top-down grasp workspace.
    "cube_spawn_x": (-0.22, -0.02),
    "cube_spawn_y": (-0.22, 0.22),
    # Carry targets remain inside the conservative loaded-arm workspace.
    "target_x": (-0.16, 0.15),
    "target_y": (-0.30, 0.30),
    # Strong separation maintains spatial diversity between the two targets.
    "minimum_target_separation": 0.40,
    # The positive-X, large-|Y| corners are the furthest loaded-arm reaches.
    # Clip only those small corner caps instead of shrinking the whole target
    # distribution, preserving broad spatial and angular diversity.
    "target_corner_y_threshold": 0.24,
    "target_far_x_at_corner": 0.12,
    # Ensure the first reveal always produces a visible approach rather than
    # spawning almost directly beneath the already-lifted cube.
    "minimum_spawn_target_separation": 0.32,
    # The full beacon tower is hidden below the table at q=0 and rises far
    # enough to expose its pedestal, mast and luminous lens.  The keyframe
    # begins only after the lens and upper guard are fully above the tabletop.
    "beacon_hidden_root_z": -0.090,
    "beacon_lift_travel": 0.095,
    "beacon_clear_qpos": 0.090,
    "beacon_raise_steps": 4,
    "beacon_hold_steps": (18, 24),
    "beacon_lower_steps": 4,
    "beacon_moving_mass": 0.035,
    "beacon_drive_stiffness": 420.0,
    "beacon_drive_damping": 10.0,
    "beacon_drive_force_limit": 35.0,
    "memory_delay_steps": (16, 24),
    # One command cap is used before, during, and after both beacon reveals.
    "task_max_position_command": 0.14,
    "carry_height": 0.17,
    # The first contact uses a dedicated tight physical approach so it cannot
    # be declared complete while the cube is still hovering above the table.
    "first_contact_tcp_height": 0.0275,
    "first_contact_position_tolerance": 0.0005,
    "placement_tcp_height": 0.032,
    "placement_position_tolerance": 0.015,
    "first_target_contact_clearance": 0.0001,
    "placement_tolerance": 0.045,
    "final_table_contact_force_threshold": 0.05,
    "final_bottom_clearance": 0.002,
    "final_linear_speed_threshold": 0.03,
    "final_angular_speed_threshold": 0.5,
    "final_stable_steps": 4,
    "final_gripper_open_sum": 0.07,
    "final_clear_tcp_height": 0.10,
}

__all__ = ["CFG"]
