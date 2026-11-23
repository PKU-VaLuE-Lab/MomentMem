"""Configuration for the constrained physical spring-slider task."""

CFG = {
    # The world-Y rail is visually legible and remains in the robust workspace.
    "rail_axis": "y",
    # Rotate the articulation root by pi about Z so positive joint travel is
    # world -Y.  The candidate positions then use the long rail from its
    # positive-Y natural end toward the camera-visible negative-Y anchor.
    "rail_direction_world": -1.0,
    "slider_root_pose": (-0.15, 0.10, 0.012),
    "slider_root_quaternion_wxyz": (0.0, 0.0, 0.0, 1.0),
    "joint_root_half_size": (0.050, 0.020, 0.012),
    # Separate collision-bearing rail, centered so its complete extent
    # y=[-0.33, 0.21] stays inside the canonical workspace.
    "rail_pose": (-0.15, -0.06, 0.012),
    "rail_half_size": (0.050, 0.270, 0.012),
    "slider_half_size": (0.035, 0.030, 0.027),
    "slider_offset": (0.0, 0.0, 0.045),
    "joint_limits": (0.0, 0.320),
    # Target compressions map to robust world-Y positions in [-0.16, 0.05].
    "target_range": (0.050, 0.260),
    "rail_jitter": 0.006,
    "anchor_pose": (-0.15, -0.275, 0.055),
    "anchor_half_size": (0.050, 0.018, 0.055),
    "spring_marker_count": 14,
    "spring_marker_start_y": 0.055,
    "spring_marker_spacing": 0.024,
    "spring_marker_lateral_offset": 0.014,
    "demonstration_settle_steps": 42,
    "demonstration_hold_steps": (30, 44),
    "return_steps": 52,
    "position_tolerance": 0.005,
    # Closed-loop evaluation checks the maximum physical compression reached
    # before the gripper clears the slider. The 5 mm tolerance above remains
    # the stricter expert-generation and phase-transition threshold.
    "evaluation_position_tolerance": 0.045,
    # Keep transfers and horizontal compression at the suite's established
    # command cap; physical push speed is tuned through rail damping below.
    "max_position_command": 0.14,
    # The lower execution damping below lets the physical slider reach the
    # requested ~3.4 cm/s without increasing the controller command cap.
    "push_max_position_command": 0.14,
    # Command past the remembered position to compensate for spring load.
    "push_command_preload": 0.030,
    # Keep a short horizontal continuation before lifting, without driving the
    # physical maximum beyond the 45 mm closed-loop evaluation tolerance.
    "post_target_push_margin": 0.006,
    "post_target_push_dwell_steps": 3,
    "pusher_face_offset": 0.044,
    "push_tcp_height": 0.058,
    "push_contact_tcp_max_height": 0.110,
    # Command above the clearance gate so replayed actions reliably clear the slider.
    "post_push_lift_command_height": 0.215,
    "post_push_lift_height": 0.18,
    # Finish near the commanded lift height and hold for stable physical states.
    "post_push_lift_position_tolerance": 0.008,
    "post_push_lift_success_margin": 0.006,
    # Require a short in-band contact dwell before the expert begins lifting.
    "pre_lift_contact_dwell_steps": 4,
    # Expert-generation success requires consecutive raised-arm, in-band states.
    "post_push_lift_stable_steps": 12,
    # Earliest expert success time, after the push, lift, and slider settling.
    "success_min_execution_steps": 280,
    "spring_stiffness": 18.0,
    "spring_damping": 5.0,
    # During robot execution, extra physical rail damping suppresses the
    # accelerating contact transient and makes the compression stroke nearly
    # uniform. The automatic observation demonstration keeps spring_damping.
    # Calibrated for an approximately 3.4 cm/s physical compression stroke.
    # These values change only the physical push resistance, leaving approach
    # motion, target sampling, and success logic unchanged.
    "execution_spring_damping": 10.0,
    "execution_spring_damping_gain_per_meter": 1300.0,
    # Once the equilibrium switches to the remembered target, use moderate
    # damping so the physical joint settles promptly instead of remaining
    # severely overdamped at large compression.
    "post_push_spring_damping": 8.0,
    "post_push_spring_damping_gain_per_meter": 0.0,
    "execution_spring_force_limit": 500.0,
    "spring_force_limit": 7.0,
}

__all__ = ["CFG"]
