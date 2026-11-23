"""Configuration for AST_mine_sweeper."""

CFG = {
    # World-Y rails ordered from nearer to farther relative to the robot.
    "line_x": (-0.105, 0.105),
    "rail_center_y": -0.010,
    "rail_half_length": 0.270,
    # The visible rail spans y=[-0.280, 0.260].  These are probe-centre
    # endpoints: after accounting for the 2.3-cm probe radius, its outer edge
    # comes within about 1.2 cm of each visible rail end.  The expert scans the
    # near rail low->high, transfers laterally there, then scans the far rail
    # high->low; neither rail is retraced.
    "scan_y_endpoints": (-0.245, 0.225),
    # The pickup-to-first-endpoint carry is one uninterrupted Cartesian
    # segment.  Its command is clipped by vector norm rather than per axis,
    # preserving a straight diagonal without intermediate hand-off ripples.
    "first_endpoint_direct_tolerance": 0.010,
    "trigger_y_range": (-0.190, 0.170),
    "trigger_min_separation": 0.08,
    "probe_spawn": (-0.20, 0.27),
    # A squat probe remains stable after physical release.
    "probe_radius": 0.023,
    "probe_half_height": 0.022,
    "probe_mass": 0.080,
    # Colored cubes use a reachable staging region separate from the rails and
    # away from the robot's folded-arm joint limit.
    "cube_spawn_xy_low": (-0.220, -0.130),
    "cube_spawn_xy_high": (-0.130, 0.130),
    "cube_spawn_min_separation": 0.095,
    "cube_spawn_probe_clearance": 0.100,
    "cube_spawn_rail_clearance": 0.080,
    "cube_spawn_endpoint_clearance": 0.100,
    # The straight physical scan crosses this compact reveal radius reliably.
    "reveal_distance": 0.020,
    # A high transfer over a hidden XY point must not reveal it.  With the TCP
    # at 4 cm during a scan, the grasped probe centre remains below this bound.
    "reveal_max_probe_z": 0.065,
    # Use one fixed speed for each complete rail pass. The command cap is
    # calibrated against the physical Panda/probe trajectory; validation can
    # tune this value without changing the clue-trigger logic.
    "scan_target_speed_mps": 0.10,
    "scan_max_position_command": 0.125,
    "memory_delay_steps": (16, 24),
    "placement_tolerance": 0.045,
    "placement_contact_force_threshold": 0.05,
    "placement_linear_speed_tolerance": 0.03,
    "placement_angular_speed_tolerance": 1.0,
    "placement_stable_steps": 4,
}

__all__ = ["CFG"]
