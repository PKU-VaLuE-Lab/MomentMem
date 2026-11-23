"""Configuration for SD_explore_cube."""

CFG = {
    # Carried-cup workspace spanning four quadrants while retaining elbow bend.
    "layout_xy_low": (-0.16, -0.205),
    "layout_xy_high": (0.08, 0.205),
    "robot_base_xy": (-0.615, 0.0),
    "carried_horizontal_reach_range": (0.39, 0.735),
    # Include clearance for the open hand while descending to the hidden cube.
    "minimum_cup_separation": 0.180,
    "minimum_twice_triangle_area": 0.018,
    "cup_inner_radius": 0.047,
    "cup_outer_radius": 0.057,
    "cup_height": 0.078,
    "cup_wall": 0.006,
    "cup_wall_segments": 14,
    # A released cup counts as properly replaced only when its open local -Z
    # mouth points down, its rim/root remains at tabletop height, and PhysX
    # reports real cup-table support.  This prevents discarding an inspected
    # cover from becoming a spatial-memory shortcut.
    "cup_mouth_down_min_axis_z": 0.966,
    "cup_table_root_z_bounds": (-0.006, 0.020),
    "cup_table_contact_force_threshold": 0.05,
    # A checked non-target cup must also remain at its original footprint.
    # Merely standing it upright elsewhere would otherwise leave an external
    # spatial marker that can replace the intended memory.
    "cup_replacement_xy_tolerance": 0.030,
    "handle_post_half_height": 0.017,
    "handle_tcp_height": 0.123,
    "cube_half_size": 0.018,
    # A small visible gap above the table is sufficient to count as lifted.
    "cube_table_clearance": 0.005,
    "reveal_hold_steps": (14, 20),
    "memory_delay_steps": (16, 24),
    # Final target-cup parking is sampled per episode from a broad safe side
    # region instead of always choosing one fixed corner.
    "cup_side_goal_low": (-0.22, -0.225),
    "cup_side_goal_high": (0.08, 0.225),
    "cup_side_goal_min_clearance": 0.155,
}

__all__ = ["CFG"]
