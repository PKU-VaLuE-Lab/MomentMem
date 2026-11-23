"""Configuration for the physically articulated AO_hidden_pattern task."""

CFG = {
    "texture_bank_size": 8,
    "texture_grid_size": 5,
    # Broad, shallow trays keep every insert fully visible from the base camera.
    "box_inner_half_xy": (0.085, 0.080),
    "box_wall_height": 0.032,
    "box_wall_thickness": 0.006,
    "lid_thickness": 0.006,
    "lid_open_angle": 1.42,
    # The 19 cm pitch leaves a physical air gap between adjacent outer walls,
    # even under the opposing extremes of per-episode XY jitter.
    "slot_y": (-0.19, 0.0, 0.19),
    "slot_x": -0.055,
    "slot_jitter": 0.004,
    "open_settle_steps": 34,
    "open_hold_steps": (28, 42),
    "close_steps": 32,
    "closed_gap_steps": (8, 14),
    "query_raise_steps": 24,
    "query_hold_steps": (28, 42),
    "query_lower_steps": 24,
    "memory_delay_steps": (16, 24),
    # A clearly visible ten-centimetre physical displacement, reached while
    # the closed fingers remain in contact with the shallow closed-box edge.
    "push_distance": 0.10,
    "wrong_box_tolerance": 0.055,
}

__all__ = ["CFG"]
